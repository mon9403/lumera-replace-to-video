import os
import uuid
import json
import httpx
import sqlite3
import secrets
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic_settings import BaseSettings
from openai import OpenAI

from models import ComposeResponse, TaskResponse
from utils import save_base64_image


# ----------------------
# Settings
# ----------------------
class Settings(BaseSettings):
    OPENAI_API_KEY: str
    PIAPI_API_KEY: str
    SPACE_HOST: str
    PORT: int = 8080

    class Config:
        env_file = ".env"


settings = Settings()

app = FastAPI(title="Lumera AI – Product Replace to Video")

# Allow cross-origin for quick testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = Path(os.getenv("OUTPUT_DIR", str(BASE_DIR / "outputs")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Serve generated outputs
app.mount("/outputs", StaticFiles(directory=str(OUT_DIR)), name="outputs")

# ✅ Force correct base_url so we never hit our own Render host by mistake
client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
    base_url="https://api.openai.com/v1"
)

# SQLite for share links
DB_PATH = Path(os.getenv("DB_PATH", str(BASE_DIR / "shares.db")))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
con = sqlite3.connect(DB_PATH)
con.execute(
    """
    CREATE TABLE IF NOT EXISTS shares (
        slug TEXT PRIMARY KEY,
        task_id TEXT,
        composite_url TEXT,
        prompt TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        expires_at DATETIME
    )
    """
)
con.commit()
con.close()


def db():
    return sqlite3.connect(DB_PATH)


def new_slug(n=6):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    return ''.join(secrets.choice(alphabet) for _ in range(n))


# ----------------------
# Helpers
# ----------------------
def compose_with_openai(reference_bytes: bytes, replacement_bytes: bytes, target_size: str = "1024x1024") -> str:
    """
    Compose via Responses API using gpt-4.1 + image_generation tool.
    Returns base64 PNG (no data URL).
    """
    import base64

    def to_data_url(b: bytes, mime="image/png"):
        return f"data:{mime};base64," + base64.b64encode(b).decode("utf-8")

    ref_data_url = to_data_url(reference_bytes)
    rep_data_url = to_data_url(replacement_bytes)

    prompt = (
        "Replace the old product in the scene (first image) with the new product from the second image. "
        "Match perspective, lighting, shadows, and reflections. Maintain the original background. "
        f"Output a clean photorealistic composite (target size {target_size})."
    )

    try:
        resp = client.responses.create(
            model="gpt-4.1",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": ref_data_url},
                    {"type": "input_image", "image_url": rep_data_url},
                ],
            }],
            tools=[{"type": "image_generation", "model": "gpt-image-1"}],
            tool_choice={"type": "image_generation"},
        )
    except Exception as e:
        # Make the error message explicit so we can see if it is a 404 from OpenAI
        status = getattr(getattr(e, "response", None), "status_code", None)
        url = getattr(getattr(getattr(e, "response", None), "request", None), "url", None)
        body = None
        try:
            body = getattr(e, "response", None).text[:300]
        except Exception:
            pass
        raise RuntimeError(f"OpenAI call failed (status={status}, url={url}) body={body}")

    # ---- Extract base64 image from the Responses output ----
    b64 = None
    out = getattr(resp, "output", None) or []
    for item in out:
        itype = getattr(item, "type", None) or (isinstance(item, dict) and item.get("type"))
        # Case 1: new schema → result field
        if itype == "image_generation_call" and (hasattr(item, "result") or (isinstance(item, dict) and "result" in item)):
            b64 = getattr(item, "result", None) or (item.get("result") if isinstance(item, dict) else None)
            if b64:
                break
        # Case 2: old schema → output_image
        content = getattr(item, "content", None) or (isinstance(item, dict) and item.get("content")) or []
        for c in content:
            ctype = getattr(c, "type", None) or (isinstance(c, dict) and c.get("type"))
            if ctype == "output_image":
                img = getattr(c, "image", None) or (isinstance(c, dict) and c.get("image")) or {}
                b64 = getattr(img, "b64_json", None) or (isinstance(img, dict) and img.get("b64_json"))
                if b64:
                    break
        if b64:
            break

    if not b64:
        raise RuntimeError("No image returned from image_generation tool")

    return b64


def draft_kling_prompt_with_openai(scene_notes: str, aspect: str, duration: int) -> str:
    sys = (
        "You are a creative director writing concise prompts for Kling v2.1. "
        "Keep under 1400 chars. Be explicit about camera moves, lighting, and pacing."
    )
    user = f"""
Create a cinematic animation prompt for Kling v2.1.
Aspect: {aspect}
Duration: {duration}s
Scene details / product context: {scene_notes}
"""
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
        temperature=0.8,
    )
    return r.choices[0].message.content.strip()


async def create_kling_task(prompt: str, image_url: str, aspect: str, duration: int) -> str:
    url = "https://api.piapi.ai/api/kling/v2.1/video"
    headers = {
        "Authorization": f"Bearer {settings.PIAPI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "image_url": image_url,
        "mode": "image_to_video",
        "aspect_ratio": aspect,
        "duration": duration,
    }
    async with httpx.AsyncClient(timeout=120) as x:
        resp = await x.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        data = resp.json()
        task_id = data.get("task_id") or data.get("id")
        if not task_id:
            raise HTTPException(status_code=500, detail=f"Unexpected PiAPI response: {data}")
        return task_id


async def get_kling_task(task_id: str) -> dict:
    url = f"https://api.piapi.ai/api/kling/v2.1/task/{task_id}"
    headers = {"Authorization": f"Bearer {settings.PIAPI_API_KEY}"}
    async with httpx.AsyncClient(timeout=60) as x:
        resp = await x.get(url, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()


# ----------------------
# API Endpoints
# ----------------------
@app.post("/api/compose", response_model=ComposeResponse)
async def compose(
    reference: UploadFile = File(...),
    replacement: UploadFile = File(...),
    aspect: str = Form("9:16"),
    duration: int = Form(5),
    notes: Optional[str] = Form(None),
):
    ref_bytes = await reference.read()
    rep_bytes = await replacement.read()

    try:
        b64 = compose_with_openai(ref_bytes, rep_bytes, target_size="1024x1024")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI composition failed: {e}")

    file_id = uuid.uuid4().hex
    composite_name = f"composite_{file_id}.png"
    composite_path = OUT_DIR / composite_name
    save_base64_image(b64, str(composite_path))

    composite_url = f"{settings.SPACE_HOST}/outputs/{composite_name}"

    scene_notes = notes or "Product placed naturally into the reference scene."
    prompt = draft_kling_prompt_with_openai(scene_notes, aspect, duration)

    task_id = await create_kling_task(prompt, composite_url, aspect, duration)
    return ComposeResponse(composite_url=composite_url, task_id=task_id, prompt=prompt)


@app.get("/api/task/{task_id}", response_model=TaskResponse)
async def task_status(task_id: str):
    data = await get_kling_task(task_id)
    status = data.get("status") or data.get("state") or "unknown"
    video_url = data.get("video_url") or (data.get("result", {}) if isinstance(data.get("result"), dict) else {}).get("url")
    detail = json.dumps(data)
    return TaskResponse(status=status, video_url=video_url, detail=detail)


@app.post("/api/share")
async def create_share(payload: dict):
    task_id = payload.get("task_id")
    composite_url = payload.get("composite_url")
    prompt = payload.get("prompt")
    if not (task_id and composite_url):
        raise HTTPException(status_code=400, detail="task_id and composite_url required")
    slug = new_slug()
    with db() as c:
        c.execute("INSERT INTO shares(slug, task_id, composite_url, prompt) VALUES(?,?,?,?)",
                  (slug, task_id, composite_url, prompt or ""))
        c.commit()
    return {"slug": slug, "url": f"{settings.SPACE_HOST}/v/{slug}"}


@app.get("/v/{slug}", response_class=HTMLResponse)
async def view_share(slug: str):
    with db() as c:
        cur = c.execute("SELECT task_id, composite_url, prompt FROM shares WHERE slug=?", (slug,))
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    task_id, composite_url, prompt = row
    data = await get_kling_task(task_id)
    video_url = data.get("video_url") or (data.get("result", {}) if isinstance(data.get("result"), dict) else {}).get("url") or ""
    return HTMLResponse(f"<h1>Share</h1><img src='{composite_url}' /><pre>{prompt}</pre><video src='{video_url}' controls></video>")


# ---- Debug endpoints to test connectivity ----
@app.get("/api/debug/openai-ping")
def openai_ping():
    try:
        r = client.responses.create(
            model="gpt-4.1",
            input=[{"role": "user", "content": [{"type": "input_text", "text": "ping"}]}]
        )
        return {"ok": True, "model": getattr(r, "model", None)}
    except Exception as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        url = getattr(getattr(getattr(e, "response", None), "request", None), "url", None)
        body = None
        try:
            body = getattr(e, "response", None).text[:300]
        except Exception:
            pass
        return {"ok": False, "where": "openai", "status": status, "url": str(url), "error": str(e), "body": body}


@app.get("/api/debug/piapi-ping")
async def piapi_ping():
    import httpx
    test_url = "https://api.piapi.ai/api/kling/v2.1/task/does-not-exist"
    headers = {"Authorization": f"Bearer {os.getenv('PIAPI_API_KEY','')}"}
    try:
        async with httpx.AsyncClient(timeout=20) as x:
            resp = await x.get(test_url, headers=headers)
        return {"ok": resp.status_code in (401, 404), "status": resp.status_code, "url": test_url, "body": resp.text[:300]}
    except Exception as e:
        return {"ok": False, "where": "piapi", "error": str(e)}


# ----------------------
# Serve frontend last (avoid swallowing /api routes)
# ----------------------
FRONTEND_DIR = (BASE_DIR.parent / "frontend")
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

    @app.get("/", response_class=FileResponse)
    async def root():
        return FileResponse(FRONTEND_DIR / "index.html")
else:
    @app.get("/", response_class=HTMLResponse)
    async def root_fallback():
        return HTMLResponse("<h1>Lumera API</h1><p>Frontend not found. Try /api/health or /docs.</p>")


@app.get("/api/health")
async def health():
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)
