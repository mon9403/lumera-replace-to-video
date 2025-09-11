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

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Files
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = Path(os.getenv("OUTPUT_DIR", str(BASE_DIR / "outputs")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Serve outputs statically
app.mount("/outputs", StaticFiles(directory=str(OUT_DIR)), name="outputs")

# OpenAI client
client = OpenAI(api_key=settings.OPENAI_API_KEY)

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
    import base64, json, sys

    def to_data_url(b: bytes, mime="image/png"):
        return f"data:{mime};base64," + base64.b64encode(b).decode("utf-8")

    ref_data_url = to_data_url(reference_bytes)
    rep_data_url = to_data_url(replacement_bytes)

    prompt = (
        "Replace the old product in the scene (first image) with the new product from the second image. "
        "Match perspective, lighting, shadows, and reflections. Maintain the original background. "
        f"Output a clean photorealistic composite (target size {target_size})."
    )

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

    # ---- Robust extraction: try multiple known shapes ----
    def _extract_b64(o) -> str | None:
        # 1) Newer SDK: resp.output -> list[message]; content -> list[output_image]
        out = getattr(o, "output", None)
        if out:
            for item in out:
                itype = getattr(item, "type", None) or (isinstance(item, dict) and item.get("type"))
                content = getattr(item, "content", None) or (isinstance(item, dict) and item.get("content")) or []
                if itype == "message" and content:
                    for c in content:
                        ctype = getattr(c, "type", None) or (isinstance(c, dict) and c.get("type"))
                        if ctype == "output_image":
                            img = getattr(c, "image", None) or (isinstance(c, dict) and c.get("image")) or {}
                            b64 = getattr(img, "b64_json", None) or (isinstance(img, dict) and img.get("b64_json"))
                            if b64:
                                return b64

        # 2) Some variants: resp.output[0].content is a dict with "image"
        try:
            d = o.model_dump() if hasattr(o, "model_dump") else None
        except Exception:
            d = None
        if isinstance(d, dict):
            # Walk everything to find any key named b64_json
            stack = [d]
            while stack:
                cur = stack.pop()
                if isinstance(cur, dict):
                    if "b64_json" in cur and isinstance(cur["b64_json"], str):
                        return cur["b64_json"]
                    stack.extend(cur.values())
                elif isinstance(cur, list):
                    stack.extend(cur)
        return None

    b64 = _extract_b64(resp)

    # Optional: log a compact sample of the response if extraction failed (helps once, then you can remove)
    if not b64:
        try:
            sample = resp.model_dump() if hasattr(resp, "model_dump") else {}
            # Trim huge blobs before logging
            def _trim(x):
                if isinstance(x, dict):
                    return {k: _trim(v) for k, v in list(x.items())[:10]}
                if isinstance(x, list):
                    return [_trim(v) for v in x[:5]]
                if isinstance(x, str) and len(x) > 200:
                    return x[:200] + "...(trimmed)"
                return x
            print("DEBUG compose_with_openai resp sample:", json.dumps(_trim(sample))[:4000], file=sys.stderr)
        except Exception:
            pass
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

# ----------------------
# Serve frontend (after APIs, avoids swallowing POST)
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
