import os
import uuid
import json
import httpx
import sqlite3
import secrets
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
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

# CORS (adjust origins later for production)
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

FRONTEND_DIR = (BASE_DIR.parent / "frontend")
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:
    @app.get("/", response_class=HTMLResponse)
    async def root():
        return "<h1>Lumera API</h1><p>Frontend folder not found. Try /api/health or /docs.</p>"

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
    Compose the replacement product into the reference scene via Responses API
    using gpt-4.1 + image_generation tool. Returns base64 PNG (no data URL).
    """
    import base64

    def to_data_url(b: bytes, mime="image/png"):
        return f"data:{mime};base64," + base64.b64encode(b).decode("utf-8")

    ref_data_url = to_data_url(reference_bytes)
    rep_data_url = to_data_url(replacement_bytes)

    prompt = (
        "Replace the old product in the scene (first image) with the new product from the second image. "
        "Match perspective, lighting, shadows, and reflections. Maintain the original background. "
        "Place the new product naturally where the old one was; remove any remnants. Output a clean photorealistic composite."
    )

    # Use the Responses API with the image_generation tool
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
        tools=[{"type": "image_generation", "image": {"size": target_size}}],
    )

    # Extract the generated image (base64)
    # The SDK returns tool output items; find the first image.
    # Depending on SDK minor versions, the shape can differ slightly.
    out = resp.output if hasattr(resp, "output") else resp  # be tolerant
    # Walk the structure to find a b64 image
    b64 = None
    try:
        for item in out:  # list of output items
            if getattr(item, "type", "") == "message":
                for c in getattr(item, "content", []):
                    if getattr(c, "type", "") == "output_image":
                        # c.image is an object with b64_json
                        img_obj = getattr(c, "image", None)
                        if img_obj and getattr(img_obj, "b64_json", None):
                            b64 = img_obj.b64_json
                            break
            if b64:
                break
    except Exception:
        pass

    if not b64:
        # Fallback for alt shapes (older/newer SDKs)
        # Try to look for dicts
        try:
            for item in out:
                if isinstance(item, dict) and item.get("type") == "message":
                    for c in item.get("content", []):
                        if c.get("type") == "output_image":
                            img_obj = c.get("image") or {}
                            b64 = img_obj.get("b64_json")
                            break
                if b64:
                    break
        except Exception:
            pass

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

Structure:
- Camera: one primary move + subtle secondary motion.
- Environment motion: 1–2 tasteful effects (particles, light rays, gentle wind).
- Subject motion: small parallax, slow spin, or reflective shimmer—avoid warping.
- Lighting: specify source, color temp, contrast, and any glow.
- Safety: photorealistic, no text, no extra objects.
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
    reference: UploadFile = File(..., description="Reference scene with old product"),
    replacement: UploadFile = File(..., description="New product photo"),
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

    scene_notes = notes or "Product placed naturally into the reference scene; preserve realism."
    try:
        prompt = draft_kling_prompt_with_openai(scene_notes, aspect, duration)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prompt generation failed: {e}")

    try:
        task_id = await create_kling_task(prompt, composite_url, aspect, duration)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Kling task creation failed: {e}")

    return ComposeResponse(composite_url=composite_url, task_id=task_id, prompt=prompt)

@app.get("/api/task/{task_id}", response_model=TaskResponse)
async def task_status(task_id: str):
    data = await get_kling_task(task_id)
    status = data.get("status") or data.get("state") or "unknown"
    video_url = data.get("video_url") or (data.get("result", {}) if isinstance(data.get("result"), dict) else {}).get("url")
    detail = json.dumps(data)
    return TaskResponse(status=status, video_url=video_url, detail=detail)

# ---- Sharing ----
@app.post("/api/share")
async def create_share(payload: dict):
    task_id = payload.get("task_id")
    composite_url = payload.get("composite_url")
    prompt = payload.get("prompt")
    if not (task_id and composite_url):
        raise HTTPException(status_code=400, detail="task_id and composite_url required")

    slug = new_slug()
    with db() as c:
        c.execute(
            "INSERT INTO shares(slug, task_id, composite_url, prompt) VALUES(?,?,?,?)",
            (slug, task_id, composite_url, prompt or ""),
        )
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
    status = (data.get("status") or data.get("state") or "unknown").lower()
    video_url = data.get("video_url") or (data.get("result", {}) if isinstance(data.get("result"), dict) else {}).get("url") or ""

    title = "Lumera AI – Product Replace to Video"
    desc = f"Status: {status.upper()}"

    def esc(s): 
        return (s or "").replace("<","&lt;").replace(">","&gt;")

    html = f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>{esc(title)}</title>
      <meta property="og:title" content="{esc(title)}" />
      <meta property="og:description" content="{esc(desc)}" />
      <meta property="og:type" content="video.other" />
      <meta property="og:image" content="{video_url or composite_url}" />
      <meta property="twitter:card" content="summary_large_image" />
      <meta property="twitter:title" content="{esc(title)}" />
      <meta property="twitter:description" content="{esc(desc)}" />
      <meta property="twitter:image" content="{video_url or composite_url}" />
      <style>
        body{{font-family:ui-sans-serif,system-ui,Arial;padding:24px;max-width:720px;margin:auto}}
        .card{{border:1px solid #e5e7eb;border-radius:16px;padding:16px;margin:12px 0;box-shadow:0 2px 8px rgba(0,0,0,.04)}}
        img,video{{max-width:100%;border-radius:12px}}
        pre{{white-space:pre-wrap;font-family:ui-monospace,Menlo,monospace}}
      </style>
    </head>
    <body>
      <h1>🔗 Share – Lumera AI</h1>
      <div class="card">
        <h3>🖼️ Composite</h3>
        <img src="{composite_url}" alt="Composite" />
      </div>
      <div class="card">
        <h3>🎬 Prompt</h3>
        <pre>{esc(prompt)}</pre>
      </div>
      <div class="card">
        <h3>📺 Video</h3>
        {f'<video controls playsinline src="{video_url}"></video>' if video_url else '<em>Rendering… refresh in a moment.</em>'}
      </div>
      <div class="card">
        <a href="/">Create your own →</a>
      </div>
      <script>
        const hasVideo = {str(bool(video_url)).lower()};
        if (!hasVideo) setTimeout(() => location.reload(), 5000);
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

# Health
@app.get("/api/health")
async def health():
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)
