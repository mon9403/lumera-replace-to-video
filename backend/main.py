# main.py
import os
import io
import uuid
import json
import httpx
import sqlite3
import secrets
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic_settings import BaseSettings
from openai import OpenAI
from PIL import Image

from models import ComposeResponse, TaskResponse
# from utils import save_base64_image  # not needed in this flow

# ----------------------
# Settings
# ----------------------
class Settings(BaseSettings):
    OPENAI_API_KEY: str
    PIAPI_API_KEY: str
    SPACE_HOST: str = ""  # optional fallback; we prefer the request host
    PORT: int = 8080

    class Config:
        env_file = ".env"

settings = Settings()

# ----------------------
# App & Static
# ----------------------
app = FastAPI(title="Lumera AI – Image → Kling Video")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = Path(os.getenv("OUTPUT_DIR", str(BASE_DIR / "outputs")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Serve /outputs publicly
app.mount("/outputs", StaticFiles(directory=str(OUT_DIR)), name="outputs")

# ----------------------
# OpenAI client (force official base URL)
# ----------------------
for k in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_API_HOST", "OPENAI_URL"):
    os.environ.pop(k, None)

client = OpenAI(api_key=settings.OPENAI_API_KEY, base_url="https://api.openai.com/v1")

# ----------------------
# Share-link DB
# ----------------------
DB_PATH = Path(os.getenv("DB_PATH", str(BASE_DIR / "shares.db")))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
with sqlite3.connect(DB_PATH) as _con:
    _con.execute(
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
    _con.commit()

def db():
    return sqlite3.connect(DB_PATH)

def new_slug(n=6):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    return ''.join(secrets.choice(alphabet) for _ in range(n))

# ----------------------
# Helpers
# ----------------------
def public_url(request: Request, path: str) -> str:
    """Build public URL from the CURRENT request host; ignores stale SPACE_HOST."""
    return str(request.base_url).rstrip("/") + "/" + path.lstrip("/")

MIN_SIDE = 512
MAX_SIDE = 1920

def normalize_reference_image(raw: bytes) -> bytes:
    """
    - Convert to RGB PNG
    - Ensure min side >= 512, max side <= 1920 (keeps aspect ratio)
    """
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = im.size

    scale = 1.0
    if min(w, h) < MIN_SIDE:
        scale = max(scale, MIN_SIDE / float(min(w, h)))
    if max(w, h) * scale > MAX_SIDE:
        scale = min(scale, MAX_SIDE / float(max(w, h)))

    if abs(scale - 1.0) > 1e-3:
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        im = im.resize((new_w, new_h), Image.LANCZOS)

    out = io.BytesIO()
    im.save(out, format="PNG", optimize=True)
    return out.getvalue()

def generate_kling_prompt_from_image(reference_bytes: bytes, aspect: str, duration: int, notes: Optional[str]) -> str:
    """
    Use OpenAI (gpt-4.1) to:
      - Analyze the image
      - Produce a concise, generic Kling prompt (no brand/color specifics)
    """
    import base64

    def to_data_url(b: bytes, mime="image/png"):
        return f"data:{mime};base64," + base64.b64encode(b).decode("utf-8")

    ref_data_url = to_data_url(reference_bytes)

    user_instruction = f"""
You are crafting a **Kling v2.1** animation prompt from ONE reference image.

Goal: write a cinematic animation prompt that captures the *style* and *motion* of the scene,
**without naming the exact product or giving brand-specific or color-specific details**.

1) Understand the Image Context
- Identify the type of subject (e.g. “the product”, “the object”, “the item”) **but never describe brand, color or model**.
- Summarise the environment (studio, outdoor, city, cozy room, etc.).
- Capture overall mood (warm, luxury, dreamy, energetic) without product-specific adjectives.

2) Build the Animation Prompt (concise, under ~1200 chars)
Always create MOTION in three layers:

A. Camera Dynamics (choose 1–2 moves total)
- zoom in/out, pan/tilt, orbit, dolly push/pull, rack focus

B. Scene & Subject Motion (pick 2–3 tasteful details)
- environment: fabric/smoke/fog, petals/leaves/snow/particles, waves/ripples/wind/reflections, shadows passing, neon flicker
- subject: subtle rotation/glow/shimmer of *the product*; hair/cloth reacts; light rays crossing subject

C. Lighting & Atmosphere (pick 1–2)
- golden hour glow, neon reflections, candlelight flicker, rain droplets reflections, lens flare, dreamy haze, spotlight beams

Constraints:
- Say only “the product” or “the object” when referring to the main subject.
- Do not mention brand names, logos, text, or specific colours.
- Photorealistic. Respect the overall composition and mood.
- Keep motion subtle and elegant, avoid warping.
- Output should be a SINGLE paragraph Kling prompt.

Format:
- Write as 2–3 short sentences (<= ~200 chars each), not bullets.

Target:
- Aspect: {aspect}
- Duration: {duration}s
{"- Extra creative direction: " + notes if notes else ""}
"""

    resp = client.responses.create(
        model="gpt-4.1",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_instruction},
                {"type": "input_image", "image_url": ref_data_url},
            ],
        }],
    )

    # Prefer output_text if available
    text = ""
    if hasattr(resp, "output_text") and isinstance(resp.output_text, str) and resp.output_text.strip():
        text = resp.output_text.strip()
    else:
        out = getattr(resp, "output", None) or []
        for item in out:
            itype = getattr(item, "type", None) or (isinstance(item, dict) and item.get("type"))
            if itype != "message":
                continue
            content = getattr(item, "content", None) or (isinstance(item, dict) and item.get("content")) or []
            for c in content:
                ctype = getattr(c, "type", None) or (isinstance(c, dict) and c.get("type"))
                if ctype == "output_text":
                    txt = getattr(c, "text", None) or (isinstance(c, dict) and c.get("text"))
                    if isinstance(txt, str) and txt.strip():
                        text = txt.strip()
                        break
            if text:
                break

    if not text:
        raise RuntimeError("No prompt text returned from OpenAI.")
    return text

# ----------------------
# PiAPI v1 (Kling)
# ----------------------
async def create_kling_task(prompt: str, image_url: str, aspect: str, duration: int) -> str:
    """
    Create Kling task (image-to-video) via PiAPI v1.
    POST https://api.piapi.ai/api/v1/task  with header x-api-key
    """
    url = "https://api.piapi.ai/api/v1/task"
    headers = {"x-api-key": settings.PIAPI_API_KEY, "Content-Type": "application/json"}

    version = os.getenv("KLING_VERSION", "2.1")
    mode = os.getenv("KLING_MODE", "std")

    payload = {
        "model": "kling",
        "task_type": "video_generation",
        "input": {
            "prompt": prompt or "",
            "duration": 5 if duration not in (5, 10) else duration,
            "aspect_ratio": aspect if aspect in ("16:9", "9:16", "1:1") else "9:16",
            "mode": mode if mode in ("std", "pro") else "std",
            "version": version,
            "image_url": image_url,
        },
    }

    async with httpx.AsyncClient(timeout=120) as x:
        resp = await x.post(url, headers=headers, json=payload)

    if resp.status_code != 200:
        # bubble up PiAPI body
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    data = resp.json() or {}
    d = data.get("data") or {}

    if d.get("status") == "failed":
        err = d.get("error") or {}
        raise HTTPException(status_code=402, detail=f"PiAPI error: {err}")

    task_id = d.get("task_id")
    if not task_id:
        raise HTTPException(status_code=500, detail=f"Unexpected PiAPI response (no task_id): {data}")
    return task_id

async def get_kling_task(task_id: str) -> dict:
    url = f"https://api.piapi.ai/api/v1/task/{task_id}"
    headers = {"x-api-key": settings.PIAPI_API_KEY}
    async with httpx.AsyncClient(timeout=60) as x:
        resp = await x.get(url, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()

# ----------------------
# Parser for PiAPI task JSON
# ----------------------
def _parse_kling_task(raw: dict) -> dict:
    """
    Normalize PiAPI get-task JSON to:
      status: completed|processing|pending|failed|staged|unknown|stuck
      video_url: str|None
      error_message: str|None
      created_at/started_at/ended_at: ISO strings or ""
    Handles both output.video_url and output.works[0].video.resource shapes.
    Adds client-side 'stuck' if processing > 20 min.
    """
    d = (raw or {}).get("data") or {}
    status = (d.get("status") or "").lower() or "unknown"
    out = d.get("output") or {}

    # video url
    video_url = out.get("video_url")
    if not video_url:
        works = out.get("works") or []
        if works and isinstance(works, list):
            vid = ((works[0] or {}).get("video") or {}).get("resource")
            if isinstance(vid, str) and vid:
                video_url = vid

    # error message
    err = d.get("error") or {}
    error_message = (err.get("raw_message") or err.get("message") or "").strip() or None

    # timestamps
    meta = d.get("meta") or {}
    created_at = meta.get("created_at") or ""
    started_at = meta.get("started_at") or ""
    ended_at = meta.get("ended_at") or ""

    # stuck detection
    STUCK_MINUTES = 20
    try:
        if status == "processing" and started_at:
            t0 = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            mins = (datetime.now(timezone.utc) - t0).total_seconds() / 60.0
            if mins >= STUCK_MINUTES:
                status = "stuck"
                if not error_message:
                    error_message = f"Kling task has been processing for ~{int(mins)} minutes. Likely queue/backlog or upstream issue."
    except Exception:
        pass

    return {
        "status": status,
        "video_url": video_url,
        "error_message": error_message,
        "created_at": created_at,
        "started_at": started_at,
        "ended_at": ended_at,
    }

# ----------------------
# API Endpoints
# ----------------------
@app.post("/api/compose", response_model=ComposeResponse)
async def compose(
    request: Request,
    reference: UploadFile = File(..., description="Reference image only"),
    aspect: str = Form("9:16"),
    duration: int = Form(5),
    notes: Optional[str] = Form(None),
):
    # 1) Read & normalize image
    raw = await reference.read()
    try:
        png_bytes = normalize_reference_image(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"image_normalize failed: {e}")

    # 2) Save PNG we serve publicly
    file_id = uuid.uuid4().hex
    ref_name = f"reference_{file_id}.png"
    ref_path = OUT_DIR / ref_name
    with open(ref_path, "wb") as f:
        f.write(png_bytes)

    # 3) Build public URL from THIS request's host
    reference_url = public_url(request, f"/outputs/{ref_name}")
    print("DEBUG reference_url →", reference_url)

    # 4) Minimal reachability check (warm static & catch 404 early)
    try:
        async with httpx.AsyncClient(timeout=20) as x:
            r = await x.get(reference_url)
        if r.status_code != 200 or not r.content:
            raise HTTPException(status_code=500, detail=f"reference_url_unreachable: {reference_url} (status {r.status_code})")
        ctype = r.headers.get("content-type", "")
        if "image" not in ctype:
            print("WARN: content-type for reference_url is", ctype)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"reference_url_check failed: {e}")

    # 5) Draft Kling prompt (generic)
    try:
        kling_prompt = generate_kling_prompt_from_image(png_bytes, aspect, duration, notes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"prompt_from_image failed: {e}")

    # 6) Create Kling task
    try:
        task_id = await create_kling_task(kling_prompt, reference_url, aspect, duration)
    except HTTPException as e:
        raise HTTPException(status_code=e.status_code, detail=f"piapi_create failed: {e.detail}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"piapi_create failed: {e}")

    return ComposeResponse(composite_url=reference_url, task_id=task_id, prompt=kling_prompt)

@app.get("/api/task/{task_id}", response_model=TaskResponse)
async def task_status(task_id: str, raw: Optional[int] = Query(None)):
    """
    Normalized status + video_url + raw JSON in 'detail'.
    Add ?raw=1 to inspect exactly what PiAPI returned.
    """
    raw_json = await get_kling_task(task_id)
    parsed = _parse_kling_task(raw_json)

    status = parsed["status"]
    video_url = parsed["video_url"]

    # If failed/stuck, inject a client hint so UI can show a helpful message
    if status in ("failed", "stuck") and parsed["error_message"]:
        raw_json = {
            **raw_json,
            "_client_hint": {
                "status_parsed": status,
                "error_message": parsed["error_message"],
                "created_at": parsed["created_at"],
                "started_at": parsed["started_at"],
                "ended_at": parsed["ended_at"],
            }
        }

    return TaskResponse(status=status, video_url=video_url, detail=json.dumps(raw_json))

# ---- Share (kept same; composite_url now = reference_url) ----
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
    host = settings.SPACE_HOST.rstrip("/") if settings.SPACE_HOST else ""
    url = host if host else ""  # try env first (for share links)
    if not url:
        # fallback: relative, caller can prefix with current origin
        url = ""
    return {"slug": slug, "url": f"{(url or '').rstrip('/')}/v/{slug}" if url else f"/v/{slug}"}

@app.get("/v/{slug}", response_class=HTMLResponse)
async def view_share(slug: str):
    with db() as c:
        cur = c.execute("SELECT task_id, composite_url, prompt FROM shares WHERE slug=?", (slug,))
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    task_id, image_url, prompt = row
    raw = await get_kling_task(task_id)
    parsed = _parse_kling_task(raw)
    video_url = parsed["video_url"] or ""

    def esc(s: str) -> str:
        return (s or "").replace("<","&lt;").replace(">","&gt;")

    html = f"""
    <!doctype html>
    <html><head><meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Lumera Share</title>
    <style>body{{font-family:ui-sans-serif,system-ui,Arial;padding:24px;max-width:720px;margin:auto}}
    .card{{border:1px solid #e5e7eb;border-radius:16px;padding:16px;margin:12px 0;box-shadow:0 2px 8px rgba(0,0,0,.04)}}
    img,video{{max-width:100%;border-radius:12px}}</style></head>
    <body>
      <h1>🔗 Share – Lumera</h1>
      <div class="card"><h3>🖼️ Reference</h3><img src="{image_url}" /></div>
      <div class="card"><h3>🎬 Kling Prompt</h3><pre style="white-space:pre-wrap">{esc(prompt)}</pre></div>
      <div class="card"><h3>📺 Video</h3>{(f'<video src="{video_url}" controls playsinline></video>' if video_url else '<em>Rendering… refresh later.</em>')}</div>
      <div class="card"><a href="/">Create your own →</a></div>
      <script>if(!{str(bool(video_url)).lower()})setTimeout(()=>location.reload(),8000);</script>
    </body></html>
    """
    return HTMLResponse(html)

# ---- Debug routes ----
@app.get("/api/debug/host")
def debug_host(request: Request):
    return {
        "SPACE_HOST": settings.SPACE_HOST,
        "request_base_url": str(request.base_url),
    }

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
    test_url = "https://api.piapi.ai/api/v1/task/does-not-exist"
    headers = {"x-api-key": settings.PIAPI_API_KEY}
    try:
        async with httpx.AsyncClient(timeout=20) as x:
            resp = await x.get(test_url, headers=headers)
        return {"ok": resp.status_code in (401, 404), "status": resp.status_code, "url": test_url, "body": resp.text[:300]}
    except Exception as e:
        return {"ok": False, "where": "piapi", "error": str(e)}

# ---- Frontend mounting (if present) ----
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
