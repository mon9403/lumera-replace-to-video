import base64
from PIL import Image
from typing import Tuple

def save_base64_image(b64: str, out_path: str) -> None:
    if b64.startswith("data:image"):
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    with open(out_path, "wb") as f:
        f.write(raw)

def fit_to_canvas(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return image.copy().resize(size, Image.LANCZOS)
