from __future__ import annotations

import subprocess
from pathlib import Path
from shutil import which

from PIL import Image, ImageFile

from .utils import ensure_dir


ImageFile.LOAD_TRUNCATED_IMAGES = True


def open_rgb(path: str | Path, heic_cache_dir: str | Path = "runs/_heic_cache") -> Image.Image:
    """
    Open an image as RGB.

    Supports common formats via PIL. For .heic/.heif on macOS, falls back to `sips`
    conversion into a cache directory when PIL cannot decode the file.
    """
    p = Path(path)
    try:
        with Image.open(p) as im:
            im = im.convert("RGB")
            im.load()
            return im
    except Exception:
        ext = p.suffix.lower()
        if ext not in {".heic", ".heif"}:
            raise

    # HEIC/HEIF fallback: macOS `sips` conversion.
    sips = which("sips")
    if not sips:
        raise OSError(f"Unable to decode {p} (HEIC/HEIF); install a HEIC-capable Pillow plugin or enable macOS sips")

    cache_dir = ensure_dir(heic_cache_dir)
    st = p.stat()
    out_path = cache_dir / f"{p.stem}_{st.st_size}_{st.st_mtime_ns}.jpg"
    if not out_path.exists():
        subprocess.run(
            [sips, "-s", "format", "jpeg", str(p), "--out", str(out_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    with Image.open(out_path) as im:
        im = im.convert("RGB")
        im.load()
        return im

