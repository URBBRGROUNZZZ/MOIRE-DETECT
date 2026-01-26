from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from PIL import Image, ImageFile

from .utils import is_image_file


ImageFile.LOAD_TRUNCATED_IMAGES = True


def iter_image_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and is_image_file(p.name):
            yield p


def is_readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.load()
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="train", help="e.g. train or validate")
    ap.add_argument("--move-to", type=str, default="", help="optional dir to move bad images into")
    args = ap.parse_args()

    root = Path(args.root)
    bad: List[Path] = []
    for p in iter_image_files(root):
        if not is_readable_image(p):
            bad.append(p)

    print(f"scanned={root} bad={len(bad)}")
    for p in bad[:50]:
        print(f"BAD\t{p}")
    if len(bad) > 50:
        print(f"... and {len(bad) - 50} more")

    if args.move_to and bad:
        dst = Path(args.move_to)
        dst.mkdir(parents=True, exist_ok=True)
        for p in bad:
            target = dst / p.name
            target = target if not target.exists() else dst / f"{p.stem}_{p.stat().st_size}{p.suffix}"
            p.rename(target)
        print(f"moved {len(bad)} files to {dst}")


if __name__ == "__main__":
    main()

