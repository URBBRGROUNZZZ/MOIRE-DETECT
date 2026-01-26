from __future__ import annotations

import argparse
import os
import random
from collections import Counter
from pathlib import Path

from PIL import Image

from .utils import is_image_file


def list_images(root: str | Path):
    root = Path(root)
    items = []
    for cls in ["0", "1"]:
        d = root / cls
        if not d.is_dir():
            continue
        for fn in d.iterdir():
            if fn.is_file() and is_image_file(fn.name):
                items.append((fn, int(cls)))
    return items


def analyze_split(name: str, root: str | Path, sample_n: int = 500) -> None:
    items = list_images(root)
    print(f"[{name}] root={root} total={len(items)}")
    counts = Counter([y for _, y in items])
    print(" class_counts:", dict(counts))

    if not items:
        return

    sample = random.sample(items, min(sample_n, len(items)))
    sizes = []
    modes = Counter()
    bad = 0
    for p, _ in sample:
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                sizes.append(im.size)
                modes["RGB"] += 1
        except Exception:
            bad += 1

    if sizes:
        ws = [w for w, _ in sizes]
        hs = [h for _, h in sizes]
        print(" sample_size_minmax:", {"w_min": min(ws), "w_max": max(ws), "h_min": min(hs), "h_max": max(hs)})
        print(" sample_common_sizes:", Counter(sizes).most_common(10))
    print(" sample_unreadable:", bad)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", type=str, default="train")
    ap.add_argument("--val-dir", type=str, default="validate")
    ap.add_argument("--sample-n", type=int, default=500)
    args = ap.parse_args()

    analyze_split("train", args.train_dir, sample_n=args.sample_n)
    analyze_split("validate", args.val_dir, sample_n=args.sample_n)


if __name__ == "__main__":
    # Ensure deterministic-ish sampling in analysis output when needed.
    seed = int(os.environ.get("MOIRE_ANALYZE_SEED", "0"))
    random.seed(seed)
    main()

