from __future__ import annotations

import math
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from PIL import Image
from torchvision import transforms

from .data import IMAGENET_MEAN, IMAGENET_STD
from .utils import is_image_file


def build_eval_transform(
    img_size: int,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
) -> transforms.Compose:
    resize = int(round(img_size / 0.875))
    return transforms.Compose(
        [
            transforms.Resize(resize, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def iter_image_paths(root_dir: str | Path) -> List[Tuple[Path, int]]:
    root_dir = Path(root_dir)
    out: List[Tuple[Path, int]] = []
    for cls in ["0", "1"]:
        d = root_dir / cls
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and is_image_file(p.name):
                out.append((p, int(cls)))
    return out


def generate_positions(length: int, window: int, stride: int) -> List[int]:
    if length <= window:
        return [0]
    stride = max(1, stride)
    n = int(math.floor((length - window) / stride)) + 1
    pos = [i * stride for i in range(n)]
    last = length - window
    if pos[-1] != last:
        pos.append(last)
    return pos


def crop_tile(im: Image.Image, x0: int, y0: int, w: int, h: int) -> Image.Image:
    W, H = im.size
    if W < w or H < h:
        return im.resize((w, h), resample=Image.Resampling.NEAREST)
    return im.crop((x0, y0, x0 + w, y0 + h))


@torch.no_grad()
def max_prob_tiled(
    model: torch.nn.Module,
    im: Image.Image,
    tfm: transforms.Compose,
    device: torch.device,
    window_sizes: Sequence[int],
    stride_ratio: float,
    tile_batch: int = 64,
    early_stop_threshold: float | None = None,
) -> float:
    W, H = im.size
    best = 0.0
    for win in window_sizes:
        stride = int(round(win * stride_ratio))
        xs = generate_positions(W, win, stride)
        ys = generate_positions(H, win, stride)

        tiles: List[torch.Tensor] = []
        for y0 in ys:
            for x0 in xs:
                tile = crop_tile(im, x0, y0, win, win)
                tiles.append(tfm(tile))

        for i in range(0, len(tiles), max(int(tile_batch), 1)):
            batch = torch.stack(tiles[i : i + tile_batch], dim=0).to(device)
            logits = model(batch)
            prob1 = torch.softmax(logits, dim=1)[:, 1]
            max_in_batch = float(prob1.max().item())
            if max_in_batch > best:
                best = max_in_batch
                if early_stop_threshold is not None and best >= float(early_stop_threshold):
                    return best
    return best
