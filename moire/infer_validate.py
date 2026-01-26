from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from PIL import Image
from tqdm import tqdm

from .metrics import compute_binary_metrics
from .model import ModelConfig, ViTFFTClassifier
from .utils import get_device, load_checkpoint
from .tiling import build_eval_transform, iter_image_paths, max_prob_tiled


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", type=str, default="validate")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--window-sizes", type=str, default="224,320,448,512")
    ap.add_argument("--stride-ratio", type=float, default=0.5)
    ap.add_argument("--tile-reduce", type=str, default="max", choices=["max", "mean", "topk_mean", "p95"])
    ap.add_argument("--tile-topk", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5, help="classification threshold for metrics")
    ap.add_argument(
        "--early-stop-threshold",
        type=float,
        default=None,
        help="early-stop threshold (default: same as --threshold)",
    )
    ap.add_argument("--no-early-stop", dest="early_stop", action="store_false", default=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tile-batch", type=int, default=64)
    ap.add_argument(
        "--sweep-thresholds",
        type=str,
        default="",
        help="optional, e.g. 0.3,0.4,0.5,0.6 to print metrics sweep without re-running inference",
    )
    args = ap.parse_args()

    device = get_device(args.device)
    ckpt = load_checkpoint(args.ckpt, map_location="cpu")
    meta = ckpt["meta"]

    model_cfg = ModelConfig(
        model_name=meta["model_name"],
        img_size=int(meta["img_size"]),
        freq_size=int(meta["freq_size"]),
        freq_dim=int(meta["freq_dim"]),
        num_classes=int(meta["num_classes"]),
        mean=tuple(meta["mean"]),
        std=tuple(meta["std"]),
        pretrained=False,
        srm_enabled=bool(meta.get("srm_enabled", False)),
        srm_kernels=int(meta.get("srm_kernels", 0) or 0),
        srm_scale_init=float(meta.get("srm_scale_init", 0.0) or 0.0),
        gabor_enabled=bool(meta.get("gabor_enabled", False)),
        gabor_kernels=int(meta.get("gabor_kernels", 0) or 0),
        gabor_ksize=int(meta.get("gabor_ksize", 0) or 0),
        gabor_sigma=float(meta.get("gabor_sigma", 0.0) or 0.0),
        gabor_lambda=float(meta.get("gabor_lambda", 0.0) or 0.0),
        gabor_gamma=float(meta.get("gabor_gamma", 0.0) or 0.0),
        gabor_psi=float(meta.get("gabor_psi", 0.0) or 0.0),
        gabor_scale_init=float(meta.get("gabor_scale_init", 0.0) or 0.0),
        freq_attn_enabled=bool(meta.get("freq_attn_enabled", False)),
        freq_attn_low_freq_ratio=float(meta.get("freq_attn_low_freq_ratio", 0.0) or 0.0),
        freq_attn_scale_init=float(meta.get("freq_attn_scale_init", 0.0) or 0.0),
    )
    model = ViTFFTClassifier(model_cfg)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    model.eval()

    tfm = build_eval_transform(model_cfg.img_size, mean=model_cfg.mean, std=model_cfg.std)
    window_sizes = parse_int_list(args.window_sizes)
    window_sizes = sorted(set(window_sizes))
    items = iter_image_paths(args.val_dir)
    if args.limit and args.limit > 0:
        per = (int(args.limit) + 1) // 2
        items0 = [(p, y) for p, y in items if y == 0][:per]
        items1 = [(p, y) for p, y in items if y == 1][:per]
        items = (items0 + items1)[: args.limit]

    y_true: List[int] = []
    y_prob: List[float] = []
    early_stop_threshold = float(args.threshold if args.early_stop_threshold is None else args.early_stop_threshold)
    for p, y in tqdm(items, desc="infer", total=len(items)):
        with Image.open(p) as im:
            im = im.convert("RGB")
        prob = max_prob_tiled(
            model=model,
            im=im,
            tfm=tfm,
            device=device,
            window_sizes=window_sizes,
            stride_ratio=args.stride_ratio,
            tile_batch=args.tile_batch,
            tile_reduce=args.tile_reduce,
            tile_topk=args.tile_topk,
            early_stop_threshold=(early_stop_threshold if args.early_stop else None),
        )
        y_true.append(int(y))
        y_prob.append(float(prob))

    if args.sweep_thresholds:
        for t in parse_float_list(args.sweep_thresholds):
            res = compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=t)
            out = res.as_dict()
            out["threshold"] = float(t)
            print(out)
    else:
        res = compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=float(args.threshold))
        print(res.as_dict())


if __name__ == "__main__":
    main()
