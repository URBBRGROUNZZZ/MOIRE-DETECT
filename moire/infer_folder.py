from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from .model import ModelConfig, ViTFFTClassifier
from .tiling import build_eval_transform, max_prob_tiled
from .utils import ensure_dir, get_device, is_image_file, load_checkpoint, write_json


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def iter_images(root: Path, recursive: bool, skip_dir: Optional[Path]) -> List[Path]:
    it: Iterable[Path] = root.rglob("*") if recursive else root.glob("*")
    out: List[Path] = []
    for p in it:
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if not is_image_file(p.name):
            continue
        if skip_dir is not None:
            try:
                p.relative_to(skip_dir)
                continue
            except ValueError:
                pass
        out.append(p)
    return sorted(out)


def summarize_by_group(
    rows: Sequence[Dict[str, object]], thresholds: Sequence[float]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    groups: Dict[str, List[Tuple[float, int]]] = {}
    for r in rows:
        group = str(r.get("group", ""))
        prob = float(r["prob"])
        groups.setdefault(group, []).append((prob, 1))

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for group, items in groups.items():
        probs = [p for p, _ in items]
        for t in thresholds:
            pred1 = sum(1 for p in probs if p >= float(t))
            pred0 = len(probs) - pred1
            out.setdefault(group, {})[str(t)] = {
                "n": float(len(probs)),
                "pred0": float(pred0),
                "pred1": float(pred1),
                "pred1_rate": float(pred1 / max(len(probs), 1)),
            }
    return out


def safe_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    for i in range(1, 10_000):
        cand = parent / f"{stem}__dup{i:04d}{suffix}"
        if not cand.exists():
            return cand
    raise RuntimeError(f"Unable to find free filename for {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default="")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--window-sizes", type=str, default="224,320,448,512")
    ap.add_argument("--stride-ratio", type=float, default=0.5)
    ap.add_argument("--tile-reduce", type=str, default="max", choices=["max", "mean", "topk_mean", "p95"])
    ap.add_argument("--tile-topk", type=int, default=5)
    ap.add_argument("--freq-attn-scale", type=float, default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument(
        "--sweep-thresholds",
        type=str,
        default="",
        help="optional, e.g. 0.5,0.9,0.96 to print per-group splits without re-running inference",
    )
    ap.add_argument(
        "--early-stop-threshold",
        type=float,
        default=None,
        help="early-stop threshold (default: same as --threshold)",
    )
    ap.add_argument("--no-early-stop", dest="early_stop", action="store_false", default=True)
    ap.add_argument("--tile-batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-recursive", dest="recursive", action="store_false", default=True)
    ap.add_argument("--action", type=str, choices=["none", "copy", "move"], default="none")
    ap.add_argument("--flat", action="store_true", default=False, help="do not preserve relative paths in out-dir")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"--input-dir is not a directory: {input_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else None
    if args.action != "none":
        if out_dir is None:
            raise SystemExit("--out-dir is required when --action is not 'none'")
        ensure_dir(out_dir / "0")
        ensure_dir(out_dir / "1")
        ensure_dir(out_dir)

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
        input_highpass_enabled=bool(meta.get("input_highpass_enabled", False)),
        input_highpass_ksize=int(meta.get("input_highpass_ksize", 0) or 0),
        input_highpass_sigma=float(meta.get("input_highpass_sigma", 0.0) or 0.0),
        input_highpass_scale_init=float(meta.get("input_highpass_scale_init", 0.0) or 0.0),
    )
    model = ViTFFTClassifier(model_cfg)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    if args.freq_attn_scale is not None and hasattr(model, "freq_attn_scale"):
        with torch.no_grad():
            model.freq_attn_scale.copy_(
                torch.tensor(float(args.freq_attn_scale), dtype=model.freq_attn_scale.dtype, device=device)
            )
    model.eval()

    tfm = build_eval_transform(model_cfg.img_size, mean=model_cfg.mean, std=model_cfg.std)
    window_sizes = sorted(set(parse_int_list(args.window_sizes)))
    early_stop_threshold = float(
        args.threshold if args.early_stop_threshold is None else args.early_stop_threshold
    )

    skip_dir: Optional[Path] = None
    if out_dir is not None:
        try:
            out_dir.relative_to(input_dir)
            skip_dir = out_dir
        except ValueError:
            skip_dir = None

    files = iter_images(input_dir, recursive=bool(args.recursive), skip_dir=skip_dir)
    if args.limit and args.limit > 0:
        files = files[: int(args.limit)]

    rows: List[Dict[str, object]] = []
    n_pred0 = 0
    n_pred1 = 0

    for p in tqdm(files, desc="infer", total=len(files)):
        rel = p.relative_to(input_dir)
        group = rel.parts[0] if len(rel.parts) > 1 else ""
        with Image.open(p) as im:
            im = im.convert("RGB")
        prob = max_prob_tiled(
            model=model,
            im=im,
            tfm=tfm,
            device=device,
            window_sizes=window_sizes,
            stride_ratio=float(args.stride_ratio),
            tile_batch=int(args.tile_batch),
            tile_reduce=args.tile_reduce,
            tile_topk=args.tile_topk,
            early_stop_threshold=(early_stop_threshold if args.early_stop else None),
        )
        pred = 1 if float(prob) >= float(args.threshold) else 0
        if pred == 1:
            n_pred1 += 1
        else:
            n_pred0 += 1

        rows.append(
            {
                "path": str(rel),
                "group": group,
                "prob": float(prob),
                "pred": int(pred),
            }
        )

        if out_dir is not None and args.action != "none":
            if args.flat:
                dst_rel = Path(p.name)
            else:
                dst_rel = rel
            dst = out_dir / str(pred) / dst_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst = safe_destination(dst)
            if args.action == "move":
                shutil.move(str(p), str(dst))
            else:
                shutil.copy2(str(p), str(dst))

    print(
        {
            "n": len(rows),
            "pred0": n_pred0,
            "pred1": n_pred1,
            "threshold": float(args.threshold),
            "window_sizes": window_sizes,
            "stride_ratio": float(args.stride_ratio),
            "tile_reduce": str(args.tile_reduce),
            "tile_topk": int(args.tile_topk),
            "freq_attn_scale": (float(args.freq_attn_scale) if args.freq_attn_scale is not None else None),
            "ckpt": str(args.ckpt),
            "action": str(args.action),
            "out_dir": (str(out_dir) if out_dir is not None else ""),
        }
    )

    thresholds = [float(args.threshold)]
    if args.sweep_thresholds:
        thresholds = parse_float_list(args.sweep_thresholds)
    by_group = summarize_by_group(rows, thresholds=thresholds)
    if by_group:
        print({"by_group": by_group})

    if out_dir is None:
        out_dir = ensure_dir(input_dir / "_infer_out")
    else:
        ensure_dir(out_dir)

    pred_csv = out_dir / "preds.csv"
    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "group", "prob", "pred"])
        w.writeheader()
        w.writerows(rows)

    write_json(
        out_dir / "infer_config.json",
        {
            "input_dir": str(input_dir),
            "ckpt": str(args.ckpt),
            "device": str(args.device),
            "window_sizes": window_sizes,
            "stride_ratio": float(args.stride_ratio),
            "tile_reduce": str(args.tile_reduce),
            "tile_topk": int(args.tile_topk),
            "freq_attn_scale": (float(args.freq_attn_scale) if args.freq_attn_scale is not None else None),
            "threshold": float(args.threshold),
            "early_stop": bool(args.early_stop),
            "early_stop_threshold": (float(early_stop_threshold) if args.early_stop else None),
            "tile_batch": int(args.tile_batch),
            "recursive": bool(args.recursive),
            "action": str(args.action),
            "flat": bool(args.flat),
            "model": asdict(model_cfg),
        },
    )
    print({"preds_csv": str(pred_csv)})


if __name__ == "__main__":
    main()
