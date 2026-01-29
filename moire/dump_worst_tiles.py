from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from .model import ModelConfig, ViTFFTClassifier
from .tiling import build_eval_transform, crop_tile, generate_positions
from .utils import ensure_dir, get_device, load_checkpoint, write_json


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def load_preds(preds_csv: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with preds_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # infer_folder writes: path, group, prob, pred
            if "path" not in row or "group" not in row or "prob" not in row:
                continue
            rows.append(
                {
                    "path": row["path"],
                    "group": row["group"],
                    "prob": float(row["prob"]),
                }
            )
    return rows


def batched_tile_probs(
    model: torch.nn.Module,
    im: Image.Image,
    tfm,
    device: torch.device,
    win: int,
    stride_ratio: float,
    tile_batch: int,
) -> Iterable[Tuple[int, int, float]]:
    W, H = im.size
    stride = int(round(int(win) * float(stride_ratio)))
    xs = generate_positions(W, int(win), stride)
    ys = generate_positions(H, int(win), stride)

    coords: List[Tuple[int, int]] = []
    tensors: List[torch.Tensor] = []

    def flush() -> Iterable[Tuple[int, int, float]]:
        if not tensors:
            return []
        batch = torch.stack(tensors, dim=0).to(device)
        logits = model(batch)
        prob1 = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
        out = [(x0, y0, float(p)) for (x0, y0), p in zip(coords, prob1)]
        coords.clear()
        tensors.clear()
        return out

    for y0 in ys:
        for x0 in xs:
            tile = crop_tile(im, x0, y0, int(win), int(win))
            tensors.append(tfm(tile))
            coords.append((int(x0), int(y0)))
            if len(tensors) >= max(int(tile_batch), 1):
                for item in flush():
                    yield item
    for item in flush():
        yield item


def safe_name(s: str) -> str:
    # Keep it filesystem-friendly while preserving readability.
    return (
        s.replace("/", "__")
        .replace("\\", "__")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("|", "_")
    )


def _already_dumped_max(out_dir: Path, subdir: str, rel: Path) -> bool:
    prefix = safe_name(rel.as_posix()) + "__MAX__"
    return any((out_dir / subdir).glob(prefix + "*"))


def _already_dumped_min(out_dir: Path, subdir: str, rel: Path) -> bool:
    prefix = safe_name(rel.as_posix()) + "__MIN__"
    return any((out_dir / subdir).glob(prefix + "*"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=str, required=True, help="root folder used in infer_folder")
    ap.add_argument("--preds-csv", type=str, required=True, help="infer_folder output preds.csv")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--window-sizes", type=str, default="224,320,448")
    ap.add_argument("--stride-ratio", type=float, default=0.5)
    ap.add_argument("--tile-batch", type=int, default=64)
    ap.add_argument("--k", type=int, default=5, help="top-k most extreme per group")
    ap.add_argument("--which", type=str, choices=["both", "fp", "fn"], default="both")
    ap.add_argument("--save-min-tile", action="store_true", default=False)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    preds_csv = Path(args.preds_csv)
    out_dir = ensure_dir(args.out_dir)
    ensure_dir(out_dir / "FP_actual_normal_pred_flip")
    ensure_dir(out_dir / "FN_actual_flip_pred_normal")
    if args.save_min_tile:
        ensure_dir(out_dir / "FP_min_tile")
        ensure_dir(out_dir / "FN_min_tile")

    rows = load_preds(preds_csv)
    # FP candidates: actual normal group, highest prob (most moire-like).
    # FN candidates: actual flip group, lowest prob (most normal-like).
    normals_all = [r for r in rows if r.get("group") == "正常"]
    flips_all = [r for r in rows if r.get("group") == "翻拍"]
    normals_all.sort(key=lambda r: float(r["prob"]), reverse=True)
    flips_all.sort(key=lambda r: float(r["prob"]))

    def pick_existing(items: Sequence[Dict[str, object]], k: int) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        for it in items:
            rel = Path(str(it["path"]))
            if (input_dir / rel).exists():
                out.append(it)
                if len(out) >= k:
                    break
        return out

    k = max(int(args.k), 1)
    pick_fp = pick_existing(normals_all, k=k)
    pick_fn = pick_existing(flips_all, k=k)

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
    model.eval()

    tfm = build_eval_transform(model_cfg.img_size, mean=model_cfg.mean, std=model_cfg.std)
    window_sizes = sorted(set(parse_int_list(args.window_sizes)))

    def dump_one(item: Dict[str, object], kind: str) -> Dict[str, object]:
        rel = Path(str(item["path"]))
        src = input_dir / rel
        if not src.exists():
            return {"path": str(rel), "missing": True}

        max_sub = "FP_actual_normal_pred_flip" if kind == "fp" else "FN_actual_flip_pred_normal"
        min_sub = "FP_min_tile" if kind == "fp" else "FN_min_tile"
        if _already_dumped_max(out_dir, max_sub, rel) and (not args.save_min_tile or _already_dumped_min(out_dir, min_sub, rel)):
            return {"path": str(rel), "group": str(item.get("group", "")), "skipped": True}

        with Image.open(src) as im:
            im = im.convert("RGB")

        best = (-1.0, None)  # (prob, (win,x0,y0))
        worst = (1e9, None)
        target = float(item.get("prob", -1.0))
        found_target = False
        for win in window_sizes:
            for x0, y0, p in batched_tile_probs(
                model=model,
                im=im,
                tfm=tfm,
                device=device,
                win=int(win),
                stride_ratio=float(args.stride_ratio),
                tile_batch=int(args.tile_batch),
            ):
                if p > best[0]:
                    best = (p, (int(win), int(x0), int(y0)))
                    # infer_folder's image prob is the max over all tiles/windows; if we re-find it
                    # and we're not asked to also find the min tile, we can stop early.
                    if not args.save_min_tile and target >= 0.0 and best[0] >= (target - 1e-6):
                        found_target = True
                        break
                if p < worst[0]:
                    worst = (p, (int(win), int(x0), int(y0)))
            if found_target:
                break
        if found_target:
            # best already matches the recorded max; skip remaining window sizes
            pass

        out: Dict[str, object] = {
            "path": str(rel),
            "group": str(item.get("group", "")),
            "image_prob_max": float(item["prob"]),
            "tile_prob_max": float(best[0]),
            "tile_prob_min": float(worst[0]),
            "stride_ratio": float(args.stride_ratio),
            "window_sizes": window_sizes,
        }

        if best[1] is not None:
            win, x0, y0 = best[1]
            with Image.open(src) as im2:
                im2 = im2.convert("RGB")
            tile = crop_tile(im2, x0, y0, win, win)
            fname = safe_name(f"{rel.as_posix()}__MAX__win{win}__x{x0}__y{y0}__p{best[0]:.6f}.png")
            tile.save(out_dir / max_sub / fname)
            out["max_tile"] = {
                "win": win,
                "x0": x0,
                "y0": y0,
                "prob": float(best[0]),
                "file": f"{max_sub}/{fname}",
            }

        if args.save_min_tile and worst[1] is not None:
            win, x0, y0 = worst[1]
            with Image.open(src) as im2:
                im2 = im2.convert("RGB")
            tile = crop_tile(im2, x0, y0, win, win)
            fname = safe_name(f"{rel.as_posix()}__MIN__win{win}__x{x0}__y{y0}__p{worst[0]:.6f}.png")
            tile.save(out_dir / min_sub / fname)
            out["min_tile"] = {
                "win": win,
                "x0": x0,
                "y0": y0,
                "prob": float(worst[0]),
                "file": f"{min_sub}/{fname}",
            }

        return out

    results: List[Dict[str, object]] = []
    if args.which in ("both", "fp"):
        for item in tqdm(pick_fp, desc="dump_tiles(fp)"):
            results.append(dump_one(item, kind="fp"))
    if args.which in ("both", "fn"):
        for item in tqdm(pick_fn, desc="dump_tiles(fn)"):
            results.append(dump_one(item, kind="fn"))

    write_json(
        out_dir / "dump_worst_tiles_config.json",
        {
            "input_dir": str(input_dir),
            "preds_csv": str(preds_csv),
            "ckpt": str(args.ckpt),
            "device": str(args.device),
            "window_sizes": window_sizes,
            "stride_ratio": float(args.stride_ratio),
            "tile_batch": int(args.tile_batch),
            "k": int(args.k),
            "save_min_tile": bool(args.save_min_tile),
            "model": asdict(model_cfg),
        },
    )

    (out_dir / "worst_tiles.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print({"out_dir": str(out_dir), "n": len(results), "json": str(out_dir / "worst_tiles.json")})


if __name__ == "__main__":
    main()
