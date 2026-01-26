from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from PIL import Image

from .data import DataConfig, build_binary_folder, build_loaders, build_paired_train_loader
from .metrics import compute_binary_metrics
from .model import ModelConfig, ViTFFTClassifier
from .utils import (
    CheckpointMeta,
    ensure_dir,
    get_device,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    write_json,
)
from .tiling import build_eval_transform, iter_image_paths, max_prob_tiled


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> Tuple[float, List[float], List[int]]:
    model.eval()
    y_prob: List[float] = []
    y_true: List[int] = []
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        prob1 = torch.softmax(logits, dim=1)[:, 1]
        pred = (prob1 >= 0.5).long()
        correct += int((pred == y).sum().item())
        total += int(y.numel())
        y_true.extend(y.detach().cpu().numpy().astype(int).tolist())
        y_prob.extend(prob1.detach().cpu().numpy().astype(float).tolist())
    acc = correct / max(total, 1)
    return acc, y_prob, y_true


@torch.no_grad()
def evaluate_tiled(
    model: nn.Module,
    val_dir: str | Path,
    device: torch.device,
    img_size: int,
    mean,
    std,
    window_sizes: List[int],
    stride_ratio: float,
    tile_batch: int,
    early_stop_threshold: float | None,
    limit: int,
) -> Tuple[List[float], List[int]]:
    tfm = build_eval_transform(img_size=img_size, mean=mean, std=std)
    items = iter_image_paths(val_dir)
    if limit and limit > 0:
        # keep roughly balanced
        half = int(np.ceil(limit / 2))
        items0 = [(p, y) for p, y in items if y == 0][:half]
        items1 = [(p, y) for p, y in items if y == 1][:half]
        items = (items0 + items1)[:limit]

    y_prob: List[float] = []
    y_true: List[int] = []
    for p, y in tqdm(items, desc="val(tiled)", leave=False):
        with Image.open(p) as im:
            im = im.convert("RGB")
        prob = max_prob_tiled(
            model=model,
            im=im,
            tfm=tfm,
            device=device,
            window_sizes=window_sizes,
            stride_ratio=stride_ratio,
            tile_batch=tile_batch,
            early_stop_threshold=early_stop_threshold,
        )
        y_prob.append(float(prob))
        y_true.append(int(y))
    return y_prob, y_true


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    running = 0.0
    n = 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running += float(loss.item()) * int(y.size(0))
        n += int(y.size(0))
    return running / max(n, 1)


def train_one_epoch_paired(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    rank_lambda: float,
    rank_margin: float,
) -> float:
    model.train()
    running = 0.0
    n = 0
    for clean, moire in tqdm(loader, desc="train(paired)", leave=False):
        clean = clean.to(device, non_blocking=True)
        moire = moire.to(device, non_blocking=True)
        b = int(clean.size(0))

        x = torch.cat([clean, moire], dim=0)
        y = torch.cat(
            [
                torch.zeros((b,), dtype=torch.long, device=device),
                torch.ones((b,), dtype=torch.long, device=device),
            ],
            dim=0,
        )

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss_ce = criterion(logits, y)

        logits_clean = logits[:b]
        logits_moire = logits[b:]
        score_clean = logits_clean[:, 1] - logits_clean[:, 0]
        score_moire = logits_moire[:, 1] - logits_moire[:, 0]
        loss_rank = F.softplus(float(rank_margin) - (score_moire - score_clean)).mean()

        loss = loss_ce + float(rank_lambda) * loss_rank
        loss.backward()
        optimizer.step()

        running += float(loss.item()) * int(2 * b)
        n += int(2 * b)
    return running / max(n, 1)


def limit_loader(loader: DataLoader, limit: int, shuffle: bool, seed: int) -> DataLoader:
    if limit <= 0:
        return loader
    ds = loader.dataset
    limit = min(limit, len(ds))
    g = torch.Generator()
    g.manual_seed(int(seed))
    idx = torch.randperm(len(ds), generator=g).tolist()[:limit]
    subset = Subset(ds, idx)
    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=shuffle,
        num_workers=loader.num_workers,
        pin_memory=getattr(loader, "pin_memory", False),
        drop_last=loader.drop_last,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", type=str, default="train")
    ap.add_argument("--val-dir", type=str, default="validate")
    ap.add_argument("--save-dir", type=str, default="runs/exp")
    ap.add_argument("--model", type=str, default="deit_small_patch16_224")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--freq-size", type=int, default=128)
    ap.add_argument("--freq-dim", type=int, default=256)
    ap.add_argument("--srm", dest="srm_enabled", action="store_true", default=False)
    ap.add_argument("--no-srm", dest="srm_enabled", action="store_false")
    ap.add_argument("--srm-kernels", type=int, default=3)
    ap.add_argument("--srm-scale-init", type=float, default=0.0)
    ap.add_argument("--gabor", dest="gabor_enabled", action="store_true", default=False)
    ap.add_argument("--no-gabor", dest="gabor_enabled", action="store_false")
    ap.add_argument("--gabor-kernels", type=int, default=4, help="number of orientations")
    ap.add_argument("--gabor-ksize", type=int, default=9)
    ap.add_argument("--gabor-sigma", type=float, default=2.0)
    ap.add_argument("--gabor-lambda", dest="gabor_lambda", type=float, default=4.0)
    ap.add_argument("--gabor-gamma", type=float, default=0.5)
    ap.add_argument("--gabor-psi", type=float, default=0.0)
    ap.add_argument("--gabor-scale-init", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--train-mode", type=str, default="single", choices=["single", "paired"])
    ap.add_argument("--pair-rank-lambda", type=float, default=0.5)
    ap.add_argument("--pair-rank-margin", type=float, default=1.0)
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--val-limit", type=int, default=0)
    ap.add_argument("--limit-seed", type=int, default=123)
    ap.add_argument("--val-mode", type=str, default="tiled", choices=["tiled", "crop"])
    ap.add_argument("--val-window-sizes", type=str, default="224,320,448")
    ap.add_argument("--val-stride-ratio", type=float, default=0.5)
    ap.add_argument(
        "--val-threshold",
        type=float,
        default=0.5,
        help="only used to compute thresholded metrics (f1/precision/recall/accuracy); AUC is threshold-free",
    )
    ap.add_argument("--val-tile-batch", type=int, default=64)
    ap.add_argument(
        "--best-metric",
        type=str,
        default="auc",
        choices=["auc", "f1", "accuracy", "recall", "precision"],
        help="metric used to select best.pt; prefer auc to avoid arbitrary thresholds",
    )
    ap.add_argument(
        "--val-early-stop-threshold",
        type=float,
        default=None,
        help="optional: stop tile scan once prob >= this value (speeds up; can slightly distort AUC)",
    )
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--pretrained", action="store_true", default=True)
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    ap.add_argument(
        "--init-ckpt",
        type=str,
        default="",
        help="optional: initialize model weights from a checkpoint (loads state_dict with strict=False)",
    )
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)
    save_dir = ensure_dir(args.save_dir)

    data_cfg = DataConfig(img_size=args.img_size)
    if args.train_mode == "paired":
        train_loader = build_paired_train_loader(
            args.train_dir,
            cfg=data_cfg,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_ds = build_binary_folder(args.val_dir, train=False, cfg=data_cfg)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
    else:
        train_loader, val_loader = build_loaders(
            args.train_dir,
            args.val_dir,
            cfg=data_cfg,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
    train_loader = limit_loader(train_loader, limit=args.train_limit, shuffle=True, seed=args.limit_seed)
    val_loader = limit_loader(val_loader, limit=args.val_limit, shuffle=False, seed=args.limit_seed + 1)

    model_cfg = ModelConfig(
        model_name=args.model,
        img_size=args.img_size,
        freq_size=args.freq_size,
        freq_dim=args.freq_dim,
        num_classes=2,
        mean=data_cfg.mean,
        std=data_cfg.std,
        pretrained=args.pretrained,
        srm_enabled=bool(args.srm_enabled),
        srm_kernels=int(args.srm_kernels),
        srm_scale_init=float(args.srm_scale_init),
        gabor_enabled=bool(args.gabor_enabled),
        gabor_kernels=int(args.gabor_kernels),
        gabor_ksize=int(args.gabor_ksize),
        gabor_sigma=float(args.gabor_sigma),
        gabor_lambda=float(args.gabor_lambda),
        gabor_gamma=float(args.gabor_gamma),
        gabor_psi=float(args.gabor_psi),
        gabor_scale_init=float(args.gabor_scale_init),
    )
    model = ViTFFTClassifier(model_cfg).to(device)

    if args.init_ckpt:
        ckpt = load_checkpoint(args.init_ckpt, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(
            {
                "init_ckpt": str(args.init_ckpt),
                "missing_keys": len(missing),
                "unexpected_keys": len(unexpected),
            }
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    meta = CheckpointMeta(
        model_name=model_cfg.model_name,
        img_size=model_cfg.img_size,
        freq_size=model_cfg.freq_size,
        freq_dim=model_cfg.freq_dim,
        num_classes=model_cfg.num_classes,
        mean=model_cfg.mean,
        std=model_cfg.std,
        srm_enabled=bool(model_cfg.srm_enabled),
        srm_kernels=int(model_cfg.srm_kernels) if model_cfg.srm_enabled else 0,
        srm_scale_init=float(model_cfg.srm_scale_init) if model_cfg.srm_enabled else 0.0,
        gabor_enabled=bool(model_cfg.gabor_enabled),
        gabor_kernels=int(model_cfg.gabor_kernels) if model_cfg.gabor_enabled else 0,
        gabor_ksize=int(model_cfg.gabor_ksize) if model_cfg.gabor_enabled else 0,
        gabor_sigma=float(model_cfg.gabor_sigma) if model_cfg.gabor_enabled else 0.0,
        gabor_lambda=float(model_cfg.gabor_lambda) if model_cfg.gabor_enabled else 0.0,
        gabor_gamma=float(model_cfg.gabor_gamma) if model_cfg.gabor_enabled else 0.0,
        gabor_psi=float(model_cfg.gabor_psi) if model_cfg.gabor_enabled else 0.0,
        gabor_scale_init=float(model_cfg.gabor_scale_init) if model_cfg.gabor_enabled else 0.0,
    )
    write_json(save_dir / "config.json", {"train": vars(args), "model": asdict(model_cfg), "data": asdict(data_cfg)})

    best = -1.0
    window_sizes = [int(x.strip()) for x in args.val_window_sizes.split(",") if x.strip()]
    window_sizes = sorted(set(window_sizes))
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if args.train_mode == "paired":
            train_loss = train_one_epoch_paired(
                model,
                train_loader,
                device,
                optimizer,
                criterion,
                rank_lambda=args.pair_rank_lambda,
                rank_margin=args.pair_rank_margin,
            )
        else:
            train_loss = train_one_epoch(model, train_loader, device, optimizer, criterion)
        scheduler.step()

        if args.val_mode == "crop":
            val_acc, y_prob, y_true = evaluate(model, val_loader, device)
            metrics = compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=args.val_threshold).as_dict()
            metrics["val_mode"] = "crop"
            metrics["val_acc"] = float(val_acc)
        else:
            # tiled validation on original validate images (matches final inference objective)
            y_prob, y_true = evaluate_tiled(
                model=model,
                val_dir=args.val_dir,
                device=device,
                img_size=model_cfg.img_size,
                mean=model_cfg.mean,
                std=model_cfg.std,
                window_sizes=window_sizes,
                stride_ratio=args.val_stride_ratio,
                tile_batch=args.val_tile_batch,
                early_stop_threshold=args.val_early_stop_threshold,
                limit=args.val_limit,
            )
            metrics = compute_binary_metrics(y_true=y_true, y_prob=y_prob, threshold=args.val_threshold).as_dict()
            metrics["val_mode"] = "tiled"
            metrics["val_acc"] = float(metrics["accuracy"])

        elapsed = time.time() - t0
        lr = float(optimizer.param_groups[0]["lr"])
        log = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "lr": lr,
            "time_sec": float(elapsed),
            **metrics,
        }
        print(log)

        score = float(metrics.get(args.best_metric, float("nan")))
        if np.isnan(score):
            score = float(metrics["f1"])
        if score > best:
            best = float(score)
            save_checkpoint(save_dir / "best.pt", model, meta, epoch=epoch, best_metric=best, optimizer=optimizer)
            write_json(save_dir / "best_metrics.json", log)

    save_checkpoint(save_dir / "last.pt", model, meta, epoch=args.epochs, best_metric=best, optimizer=optimizer)
    print(f"done. best_metric={best:.6f} ckpt={save_dir/'best.pt'}")


if __name__ == "__main__":
    main()
