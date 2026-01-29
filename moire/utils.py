from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class CheckpointMeta:
    model_name: str
    img_size: int
    freq_size: int
    freq_dim: int
    num_classes: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    # Optional architectural flags (kept in meta so inference can reconstruct the same model).
    srm_enabled: bool = False
    srm_kernels: int = 0
    srm_scale_init: float = 0.0
    gabor_enabled: bool = False
    gabor_kernels: int = 0
    gabor_ksize: int = 0
    gabor_sigma: float = 0.0
    gabor_lambda: float = 0.0
    gabor_gamma: float = 0.0
    gabor_psi: float = 0.0
    gabor_scale_init: float = 0.0
    freq_attn_enabled: bool = False
    freq_attn_low_freq_ratio: float = 0.0
    freq_attn_scale_init: float = 0.0
    input_highpass_enabled: bool = False
    input_highpass_ksize: int = 0
    input_highpass_sigma: float = 0.0
    input_highpass_scale_init: float = 0.0


def save_checkpoint(
    save_path: str | Path,
    model: torch.nn.Module,
    meta: CheckpointMeta,
    epoch: int,
    best_metric: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    payload: Dict[str, Any] = {
        "meta": asdict(meta),
        "epoch": epoch,
        "best_metric": best_metric,
        "state_dict": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, str(save_path))


def load_checkpoint(
    ckpt_path: str | Path, map_location: str | torch.device = "cpu"
) -> Dict[str, Any]:
    return torch.load(str(ckpt_path), map_location=map_location)


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def is_image_file(name: str) -> bool:
    ext = os.path.splitext(name.lower())[1]
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}
