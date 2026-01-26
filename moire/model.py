from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except Exception as e:  # pragma: no cover
    timm = None


def _fftshift2(x: torch.Tensor) -> torch.Tensor:
    # x: [B, 1, H, W]
    h = x.shape[-2]
    w = x.shape[-1]
    return torch.roll(torch.roll(x, shifts=h // 2, dims=-2), shifts=w // 2, dims=-1)


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "deit_small_patch16_224"
    img_size: int = 224
    freq_size: int = 128
    freq_dim: int = 256
    num_classes: int = 2
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    pretrained: bool = True
    # SRM residual branch (optional).
    srm_enabled: bool = False
    srm_kernels: int = 3
    srm_scale_init: float = 0.0
    # Gabor residual branch (optional).
    gabor_enabled: bool = False
    gabor_kernels: int = 4
    gabor_ksize: int = 9
    gabor_sigma: float = 2.0
    gabor_lambda: float = 4.0
    gabor_gamma: float = 0.5
    gabor_psi: float = 0.0
    gabor_scale_init: float = 0.0


class FFTBranch(nn.Module):
    def __init__(self, freq_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, freq_dim),
            nn.LayerNorm(freq_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TLU(nn.Module):
    """Truncated Linear Unit commonly used in forensics residual branches."""

    def __init__(self, threshold: float = 3.0) -> None:
        super().__init__()
        self.threshold = float(threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = float(self.threshold)
        return x.clamp(min=-t, max=t)


def _build_srm_kernels_5x5(n: int) -> torch.Tensor:
    """
    Return up to n fixed 5x5 SRM-like high-pass kernels.
    The exact set varies across papers; this small set is a commonly used starter for forensics.
    """
    k1 = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    ) / 4.0
    k2 = torch.tensor(
        [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ],
        dtype=torch.float32,
    ) / 12.0
    k3 = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    ) / 2.0
    ks = [k1, k2, k3]
    n = int(n)
    if n <= 0:
        return torch.empty((0, 5, 5), dtype=torch.float32)
    if n <= len(ks):
        return torch.stack(ks[:n], dim=0)
    # If asked for more, just repeat deterministically (better than silently failing).
    out = []
    for i in range(n):
        out.append(ks[i % len(ks)])
    return torch.stack(out, dim=0)


def _build_gabor_kernels(
    ksize: int,
    thetas: Tuple[float, ...],
    sigma: float,
    lambd: float,
    gamma: float,
    psi: float,
) -> torch.Tensor:
    half = int(ksize) // 2
    axis = torch.arange(-half, half + 1, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis)
    kernels = []
    for theta in thetas:
        x_t = xx * math.cos(theta) + yy * math.sin(theta)
        y_t = -xx * math.sin(theta) + yy * math.cos(theta)
        gb = torch.exp(-(x_t**2 + (gamma**2) * y_t**2) / (2.0 * (sigma**2)))
        gb = gb * torch.cos((2.0 * math.pi * x_t / lambd) + psi)
        gb = gb - gb.mean()
        gb = gb / (gb.norm() + 1e-6)
        kernels.append(gb)
    return torch.stack(kernels, dim=0)


class SRMBranch(nn.Module):
    def __init__(self, in_ch: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ViTFFTClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if timm is None:
            raise RuntimeError("timm is required. Please `pip install timm`.")

        self.cfg = cfg
        mean = torch.tensor(cfg.mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(cfg.std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

        # num_classes=0 -> return features for most timm models.
        # Keep attribute name `vit` for backward-compatible checkpoints (older runs used ViT backbones).
        try:
            self.vit = timm.create_model(cfg.model_name, pretrained=cfg.pretrained, num_classes=0)
        except Exception:
            self.vit = timm.create_model(cfg.model_name, pretrained=False, num_classes=0)

        vit_dim = getattr(self.vit, "num_features", None)
        if vit_dim is None:
            raise RuntimeError(f"Unable to infer feature dim for model {cfg.model_name}")

        self.freq_branch = FFTBranch(cfg.freq_dim)
        self.srm_enabled = bool(cfg.srm_enabled)
        if self.srm_enabled:
            kernels = _build_srm_kernels_5x5(int(cfg.srm_kernels))  # [K,5,5]
            # Apply kernels independently per RGB channel: out_ch = 3*K, groups=3.
            w = kernels[:, None, :, :].repeat(3, 1, 1, 1)  # [3K,1,5,5]
            self.srm = nn.Conv2d(
                in_channels=3,
                out_channels=w.shape[0],
                kernel_size=5,
                stride=1,
                padding=2,
                bias=False,
                groups=3,
            )
            with torch.no_grad():
                self.srm.weight.copy_(w)
            for p in self.srm.parameters():
                p.requires_grad = False
            self.srm_tlu = _TLU(threshold=3.0)
            self.srm_branch = SRMBranch(in_ch=int(w.shape[0]), out_dim=int(vit_dim))
            self.srm_scale = nn.Parameter(torch.tensor(float(cfg.srm_scale_init), dtype=torch.float32))

        self.gabor_enabled = bool(cfg.gabor_enabled)
        if self.gabor_enabled:
            num = max(int(cfg.gabor_kernels), 1)
            ksize = max(int(cfg.gabor_ksize), 1)
            if ksize % 2 == 0:
                ksize += 1
            sigma = max(float(cfg.gabor_sigma), 1e-6)
            lambd = max(float(cfg.gabor_lambda), 1e-6)
            gamma = max(float(cfg.gabor_gamma), 1e-6)
            psi = float(cfg.gabor_psi)
            thetas = tuple(i * math.pi / num for i in range(num))
            kernels = _build_gabor_kernels(ksize, thetas, sigma, lambd, gamma, psi)
            w = kernels[:, None, :, :].repeat(3, 1, 1, 1)  # [3K,1,ks,ks]
            self.gabor = nn.Conv2d(
                in_channels=3,
                out_channels=w.shape[0],
                kernel_size=int(ksize),
                stride=1,
                padding=int(ksize) // 2,
                bias=False,
                groups=3,
            )
            with torch.no_grad():
                self.gabor.weight.copy_(w)
            for p in self.gabor.parameters():
                p.requires_grad = False
            self.gabor_branch = SRMBranch(in_ch=int(w.shape[0]), out_dim=int(vit_dim))
            self.gabor_scale = nn.Parameter(torch.tensor(float(cfg.gabor_scale_init), dtype=torch.float32))

        self.fusion = nn.Sequential(
            nn.Linear(vit_dim + cfg.freq_dim, vit_dim),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(vit_dim, cfg.num_classes),
        )

    def _denormalize_to_unit(self, x_norm: torch.Tensor) -> torch.Tensor:
        # x_norm: [B,3,H,W] normalized by mean/std; convert back to approx [0,1].
        x = x_norm * self.std + self.mean
        return x.clamp(0.0, 1.0)

    def _compute_fft_map(self, x_norm: torch.Tensor) -> torch.Tensor:
        # x_norm: [B,3,H,W] normalized by mean/std; convert back to approx [0,1] for FFT.
        x = self._denormalize_to_unit(x_norm)
        x = F.interpolate(x, size=(self.cfg.freq_size, self.cfg.freq_size), mode="bilinear", align_corners=False)
        gray = (0.2989 * x[:, 0] + 0.5870 * x[:, 1] + 0.1140 * x[:, 2]).unsqueeze(1)  # [B,1,H,W]

        fft = torch.fft.fft2(gray, dim=(-2, -1))
        mag = torch.abs(fft)
        mag = torch.log1p(mag)
        mag = _fftshift2(mag)
        # per-sample normalization
        mag = mag - mag.mean(dim=(-2, -1), keepdim=True)
        mag = mag / (mag.std(dim=(-2, -1), keepdim=True) + 1e-6)
        return mag

    def _vit_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return a [B,C] feature vector for both ViT-like backbones (token outputs) and CNN backbones
        (spatial feature maps). Name kept for backward compatibility.
        """
        feats = self.vit.forward_features(x) if hasattr(self.vit, "forward_features") else self.vit(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        if getattr(feats, "ndim", 0) == 3:
            # timm VisionTransformer forward_head expects token sequence [B,N,C];
            # passing CLS [B,C] can collapse to [B], so only call it on token sequences.
            if hasattr(self.vit, "forward_head"):
                try:
                    return self.vit.forward_head(feats, pre_logits=True)
                except TypeError:
                    pass
            # tokens [B, N, C] -> CLS
            return feats[:, 0]
        if getattr(feats, "ndim", 0) == 4:
            # CNN feature map [B,C,H,W] -> global average pool.
            return feats.mean(dim=(-2, -1))
        return feats  # already [B,C] for many timm models

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vit_feat = self._vit_features(x)
        x_unit = None
        if self.srm_enabled or self.gabor_enabled:
            x_unit = self._denormalize_to_unit(x)
        if self.srm_enabled:
            # Scale residual to roughly match common SRM implementations that operate on 8-bit space.
            resid = self.srm_tlu(self.srm(x_unit * 255.0))
            srm_feat = self.srm_branch(resid)
            vit_feat = vit_feat + self.srm_scale * srm_feat
        if self.gabor_enabled:
            gabor = self.gabor(x_unit)
            gabor_feat = self.gabor_branch(gabor)
            vit_feat = vit_feat + self.gabor_scale * gabor_feat
        fft_map = self._compute_fft_map(x)
        freq_feat = self.freq_branch(fft_map)
        fused = torch.cat([vit_feat, freq_feat], dim=1)
        return self.fusion(fused)
