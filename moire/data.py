from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageFile
import random
import re
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

from .utils import is_image_file


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Be tolerant to partially-written/truncated files; some datasets contain a few bad samples.
ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass(frozen=True)
class DataConfig:
    img_size: int = 224
    mean: Tuple[float, float, float] = IMAGENET_MEAN
    std: Tuple[float, float, float] = IMAGENET_STD


def pil_rgb_loader(fp: str) -> Image.Image:
    # Must be top-level to be picklable when DataLoader uses multiprocessing (num_workers > 0).
    try:
        with Image.open(fp) as im:
            im = im.convert("RGB")
            im.load()
            return im
    except Exception:
        # Fallback: try OpenCV decoding (sometimes succeeds where PIL fails).
        try:
            import cv2
            import numpy as np

            data = np.fromfile(fp, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                raise OSError("cv2.imdecode returned None")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img)
        except Exception:
            raise


def build_transforms(cfg: DataConfig, train: bool) -> transforms.Compose:
    if train:
        t = [
            transforms.RandomResizedCrop(
                cfg.img_size, scale=(0.6, 1.0), ratio=(0.75, 1.3333333333)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    else:
        resize = int(round(cfg.img_size / 0.875))
        t = [
            transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(cfg.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
        ]
    return transforms.Compose(t)


class PairedTrainTransform:
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self.crop = transforms.RandomResizedCrop(
            cfg.img_size, scale=(0.6, 1.0), ratio=(0.75, 1.3333333333)
        )
        self.color_jitter = transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02)

    def __call__(self, clean: Image.Image, moire: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        i, j, h, w = self.crop.get_params(clean, self.crop.scale, self.crop.ratio)
        clean = TF.resized_crop(
            clean,
            i,
            j,
            h,
            w,
            size=(self.cfg.img_size, self.cfg.img_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        )
        moire = TF.resized_crop(
            moire,
            i,
            j,
            h,
            w,
            size=(self.cfg.img_size, self.cfg.img_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        )
        if random.random() < 0.5:
            clean = TF.hflip(clean)
            moire = TF.hflip(moire)

        fn_idx, b, c, s, h = transforms.ColorJitter.get_params(
            self.color_jitter.brightness,
            self.color_jitter.contrast,
            self.color_jitter.saturation,
            self.color_jitter.hue,
        )
        for fn_id in fn_idx:
            if fn_id == 0 and b is not None:
                clean = TF.adjust_brightness(clean, b)
                moire = TF.adjust_brightness(moire, b)
            elif fn_id == 1 and c is not None:
                clean = TF.adjust_contrast(clean, c)
                moire = TF.adjust_contrast(moire, c)
            elif fn_id == 2 and s is not None:
                clean = TF.adjust_saturation(clean, s)
                moire = TF.adjust_saturation(moire, s)
            elif fn_id == 3 and h is not None:
                clean = TF.adjust_hue(clean, h)
                moire = TF.adjust_hue(moire, h)

        clean_t = TF.to_tensor(clean)
        moire_t = TF.to_tensor(moire)
        clean_t = TF.normalize(clean_t, mean=self.cfg.mean, std=self.cfg.std)
        moire_t = TF.normalize(moire_t, mean=self.cfg.mean, std=self.cfg.std)
        return clean_t, moire_t


class SafeImageFolder(datasets.ImageFolder):
    """
    ImageFolder that skips unreadable/corrupted images by resampling another index.
    This prevents training from crashing when a dataset contains a few bad files.
    """

    def __init__(self, *args, max_retries: int = 10, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_retries = int(max_retries)

    def __getitem__(self, index: int):
        last_exc: Exception | None = None
        for _ in range(max(self.max_retries, 1)):
            try:
                return super().__getitem__(index)
            except Exception as e:
                last_exc = e
                # Resample another index (avoid infinite loops on a single bad file).
                index = torch.randint(low=0, high=len(self.samples), size=(1,)).item()
        raise OSError(f"Failed to load image after {self.max_retries} retries") from last_exc


def build_imagefolder(path: str | Path, train: bool, cfg: DataConfig) -> datasets.ImageFolder:
    p = Path(path)
    tfm = build_transforms(cfg, train=train)
    return SafeImageFolder(root=str(p), transform=tfm, loader=pil_rgb_loader)


class BinaryFolderDataset(Dataset[Tuple[torch.Tensor, int]]):
    """
    Dataset for the standard {0,1} folder layout, ignoring any other subfolders.

    This avoids accidental multi-class labels if the directory contains extra folders
    like `0_synth_*` or other artifacts.
    """

    def __init__(self, root: str | Path, train: bool, cfg: DataConfig, max_retries: int = 10) -> None:
        self.root = Path(root)
        self.max_retries = int(max_retries)
        self.tfm = build_transforms(cfg, train=train)

        items: List[Tuple[Path, int]] = []
        for cls in ["0", "1"]:
            d = self.root / cls
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                if p.is_file() and is_image_file(p.name):
                    items.append((p, int(cls)))

        if not items:
            raise RuntimeError(f"No images found under {self.root}/{{0,1}}")
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        last_exc: Exception | None = None
        for _ in range(max(self.max_retries, 1)):
            p, y = self.items[index]
            try:
                im = pil_rgb_loader(str(p))
                x = self.tfm(im)
                return x, int(y)
            except Exception as e:
                last_exc = e
                index = torch.randint(low=0, high=len(self.items), size=(1,)).item()
        raise OSError(f"Failed to load image after {self.max_retries} retries") from last_exc


def build_binary_folder(root: str | Path, train: bool, cfg: DataConfig) -> Dataset[Tuple[torch.Tensor, int]]:
    return BinaryFolderDataset(root=root, train=train, cfg=cfg)


class PairedCleanMoireDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """
    For datasets laid out as:
      train/0/<base>.png
      train/1/<base>__vXX__pattern_..._moire.png

    Returns (clean_tensor, moire_tensor) with *synchronized* random crops/augs.
    """

    def __init__(self, train_dir: str | Path, cfg: DataConfig, max_retries: int = 10) -> None:
        self.train_dir = Path(train_dir)
        self.cfg = cfg
        self.max_retries = int(max_retries)
        self.tfm = PairedTrainTransform(cfg)

        clean_dir = self.train_dir / "0"
        moire_dir = self.train_dir / "1"
        if not clean_dir.is_dir() or not moire_dir.is_dir():
            raise FileNotFoundError(f"Expected {clean_dir} and {moire_dir} to exist")

        clean_by_stem: Dict[str, Path] = {p.stem: p for p in clean_dir.glob("*.png")}
        rx = re.compile(r"^(?P<base>.+?)__v\d+__pattern_.*_moire$", re.I)
        pairs: List[Tuple[Path, Path]] = []
        for mp in sorted(moire_dir.glob("*.png")):
            m = rx.match(mp.stem)
            if not m:
                continue
            base = m.group("base")
            cp = clean_by_stem.get(base)
            if cp is None:
                continue
            pairs.append((cp, mp))

        if not pairs:
            raise RuntimeError(f"No clean/moire pairs found under {self.train_dir}")
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        last_exc: Exception | None = None
        for _ in range(max(self.max_retries, 1)):
            cp, mp = self.pairs[index]
            try:
                clean = pil_rgb_loader(str(cp))
                moire = pil_rgb_loader(str(mp))
                return self.tfm(clean, moire)
            except Exception as e:
                last_exc = e
                index = torch.randint(low=0, high=len(self.pairs), size=(1,)).item()
        raise OSError(f"Failed to load paired image after {self.max_retries} retries") from last_exc


def build_paired_train_loader(
    train_dir: str | Path,
    cfg: DataConfig,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
) -> DataLoader:
    train_ds = PairedCleanMoireDataset(train_dir=train_dir, cfg=cfg)
    return DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )


def build_loaders(
    train_dir: str | Path,
    val_dir: str | Path,
    cfg: DataConfig,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = build_binary_folder(train_dir, train=True, cfg=cfg)
    val_ds = build_binary_folder(val_dir, train=False, cfg=cfg)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader
