# Conversation Context (moire)

This file captures the working context and key commands/results from the current Codex session so it can be reused on another machine or future runs.

## Dataset / Folder Notes

- Training/validation datasets use ImageFolder layout:
  - `train/0` = class 0 (no moire)
  - `train/1` = class 1 (moire)
  - `validate/0`, `validate/1`
- Merchant test set was moved into repo root:
  - `商户图片测试/正常` and `商户图片测试/翻拍`
  - Current counted images: total 59 (正常 39, 翻拍 20)
- Older `runs/*/infer_config.json` files may contain absolute paths from earlier location (`/Users/karl/Downloads/商户图片测试`). These are only logs and do not affect current code execution.
- `商户图片测试/rename_map_1768287298.csv` contains old absolute paths (data artifact, not used by code).

## One-off Data Move Performed

Moved ReCTS images into training class 0 with continued naming scheme:

- Source: `/Users/karl/Downloads/ReCTS/img` (20,000 images)
- Target: `/Users/karl/Documents/moire/train/0`
- Naming scheme in `train/0` was `image_test_part001_%08d_target.<ext>`
- Existing max index was `00009989`; new files were renamed to:
  - `image_test_part001_00009990_target.jpg` .. `image_test_part001_00029989_target.jpg`

## Code Added / Changed

### New modules

- `moire/infer_folder.py`
  - Run tiled inference on an arbitrary folder (optionally recursive), output `preds.csv`, optional threshold sweeps, optional copy/move into `out_dir/{0,1}/...`.
- `moire/dump_worst_tiles.py`
  - Given `infer_folder` `preds.csv` + a checkpoint, find and export the max-probability tile (and optionally min tile) for the most extreme FP/FN samples.

### Compatibility change (CNN backbones)

- `moire/model.py`
  - `ViTFFTClassifier._vit_features()` generalized to also support CNN backbones from timm (e.g. `resnet34`) by global-average-pooling 4D feature maps to `[B,C]`.
  - Kept attribute name `self.vit` for backward-compatible checkpoints.

### SRM residual branch support

Added optional SRM residual branch as a supplement to ViT (+ existing FFT branch):

- `moire/model.py`
  - `ModelConfig` extended with: `srm_enabled`, `srm_kernels`, `srm_scale_init`.
  - Fixed SRM-like 5x5 kernels via grouped conv (`groups=3`), frozen weights (no training).
  - Residual feature branch (`SRMBranch`) is trained; combined into ViT features via learnable scalar `srm_scale`:
    - `vit_feat = vit_feat + srm_scale * srm_feat`
  - SRM uses denormalized RGB in `[0,1]`, scaled to `*255` before filtering; residual is passed through TLU clamp.
- `moire/utils.py`
  - `CheckpointMeta` extended to store SRM flags so inference can reconstruct exact architecture.
- `moire/train.py`
  - New CLI flags: `--srm/--no-srm`, `--srm-kernels`, `--srm-scale-init`
  - New CLI flag: `--init-ckpt` to initialize from an existing checkpoint (`strict=False`).
  - Meta now records SRM info.
- `moire/infer_validate.py`, `moire/infer_folder.py`, `moire/dump_worst_tiles.py`
  - Load SRM flags from checkpoint meta to build matching model at inference time.

## Training Commands Used

### Base (exp3) (no SRM)

Trained previously; used for comparisons and as SRM init base:

- Checkpoint: `runs/exp3/best.pt`

### exp3 + SRM (initialized from exp3)

Example training command used conceptually:

```sh
.venv/bin/python -m moire.train \
  --train-dir train --val-dir validate \
  --save-dir runs/exp3_srm1 \
  --model deit_small_patch16_224 \
  --train-mode paired \
  --srm --srm-kernels 3 --srm-scale-init 0.1 \
  --init-ckpt runs/exp3/best.pt --no-pretrained \
  --epochs 6 --batch-size 64 --lr 3e-4 --weight-decay 0.05 \
  --val-mode tiled --val-window-sizes 224,320,448 --best-metric auc \
  --device mps
```

Produced:
- `runs/exp3_srm1/best.pt`
- `runs/exp3_srm1/last.pt`

### ViT-Base run (vitb_exp1)

Trained previously:
- `runs/vitb_exp1/best.pt` / `last.pt`
- `model = vit_base_patch16_224`, paired mode, `batch_size=16`, `lr=5e-5`, `weight_decay=0.1`, tiled val windows `224,320,448`.

### ResNet-34 run (resnet34_exp1)

```sh
.venv/bin/python -m moire.train \
  --train-dir train --val-dir validate \
  --save-dir runs/resnet34_exp1 \
  --model resnet34 --train-mode paired \
  --val-mode tiled --val-window-sizes 224,320,448 \
  --best-metric auc --device mps
```

Note: validation metrics during training looked suspicious (often all predicted positive at threshold=0.5); merchant test performance was poor.

## Merchant Test Inference Commands

Use relative path now that folder is in repo root:

```sh
.venv/bin/python -m moire.infer_folder \
  --input-dir "商户图片测试" \
  --ckpt runs/exp3_srm1/last.pt \
  --device mps \
  --window-sizes 224,320,448 \
  --no-early-stop \
  --threshold 0.5 \
  --sweep-thresholds 0.5,0.6,0.7,0.8,0.9,0.95,0.96,0.97,0.98,0.99 \
  --out-dir runs/exp3_srm1/merchant_test_infer_root \
  --action none
```

## Key Observations / Results (Merchant Test)

On the current merchant test set (59 images: 正常39/翻拍20):

- exp3 (no SRM, best.pt):
  - AUC ~ 0.799
  - At threshold 0.96: TP 15, FP 12, TN 27, FN 5 (F1 ~ 0.638, Acc ~ 0.712)
- exp3_srm1 (SRM, last.pt):
  - AUC ~ 0.826
  - At threshold 0.5: TP 13, FP 5, TN 34, FN 7 (F1 ~ 0.684, Acc ~ 0.797)
  - At threshold 0.96: TP 11, FP 2, TN 37, FN 9 (F1 ~ 0.667, Acc ~ 0.814)

Interpretation:
- SRM version (last.pt) significantly reduced false positives on merchant set; AUC improved on this set.

## Utilities

### Export worst tiles for inspection

Example for ViT-Base on merchant set:

```sh
.venv/bin/python -m moire.dump_worst_tiles \
  --input-dir "商户图片测试" \
  --preds-csv runs/vitb_exp1/merchant_test_infer_ws224_320_448/preds.csv \
  --ckpt runs/vitb_exp1/best.pt \
  --out-dir runs/vitb_exp1/merchant_test_worst_tiles \
  --device mps --window-sizes 224,320,448 --stride-ratio 0.5 \
  --tile-batch 64 --k 5 --save-min-tile
```

Outputs tiles under:
- `runs/vitb_exp1/merchant_test_worst_tiles/FP_actual_normal_pred_flip/`
- `runs/vitb_exp1/merchant_test_worst_tiles/FN_actual_flip_pred_normal/`

