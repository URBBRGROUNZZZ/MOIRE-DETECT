# moire（二分类：是否包含摩尔纹）

目录结构（当前已兼容你的数据组织方式）：

```
train/
  0/  # 无摩尔纹
  1/  # 有摩尔纹
validate/
  0/
  1/
```

## 1) 环境

已按你的要求使用虚拟环境（Python 3.10）：

```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt
```

## 2) 数据分析

```sh
.venv/bin/python -m moire.analyze_data --train-dir train --val-dir validate
```

## 模型架构与工作模式（ViT + 频域分析）

整体是“两路特征融合”的二分类模型：空间域用 ViT 提取语义/纹理特征，频域用 FFT 幅度谱提取周期性线索，最后融合判别是否包含摩尔纹。

**网络架构（`moire/model.py`）**

- 空间分支（ViT）：来自 `timm`（默认 `deit_small_patch16_224`），输出全局特征向量 `vit_feat`。
- 频域分支（FFT→CNN）：把图像转灰度后做 2D FFT，取 `log(1+|FFT|)` 并 `fftshift` 得到频谱图 `fft_map`，再用轻量 CNN + 全局池化得到向量 `freq_feat`。
- 频域引导注意力（可选）：对高频成分做反变换得到空间注意力图，给 ViT token 施加加权，强调可能含摩尔纹的区域。
- 融合头：拼接 `[vit_feat, freq_feat]` 后用 MLP 输出 2 类 logits。

**训练/验证（`moire/train.py`）**

- 训练使用 ImageFolder：`train/0`=无摩尔纹，`train/1`=有摩尔纹；loss 为 `CrossEntropyLoss`，优化器 `AdamW`。
- 会保存 `runs/.../best.pt`（包含模型权重与 meta：`img_size/freq_size/mean/std` 等），推理时自动复原配置。

**多尺度切片推理（`moire/infer_validate.py`）**

- 对大图用多个窗口大小滑动裁剪（例如 `224,320,448,512`），每个 tile 单独推理得到 `P(moire=1)`。
- 图级概率取所有 tile 的 `max_prob`（只要有一块检测到摩尔纹就判 1），并支持 `threshold` 提前停止加速。

## 3) 训练（ViT + 频域分支融合）

```sh
.venv/bin/python -m moire.train \\
  --train-dir train --val-dir validate \\
  --model deit_small_patch16_224 --img-size 224 --freq-size 128 \\
  --epochs 10 --batch-size 64 --lr 3e-4 \\
  --save-dir runs/exp1
```

默认会用与最终目标一致的 **tiled 验证**（对 `validate/` 大图做多窗口滑动、取最大概率），并据此保存 `best.pt`。如需更快但不一致的验证方式，可显式设置 `--val-mode crop`。

频域引导注意力默认开启，可用 `--no-freq-attn` 关闭，或用 `--freq-attn-low-freq-ratio/--freq-attn-scale-init` 调整强度。

如需更稳健的 tile 聚合，可以改为 `topk_mean` 或 `p95`：

```sh
.venv/bin/python -m moire.train \
  --train-dir train --val-dir validate \
  --val-tile-reduce topk_mean --val-tile-topk 5 \
  --save-dir runs/exp1
```

### 成对训练（paired）

成对训练要求文件名匹配规则（仅 PNG）：

```
train/0/<base>.png
train/1/<base>__vXX__pattern_..._moire.png
```

示例：
```sh
.venv/bin/python -m moire.train \
  --train-dir train --val-dir validate \
  --train-mode paired \
  --model deit_small_patch16_224 --img-size 224 --freq-size 128 \
  --val-tile-reduce topk_mean --val-tile-topk 5 \
  --save-dir runs/exp_vit_paired_freqattn
```

### Swin（本地权重）训练

若网络访问 HuggingFace 超时，可先手动下载 `pytorch_model.bin`，然后用 `--no-pretrained --init-ckpt`：

```sh
.venv/bin/python -m moire.train \
  --train-dir train --val-dir validate \
  --train-mode paired \
  --model swin_tiny_patch4_window7_224 --img-size 224 --freq-size 128 \
  --no-pretrained \
  --init-ckpt /Users/karl/Downloads/pytorch_model.bin \
  --val-tile-reduce topk_mean --val-tile-topk 5 \
  --save-dir runs/exp_swin_paired_freqattn
```

`--init-ckpt` 会自动识别 timm backbone 权重并只加载到主干，避免与自定义分支冲突。

### SwinV2-Base (256)

```sh
.venv/bin/python -m moire.train \
  --train-dir train --val-dir validate \
  --train-mode paired \
  --model swinv2_base_window8_256 --img-size 256 --freq-size 128 \
  --val-tile-reduce topk_mean --val-tile-topk 5 \
  --batch-size 8 --num-workers 4 --device cuda \
  --save-dir runs/exp_swinv2_base_paired_freqattn
```

### 常用模型与尺寸示例

ViT (224):
```sh
.venv/bin/python -m moire.train \
  --train-dir train --val-dir validate \
  --model deit_small_patch16_224 --img-size 224 --freq-size 128 \
  --save-dir runs/exp_vit224
```

ViT (384):
```sh
.venv/bin/python -m moire.train \
  --train-dir train --val-dir validate \
  --model deit_small_patch16_384 --img-size 384 --freq-size 192 \
  --val-window-sizes 384,512,640 \
  --save-dir runs/exp_vit384
```

Swin-T (224):
```sh
.venv/bin/python -m moire.train \
  --train-dir train --val-dir validate \
  --model swin_tiny_patch4_window7_224 --img-size 224 --freq-size 128 \
  --save-dir runs/exp_swin_tiny
```

### 关键配置项速查

- `--model`: timm 模型名（如 `deit_small_patch16_224` / `swin_tiny_patch4_window7_224`）
- `--img-size`: 输入尺寸；ViT/Swin 需是 patch size 的整数倍
- `--freq-size`/`--freq-dim`: 频域分支尺寸与通道
- `--freq-attn*`: 频域引导注意力开关与强度（默认开启）
- `--val-window-sizes`: tiled 验证的窗口大小列表
- `--val-tile-reduce`: tile 聚合方式（`max`/`mean`/`topk_mean`/`p95`）
- `--val-tile-topk`: `topk_mean` 的 k
- `--no-pretrained --init-ckpt`: 使用本地权重（避免下载）

## 4) 在 validate 上做多尺度切片推理验证

```sh
.venv/bin/python -m moire.infer_validate \\
  --val-dir validate \\
  --ckpt runs/exp1/best.pt \\
  --window-sizes 224,320,448,512 \\
  --stride-ratio 0.5 --threshold 0.5
```

## 5) 对任意文件夹做推理（含子目录）

```sh
.venv/bin/python -m moire.infer_folder \
  --input-dir "商户图片测试" \
  --ckpt runs/exp1/best.pt \
  --window-sizes 224,320,448,512 \
  --tile-reduce topk_mean --tile-topk 5
```
