# Repository Guidelines

## Project Structure & Module Organization

- `moire/`: Python package containing all training/inference code.
  - `moire/model.py`: ViT + FFT (frequency) feature fusion classifier.
  - `moire/train.py`: Training loop + checkpointing to `runs/`.
  - `moire/infer_validate.py`: Multi-scale tiled inference over `validate/`.
  - `moire/analyze_data.py`: Dataset sanity checks (counts, sizes).
- `train/` and `validate/`: ImageFolder-style datasets with class folders `0/` (no moiré) and `1/` (moiré).
- `runs/`: Experiment outputs (e.g., `best.pt`, `last.pt`, `config.json`).
- `.venv/`: Local virtual environment (do not commit).

## Build, Test, and Development Commands

Create env + install deps:
```sh
python3.10 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt
```

Analyze dataset:
```sh
.venv/bin/python -m moire.analyze_data --train-dir train --val-dir validate
```

Train (saves to `runs/exp1/`):
```sh
.venv/bin/python -m moire.train --train-dir train --val-dir validate --save-dir runs/exp1
```

Validate with multi-window tiling:
```sh
.venv/bin/python -m moire.infer_validate --val-dir validate --ckpt runs/exp1/best.pt --window-sizes 224,320,448,512
```

## Coding Style & Naming Conventions

- Python: 4-space indentation, PEP 8 naming (`snake_case` for functions/vars, `CamelCase` for classes).
- Prefer small, testable functions and explicit types where helpful (see `moire/utils.py`).
- No formatter/linter is currently configured; keep changes minimal and consistent with surrounding code.

## Testing Guidelines

- No dedicated test suite yet. Use quick smoke runs:
  - `moire.train` supports `--train-limit`/`--val-limit` for fast checks.
  - `moire.infer_validate --limit N` to validate a small subset.

## Commit & Pull Request Guidelines

- This folder may not be initialized as a Git repo. If you add Git, use clear messages (e.g., Conventional Commits: `feat: ...`, `fix: ...`).
- PRs (if used) should include: what changed, how to run, and relevant metrics/output paths (e.g., `runs/expX/best_metrics.json`).

