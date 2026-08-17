# Research — training & evaluation code from slavica

Archive of the training and evaluation code that produced the models currently
used by the app, recovered from the GPU server **slavica.zesoi.fer.hr**
(`/home/datice/`). Until this branch, this code existed only on the server —
`src/lumen/` docstrings referenced it (e.g. "Matches model_10_6/preprocess.py
exactly") but the scripts themselves were never committed.

These are standalone run scripts with hardcoded server paths (`/home/datice/...`),
kept as-is for provenance. Shared library code lives in `src/lumen/`; anything
here that the app needs at inference time has already been ported there
(`preprocess_fused`, `preprocess_mobile`, `model_meta.py`).

## Contents

| Directory | What it is |
|-----------|-----------|
| [`model_10_6/`](model_10_6/) | Training pipeline for the **production fused model** (`checkpoint_20260610_230527.pt`): DINOv2-S (frozen) + TinyCNN + MetaMLP late fusion, 448px, 67k images. Includes preprocessing, split generation, training script, and run logs. |
| [`mobile_eval/`](mobile_eval/) | Out-of-domain evaluation of the fused model on MILK10k **mobile close-ups**, plus the fine-tuning script that produced `checkpoint_mobile_best.pt` (the mobile model shipped in `web_app/backend/`). See `FINDINGS.md`. |
| [`baseline_eval/`](baseline_eval/) | Evaluation harness for the original LUMEN baseline model (`lumen_model.pth`) across ISIC 2019/2020 + MILK10k, and comparison tooling used to validate the repo refactor against slavica reference predictions. |
| [`patches/`](patches/) | Uncommitted working-tree changes recovered from server clones that were never pushed. |
| `requirements_slavica.txt` | `pip freeze` of the DATICE conda env on slavica (torch 2.6.0+cu124, RTX A6000) — the environment all of this ran in. |

## What was deliberately left out

- **Checkpoints and datasets** — `checkpoint_20260610_230527.pt`,
  `preprocessed448_67k/`, prediction CSVs, plots. Too large for git; they remain
  on slavica. (`checkpoint_mobile_best.pt` is already tracked in `web_app/backend/`.)
- **`~/Lumen-i-/training/*.py` (April 2026)** — old-structure notebook-to-script
  conversions (`train_melanoma.py`, `evaluate_new.py`). Superseded by the repo
  refactor (`src/lumen/training/`) and by `model_10_6/train.py`.
- Local path/parameter tweaks in the old-structure server clone (run-specific
  edits, not logic changes).
