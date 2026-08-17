# Research — historical evaluation tooling from slavica

Historical one-off tooling recovered from the GPU server **slavica.zesoi.fer.hr**
(`/home/datice/`). The *production* code recovered alongside it has been
integrated into the repo proper and no longer lives here:

- Fused (dermatoscopic) model training → `scripts/preprocess_fused_dataset.py`,
  `scripts/make_split.py`, `scripts/train_fused.py` + `src/lumen/training/fused.py`
- Mobile model fine-tune & OOD evaluation → `scripts/train_mobile.py`,
  `scripts/eval_mobile.py`; findings and run records → `docs/training/`

The verbatim server originals are preserved in this branch's early commits.

## Contents

| Item | What it is |
|------|-----------|
| [`baseline_eval/`](baseline_eval/) | Evaluation harness for the original LUMEN baseline model across ISIC 2019/2020 + MILK10k, and the comparison tooling used to validate the repo refactor against slavica reference predictions. One-off, superseded by the fused model. |
| [`patches/`](patches/) | Uncommitted working-tree changes recovered from server clones (legacy CustomCNN 224→448 migration — partially superseded, see its README). |
| `requirements_slavica.txt` | `pip freeze` of the DATICE conda env on slavica (torch 2.6.0+cu124, RTX A6000) — the environment the shipped models were trained in. |

Not committed (size — they remain on slavica): checkpoints, preprocessed
datasets, prediction CSVs, plots.
