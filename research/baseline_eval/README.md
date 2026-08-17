# baseline_eval — LUMEN baseline evaluation & refactor validation

Evaluation harness for the original LUMEN baseline model
(`/home/datice/models/lumen_model.pth`, CustomCNN @ 224px) across all raw
images from ISIC 2019, ISIC 2020 and MILK10k, plus tooling used to verify that
the repo refactor reproduces slavica's reference predictions.

Adapted from the server's `note.txt`:

| Script | What it does |
|--------|-------------|
| `evaluate.py` | Preprocess (modified resize order to fit all datasets, final 224×224, parallel) + inference over a folder of raw images. Writes `predictions.csv`. Filename prefix encodes the source dataset/year. |
| `analyse.py` | Per-dataset (2019 / 2020 / MILK10k) and combined metrics from `all_GT.csv` (ground truth joined with predictions). |
| `analyse_refactor.py` | Same metric blocks as `analyse.py`, but for the refactored `scripts/predict.py` output (`predictions_refactor.csv`). |
| `compare_runs.py` | Compares slavica's reference `predictions.csv` against the refactor's output: row counts, binary agreement, probability deltas, flipped predictions. |

Data artifacts (`all_imgs/`, `all_GT.csv`, `predictions*.csv`) remain on
slavica at `/home/datice/data/baseline_eval/` — not committed (size).

The `preprocess → 896 → hair removal → 448` order established by
`evaluate.py` is what `model_10_6/preprocess.py` (and therefore
`lumen.preprocessing.preprocess_fused`) replicates.
