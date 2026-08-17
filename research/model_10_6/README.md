# model_10_6 — fused metadata model training pipeline

Training pipeline for the production model `checkpoint_20260610_230527.pt`
(2026-06-10), served by the web app via `src/lumen/model_meta.py`.

**Architecture** (see `train.py` docstring): DINOv2-S (frozen, 384-d) +
TinyCNN (trainable, AvgPool+MaxPool, 192-d) → VisionProj (576→384→256) =
vision embedding; metadata (age, sex, anatom_site; 11-dim) → MetaMLP (11→32→16);
late fusion concat(256+16) → Classifier (272→256→128→1).
Only ~722k trainable parameters on top of the frozen ~21M backbone.

## Pipeline (run in order)

1. **`preprocess.py`** — 67k images (ISIC 2019 + 2020 + MILK10k dermoscopic,
   from `final_metadata.csv`) → central square crop → resize 896 → hair removal
   (black-hat + inpaint, proportional kernel) → 448×448 LANCZOS4.
   Ported to the library as `lumen.preprocessing.preprocess_fused`.
2. **`make_split.py`** — train/val/test ≈ 52k/5.2k/10.3k via
   `StratifiedGroupKFold(13)` grouped by `patient_id`, stratified by target
   (verified: no `lesion_id` spans more than one `patient_id`).
3. **`train.py`** — AdamW, linear warmup (2 ep) + cosine (T_max=40),
   BCEJLoss (λ=0.9, tpr_weight=2.5), early stop on val loss, batch 64.

## Results (`logs/results_20260610_230527.txt`)

Best epoch 7 (val loss 1.4265), ~34 min on RTX A6000.

| Split | n | Acc | Prec | Rec (TPR) | F1 | FPR |
|-------|---|-----|------|-----------|----|-----|
| train | 51,597 | 0.887 | 0.704 | 0.934 | 0.803 | 0.128 |
| val   | 5,162  | 0.879 | 0.693 | 0.912 | 0.788 | 0.132 |
| test  | 10,326 | 0.876 | 0.687 | 0.912 | 0.784 | 0.136 |

Inputs it expects on slavica: `/home/datice/model_10_6/final_metadata.csv` and
the raw datasets under `/home/datice/data/original_data/`.
