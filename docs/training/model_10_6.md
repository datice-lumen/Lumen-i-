# Dermatoscopic model — training run model_10_6 (2026-06-10)

Training record for the production fused model `checkpoint_20260610_230527.pt`,
served by the web app via `src/lumen/model_meta.py`. Trained on slavica
(RTX A6000, DATICE conda env — `research/requirements_slavica.txt`).

**Architecture**: DINOv2-S (frozen, 384-d) + TinyCNN (trainable,
AvgPool+MaxPool, 192-d) → VisionProj (576→384→256) = vision embedding;
metadata (age, sex, anatom_site; 11-dim) → MetaMLP (11→32→16); late fusion
concat(256+16) → Classifier (272→256→128→1). Only ~722k trainable parameters
on top of the frozen ~21M backbone.

## Reproduce

The run scripts live in the repo (hyperparameter defaults = this run's config):

```bash
# 1. Preprocess the 67k images (ISIC 2019 + 2020 + MILK10k dermoscopic) to 448px
python scripts/preprocess_fused_dataset.py \
    --metadata final_metadata.csv \
    --images <2019_input> --images <2020_train> --images <MILK10k_input> \
    --output preprocessed448

# 2. Patient-grouped stratified split (train/val/test ≈ 52k/5.2k/10.3k)
python scripts/make_split.py --metadata final_metadata.csv

# 3. Train (AdamW, warmup+cosine, BCEJLoss λ=0.9 tpr_weight=2.5, early stop)
python scripts/train_fused.py \
    --metadata final_metadata.csv --img-dir preprocessed448 --output-dir runs/fused
```

## Results (`model_10_6_logs/results_20260610_230527.txt`)

Best epoch 7 (val loss 1.4265), ~34 min.

| Split | n | Acc | Prec | Rec (TPR) | F1 | FPR |
|-------|---|-----|------|-----------|----|-----|
| train | 51,597 | 0.887 | 0.704 | 0.934 | 0.803 | 0.128 |
| val   | 5,162  | 0.879 | 0.693 | 0.912 | 0.788 | 0.132 |
| test  | 10,326 | 0.876 | 0.687 | 0.912 | 0.784 | 0.136 |

Raw logs: [`model_10_6_logs/`](model_10_6_logs/). The original standalone run
scripts (hardcoded server paths) are preserved in this branch's early history;
the checkpoint and `final_metadata.csv`/`preprocessed448_67k` remain on slavica
at `/home/datice/model_10_6/`.

For the mobile-domain follow-up (evaluation + fine-tune), see
[`mobile_findings.md`](mobile_findings.md).
