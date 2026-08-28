# Skin Check — Multimodal Skin Lesion Screening

**🇬🇧 English** | [🇭🇷 Hrvatski](README.hr.md)

A multimodal deep learning system that estimates the probability that a skin lesion is malignant from a **dermoscopic image or a smartphone photo** plus optional patient context (age, sex, body site), wrapped in a web application that shows every step of its reasoning.

**Authors:** Filip Hlup, Jurica Jerinić, Tomislav Matanović, Karlo Raštegorac
**Supervisor:** Asst. Prof. Krešimir Križanović, PhD — University of Zagreb, Faculty of Electrical Engineering and Computing

🌐 **Live app:** https://datice-skin-checker.onrender.com/ · 📖 **Full write-up:** [project wiki](https://github.com/datice1/skin-check/wiki)

## Overview

- **Data.** 67,085 dermoscopic images merged from ISIC 2019, ISIC 2020 and MILK10k (24.6% malignant), with a strict patient-level train/val/test split (10/13, 1/13, 2/13) that was verified to keep every lesion in a single subset.
- **Model.** A frozen **DINOv2-S** backbone (global context, 384-d) runs in parallel with a small trainable **TinyCNN** (local texture, 192-d); the fused 576-d vector is projected to 256-d and concatenated with a 16-d **MetaMLP** encoding of age, sex and anatomical site. Only 722,513 parameters are trained.
- **Loss.** `BCE + λ·(soft FPR + w·(1 − soft TPR))` with λ = 0.9, w = 2.5 — a soft, asymmetrically weighted Youden index that penalises a missed malignancy 2.5× more than a false alarm.
- **Mobile adaptation.** Applied to phone close-ups, the dermoscopic model's sensitivity collapses from 0.912 to 0.559. Disabling hair removal and fine-tuning the heads on MILK10k smartphone images (DINOv2 stays frozen) brings it to 0.925 on an unseen mobile test set.
- **Web app.** Vue 3 + FastAPI. A single request streams the processing steps over SSE: square crop → hair removal (dermoscopic mode only) → skin-tone estimate (ITA → Fitzpatrick) → prediction → Grad-CAM. A DINOv2-based one-class gate rejects images that are not skin close-ups. Lesion history lives only in the browser; nothing is stored server-side.

| Metric | Dermoscopic test (n = 10,326) | Mobile test (n = 836) |
|---|---:|---:|
| Sensitivity / TPR | **0.912** | **0.925** |
| FPR | 0.136 | 0.378 |
| Accuracy | 0.876 | 0.843 |
| Precision | 0.687 | 0.869 |
| F1 | 0.784 | 0.896 |
| AUC | ≈ 0.94 | 0.844 |

## Repository structure

```
src/lumen/                  Core Python package (pip install -e .)
  model_meta.py             DINOv2-S + TinyCNN + MetaMLP fused model, Grad-CAM
  preprocessing.py          Centre crop, DullRazor hair removal, dermoscopic / mobile pipelines
  skin_tone.py              ITA from 8 peripheral patches, Fitzpatrick mapping
  gating/                   Skin / not-skin gate (DINOv2 embedding + Mahalanobis OOD detector)
  training/fused.py         Datasets, BCEJLoss, optimiser, epoch loop, checkpoints
scripts/                    CLI entry points (preprocess, split, train, fine-tune, evaluate, fit gate)
web_app/
  Dockerfile                Multi-stage build (Vue → Python runtime)
  backend/                  FastAPI + SSE endpoint, shipped checkpoints
  frontend/                 Vue 3 + Naive UI single-page app
docs/training/              Training run records (model_10_6, mobile fine-tune)
tests/                      pytest suite
```

See [STRUCTURE.md](STRUCTURE.md) for a module-by-module description and [USAGE.md](USAGE.md) for every CLI flag.

## Quick start

### Install

```bash
pip install -e .
```

Python 3.10+, PyTorch 2.x. A GPU is only needed for training.

### Train the dermoscopic model

```bash
# 1. Preprocess to 448 px with hair removal (one --images per source folder)
python scripts/preprocess_fused_dataset.py \
    --metadata final_metadata.csv \
    --images data/2019/ISIC_2019_Training_Input \
    --images data/2020/train \
    --images data/MILK10k/MILK10k_Training_Input \
    --output preprocessed448

# 2. Patient-grouped stratified split (adds a "split" column in place)
python scripts/make_split.py --metadata final_metadata.csv

# 3. Train (AdamW 3e-4, warm-up + cosine, batch 64, early stopping on val loss)
python scripts/train_fused.py \
    --metadata final_metadata.csv \
    --img-dir preprocessed448 \
    --output-dir runs/fused
```

### Fine-tune and evaluate the mobile model

```bash
python scripts/eval_mobile.py  --checkpoint runs/fused/checkpoint_<ts>.pt --eval-csv mobile_eval.csv --images data/MILK10k/MILK10k_Training_Input
python scripts/train_mobile.py --pretrained runs/fused/checkpoint_<ts>.pt --eval-csv mobile_eval.csv --images data/MILK10k/MILK10k_Training_Input --output-dir runs/mobile
```

### Run the web app

```bash
# build context is the repo root
docker build -f web_app/Dockerfile -t skin-check .
docker run -p 8000:8000 skin-check
# open http://localhost:8000
```

Model weights ship inside `web_app/backend/`; DINOv2-S is baked into the image at build time. For deployment on Render see `render.yaml` (a 2 GB instance is required).

### Use from Python

```python
import cv2
import torch
from lumen.model_meta import load_fused_model, image_to_tensor, encode_metadata
from lumen.preprocessing import preprocess_mobile

model, meta_cfg = load_fused_model("web_app/backend/checkpoint_mobile_best.pt", device="cpu")
rgb = cv2.cvtColor(cv2.imread("lesion.jpg"), cv2.COLOR_BGR2RGB)
img = image_to_tensor(preprocess_mobile(rgb))
meta, meta_used = encode_metadata(54, "male", "torso", meta_cfg)  # meta_used tells you which fields were actually used
with torch.no_grad():
    prob = torch.sigmoid(model(img, meta)).item()
print(f"P(malignant) = {prob:.2f}")
```

## Documentation

- [Project Documentation](https://github.com/datice1/skin-check/wiki/Project-Documentation) — motivation, data, architecture, loss, experiments, mobile adaptation, web app, discussion (English)
- [Dokumentacija projekta](https://github.com/datice1/skin-check/wiki/Dokumentacija-projekta) — the same document in Croatian
- [Technical Documentation](https://github.com/datice1/skin-check/wiki/Technical-Documentation) — code layout, API, training commands, SSE contract, deployment

## License

See [LICENSE](LICENSE).

## Disclaimer

This is a research and educational tool, not a certified medical device. The model has not been clinically validated; its output is a probability estimate, not a diagnosis, and does not replace an examination by a dermatologist.
