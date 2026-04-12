# Fairness-Driven Melanoma Classification

A deep CNN approach with equalized skin tone performances, developed for the **Lumen Data Science Challenge 2025**.

**Authors:** Jurica Jerinic, Filip Hlup, Tomislav Matanovic, Karlo Rastegorac
**Group:** Datice

## Overview

This project develops a robust deep learning model that classifies dermatoscopic skin lesion images as **benign or malignant**, with a strong emphasis on **fairness across skin tones**. The model is a custom CNN (6.7M parameters) trained from scratch on the [ISIC 2020 dataset](https://challenge2020.isic-archive.com/), incorporating a fairness-aware loss function based on Equalized Odds.

| Metric | Test Score |
|--------|-----------|
| Accuracy | 0.83 |
| AUC | 0.86 |
| TPR (Sensitivity) | 0.69 |
| FPR | 0.16 |
| Equalized Odds Gap | 0.51 |

A [live web application](https://lumen-i.onrender.com) is also available for interactive predictions with Grad-CAM explainability.

## Repository Structure

```
src/lumen/                        # Core Python package (pip install -e .)
  preprocessing.py                # Hair removal, cropping, resize pipeline
  skin_tone.py                    # ITA calculation & Fitzpatrick mapping
  model.py                        # CustomCNN + PretrainedEfficientNet architectures
  inference.py                    # Tensor prep, prediction, Grad-CAM
  folding.py                      # Triple-stratified k-fold splitting
  training/                       # Training-specific modules
    loss.py                       # Fairness-aware custom loss function
    augmentation.py               # Data augmentation transforms
    dataset.py                    # PyTorch Dataset with parallel loading
    evaluation.py                 # Metrics, fairness evaluation, plotting
    trainer.py                    # Training loop with early stopping

scripts/                          # CLI entrypoints
  preprocess_dataset.py           # Batch dataset preprocessing
  predict.py                      # Batch inference
  train.py                        # Model training

configs/default.yaml              # Training/inference configuration
web_app/
  Dockerfile                      # Multi-stage Docker build
  backend/                        # FastAPI + PyTorch backend
  frontend/                       # Vue.js + Naive UI frontend
```

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.6+
- [ISIC 2020 Training Dataset](https://challenge2020.isic-archive.com/)

### Install Dependencies

```bash
pip install -e .
```

This installs the `lumen` package in editable mode with all dependencies.

### 1. Preprocess Data

Place the ISIC 2020 dataset and metadata files alongside the preprocessing script, then run:

```bash
python scripts/preprocess_dataset.py
```

This performs duplicate removal, hair artifact removal, ITA-based Fitzpatrick skin tone estimation, per-patient image capping, and triple-stratified k-fold splitting. Output is a structured folder with preprocessed images and metadata per fold.

Configuration constants are at the top of the script: `PERCENT`, `OUTPUT_FOLDER`, `TARGET_SIZE`, `NUM_FOLDS`, `PARALLEL`.

### 2. Train the Model

```bash
python scripts/train.py \
    --data-dir preprocessing/2all_k_FOLD_dataset_PREPROCESSED2 \
    --metadata preprocessing/2all_k_FOLD_dataset_PREPROCESSED2/metadata_all_fold.json
```

Handles augmentation, training with the custom fairness-aware loss, cross-validation, and model saving. GPU recommended.

**Training config:** AdamW optimizer, LR 3e-5, batch size 128, up to 35 epochs with early stopping.

### 3. Run Inference

```bash
python scripts/predict.py <INPUT_FOLDER> <OUTPUT_CSV>
```

Processes a folder of `.jpg` images through the same preprocessing pipeline and outputs binary predictions to CSV. Supports parallel preprocessing (~70% of CPU cores).

### 4. Run the Web App

```bash
cd web_app
docker build -t melanoma-detector .
docker run -p 8000:8000 melanoma-detector
```

The app automatically downloads model weights from Google Drive on first run. Access at `http://localhost:8000`.

## Configuration

All training and inference settings are centralized in [`configs/default.yaml`](configs/default.yaml):

```yaml
model:
  name: "EfficientNetB0"
  pretrained: true
  custom_head:
    dropout_rate: 0.5
    output_activation: "sigmoid"

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.0005
  optimizer: "adam"
  loss: "binary_crossentropy"

inference:
  threshold: 0.5
```

## Key Technical Decisions

- **Custom CNN over pretrained EfficientNet:** Better accuracy with lower complexity for this specific task
- **Triple-stratified folding:** Prevents data leakage by enforcing patient-level separation across folds
- **Fairness-aware loss function:** Incorporates Equalized Odds Gap regularization, per-class recall penalty, and augmentation-aware weighting
- **Hair removal preprocessing:** Morphological black-hat filtering + inpainting to reduce artifact noise
- **ITA-based skin tone estimation:** Computed from 8 peripheral subregions to avoid lesion interference

## Documentation

Full documentation is available in the [project wiki](https://github.com/datice-lumen/Lumen-i-/wiki):

- [Project Documentation](https://github.com/datice-lumen/Lumen-i-/wiki/Project-Documentation) -- methodology, results, fairness evaluation
- [Technical Documentation](https://github.com/datice-lumen/Lumen-i-/wiki/Technical-Documentation) -- implementation details, code walkthroughs, deployment

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## Disclaimer

This tool is intended for educational and research purposes only. It is not validated for clinical use and should not replace professional medical diagnosis.
