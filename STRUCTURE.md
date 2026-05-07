# Project Structure

**🇬🇧 English** | [🇭🇷 Hrvatski](STRUCTURE.hr.md)

## Overview

All shared logic lives in the `lumen` Python package (`src/lumen/`). Scripts and the web app are thin wrappers that import from it. No function is defined in more than one place.

```
src/lumen/                        Core library
src/lumen/training/               Training-only modules (loss, dataset, augmentation, trainer)
scripts/                          CLI entrypoints (import from lumen)
web_app/                          FastAPI backend + Vue frontend (imports from lumen)
configs/default.yaml              Default hyperparameters (dataset paths, model, preprocessing, training)
```

---

## `src/lumen/` — Core Package

### `preprocessing.py`

Image processing pipeline for dermatoscopic images.

| Function | What it does |
|----------|-------------|
| `remove_hair(image)` | Black-hat morphology + inpainting to remove hair artifacts |
| `square_crop(image)` | Center-crop to square (removes dermatoscope edges) |
| `check_resize(image)` | Validates 3:2 aspect ratio and minimum 500px |
| `preprocess(image, ...)` | Full pipeline: crop → hair removal → resize → optional ITA |
| `preprocess_from_path(path, ...)` | Load from disk + intermediate resize + full pipeline |
| `preprocess_for_inference(path, ...)` | Same but returns RGB, skips ITA (for model input) |
| `parallel_preprocess(paths, ...)` | Multiprocessing wrapper for batch preprocessing |
| `parallel_preprocess_for_inference(names, folder, ...)` | Multiprocessing wrapper for batch inference |

### `skin_tone.py`

ITA-based skin tone estimation and Fitzpatrick classification.

| Function | What it does |
|----------|-------------|
| `calculate_ita_subregions(image)` | Computes ITA from 8 peripheral subregions, returns top-2 average |
| `get_fitzpatrick(ita)` | Maps ITA → Fitzpatrick type (1–6 integer) |
| `get_fitzpatrick_label(ita)` | Maps ITA → human-readable label like `"III (Intermediate)"` |
| `assign_fitz_group(ita)` | Maps ITA → training group (12, 3, 4, or 56) — groups I–II and V–VI together |

### `model.py`

Neural network architectures and weight management.

| Class/Function | What it does |
|----------------|-------------|
| `CustomCNN` | 6.7M-parameter CNN trained from scratch. 4 conv blocks + 3-layer classifier. Primary model. |
| `PretrainedEfficientNet` | EfficientNet-B0 backbone with custom classification head. Alternative model. |
| `load_model(model_class, path)` | Instantiate + load weights + set eval mode. Works with either architecture. |
| `download_weights_from_gdrive(path, file_id)` | Download `.pth` from Google Drive (used by web app on first startup) |

Both models take 224x224 RGB input and output raw logits (apply sigmoid yourself or use `inference.predict()`).

### `inference.py`

Running predictions and explainability.

| Function | What it does |
|----------|-------------|
| `prepare_tensor(img_np)` | Converts (H,W,3) numpy array → (1,3,H,W) float tensor normalized to [0,1] |
| `predict(model, tensor, threshold)` | Forward pass → returns `(probability, predicted_class)` |
| `apply_gradcam(model, tensor, layer)` | Generates Grad-CAM heatmap for the given layer. Returns (cam, class_idx) |

### `folding.py`

Patient-level stratified k-fold splitting.

| Function | What it does |
|----------|-------------|
| `build_patient_dict(df)` | Builds `{patient_id: [n_class0, n_class1]}` from a DataFrame |
| `triple_stratified_fold(patient_dict, df, num_folds)` | Round-robin fold assignment. Class-1 patients distributed first, then class-0. No patient spans multiple folds. |

---

## `src/lumen/training/` — Training Modules

### `loss.py`

| Class | What it does |
|-------|-------------|
| `FairnessAwareLoss` | Custom weighted BCE with three components: (1) per-class weighting, (2) recall-balance regularization penalizing poor per-class recall, (3) equalized odds regularization penalizing TPR/FPR gaps across skin tone groups. Augmented samples get reduced weight. |

### `augmentation.py`

All functions operate on float32 numpy arrays in [0,1] range.

| Function | What it does |
|----------|-------------|
| `random_rotate(img)` | Rotate by random multiple of 90° |
| `flip_vertical(img)` | Vertical flip |
| `flip_horizontal(img)` | Horizontal flip |
| `contrast_change(img)` | Random contrast ±2–10% |
| `brightness_change(img)` | Random brightness ±5–10% |
| `add_gaussian_noise(img)` | Gaussian noise, std 0.01–0.05 |
| `color_jitter(img)` | Random hue/saturation/value shift in HSV space |
| `augment(img, n)` | Produce n augmented copies: random rotation + 3+ random transforms |

### `dataset.py`

| Class/Function | What it does |
|----------------|-------------|
| `SkinImageDataset` | PyTorch Dataset. Loads all images into memory at init using parallel threads. Augments class-1 and underrepresented groups on the fly. Returns `(img_tensor, label, fitz_group, is_augmented)`. |
| `load_and_prepare_image(name, fold, dir)` | Load a single preprocessed image from the fold directory structure |
| `calculate_class_weights(loader, device)` | Compute sklearn balanced class weights from a DataLoader |

### `evaluation.py`

| Function | What it does |
|----------|-------------|
| `evaluate_loss(loader, model, criterion, device)` | Compute average loss over a DataLoader |
| `detailed_evaluation(loader, model, device, threshold)` | Classification report + AUC |
| `evaluate_fairness(loader, model, criterion, device)` | Per-Fitzpatrick-group precision, recall, F1, accuracy, FPR |
| `evaluate_thresholds(model, loader, device)` | Sweep thresholds 0.2–0.8, plot curves, return best threshold/F1 |
| `plot_metrics(history)` | Plot 2x2 grid: loss, AUC, F1, class-1 recall over epochs |

### `trainer.py`

| Function | What it does |
|----------|-------------|
| `train_epoch(loader, model, criterion, optimizer, device)` | One training epoch. Returns (avg_loss, predictions, labels). |
| `k_fold_training(metadata, base_dir, train_folds, val_folds, ...)` | Full training run: builds datasets, trains with early stopping + LR reduction, logs fairness metrics per epoch, saves best weights. Returns (model, history). |

---

## `scripts/` — CLI Entrypoints

| Script | Purpose | Key args |
|--------|---------|----------|
| `preprocess_dataset.py` | Raw images → preprocessed folds + metadata | `--images`, `--labels`, `--duplicates` |
| `train.py` | Metadata + folds → trained model weights | `--data-dir`, `--metadata` |
| `predict.py` | Images + weights → predictions CSV | `--images`, `--weights` |

All scripts accept `--help` for full flag documentation.

---

## `web_app/` — Web Application

```
web_app/
├── Dockerfile              Multi-stage build (Node → Python → runtime)
├── backend/
│   ├── app.py              FastAPI setup, model loading on startup
│   └── router.py           SSE endpoint: upload → preprocess → predict → Grad-CAM
└── frontend/               Vue 3 + Naive UI single-page app
```

The backend imports `lumen.preprocessing`, `lumen.skin_tone`, `lumen.inference`, and `lumen.model` — no duplicated logic.

---

## Data Flow

```
Raw .jpg images
      │
      ▼
┌─────────────────────┐
│  preprocess_dataset │  scripts/preprocess_dataset.py
│  (preprocessing.py) │  remove hair, crop, resize, compute ITA
│  (skin_tone.py)     │  estimate Fitzpatrick type
│  (folding.py)       │  patient-level k-fold split
└─────────┬───────────┘
          ▼
   Preprocessed folds + metadata.json
          │
          ▼
┌─────────────────────┐
│       train         │  scripts/train.py
│  (dataset.py)       │  load images, augment class-1 + group 56
│  (augmentation.py)  │  random transforms
│  (loss.py)          │  fairness-aware weighted BCE
│  (trainer.py)       │  AdamW, early stopping, LR scheduling
│  (evaluation.py)    │  per-epoch metrics + fairness reporting
└─────────┬───────────┘
          ▼
     model_fold_N.pth
          │
          ▼
┌─────────────────────┐
│      predict        │  scripts/predict.py
│  (preprocessing.py) │  preprocess new images
│  (inference.py)     │  tensor prep, forward pass
└─────────┬───────────┘
          ▼
    predictions.csv
```
