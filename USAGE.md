# Usage Guide

## Setup

```bash
pip install -e .
```

---

## 1. Preprocess Dataset

Takes raw ISIC 2020 images and produces preprocessed, folded output ready for training.

```bash
python scripts/preprocess_dataset.py \
    --images path/to/train \
    --labels path/to/ISIC_2020_Training_GroundTruth.csv \
    --duplicates path/to/2020_Challenge_duplicates.csv \
    --output preprocessed
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--images` | yes | — | Folder with raw `.jpg` images |
| `--labels` | yes | — | ISIC 2020 ground truth CSV |
| `--duplicates` | yes | — | CSV listing duplicate image pairs |
| `--output` | no | `preprocessed` | Output folder |
| `--percent` | no | `1.0` | Fraction of dataset to use (0.0–1.0) |
| `--target-size` | no | `200` | Output image size in pixels |
| `--num-folds` | no | `5` | Number of k-folds |
| `--max-class0` | no | `20` | Max class-0 images per patient |
| `--no-parallel` | no | — | Disable multiprocessing |
| `--seed` | no | `42` | Random seed |

**Output:**
```
preprocessed/
├── 0_fold/          # Preprocessed images per fold
├── 1_fold/
├── ...
├── folds.csv        # Patient-to-fold assignments
└── metadata.json    # ITA values, labels, fold IDs
```

---

## 2. Train

Runs k-fold cross-validation with the fairness-aware loss function.

```bash
python scripts/train.py \
    --data-dir preprocessed \
    --metadata preprocessed/metadata.json
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--data-dir` | yes | — | Preprocessed fold directory (from step 1) |
| `--metadata` | yes | — | Path to `metadata.json` (from step 1) |
| `--output-dir` | no | `models` | Where to save `.pth` weights |
| `--target-ratio` | no | `0.15` | Target class-1 ratio after augmentation |
| `--num-folds` | no | `5` | Number of folds |
| `--test-fold` | no | `4` | Fold held out for final testing |
| `--epochs` | no | `35` | Max training epochs |
| `--batch-size` | no | `128` | Batch size |
| `--lr` | no | `3e-5` | Learning rate (AdamW) |
| `--seed` | no | `42` | Random seed |

**Output:**
```
models/
├── model_fold_0.pth
├── model_fold_1.pth
├── model_fold_2.pth
└── model_fold_3.pth
```

---

## 3. Predict

Runs batch inference on a folder of `.jpg` images.

```bash
python scripts/predict.py \
    --images path/to/images \
    --weights models/model_fold_0.pth \
    --output predictions.csv
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--images` | yes | — | Folder with `.jpg` images to classify |
| `--weights` | yes | — | Path to trained `.pth` model weights |
| `--output` | no | `predictions.csv` | Output CSV path |
| `--threshold` | no | `0.5` | Classification threshold |
| `--no-parallel` | no | — | Disable multiprocessing |

**Output:** CSV with columns `image_name, target` (0 = benign, 1 = malignant).

---

## 4. Web App

```bash
cd web_app
docker build -t melanoma-detector .
docker run -p 8000:8000 melanoma-detector
```

Opens at `http://localhost:8000`. Model weights are downloaded automatically on first run.

---

## Python API

Everything is importable from the `lumen` package:

```python
from lumen.preprocessing import preprocess, remove_hair, square_crop
from lumen.skin_tone import get_fitzpatrick, calculate_ita_subregions
from lumen.model import CustomCNN, PretrainedEfficientNet, load_model
from lumen.inference import prepare_tensor, predict, apply_gradcam
from lumen.training.loss import FairnessAwareLoss
from lumen.training.augmentation import augment
from lumen.training.dataset import SkinImageDataset
from lumen.training.evaluation import detailed_evaluation, evaluate_fairness
from lumen.training.trainer import k_fold_training
from lumen.folding import triple_stratified_fold
```

### Quick single-image prediction

```python
import cv2
from lumen.model import CustomCNN, load_model
from lumen.preprocessing import preprocess
from lumen.inference import prepare_tensor, predict

model = load_model(CustomCNN, "models/model_fold_0.pth")
image = cv2.cvtColor(cv2.imread("lesion.jpg"), cv2.COLOR_BGR2RGB)
processed = preprocess(image, target_size=(224, 224), compute_ita=False)
prob, cls = predict(model, prepare_tensor(processed))
print(f"{'Malignant' if cls else 'Benign'} ({prob:.1%})")
```
