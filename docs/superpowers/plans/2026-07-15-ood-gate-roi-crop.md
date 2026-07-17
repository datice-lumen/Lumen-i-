# OOD Skin-Gate + ROI Background-Crop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `preprocess_and_gate(image, gate, point=None)` that crops the skin ROI out of a phone photo (SAM 2) and returns either the crop or `"unclassified"` if the image isn't skin (DINOv2 + one-class Mahalanobis).

**Architecture:** Two frozen stages. SAM 2 (tiny) segments the skin region around a point (default: image center) and returns a rectangular crop with margin — no black cutout. DINOv2-S embeds the crop to 384-d; a one-class Mahalanobis gate (fit = mean + shrunk covariance + percentile threshold) decides in-distribution vs OOD. Nothing is trained.

**Tech Stack:** Python 3.10 (CPU), PyTorch (torch.hub DINOv2 `dinov2_vits14_reg`), ultralytics (SAM 2 `sam2_t.pt`), scikit-learn (`LedoitWolf`), numpy, Pillow, OpenCV, matplotlib, pytest.

## Global Constraints

- **No training of any kind.** DINOv2 and SAM 2 are used frozen. The gate is *fit* = summary statistics only. The existing melanoma model is never loaded or touched.
- **Local CPU only. Nothing runs on slavica.** The single exception is the one-time data-copy step, which *reads* the SSHFS mount (`/home/hlupek/slavica/remote/...`) and writes locally.
- **Internal image format is RGB uint8 numpy `HxWx3`** end-to-end (load → crop → embed).
- **DINOv2 input:** resize to `448x448` + ImageNet mean `[0.485,0.456,0.406]` / std `[0.229,0.224,0.225]` (matches the classifier's `train.py`). Model id: `dinov2_vits14_reg`, output dim `384`.
- **Branch:** all work on `feat/ood-gate`. Artifacts live under gitignored `data/` and `reports/`.
- **Python environment (CRITICAL):** this box is GPU-less and its system Python has a GPU-oriented `torch 2.11+cu130` that must NOT be used. All `python`/`pytest`/`pip` commands run via the repo venv at `.venv/` — use `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/pip`, never bare `python`/`python3`/`pytest`/`pip`. The venv already contains CPU-only `torch`/`torchvision`, `ultralytics`, `scikit-learn`, `pytest`, and an editable install of `lumen`. Do NOT create a venv, and do NOT `pip install torch`/`torchvision`/`ultralytics` again (that pulls multi-GB CUDA wheels).
- **Layout:** src-layout, editable install active in the venv — `from lumen.gating... import ...` works via `.venv/bin/python`. Tests under `tests/` run with `.venv/bin/pytest`.
- **Mount paths (read-only, data copy only):** MILK10k metadata `/home/hlupek/slavica/remote/data/original_data/MILK10k/MILK10k_Training_Metadata.csv`; MILK10k images `.../MILK10k_Training_Input/<IL_*>/<ISIC_*>.jpg`; preprocessed dermoscopy `/home/hlupek/slavica/remote/model_10_6/preprocessed448_67k/pre_*.jpg`.

## File Structure

- `src/lumen/gating/__init__.py` — package marker
- `src/lumen/gating/ood_gate.py` — `OODGate` (fit/score/passes/save/load), pure numpy+sklearn
- `src/lumen/gating/dino_embed.py` — `embed(image) -> np.ndarray[384]` (frozen DINOv2-S)
- `src/lumen/gating/roi_crop.py` — `crop_roi(image, point=None) -> (crop, mask, bbox, reason)` (frozen SAM 2)
- `src/lumen/gating/pipeline.py` — `preprocess_and_gate(image, gate, point=None) -> dict`
- `scripts/prepare_gate_data.py` — copy skin sample from mount + download negatives + write manifest
- `scripts/fit_gate.py` — build `data/gate.npz` from the fit split
- `scripts/eval_gate.py` — validation report + contact sheet
- `tests/gating/test_ood_gate.py`, `tests/gating/test_dino_embed.py`, `tests/gating/test_roi_crop.py`, `tests/gating/test_pipeline.py`

---

### Task 1: Branch, package scaffold, dependency

**Files:**
- Create: `src/lumen/gating/__init__.py`
- Create: `tests/gating/__init__.py`
- Modify: `pyproject.toml` (add `ultralytics` to dependencies)

**Interfaces:**
- Consumes: nothing
- Produces: importable empty package `lumen.gating`; `ultralytics` installed

- [ ] **Step 1: Create the feature branch**

Run:
```bash
cd /home/hlupek/Study/Lumen-i-
git checkout -b feat/ood-gate
```
Expected: `Switched to a new branch 'feat/ood-gate'`

- [ ] **Step 2: Create the package and test dirs**

Create `src/lumen/gating/__init__.py` with:
```python
"""OOD skin-gate + ROI background-crop (frozen models, no training)."""
```
Create `tests/gating/__init__.py` as an empty file (0 bytes).

- [ ] **Step 3: Add the ultralytics dependency**

In `pyproject.toml`, add `"ultralytics>=8.3",` to the `dependencies` list (after `"gdown",`).

- [ ] **Step 4: Install ultralytics**

Run:
```bash
cd /home/hlupek/Study/Lumen-i- && pip install "ultralytics>=8.3"
```
Expected: ends with `Successfully installed ultralytics-...` (may also pull small deps). CPU torch already present is reused.

- [ ] **Step 5: Verify the package imports**

Run:
```bash
cd /home/hlupek/Study/Lumen-i- && python -c "import lumen.gating, ultralytics; print('ok')"
```
Expected: `ok`

- [ ] **Step 6: Commit**

```bash
cd /home/hlupek/Study/Lumen-i-
git add src/lumen/gating/__init__.py tests/gating/__init__.py pyproject.toml
git commit -m "feat(gating): scaffold gating package and add ultralytics dep"
```

---

### Task 2: OODGate (one-class Mahalanobis)

**Files:**
- Create: `src/lumen/gating/ood_gate.py`
- Test: `tests/gating/test_ood_gate.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `class OODGate` with fields `mean: np.ndarray (D,)`, `precision: np.ndarray (D,D)`, `threshold: float`
  - `OODGate.fit(features: np.ndarray[N,D], percentile: float = 99.0) -> OODGate`
  - `OODGate.score(feature: np.ndarray[D]) -> float`
  - `OODGate.passes(feature: np.ndarray[D]) -> bool`
  - `OODGate.save(path: str) -> None` / `OODGate.load(path: str) -> OODGate` (npz)

- [ ] **Step 1: Write the failing tests**

Create `tests/gating/test_ood_gate.py`:
```python
import numpy as np
import pytest
from lumen.gating.ood_gate import OODGate


def _cluster(n=2000, d=16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(n, d))


def test_in_distribution_point_passes():
    gate = OODGate.fit(_cluster(), percentile=99.0)
    assert gate.passes(np.zeros(16)) is True


def test_far_point_is_rejected():
    gate = OODGate.fit(_cluster(), percentile=99.0)
    assert gate.passes(np.full(16, 50.0)) is False


def test_threshold_rejects_about_one_percent_of_fit():
    feats = _cluster(n=5000, d=16)
    gate = OODGate.fit(feats, percentile=99.0)
    rejected = np.mean([not gate.passes(f) for f in feats])
    assert 0.0 <= rejected <= 0.03


def test_save_load_roundtrip(tmp_path):
    gate = OODGate.fit(_cluster(), percentile=99.0)
    p = tmp_path / "gate.npz"
    gate.save(str(p))
    loaded = OODGate.load(str(p))
    f = np.ones(16)
    assert loaded.threshold == pytest.approx(gate.threshold)
    assert loaded.score(f) == pytest.approx(gate.score(f), rel=1e-6)


def test_non_finite_feature_raises():
    gate = OODGate.fit(_cluster(), percentile=99.0)
    with pytest.raises(ValueError):
        gate.score(np.array([np.nan] * 16))


def test_fit_rejects_non_2d():
    with pytest.raises(ValueError):
        OODGate.fit(np.zeros(16))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/hlupek/Study/Lumen-i- && pytest tests/gating/test_ood_gate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lumen.gating.ood_gate'`

- [ ] **Step 3: Write the implementation**

Create `src/lumen/gating/ood_gate.py`:
```python
"""One-class Mahalanobis OOD gate over frozen embeddings.

Fit = summary statistics only (mean, shrunk covariance, threshold). No training.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf


@dataclass
class OODGate:
    mean: np.ndarray       # (D,)
    precision: np.ndarray  # (D, D) inverse covariance
    threshold: float       # Mahalanobis distance cutoff

    @classmethod
    def fit(cls, features: np.ndarray, percentile: float = 99.0) -> "OODGate":
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError(f"expected 2D (N, D), got shape {features.shape}")
        mean = features.mean(axis=0)
        precision = LedoitWolf().fit(features).precision_
        distances = cls._distances(features, mean, precision)
        threshold = float(np.percentile(distances, percentile))
        return cls(mean=mean, precision=precision, threshold=threshold)

    @staticmethod
    def _distances(x: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
        centered = x - mean
        quad = np.einsum("ij,jk,ik->i", centered, precision, centered)
        return np.sqrt(np.clip(quad, 0.0, None))

    def score(self, feature: np.ndarray) -> float:
        feature = np.asarray(feature, dtype=np.float64).reshape(1, -1)
        if not np.all(np.isfinite(feature)):
            raise ValueError("feature contains NaN/inf")
        return float(self._distances(feature, self.mean, self.precision)[0])

    def passes(self, feature: np.ndarray) -> bool:
        return self.score(feature) <= self.threshold

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, precision=self.precision,
                 threshold=np.array(self.threshold, dtype=np.float64))

    @classmethod
    def load(cls, path: str) -> "OODGate":
        data = np.load(path)
        return cls(mean=data["mean"], precision=data["precision"],
                   threshold=float(data["threshold"]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/hlupek/Study/Lumen-i- && pytest tests/gating/test_ood_gate.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/hlupek/Study/Lumen-i-
git add src/lumen/gating/ood_gate.py tests/gating/test_ood_gate.py
git commit -m "feat(gating): one-class Mahalanobis OOD gate"
```

---

### Task 3: DINOv2-S embedder

**Files:**
- Create: `src/lumen/gating/dino_embed.py`
- Test: `tests/gating/test_dino_embed.py`

**Interfaces:**
- Consumes: nothing
- Produces: `embed(image) -> np.ndarray` shape `(384,)`, dtype float32. `image` is an RGB uint8 numpy array or a PIL image.

**Note:** the test downloads the DINOv2 weights on first run (GitHub reachable). It is slow once, then cached under `~/.cache/torch/hub`.

- [ ] **Step 1: Write the failing test**

Create `tests/gating/test_dino_embed.py`:
```python
import numpy as np
from lumen.gating.dino_embed import embed


def test_embed_returns_384d_finite_vector():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(200, 260, 3), dtype=np.uint8)
    feat = embed(img)
    assert feat.shape == (384,)
    assert feat.dtype == np.float32
    assert np.all(np.isfinite(feat))


def test_embed_is_deterministic():
    img = np.full((128, 128, 3), 127, dtype=np.uint8)
    a = embed(img)
    b = embed(img)
    assert np.allclose(a, b, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/hlupek/Study/Lumen-i- && pytest tests/gating/test_dino_embed.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lumen.gating.dino_embed'`

- [ ] **Step 3: Write the implementation**

Create `src/lumen/gating/dino_embed.py`:
```python
"""Frozen DINOv2-S embedder (same backbone/transform as the classifier)."""
from __future__ import annotations

import functools

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_RESIZE = 448  # matches train.py

_transform = transforms.Compose([
    transforms.Resize((_RESIZE, _RESIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


@functools.lru_cache(maxsize=1)
def _load_model():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg", verbose=False)
    model.eval()
    return model


def _to_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr.astype(np.uint8), "RGB")
    raise ValueError(f"unsupported image shape {getattr(arr, 'shape', None)}")


def embed(image) -> np.ndarray:
    """Return the 384-d DINOv2-S CLS embedding of an RGB image."""
    model = _load_model()
    tensor = _transform(_to_pil(image)).unsqueeze(0)
    with torch.no_grad():
        feat = model(tensor)
    return feat.squeeze(0).cpu().numpy().astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/hlupek/Study/Lumen-i- && pytest tests/gating/test_dino_embed.py -v`
Expected: PASS (2 passed) — first run downloads weights (slow), later runs fast.

- [ ] **Step 5: Commit**

```bash
cd /home/hlupek/Study/Lumen-i-
git add src/lumen/gating/dino_embed.py tests/gating/test_dino_embed.py
git commit -m "feat(gating): frozen DINOv2-S embedder"
```

---

### Task 4: SAM 2 ROI crop

**Files:**
- Create: `src/lumen/gating/roi_crop.py`
- Test: `tests/gating/test_roi_crop.py`

**Interfaces:**
- Consumes: nothing
- Produces: `crop_roi(image, point=None, margin=0.15, min_area_frac=0.05) -> (crop, mask, bbox, reason)`
  - `image`: RGB uint8 numpy `HxWx3`; `point`: `[x, y]` or None (default = center)
  - returns `crop` (RGB uint8 numpy), `mask` (bool numpy or None), `bbox` `[x0,y0,x1,y1]` (ints), `reason` in `{"sam_crop","no_mask_fallback"}`
- Also produces helper `center_square(image) -> (crop, bbox)` used by the pipeline fallback path.

**Note:** the test downloads `sam2_t.pt` on first run (GitHub reachable), then cached.

- [ ] **Step 1: Write the failing test**

Create `tests/gating/test_roi_crop.py`:
```python
import numpy as np
from lumen.gating.roi_crop import crop_roi, center_square


def _synthetic_subject_on_background():
    # dark background with a bright centered square "subject"
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[80:160, 120:200] = 200
    return img


def test_center_square_is_square_and_centered():
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    crop, bbox = center_square(img)
    assert crop.shape[0] == crop.shape[1] == 240
    assert bbox == [40, 0, 280, 240]


def test_crop_roi_returns_rgb_crop_and_bbox():
    img = _synthetic_subject_on_background()
    crop, mask, bbox, reason = crop_roi(img)
    assert crop.ndim == 3 and crop.shape[2] == 3
    assert len(bbox) == 4
    assert reason in {"sam_crop", "no_mask_fallback"}


def test_crop_roi_falls_back_on_blank_image():
    # a uniform image gives SAM no salient region -> fallback path must still return a crop
    img = np.full((240, 320, 3), 15, dtype=np.uint8)
    crop, mask, bbox, reason = crop_roi(img)
    assert crop.size > 0
    assert len(bbox) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/hlupek/Study/Lumen-i- && pytest tests/gating/test_roi_crop.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lumen.gating.roi_crop'`

- [ ] **Step 3: Write the implementation**

Create `src/lumen/gating/roi_crop.py`:
```python
"""SAM 2 region-of-interest crop (frozen). Removes non-skin background.

Returns a rectangular crop with margin around the segmented region — never a
hard black cutout (which would distort DINOv2 embeddings).
"""
from __future__ import annotations

import functools

import numpy as np


@functools.lru_cache(maxsize=1)
def _load_sam():
    from ultralytics import SAM
    return SAM("sam2_t.pt")


def center_square(image: np.ndarray):
    h, w = image.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return image[y0:y0 + side, x0:x0 + side], [x0, y0, x0 + side, y0 + side]


def crop_roi(image, point=None, margin: float = 0.15, min_area_frac: float = 0.05):
    image = np.asarray(image)
    h, w = image.shape[:2]
    if point is None:
        point = [w // 2, h // 2]

    result = _load_sam()(image, points=[point], labels=[1], verbose=False)[0]
    masks = getattr(result, "masks", None)

    if masks is None or masks.data is None or len(masks.data) == 0:
        crop, bbox = center_square(image)
        return crop, None, bbox, "no_mask_fallback"

    mask = masks.data[0].cpu().numpy().astype(bool)
    if mask.sum() < min_area_frac * h * w:
        crop, bbox = center_square(image)
        return crop, mask, bbox, "no_mask_fallback"

    ys, xs = np.where(mask)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    mx = int((x1 - x0) * margin)
    my = int((y1 - y0) * margin)
    x0 = max(0, x0 - mx); y0 = max(0, y0 - my)
    x1 = min(w, x1 + mx); y1 = min(h, y1 + my)
    return image[y0:y1, x0:x1], mask, [x0, y0, x1, y1], "sam_crop"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/hlupek/Study/Lumen-i- && pytest tests/gating/test_roi_crop.py -v`
Expected: PASS (3 passed) — first run downloads `sam2_t.pt` (slow), then fast.

- [ ] **Step 5: Commit**

```bash
cd /home/hlupek/Study/Lumen-i-
git add src/lumen/gating/roi_crop.py tests/gating/test_roi_crop.py
git commit -m "feat(gating): SAM 2 ROI crop with center-square fallback"
```

---

### Task 5: Unified `preprocess_and_gate` pipeline

**Files:**
- Create: `src/lumen/gating/pipeline.py`
- Test: `tests/gating/test_pipeline.py`

**Interfaces:**
- Consumes: `crop_roi` (Task 4), `embed` (Task 3), `OODGate` (Task 2)
- Produces: `preprocess_and_gate(image, gate, point=None) -> dict` with keys
  `status` (`"ok"|"unclassified"|"error"`), `crop` (RGB numpy or None), `score` (float or None),
  `threshold` (float), `bbox` (`[x0,y0,x1,y1]` or None), `reason` (str)

- [ ] **Step 1: Write the failing test**

Create `tests/gating/test_pipeline.py`. The routing logic is tested hermetically —
`crop_roi` and `embed` are monkeypatched so the test never loads SAM 2 or DINOv2:
```python
import numpy as np
import pytest
from lumen.gating import pipeline
from lumen.gating.pipeline import preprocess_and_gate


class _Gate:
    """Minimal fitted-gate stand-in: fixed score + threshold."""
    def __init__(self, score, threshold):
        self._s = score
        self.threshold = threshold
    def score(self, feature):
        return self._s


@pytest.fixture
def stub_models(monkeypatch):
    monkeypatch.setattr(
        pipeline, "crop_roi",
        lambda rgb, point=None: (rgb, None, [0, 0, rgb.shape[1], rgb.shape[0]], "sam_crop"))
    monkeypatch.setattr(pipeline, "embed", lambda crop: np.zeros(384, dtype=np.float32))


def _img():
    return np.full((64, 64, 3), 127, dtype=np.uint8)


def test_unreadable_path_returns_error(stub_models):
    out = preprocess_and_gate("/no/such/file.jpg", _Gate(0.0, 1.0))
    assert out["status"] == "error"
    assert out["reason"] == "unreadable"


def test_in_distribution_returns_ok_with_crop(stub_models):
    out = preprocess_and_gate(_img(), _Gate(0.5, 1.0))
    assert out["status"] == "ok"
    assert out["crop"] is not None
    assert out["score"] == 0.5


def test_ood_returns_unclassified_without_crop(stub_models):
    out = preprocess_and_gate(_img(), _Gate(5.0, 1.0))
    assert out["status"] == "unclassified"
    assert out["crop"] is None
    assert out["reason"] == "ood_distance>threshold"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/hlupek/Study/Lumen-i- && pytest tests/gating/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lumen.gating.pipeline'`

- [ ] **Step 3: Write the implementation**

Create `src/lumen/gating/pipeline.py`:
```python
"""Unified preprocess-and-gate entry point: SAM 2 crop -> DINOv2 -> OOD gate."""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from lumen.gating.roi_crop import crop_roi
from lumen.gating.dino_embed import embed
from lumen.gating.ood_gate import OODGate


def _load_rgb(image) -> Optional[np.ndarray]:
    if isinstance(image, np.ndarray):
        return image
    try:
        return np.array(Image.open(image).convert("RGB"))
    except Exception:
        return None


def preprocess_and_gate(image, gate: OODGate, point=None) -> dict:
    """Crop the skin ROI, then decide skin (return crop) vs unclassified.

    `image`: path or RGB numpy array. `gate`: a fitted OODGate.
    """
    rgb = _load_rgb(image)
    if rgb is None:
        return {"status": "error", "crop": None, "score": None,
                "threshold": gate.threshold, "bbox": None, "reason": "unreadable"}

    crop, _mask, bbox, _crop_reason = crop_roi(rgb, point=point)

    try:
        feat = embed(crop)
        score = gate.score(feat)
    except ValueError:
        return {"status": "error", "crop": None, "score": None,
                "threshold": gate.threshold, "bbox": bbox, "reason": "bad_feature"}

    if score <= gate.threshold:
        return {"status": "ok", "crop": crop, "score": score,
                "threshold": gate.threshold, "bbox": bbox, "reason": "in_distribution"}
    return {"status": "unclassified", "crop": None, "score": score,
            "threshold": gate.threshold, "bbox": bbox, "reason": "ood_distance>threshold"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/hlupek/Study/Lumen-i- && pytest tests/gating/test_pipeline.py -v`
Expected: PASS (3 passed) — fast, no model downloads (`crop_roi`/`embed` are monkeypatched).

- [ ] **Step 5: Commit**

```bash
cd /home/hlupek/Study/Lumen-i-
git add src/lumen/gating/pipeline.py tests/gating/test_pipeline.py
git commit -m "feat(gating): unified preprocess_and_gate pipeline"
```

---

### Task 6: Data prep — copy skin sample + download negatives

**Files:**
- Create: `scripts/prepare_gate_data.py`

**Interfaces:**
- Consumes: mount paths (Global Constraints)
- Produces:
  - `data/samples/skin/<split>/*.jpg` where `<split>` in `{fit, eval}`
  - `data/negatives/*.jpg`
  - `data/samples/manifest.csv` with columns `path,source,split` (`source` in `{milk_clinical, milk_dermoscopic, preprocessed_dermoscopy}`)

- [ ] **Step 1: Write the script**

Create `scripts/prepare_gate_data.py`:
```python
#!/usr/bin/env python3
"""Prepare gate data: copy a skin sample from the mounted slavica filesystem and
download non-skin negatives. Run once; everything downstream is local.

Usage:
    python scripts/prepare_gate_data.py            # sensible defaults
    python scripts/prepare_gate_data.py --help
"""
import argparse
import os
import shutil
import tarfile
import urllib.request
from glob import glob

import numpy as np
import pandas as pd

MOUNT = "/home/hlupek/slavica/remote"
MILK_META = f"{MOUNT}/data/original_data/MILK10k/MILK10k_Training_Metadata.csv"
MILK_IMG_ROOT = f"{MOUNT}/data/original_data/MILK10k/MILK10k_Training_Input"
PREP_DERM = f"{MOUNT}/model_10_6/preprocessed448_67k"
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"


def _resolve_milk_path(isic_id):
    hits = glob(f"{MILK_IMG_ROOT}/*/{isic_id}.jpg")
    return hits[0] if hits else None


def copy_skin(out_dir, n_clinical, n_dermoscopic, n_preprocessed, eval_frac, seed):
    rng = np.random.default_rng(seed)
    meta = pd.read_csv(MILK_META)
    rows = []

    def take(df, source, n):
        ids = df["isic_id"].tolist()
        rng.shuffle(ids)
        picked = 0
        for isic_id in ids:
            if picked >= n:
                break
            src = _resolve_milk_path(isic_id)
            if src:
                rows.append((src, source))
                picked += 1

    take(meta[meta["image_type"] == "clinical: close-up"], "milk_clinical", n_clinical)
    take(meta[meta["image_type"] == "dermoscopic"], "milk_dermoscopic", n_dermoscopic)

    prep = sorted(glob(f"{PREP_DERM}/pre_*.jpg"))
    rng.shuffle(prep)
    for src in prep[:n_preprocessed]:
        rows.append((src, "preprocessed_dermoscopy"))

    manifest = []
    for i, (src, source) in enumerate(rows):
        split = "eval" if rng.random() < eval_frac else "fit"
        dst_dir = os.path.join(out_dir, "samples", "skin", split)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, f"{source}_{i:05d}.jpg")
        shutil.copyfile(src, dst)
        manifest.append({"path": dst, "source": source, "split": split})

    df = pd.DataFrame(manifest)
    mpath = os.path.join(out_dir, "samples", "manifest.csv")
    df.to_csv(mpath, index=False)
    print(f"skin: copied {len(df)} images "
          f"(fit={sum(df['split']=='fit')}, eval={sum(df['split']=='eval')}) -> {mpath}")


def download_negatives(out_dir, n):
    neg_dir = os.path.join(out_dir, "negatives")
    os.makedirs(neg_dir, exist_ok=True)
    tgz = os.path.join(out_dir, "imagenette2-160.tgz")
    try:
        if not os.path.exists(tgz):
            print(f"downloading negatives from {IMAGENETTE_URL} ...")
            urllib.request.urlretrieve(IMAGENETTE_URL, tgz)
        with tarfile.open(tgz) as t:
            members = [m for m in t.getmembers() if m.name.lower().endswith(".jpeg")]
            members = members[:n]
            for m in members:
                m.name = os.path.basename(m.name)
                t.extract(m, neg_dir)
        print(f"negatives: extracted {len(glob(os.path.join(neg_dir, '*')))} images -> {neg_dir}")
    except Exception as exc:  # network/egress failure -> manual fallback
        print(f"WARNING: negative download failed ({exc}).")
        print(f"Drop ~{n} non-skin .jpg images into {neg_dir} manually and re-run fit/eval.")


def main():
    ap = argparse.ArgumentParser(description="Prepare gate data (copy skin + download negatives)")
    ap.add_argument("--out", default="data", help="output root (default: data)")
    ap.add_argument("--n-clinical", type=int, default=500)
    ap.add_argument("--n-dermoscopic", type=int, default=300)
    ap.add_argument("--n-preprocessed", type=int, default=200)
    ap.add_argument("--n-negatives", type=int, default=150)
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(MILK_META):
        raise SystemExit(f"Mount not found at {MILK_META}. Mount slavica first "
                         f"(~/slavica/mount-slavica.sh).")

    copy_skin(args.out, args.n_clinical, args.n_dermoscopic,
              args.n_preprocessed, args.eval_frac, args.seed)
    download_negatives(args.out, args.n_negatives)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the demo prep (mount required)**

Use moderate counts so the CPU fit in Task 7 stays tractable (embedding is ~seconds/image on CPU). This copies ~400 skin images and downloads ~150 negatives:
```bash
cd /home/hlupek/Study/Lumen-i-
python scripts/prepare_gate_data.py --n-clinical 200 --n-dermoscopic 120 --n-preprocessed 80 --n-negatives 150 --out data
```
Expected: prints `skin: copied ~400 images (fit=~320, eval=~80)`; then either `negatives: extracted 150 images` or the WARNING fallback line. `data/samples/manifest.csv` exists.

Note: for a larger/more statistically meaningful eval later, re-run with the script defaults (`python scripts/prepare_gate_data.py`, which uses 500/300/200) and delete `data/cache/` so Task 7 re-embeds. Expect a longer CPU fit.

- [ ] **Step 3: Verify outputs**

Run:
```bash
cd /home/hlupek/Study/Lumen-i- && head -3 data/samples/manifest.csv && ls data/samples/skin/fit | head -3
```
Expected: CSV header `path,source,split` + rows; at least one file listed.

- [ ] **Step 4: Commit** (script only — `data/` is gitignored)

```bash
cd /home/hlupek/Study/Lumen-i-
git add scripts/prepare_gate_data.py
git commit -m "feat(gating): data-prep script (copy skin sample + download negatives)"
```

---

### Task 7: `fit_gate.py` — build `data/gate.npz`

**Files:**
- Create: `scripts/fit_gate.py`

**Interfaces:**
- Consumes: `data/samples/manifest.csv` (Task 6), `crop_roi` (Task 4), `embed` (Task 3), `OODGate` (Task 2)
- Produces: `data/gate.npz` (a saved `OODGate`); caches fit embeddings to `data/cache/fit_embeddings.npy`

- [ ] **Step 1: Write the script**

Create `scripts/fit_gate.py`:
```python
#!/usr/bin/env python3
"""Fit the one-class OOD gate on the skin fit-split. Statistics only — no training.

Usage:
    python scripts/fit_gate.py            # uses data/samples/manifest.csv
    python scripts/fit_gate.py --help
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from lumen.gating.roi_crop import crop_roi
from lumen.gating.dino_embed import embed
from lumen.gating.ood_gate import OODGate


def _features_for(paths, cache_path):
    if cache_path and os.path.exists(cache_path):
        print(f"loading cached embeddings from {cache_path}")
        return np.load(cache_path)
    feats = []
    for p in tqdm(paths, desc="embedding fit skin"):
        rgb = np.array(Image.open(p).convert("RGB"))
        crop, _mask, _bbox, _reason = crop_roi(rgb)
        feats.append(embed(crop))
    feats = np.stack(feats).astype(np.float32)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, feats)
    return feats


def main():
    ap = argparse.ArgumentParser(description="Fit the OOD gate")
    ap.add_argument("--manifest", default="data/samples/manifest.csv")
    ap.add_argument("--out", default="data/gate.npz")
    ap.add_argument("--cache", default="data/cache/fit_embeddings.npy")
    ap.add_argument("--percentile", type=float, default=99.0)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    fit_paths = df[df["split"] == "fit"]["path"].tolist()
    if not fit_paths:
        raise SystemExit("no fit-split images in manifest; run prepare_gate_data.py first")

    feats = _features_for(fit_paths, args.cache)
    gate = OODGate.fit(feats, percentile=args.percentile)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    gate.save(args.out)
    print(f"fit on {len(feats)} images; threshold={gate.threshold:.3f}; saved -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it (after Task 6 produced a manifest)**

Run: `cd /home/hlupek/Study/Lumen-i- && python scripts/fit_gate.py`
Expected: progress bar, then `fit on N images; threshold=...; saved -> data/gate.npz`. `data/gate.npz` exists.

- [ ] **Step 3: Verify the saved gate loads**

Run:
```bash
cd /home/hlupek/Study/Lumen-i- && python -c "from lumen.gating.ood_gate import OODGate; g=OODGate.load('data/gate.npz'); print('threshold', round(g.threshold,3), 'dim', g.mean.shape)"
```
Expected: prints a threshold and `dim (384,)`

- [ ] **Step 4: Commit** (script only)

```bash
cd /home/hlupek/Study/Lumen-i-
git add scripts/fit_gate.py
git commit -m "feat(gating): fit_gate script builds data/gate.npz"
```

---

### Task 8: `eval_gate.py` — validation report + contact sheet

**Files:**
- Create: `scripts/eval_gate.py`

**Interfaces:**
- Consumes: `data/gate.npz` (Task 7), `data/samples/manifest.csv` eval split, `data/negatives/*` (Task 6), `preprocess_and_gate` (Task 5)
- Produces: prints skin-pass rate + negative-reject rate; saves `reports/gate_hist.png` and `reports/gate_contact_sheet.png`

- [ ] **Step 1: Write the script**

Create `scripts/eval_gate.py`:
```python
#!/usr/bin/env python3
"""Evaluate the OOD gate: skin should pass, non-skin should be rejected.

Usage:
    python scripts/eval_gate.py
    python scripts/eval_gate.py --help
"""
import argparse
import os
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from lumen.gating.ood_gate import OODGate
from lumen.gating.pipeline import preprocess_and_gate


def _run(paths, gate):
    results = []
    for p in paths:
        out = preprocess_and_gate(p, gate)
        results.append((p, out))
    return results


def main():
    ap = argparse.ArgumentParser(description="Evaluate the OOD gate")
    ap.add_argument("--gate", default="data/gate.npz")
    ap.add_argument("--manifest", default="data/samples/manifest.csv")
    ap.add_argument("--negatives", default="data/negatives")
    ap.add_argument("--reports", default="reports")
    args = ap.parse_args()

    gate = OODGate.load(args.gate)
    df = pd.read_csv(args.manifest)
    skin_paths = df[df["split"] == "eval"]["path"].tolist()
    neg_paths = sorted(glob(os.path.join(args.negatives, "*")))

    skin = _run(skin_paths, gate)
    negs = _run(neg_paths, gate)

    skin_pass = np.mean([o["status"] == "ok" for _, o in skin]) if skin else float("nan")
    neg_reject = np.mean([o["status"] == "unclassified" for _, o in negs]) if negs else float("nan")
    print(f"skin-pass rate:    {skin_pass:.1%}  (n={len(skin)})")
    print(f"negative-reject:   {neg_reject:.1%}  (n={len(negs)})")
    print(f"threshold:         {gate.threshold:.3f}")

    os.makedirs(args.reports, exist_ok=True)

    # distance histogram
    skin_scores = [o["score"] for _, o in skin if o["score"] is not None]
    neg_scores = [o["score"] for _, o in negs if o["score"] is not None]
    plt.figure(figsize=(7, 4))
    plt.hist(skin_scores, bins=30, alpha=0.6, label="skin")
    plt.hist(neg_scores, bins=30, alpha=0.6, label="non-skin")
    plt.axvline(gate.threshold, color="k", ls="--", label="threshold")
    plt.xlabel("Mahalanobis distance"); plt.ylabel("count"); plt.legend()
    plt.title("OOD gate score distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(args.reports, "gate_hist.png"), dpi=120)
    plt.close()

    # contact sheet: 4 skin + 4 negatives (original -> verdict)
    examples = skin[:4] + negs[:4]
    n = len(examples)
    if n:
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2))
        if n == 1:
            axes = [axes]
        for ax, (p, out) in zip(axes, examples):
            ax.imshow(Image.open(p).convert("RGB"))
            ax.set_title(f"{out['status']}\nd={out['score']:.1f}" if out["score"] is not None
                         else out["status"], fontsize=9)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(args.reports, "gate_contact_sheet.png"), dpi=120)
        plt.close()
    print(f"saved reports -> {args.reports}/gate_hist.png, {args.reports}/gate_contact_sheet.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `cd /home/hlupek/Study/Lumen-i- && python scripts/eval_gate.py`
Expected: prints skin-pass rate, negative-reject rate, threshold; writes two PNGs to `reports/`.

- [ ] **Step 3: Verify report artifacts**

Run: `cd /home/hlupek/Study/Lumen-i- && ls -la reports/gate_hist.png reports/gate_contact_sheet.png`
Expected: both files exist, non-zero size.

- [ ] **Step 4: Commit** (script only)

```bash
cd /home/hlupek/Study/Lumen-i-
git add scripts/eval_gate.py
git commit -m "feat(gating): eval_gate script (rates + histogram + contact sheet)"
```

---

## Manual verification (end-to-end)

After Task 8, sanity-check the whole path on one real image:
```bash
cd /home/hlupek/Study/Lumen-i-
python -c "
from lumen.gating.ood_gate import OODGate
from lumen.gating.pipeline import preprocess_and_gate
import pandas as pd
g = OODGate.load('data/gate.npz')
p = pd.read_csv('data/samples/manifest.csv').query(\"split=='eval'\")['path'].iloc[0]
print('skin  ->', preprocess_and_gate(p, g)['status'])
import glob; n = glob.glob('data/negatives/*')[0]
print('non   ->', preprocess_and_gate(n, g)['status'])
"
```
Expected: `skin  -> ok` and `non   -> unclassified` (most of the time; exact behavior depends on the fitted threshold — the eval report quantifies it).
