# Design: OOD skin-gate + ROI background-crop (POC)

**Date:** 2026-07-15
**Status:** Approved design → pending spec review
**Scope:** Standalone proof-of-concept, run locally (CPU), code in the Lumen repo.

## Problem

The melanoma classifier confidently labels *any* image (a cat, a wall, a document) as
benign/malignant. The web app receives ordinary **phone photos of moles**, which contain
real-world clutter (fingers, clothing, table). We want a pre-classification stage that:

1. **Removes the background** — crop to the skin/lesion region, dropping clutter.
2. **Rejects non-skin inputs** — if the image isn't skin, return `unclassified` instead of
   a meaningless melanoma prediction.

This is an **out-of-distribution (OOD) input gate** plus a **region-of-interest (ROI) crop**.

## Goals

- A single entry point `preprocess_and_gate(image, point=None)` that returns either a cropped
  ROI ready for the classifier, or an `unclassified` verdict.
- No training of any kind. DINOv2 and SAM 2 are used **frozen**. The gate is *fit* = summary
  statistics only (mean, covariance, threshold). The existing melanoma model is never touched.
- Runs locally on CPU. Nothing runs on the slavica HPC.

## Non-goals

- Wiring the pipeline into the FastAPI backend / Vue frontend (integration point noted, not built).
- Closing the dermoscopy→phone **domain gap** in the classifier itself (needs fine-tuning on
  phone/clinical data — out of scope; tracked as a known limitation).
- Tight lesion-only segmentation. We crop the **skin region** (removing non-skin background),
  not the lesion boundary.

## Context (discovered)

- **Classifier backbone (slavica `train.py`):** DINOv2-S (`dinov2_vits14_reg`, frozen, 384-d) ++
  a trainable TinyCNN + metadata MLP, late-fused. So DINOv2 embeddings are already the vision
  representation the classifier relies on — the gate reuses the same representation.
- **Local repo (`Lumen-i-`):** FastAPI backend (`web_app/backend/router.py`, `POST /process`
  streams pipeline steps over SSE) + Vue frontend (has `PipelineSteps.vue`, `LiveAnalyzer.vue`).
  Model/preprocess code in `src/lumen/` (`preprocessing.py`, `inference.py`, `model.py`).
- **Local env:** CPU-only, Python 3.10; `cv2`, `sklearn`, `skimage`, `PIL`, `matplotlib`,
  `fastapi` present; `ultralytics`/`sam2` absent; GitHub + HuggingFace reachable.
- **Data (on mounted slavica, copied locally once):** MILK10k has 5,240 `clinical: close-up`
  (phone-style, real backgrounds) + 5,240 `dermoscopic`, labeled in
  `MILK10k_Training_Metadata.csv`. Plus 67k preprocessed dermoscopy images.

## Architecture — one function, two frozen stages

```
preprocess_and_gate(image, point=None) -> dict
  │
  ├─ Stage 1  ROI crop  (SAM 2 tiny, frozen; ultralytics)
  │     point = explicit (x, y) if provided, else image center
  │     SAM 2 -> mask -> bounding box -> rectangular crop + margin
  │     (keep surrounding skin; NEVER hard-mask to black)
  │     empty / tiny mask (< min_area_frac) -> center-square fallback (reason logged)
  │
  └─ Stage 2  OOD gate  (DINOv2-S, frozen + Mahalanobis stats)
        crop -> transform (resize 448 + ImageNet norm) -> DINOv2 384-d CLS feature
             -> Mahalanobis distance d to the skin distribution
             -> d <= threshold : {status:"ok",           crop, score:d, ...}
                d >  threshold : {status:"unclassified",  crop:None, score:d, ...}
```

**Why crop before gating, and no black cutout:** the crop makes inputs uniformly
skin-dominated (comparable across dermoscopy/clinical), and hard black masks create unnatural
high-contrast edges that distort DINOv2 embeddings. We keep a rectangular crop with margin.

## Components (small, single-purpose)

| File | Responsibility |
|---|---|
| `src/lumen/gating/roi_crop.py` | Load SAM 2 (tiny); `crop_roi(img, point=None) -> (crop, mask, bbox)` + fallback |
| `src/lumen/gating/dino_embed.py` | Load DINOv2-S (torch.hub, cached after first download); `embed(img) -> np.ndarray[384]` (resize 448 + ImageNet norm, CPU) |
| `src/lumen/gating/ood_gate.py` | `OODGate.fit(feats)`, `.score(feat)`, `.passes(feat)`; save/load `gate.npz` |
| `src/lumen/gating/pipeline.py` | `preprocess_and_gate(image, point=None) -> dict` wiring both stages |
| `scripts/fit_gate.py` | Offline: sample skin images -> crop -> embed -> fit stats -> save `data/gate.npz` |
| `scripts/eval_gate.py` | Run pipeline on held-out skin + negatives; print rates; save contact sheet |

## OOD gate — modeling detail

- **In-distribution = "skin imagery broadly"** (matches intent "if not skin → unclassified"),
  fit on a mix of MILK10k clinical (cropped) + dermoscopy. Not dermoscopy-only.
- **One-class Mahalanobis** with **Ledoit-Wolf shrinkage** covariance (`sklearn.covariance.LedoitWolf`)
  for stability at n < 384. Distance `d(x) = sqrt((x-μ)ᵀ Σ⁻¹ (x-μ))`.
- **Threshold** = 99th percentile of Mahalanobis distances over held-out skin (≈1% skin
  false-reject). Configurable.
- **Consistency:** fit images go through the *same* crop→embed path as inference. Crops and
  embeddings are cached to disk so re-runs are instant.

## Output contract

```python
{
  "status": "ok" | "unclassified" | "error",
  "crop":   np.ndarray | None,   # RGB HxWx3 ROI to feed the classifier (None unless ok)
  "score":  float,               # Mahalanobis distance
  "threshold": float,
  "bbox":   [x0, y0, x1, y1] | None,
  "reason": str                  # "in_distribution" | "ood_distance>threshold" | "no_mask_fallback" | "unreadable"
}
```

## Data sourcing

- **Skin (one-time copy from mount → `data/samples/`, gitignored):**
  - ~500 MILK10k `clinical: close-up` + ~300 MILK10k `dermoscopic` + ~200 preprocessed dermoscopy.
  - Split into fit (~800) and held-out eval (~200). Sizes configurable.
- **Negatives (download → `data/negatives/`, ~150):** varied non-skin images
  (animals, objects, scenery, screenshots/text, non-closeup faces) from a public source, for
  measuring false-accept rate only. The gate itself is one-class and needs no negatives to fit.

## Environment & dependencies

- Runs in the local Python env (CPU). Add **`ultralytics`** (provides SAM 2, auto-downloads the
  `sam2_t` checkpoint). Everything else already present.
- DINOv2-S loads via `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')` (downloads
  once; GitHub reachable).
- Work on a **`feat/ood-gate` branch** — `main` has uncommitted frontend WIP that must not be
  entangled. No commits unless requested.

## Error handling

- Unreadable image → `{status:"error", reason:"unreadable"}`.
- Empty / tiny SAM 2 mask (< `min_area_frac`, default 5%) → center-square crop, `reason:"no_mask_fallback"`.
- Gate accepts any 384-d vector; NaN/inf feature → `error`.
- SAM 2 checkpoint download failure → clear message with manual-download instructions.

## Testing / validation

- **Unit:** `OODGate` on synthetic data — in-distribution cluster passes; a far point is rejected;
  save/load round-trips.
- **Integration (`eval_gate.py`):** held-out MILK10k clinical + dermoscopy (expect ~99% **pass**);
  the ~150 non-skin negatives (expect high **reject**). Report skin-pass rate and
  negative-reject rate at the chosen threshold, plus a distance histogram.
- **Visual:** contact sheet (matplotlib) of `original → crop → verdict` for ~12 examples
  (clinical, dermoscopy, negatives) saved to `reports/` for eyeballing.

## Integration point (future, not in POC)

In `web_app/backend/router.py::process_image`, `preprocess_and_gate` runs before classification.
It maps naturally onto the existing SSE pipeline steps: emit "Detecting region" (crop),
"Checking it's skin" (gate); on `unclassified`, stop and return that verdict; otherwise pass the
crop to the classifier. Optional `point` supports a future "tap your mole" UI gesture.

## Risks & limitations

- **Domain gap dominates accuracy.** Even a perfect crop + gate does not make a dermoscopy-trained
  classifier accurate on phone photos. Fixing that needs fine-tuning on phone/clinical data
  (e.g. PAD-UFES-20 / MILK10k clinical) — separate effort. The gate may correctly reject many
  phone photos as OOD until then; that is honest, not a bug.
- **CPU latency.** SAM 2 on CPU is ~seconds/image — fine for the POC; the live web app would want
  a GPU or a lighter cropper. Flagged, not solved here.
- **Threshold calibration** depends on the fit sample being representative; revisit with real
  production uploads.
```
