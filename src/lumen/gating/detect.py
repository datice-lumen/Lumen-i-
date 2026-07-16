"""Skin-lesion presence detection (frozen models, no training).

The detector answers a single question: *is there a skin lesion in this image?*
It embeds the whole image with a frozen DINOv2-S and scores it against a fitted
one-class Mahalanobis gate. In-distribution -> "skin"; out-of-distribution ->
"unclassified" (e.g. a photo of a face, a wall, a keyboard). No background removal
or ROI cropping happens here — that is a separate concern.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from lumen.gating.dino_embed import embed
from lumen.gating.ood_gate import OODGate


def _load_rgb(image) -> Optional[np.ndarray]:
    """Coerce a path / PIL image / numpy array to an RGB uint8 array, or None."""
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    try:
        return np.array(Image.open(image).convert("RGB"))
    except Exception:
        return None


def detect_lesion(image, gate: OODGate) -> dict:
    """Decide whether `image` contains an in-distribution skin lesion.

    `image`: a filesystem path, a PIL image, or an RGB uint8 numpy array.
    `gate`:  a fitted `OODGate` (or any object exposing `.score(feat)` and `.threshold`).

    Returns a dict:
      status:    "skin" | "unclassified" | "error"
      is_skin:   bool
      score:     Mahalanobis distance (float) or None on error
      threshold: gate.threshold
      reason:    short human-readable explanation
    """
    rgb = _load_rgb(image)
    if rgb is None:
        return {"status": "error", "is_skin": False, "score": None,
                "threshold": gate.threshold, "reason": "unreadable"}

    try:
        feat = embed(rgb)
        score = gate.score(feat)
    except ValueError:
        return {"status": "error", "is_skin": False, "score": None,
                "threshold": gate.threshold, "reason": "bad_feature"}

    if score <= gate.threshold:
        return {"status": "skin", "is_skin": True, "score": score,
                "threshold": gate.threshold, "reason": "in_distribution"}
    return {"status": "unclassified", "is_skin": False, "score": score,
            "threshold": gate.threshold, "reason": "ood_distance>threshold"}
