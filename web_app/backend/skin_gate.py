"""Skin-image gate for the web app.

Answers one question before the melanoma pipeline runs: *is the uploaded photo
actually a close-up of skin?* Wraps the frozen DINOv2 + one-class OOD gate
(`lumen.gating`). The fitted gate is loaded once and cached.

Fail-open: if no fitted gate is available (data/gate.npz missing), `is_skin`
returns None and the caller proceeds as before — the app never breaks just
because the gate hasn't been fitted yet.
"""
from __future__ import annotations

import functools
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BACKEND_DIR.parents[1]  # backend -> web_app -> repo root


def _gate_path() -> Optional[Path]:
    """First existing gate file among: $LUMEN_GATE_PATH, backend/gate.npz, data/gate.npz."""
    candidates = []
    env = os.environ.get("LUMEN_GATE_PATH")
    if env:
        candidates.append(Path(env))
    candidates.append(_BACKEND_DIR / "gate.npz")
    candidates.append(_REPO_ROOT / "data" / "gate.npz")
    for p in candidates:
        if p.is_file():
            return p
    return None


@functools.lru_cache(maxsize=1)
def _load_gate():
    from lumen.gating.ood_gate import OODGate

    path = _gate_path()
    if path is None:
        logger.warning("Skin gate DISABLED: no fitted gate found "
                       "(set LUMEN_GATE_PATH or run scripts/fit_gate.py). "
                       "Uploads will not be screened for skin content.")
        return None
    try:
        gate = OODGate.load(str(path))
        logger.info("Loaded skin gate from %s (threshold=%.3f)", path, gate.threshold)
        return gate
    except Exception as e:
        logger.exception("Failed to load skin gate from %s: %s", path, e)
        return None


def is_skin(rgb: np.ndarray) -> Optional[dict]:
    """Screen an RGB uint8 image for skin content.

    Returns the `detect_lesion` result dict
    (`status`/`is_skin`/`score`/`threshold`/`reason`), or **None** if the gate is
    disabled (no fitted gate) — in which case the caller should proceed normally.
    """
    gate = _load_gate()
    if gate is None:
        return None
    from lumen.gating.detect import detect_lesion

    return detect_lesion(rgb, gate)
