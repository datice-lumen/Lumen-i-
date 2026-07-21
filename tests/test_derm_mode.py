"""Tests for the dermatoscope-mode addition to the web app.

Two concerns:
  1. The dermatoscopic checkpoint (najbolji_10_6.pt / model_10_6) loads and carries a
     well-formed metadata config, using the no-network head-only validator so this runs
     without pulling DINOv2 from torch.hub.
  2. The backend's mode routing helpers pick the right model input per mode.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from lumen.model_meta import load_submodules_for_validation

REPO_ROOT = Path(__file__).resolve().parent.parent
DERM_CKPT = REPO_ROOT / "web_app" / "backend" / "najbolji_10_6.pt"
BACKEND_DIR = REPO_ROOT / "web_app" / "backend"


# --- 1) the dermatoscopic checkpoint loads ---------------------------------- #

def test_derm_checkpoint_loads_and_has_meta_cfg():
    """najbolji_10_6.pt loads its trainable heads and yields a usable meta_cfg."""
    assert DERM_CKPT.exists(), f"missing derm checkpoint: {DERM_CKPT}"
    cfg = load_submodules_for_validation(str(DERM_CKPT))

    assert cfg["resize"] == 448
    assert cfg["age_std"] > 0
    # The metadata encoding is fixed-width; these categories drive the 11-dim vector.
    assert "unknown" in cfg["sex_categories"]
    assert "unknown" in cfg["site_categories"]
    assert len(cfg["site_categories"]) == 6


# --- 2) mode routing helpers ------------------------------------------------ #

def _import_router():
    """Import the backend router module (needs fastapi/multipart + backend on path)."""
    pytest.importorskip("fastapi")
    pytest.importorskip("multipart")
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    import router  # noqa: WPS433 (import inside function is intentional here)

    return router


def test_resolve_mode_normalises_input():
    router = _import_router()
    assert router.resolve_mode("phone") == "phone"
    assert router.resolve_mode("derm") == "derm"
    # Anything unrecognised (None, typo, empty) falls back to the default.
    assert router.resolve_mode(None) == "phone"
    assert router.resolve_mode("dermatoscope") == "phone"
    assert router.resolve_mode("") == "phone"


def test_select_model_input_picks_by_mode():
    router = _import_router()
    mobile_rgb = np.zeros((4, 4, 3), dtype=np.uint8)      # stand-in for preprocess_mobile
    derm_rgb = np.ones((4, 4, 3), dtype=np.uint8)         # stand-in for hair-removed 448

    # phone -> the plain-crop image; derm -> the hair-removed image.
    assert np.array_equal(router.select_model_input("phone", mobile_rgb, derm_rgb), mobile_rgb)
    assert np.array_equal(router.select_model_input("derm", mobile_rgb, derm_rgb), derm_rgb)
