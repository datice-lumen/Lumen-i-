"""Tests for the mobile model's preprocessing (crop -> 448, no hair removal).

The mobile fine-tuned checkpoint was trained on phone close-ups WITHOUT hair
removal (mobile_eval/train_mobile.py::preprocess_image). preprocess_mobile must
reproduce that exactly: a center square crop then a LANCZOS4 resize, and nothing
else — no black-hat/inpaint stage, so the pixels are untouched by hair removal.
"""

import cv2
import numpy as np

from lumen.preprocessing import preprocess_mobile, preprocess_fused, square_crop


def _synthetic_image():
    # Non-square, deterministic gradient so resize/crop effects are observable.
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 255, 400, dtype=np.uint8)[None, :]
    img[:, :, 1] = np.linspace(0, 255, 300, dtype=np.uint8)[:, None]
    img[100:120, 150:250] = 10  # dark "hair-like" streak
    return img


def test_preprocess_mobile_shape_is_target_square():
    out = preprocess_mobile(_synthetic_image(), target_size=448)
    assert out.shape == (448, 448, 3)
    assert out.dtype == np.uint8


def test_preprocess_mobile_is_plain_crop_and_resize():
    """It must equal a direct crop+resize — proving no hair removal is applied."""
    rgb = _synthetic_image()
    expected = cv2.resize(square_crop(rgb), (448, 448), interpolation=cv2.INTER_LANCZOS4)
    out = preprocess_mobile(rgb, target_size=448)
    assert np.array_equal(out, expected)


def test_preprocess_mobile_differs_from_fused_on_hair():
    """The dark streak survives in the mobile path but is inpainted in the fused path."""
    rgb = _synthetic_image()
    mobile = preprocess_mobile(rgb, target_size=448)
    fused, _, _ = preprocess_fused(rgb, target_size=448)
    # The two paths produce different pixels precisely because fused removes hair.
    assert not np.array_equal(mobile, fused)
