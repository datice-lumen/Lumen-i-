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
