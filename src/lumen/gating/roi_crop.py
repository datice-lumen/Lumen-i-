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
