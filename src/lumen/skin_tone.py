"""ITA-based skin tone estimation and Fitzpatrick type mapping."""

import numpy as np
import cv2


def calculate_ita_subregions(in_image):
    """Compute ITA from 8 peripheral subregions, return top-2 average and full list.

    Uses 20% of image width for subregion size. Samples corners and cardinal edges
    to avoid the central lesion area.

    Args:
        in_image: BGR image as numpy array.

    Returns:
        (avg_top2_ITA, all8_ita_list) — the average of the two brightest subregion
        ITA values, and the list of all 8 ITA values.
    """
    h, w = in_image.shape[:2]
    size = int(0.2 * w)

    regions = {
        "Top-Left": (0, 0, size, size),
        "Top-Right": (w - size, 0, w, size),
        "Bottom-Left": (0, h - size, size, h),
        "Bottom-Right": (w - size, h - size, w, h),
        "North": (w // 2 - size // 2, 0, w // 2 + size // 2, size),
        "South": (w // 2 - size // 2, h - size, w // 2 + size // 2, h),
        "West": (0, h // 2 - size // 2, size, h // 2 + size // 2),
        "East": (w - size, h // 2 - size // 2, w, h // 2 + size // 2),
    }

    all8_ita_list = []
    for x1, y1, x2, y2 in regions.values():
        sub_img = in_image[y1:y2, x1:x2]
        L, _, B = cv2.split(sub_img)
        L_mean = np.mean(L)
        B_mean = np.mean(B)
        ITA = np.arctan((L_mean - 50) / B_mean) * (180 / np.pi)
        all8_ita_list.append(ITA)

    avg_top2_ITA = sum(sorted(all8_ita_list, reverse=True)[:2]) / 2
    return avg_top2_ITA, all8_ita_list


def get_fitzpatrick(ita):
    """Map an ITA value to a Fitzpatrick skin type (1–6)."""
    if ita > 55:
        return 1
    elif ita > 41:
        return 2
    elif ita > 28:
        return 3
    elif ita > 19:
        return 4
    elif ita > 10:
        return 5
    else:
        return 6


def get_fitzpatrick_label(ita):
    """Map an ITA value to a human-readable Fitzpatrick label."""
    labels = {
        1: "I (Very Light)",
        2: "II (Light)",
        3: "III (Intermediate)",
        4: "IV (Tan)",
        5: "V (Brown)",
        6: "VI (Dark Brown/Black)",
    }
    return labels[get_fitzpatrick(ita)]


def assign_fitz_group(ita, scale=1.1):
    """Assign a grouped Fitzpatrick category (12, 3, 4, 56) used for training.

    Groups I–II and V–VI together because they are underrepresented individually.

    Args:
        ita: Individual Typology Angle value.
        scale: Multiplier applied to ITA before classification (default 1.1,
               as used during training).
    """
    fitz = get_fitzpatrick(ita * scale)
    if fitz <= 2:
        return 12
    elif fitz >= 5:
        return 56
    return fitz
