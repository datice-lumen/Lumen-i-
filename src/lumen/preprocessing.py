"""Image preprocessing pipeline for dermatoscopic images.

Handles hair removal, square cropping, resizing, and ITA computation.
Used by both the batch preprocessing script and the inference pipeline.
"""

import os
import multiprocessing

import cv2
import numpy as np

from lumen.skin_tone import calculate_ita_subregions


def check_resize(in_image, min_side=400, aspect_range=(1.0, 2.0)):
    """Check whether an image meets minimum quality/aspect requirements.

    Accepts dermatoscopic shapes (1:1 to 2:1 width-to-height) with each side
    at least ``min_side`` pixels. ISIC-2019 mixes 4:3, 3:2, and other ratios,
    so a tolerant range is used instead of strict equality.
    """
    height, width = in_image.shape[:2]
    if width < min_side or height < min_side:
        return False
    ratio = width / height
    return aspect_range[0] <= ratio <= aspect_range[1]


def remove_hair(in_image):
    """Remove dark hair artifacts using black-hat morphology + inpainting.

    Args:
        in_image: RGB image as numpy array. The pipeline standardises on RGB
            after the entry-point conversion in ``preprocess_*`` callers.

    Returns:
        (hair_mask, inpainted_img) — binary mask of detected hair pixels,
        and the cleaned image (same color space as input).
    """
    gray = cv2.cvtColor(in_image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 20, 255, cv2.THRESH_BINARY)
    inpainted_img = cv2.inpaint(in_image, hair_mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
    return hair_mask, inpainted_img


def square_crop(image):
    """Center-crop to a square by trimming the wider dimension.

    Removes left/right margins which often contain dermatoscope edges
    that would interfere with skin tone estimation.
    """
    h, w = image.shape[:2]
    new_w = min(h, w)
    start_x = (w - new_w) // 2
    start_y = (h - new_w) // 2
    cropped = image[start_y:start_y + new_w, start_x:start_x + new_w]
    assert cropped.shape[0] == cropped.shape[1], (
        f"Crop failed: ({cropped.shape[0]}, {cropped.shape[1]})"
    )
    return cropped


def preprocess(image, target_size=(224, 224), compute_ita=True):
    """Full preprocessing pipeline: square crop -> hair removal -> resize -> ITA.

    Args:
        image: RGB image as numpy array (any size).
        target_size: Final (width, height) tuple.
        compute_ita: If True, also compute ITA subregion values.

    Returns:
        If compute_ita is True:
            (processed_image, metadata_dict) where metadata_dict contains
            'brightest_ITA' and 'ita_list'.
        If compute_ita is False:
            processed_image only.
    """
    cropped = square_crop(image)
    hair_mask, hairless = remove_hair(cropped)
    resized = cv2.resize(hairless, target_size, interpolation=cv2.INTER_LANCZOS4)

    if not compute_ita:
        return resized

    avg_top2_ITA, all8_ita_list = calculate_ita_subregions(resized)
    metadata = {
        "brightest_ITA": avg_top2_ITA,
        "ita_list": all8_ita_list,
    }
    return resized, metadata


def preprocess_from_path(image_path, target_size=(200, 200), first_resize=(1200, 800)):
    """Load image from disk, optionally downscale, then run full pipeline.

    Used by the batch preprocessing script. Does an intermediate resize to
    (1200, 800) to speed up hair removal while preserving quality.

    Args:
        image_path: Path to a .jpg image.
        target_size: Final output size.
        first_resize: Intermediate size before processing (or None to skip).

    Returns:
        (processed_image, metadata_dict) or None if the image is invalid.
    """
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None

    orig = cv2.imread(image_path)
    if orig is None:
        print(f"Failed to read image: {image_path}")
        return None
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    if not check_resize(orig):
        print(f"Skipping {image_path}: bad dimensions or quality")
        return None

    if first_resize is not None:
        orig = cv2.resize(orig, first_resize, interpolation=cv2.INTER_AREA)

    return preprocess(orig, target_size=target_size, compute_ita=True)


def preprocess_for_inference(image_path, target_size=(224, 224)):
    """Load + preprocess an image for model inference (no ITA needed).

    Converts to RGB, does intermediate resize, then the full pipeline.

    Returns:
        Preprocessed RGB numpy array, or None if invalid.
    """
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None

    orig = cv2.imread(image_path)
    if orig is None:
        print(f"Failed to read image: {image_path}")
        return None
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    if not check_resize(orig):
        print(f"Skipping {image_path}: bad dimensions or quality")
        return None

    orig = cv2.resize(orig, (1200, 800), interpolation=cv2.INTER_AREA)

    return preprocess(orig, target_size=target_size, compute_ita=False)


# --- Parallel helpers ---

def _preprocess_single(args):
    """Worker function for parallel_preprocess."""
    image_path, idx, target_size = args
    result = preprocess_from_path(image_path, target_size=target_size)
    return result


def _preprocess_for_inference_single(full_path):
    """Worker function for parallel inference preprocessing."""
    image_name = os.path.basename(full_path)
    processed = preprocess_for_inference(full_path)
    return (image_name, processed)


def parallel_preprocess(image_paths, num_processes=None, target_size=(200, 200)):
    """Run preprocess_from_path in parallel across multiple images.

    Args:
        image_paths: List of image file paths.
        num_processes: Number of worker processes. Defaults to ~70% of CPU cores.
        target_size: Final output size.

    Returns:
        List of (processed_image, metadata_dict) results (None entries for failures).
    """
    if num_processes is None:
        num_processes = max(1, int(multiprocessing.cpu_count() * 0.7))

    inputs = [(path, idx, target_size) for idx, path in enumerate(image_paths)]
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(_preprocess_single, inputs)
    return results


def parallel_preprocess_for_inference(image_names, input_folder, num_processes=None):
    """Run inference preprocessing in parallel.

    Args:
        image_names: List of image filenames.
        input_folder: Directory containing the images.
        num_processes: Number of worker processes. Defaults to ~70% of CPU cores.

    Returns:
        List of (image_name, processed_array) tuples.
    """
    if num_processes is None:
        num_processes = max(1, int(multiprocessing.cpu_count() * 0.7))

    full_paths = [os.path.join(input_folder, name) for name in image_names]
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = []
        for idx, result in enumerate(pool.imap(_preprocess_for_inference_single, full_paths)):
            results.append(result)
            if (idx + 1) % 500 == 0:
                print(f"\t-> Preprocessed {idx + 1} images...")
    return results
