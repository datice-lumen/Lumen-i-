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


def remove_hair(in_image, kernel_size=25):
    """Remove dark hair artifacts using black-hat morphology + inpainting.

    Args:
        in_image: RGB image as numpy array. The pipeline standardises on RGB
            after the entry-point conversion in ``preprocess_*`` callers.
        kernel_size: Side of the square black-hat structuring element. The fused
            448px pipeline scales this with the intermediate resolution; the
            default (25) preserves the original 800px pipeline behaviour.

    Returns:
        (hair_mask, inpainted_img) — binary mask of detected hair pixels,
        and the cleaned image (same color space as input).
    """
    gray = cv2.cvtColor(in_image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 20, 255, cv2.THRESH_BINARY)
    inpainted_img = cv2.inpaint(in_image, hair_mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
    return hair_mask, inpainted_img


def preprocess_fused(rgb, target_size=448):
    """Preprocess an RGB image for the fused metadata model (model_10_6).

    Matches model_10_6/preprocess.py exactly: center square crop -> resize to
    ``max(800, 2*target_size)`` -> hair removal with a proportional (odd) kernel
    -> resize to ``target_size`` (LANCZOS4).

    Args:
        rgb: RGB image as a numpy array (any size).
        target_size: Final square side (default 448, the model's training size).

    Returns:
        (final_rgb, hair_mask, inpainted_rgb) — the model-ready RGB image plus the
        intermediate hair mask and inpainted image (both at the intermediate
        resolution) so the pipeline stages can be streamed to the UI.
    """
    cropped = square_crop(rgb)
    intermed = max(800, target_size * 2)
    raw_kernel = round(25 * intermed / 800)
    hair_kernel = raw_kernel if raw_kernel % 2 == 1 else raw_kernel + 1
    resized = cv2.resize(cropped, (intermed, intermed), interpolation=cv2.INTER_AREA)
    hair_mask, inpainted = remove_hair(resized, kernel_size=hair_kernel)
    final = cv2.resize(inpainted, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    return final, hair_mask, inpainted


def preprocess_mobile(rgb, target_size=448):
    """Preprocess an RGB image for the mobile fine-tuned model (checkpoint_mobile_best).

    Matches mobile_eval/train_mobile.py::preprocess_image exactly: center square crop
    -> resize to ``target_size`` (LANCZOS4). Deliberately NO hair removal — the mobile
    model was fine-tuned on phone close-ups without it, and hair removal measurably
    lowers its sensitivity (see mobile_eval/FINDINGS.md). This is the image the model
    actually sees; the hair-removal stage is kept only as a cosmetic display step.

    Args:
        rgb: RGB image as a numpy array (any size).
        target_size: Final square side (default 448, the model's training size).

    Returns:
        The model-ready RGB image (target_size x target_size).
    """
    cropped = square_crop(rgb)
    return cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)


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
    """Full preprocessing pipeline: square crop -> resize 800 -> hair removal -> resize -> ITA.

    Square crop happens on the input image before any resize, preserving
    aspect ratio for non-3:2 inputs. Matches slavica's evaluate.py order.

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
    sq800 = cv2.resize(cropped, (800, 800), interpolation=cv2.INTER_AREA)
    _, hairless = remove_hair(sq800)
    resized = cv2.resize(hairless, target_size, interpolation=cv2.INTER_LANCZOS4)

    if not compute_ita:
        return resized

    avg_top2_ITA, all8_ita_list = calculate_ita_subregions(resized)
    metadata = {
        "brightest_ITA": avg_top2_ITA,
        "ita_list": all8_ita_list,
    }
    return resized, metadata


def _load_rgb(image_path):
    """Read a JPEG and convert to RGB. Returns None on missing file or decode failure."""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    orig = cv2.imread(image_path)
    if orig is None:
        print(f"Failed to read image: {image_path}")
        return None
    return cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)


def preprocess_from_path(image_path, target_size=(200, 200)):
    """Load image from disk and run the full pipeline (with ITA metadata).

    Used by the batch dataset-preparation script.

    Args:
        image_path: Path to a .jpg image.
        target_size: Final output size.

    Returns:
        (processed_image, metadata_dict) or None if the image is invalid.
    """
    rgb = _load_rgb(image_path)
    if rgb is None:
        return None
    if not check_resize(rgb):
        print(f"Skipping {image_path}: bad dimensions or quality")
        return None

    return preprocess(rgb, target_size=target_size, compute_ita=True)


def preprocess_for_inference(image_path, target_size=(224, 224)):
    """Load + preprocess an image for model inference (no ITA).

    Returns:
        Preprocessed RGB numpy array, or None if invalid.
    """
    rgb = _load_rgb(image_path)
    if rgb is None:
        return None
    if not check_resize(rgb):
        print(f"Skipping {image_path}: bad dimensions or quality")
        return None

    return preprocess(rgb, target_size=target_size, compute_ita=False)


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
