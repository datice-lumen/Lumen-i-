"""Data augmentation for melanoma images.

All functions operate on float32 numpy arrays in [0, 1] range with shape (H, W, 3).
"""

import numpy as np
import matplotlib.colors


def random_rotate(img):
    """Rotate by a random multiple of 90 degrees."""
    angle = np.random.choice([0, 90, 180, 270])
    if angle == 0:
        return img
    return np.rot90(img, angle // 90)


def flip_vertical(img):
    return np.flip(img, axis=0)


def flip_horizontal(img):
    return np.flip(img, axis=1)


def contrast_change(img):
    """Random contrast adjustment by +/- 2–10%."""
    factor = np.random.uniform(0.02, 0.1)
    pm = np.random.choice([-1, 1])
    return np.clip(img * (1 + pm * factor), 0, 1)


def brightness_change(img):
    """Random brightness shift by +/- 5–10%."""
    delta = np.random.uniform(0.05, 0.1)
    pm = np.random.choice([-1, 1])
    return np.clip(img + pm * delta, 0, 1)


def add_gaussian_noise(img):
    """Add Gaussian noise with random std in [0.01, 0.05]."""
    std = np.random.uniform(0.01, 0.05)
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    return np.clip(img + noise, 0, 1)


def color_jitter(img):
    """Random hue/saturation/value jitter in HSV space."""
    img_hsv = matplotlib.colors.rgb_to_hsv(img)

    hue_shift = np.random.uniform(-0.06, 0.06)
    img_hsv[..., 0] = (img_hsv[..., 0] + hue_shift) % 1.0

    sat_scale = np.random.uniform(0.8, 1.2)
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] * sat_scale, 0, 1)

    val_scale = np.random.uniform(0.95, 1.05)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] * val_scale, 0, 1)

    return np.clip(matplotlib.colors.hsv_to_rgb(img_hsv), 0, 1)


ALL_AUGMENTATIONS = [
    flip_vertical,
    flip_horizontal,
    contrast_change,
    brightness_change,
    add_gaussian_noise,
    color_jitter,
]


def augment(img, n):
    """Generate n augmented copies of an image.

    Each copy is randomly rotated, then 3+ random augmentation functions are
    applied in random order.

    Args:
        img: Float32 numpy array in [0, 1], shape (H, W, 3).
        n: Number of augmented copies to produce.

    Returns:
        List of n augmented images.
    """
    augmented = []
    for _ in range(n):
        img_aug = random_rotate(img)
        n_augs = np.random.randint(3, len(ALL_AUGMENTATIONS) + 1)
        selected = np.random.choice(ALL_AUGMENTATIONS, size=n_augs, replace=False)
        for aug_fn in selected:
            img_aug = aug_fn(img_aug)
        augmented.append(img_aug)
    return augmented
