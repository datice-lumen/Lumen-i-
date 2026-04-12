"""Dataset and data loading for melanoma training.

Provides SkinImageDataset (a PyTorch Dataset) with built-in augmentation
for class-imbalanced, fairness-aware training.
"""

import os
import multiprocessing
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

from lumen.training.augmentation import augment


def load_and_prepare_image(im_name, fold_id, base_dir, target_size=(224, 224)):
    """Load a preprocessed image from the fold directory structure.

    Args:
        im_name: Image filename (e.g. "ISIC_1234567.jpg").
        fold_id: Fold number (used to locate the subfolder).
        base_dir: Root directory containing fold subfolders.
        target_size: Resize target.

    Returns:
        Float32 numpy array in [0, 1] with shape (H, W, 3), or None on error.
    """
    img_path = os.path.join(base_dir, f"{fold_id}_fold", f"pre_{im_name}")
    try:
        img = Image.open(img_path).convert("RGB").resize(target_size)
    except (UnidentifiedImageError, OSError) as e:
        print(f"Error loading image {img_path}: {e}")
        return None
    img = np.array(img, dtype=np.float32) / 255.0
    return img


def _process_entry(entry, folds_to_include, base_dir, img_size, mode, augm_needed):
    """Process a single metadata entry into training samples.

    Returns a list of sample dicts, each with 'img', 'label', 'fitz_group', 'augm_bool'.
    """
    if entry["fold_id"] not in folds_to_include:
        return []

    original_img = load_and_prepare_image(
        entry["image_name"], entry["fold_id"], base_dir, target_size=img_size
    )
    if original_img is None:
        return []

    label = entry["true_class"]
    fitz = entry["fitz_group"]

    samples = [{
        "img": original_img,
        "label": label,
        "fitz_group": fitz,
        "augm_bool": False,
    }]

    if mode == "train":
        n_aug = 0
        if label == 1:
            n_aug += augm_needed
        elif fitz == 56:
            # Group 56 (dark skin) is undersampled — give it 1 extra augmentation
            n_aug += 1

        if n_aug > 0:
            for aug_img in augment(original_img, n_aug):
                samples.append({
                    "img": aug_img,
                    "label": label,
                    "fitz_group": fitz,
                    "augm_bool": True,
                })

    return samples


class SkinImageDataset(Dataset):
    """PyTorch dataset for skin lesion images with augmentation.

    Loads all images into memory at init time using parallel workers.
    Augmentations are computed upfront (not on-the-fly).

    Args:
        metadata: List of dicts with keys 'image_name', 'fold_id', 'true_class', 'fitz_group'.
        folds_to_include: Set/list of fold IDs to include.
        base_dir: Root directory with fold subfolders.
        img_size: Target image size tuple (H, W).
        mode: "train" (with augmentation) or "val"/"test" (no augmentation).
        augm_needed: Number of augmented copies per class-1 image.
    """

    def __init__(self, metadata, folds_to_include, base_dir, img_size=(224, 224),
                 mode="train", augm_needed=1):
        self.samples = []
        self.mode = mode

        process_func = partial(
            _process_entry,
            folds_to_include=set(folds_to_include),
            base_dir=base_dir,
            img_size=img_size,
            mode=mode,
            augm_needed=augm_needed,
        )
        workers = max(1, multiprocessing.cpu_count() - 2)
        print(f"[INFO] Loading dataset with {workers} threads...")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(tqdm(executor.map(process_func, metadata), total=len(metadata)))

        for sample_list in results:
            self.samples.extend(sample_list)

        print(f"[INFO] Loaded {len(self.samples)} samples ({mode})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = torch.tensor(sample["img"].copy(), dtype=torch.float32).permute(2, 0, 1)
        label = torch.tensor(sample["label"], dtype=torch.float32)
        return img, label, sample["fitz_group"], sample["augm_bool"]


def calculate_class_weights(loader, device="cuda"):
    """Compute sklearn balanced class weights from a DataLoader.

    Returns a tensor of shape (2,) on the specified device.
    """
    all_labels = []
    for _, labels, _, _ in loader:
        all_labels.append(labels.numpy())
    all_labels = np.concatenate(all_labels, axis=0)
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=all_labels)
    return torch.tensor(weights, dtype=torch.float).to(device)
