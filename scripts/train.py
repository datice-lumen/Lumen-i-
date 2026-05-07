#!/usr/bin/env python3
"""Train a melanoma classifier with fairness-aware loss and k-fold cross-validation.

Usage:
    python scripts/train.py --data-dir path/to/preprocessed --metadata path/to/metadata.json
"""

import argparse
import json
import random
from collections import defaultdict

import numpy as np
import torch

from lumen.skin_tone import assign_fitz_group
from lumen.training.trainer import k_fold_training


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_augm_needed(data, target_ratio=0.15):
    unique_counts = defaultdict(int)
    for item in data:
        unique_counts[(item["fitz_group"], item["true_class"])] += 1

    sample0 = sum(
        count * (2 if fg == 56 else 1)
        for (fg, tc), count in unique_counts.items() if tc == 0
    )
    sample1 = sum(count for (fg, tc), count in unique_counts.items() if tc == 1)

    koef = (target_ratio * sample0) / (sample1 * (1 - target_ratio))
    augm_needed = max(int(koef - 1), 0)
    print(f"Class balance: {sample0} class-0, {sample1} class-1")
    print(f"Augmentations per class-1 image: {augm_needed}")
    return augm_needed


def main():
    parser = argparse.ArgumentParser(description="Train melanoma classifier")
    parser.add_argument("--data-dir", required=True, help="Preprocessed fold directory")
    parser.add_argument("--metadata", required=True, help="Path to metadata JSON")
    parser.add_argument("--output-dir", default="models", help="Where to save model weights (default: models)")
    parser.add_argument("--target-ratio", type=float, default=0.15, help="Target class-1 ratio (default: 0.15)")
    parser.add_argument("--num-folds", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--test-fold", type=int, default=4, help="Fold reserved for testing (default: 4)")
    parser.add_argument("--epochs", type=int, default=35, help="Max epochs (default: 35)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate (default: 3e-5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    set_seed(args.seed)
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # Load metadata
    with open(args.metadata) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} metadata entries.")

    # Assign Fitzpatrick groups
    for item in data:
        item["fitz_group"] = assign_fitz_group(item["brightest_ITA"])

    augm_needed = compute_augm_needed(data, args.target_ratio)

    # K-fold cross-validation
    all_folds = list(range(args.num_folds))
    train_val_folds = [f for f in all_folds if f != args.test_fold]

    for val_fold in train_val_folds:
        train_folds = [f for f in train_val_folds if f != val_fold]
        print(f"\n{'='*60}")
        print(f"Train: folds {train_folds} | Val: [{val_fold}] | Test: [{args.test_fold}]")
        print(f"{'='*60}")

        save_path = os.path.join(args.output_dir, f"model_fold_{val_fold}.pth")
        k_fold_training(
            metadata=data,
            base_dir=args.data_dir,
            train_folds=train_folds,
            val_folds=[val_fold],
            augm_needed=augm_needed,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            class_weights_override=torch.tensor([0.5, 8.0], dtype=torch.float),
            save_path=save_path,
        )


if __name__ == "__main__":
    main()
