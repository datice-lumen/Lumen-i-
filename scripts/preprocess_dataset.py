#!/usr/bin/env python3
"""Batch preprocessing: deduplicate, preprocess, compute ITA, k-fold split, and save.

Usage:
    python scripts/preprocess_dataset.py --images path/to/train --labels path/to/GroundTruth.csv --duplicates path/to/duplicates.csv --output path/to/output

All flags have sensible defaults — see --help.
"""

import argparse
import os
import json

import numpy as np
import pandas as pd
import cv2

from lumen.preprocessing import preprocess_from_path, parallel_preprocess
from lumen.folding import triple_stratified_fold, build_patient_dict


def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(e) for e in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    parser = argparse.ArgumentParser(description="Preprocess ISIC 2020 dataset")
    parser.add_argument("--images", required=True, help="Folder with raw .jpg images")
    parser.add_argument("--labels", required=True, help="ISIC_2020_Training_GroundTruth.csv")
    parser.add_argument("--duplicates", required=True, help="2020_Challenge_duplicates.csv")
    parser.add_argument("--output", default="preprocessed", help="Output folder (default: preprocessed)")
    parser.add_argument("--percent", type=float, default=1.0, help="Fraction of images to use (default: 1.0)")
    parser.add_argument("--target-size", type=int, default=200, help="Output image size (default: 200)")
    parser.add_argument("--num-folds", type=int, default=5, help="Number of k-folds (default: 5)")
    parser.add_argument("--max-class0", type=int, default=20, help="Max class-0 images per patient (default: 20)")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    target_size = (args.target_size, args.target_size)
    os.makedirs(args.output, exist_ok=True)

    # Load data
    duplicates_df = pd.read_csv(args.duplicates)
    duplicates_set = set(duplicates_df["ISIC_id_paired"].tolist())
    df = pd.read_csv(args.labels)

    # Sample
    tmp_df = df.sample(n=int(len(df) * args.percent), random_state=43)
    print(f"Samples before dedup: {len(tmp_df)}")

    # Remove duplicates
    visited = set()
    rows_to_drop = []
    for idx, img_name in tmp_df["image_name"].items():
        if img_name in visited or img_name in duplicates_set:
            rows_to_drop.append(idx)
            duplicates_set.discard(img_name)
        else:
            visited.add(img_name)

    tmp_df = tmp_df.drop(rows_to_drop).reset_index(drop=True)
    print(f"Samples after dedup: {len(tmp_df)}")

    # Cap class-0 images per patient
    patient_dict = build_patient_dict(tmp_df)
    rows_to_drop = []
    for patient_id, counts in patient_dict.items():
        if counts[0] <= args.max_class0:
            continue
        patient_0_rows = tmp_df[(tmp_df["patient_id"] == patient_id) & (tmp_df["target"] == 0)]
        n_remove = len(patient_0_rows) - args.max_class0
        drop_idx = np.random.choice(patient_0_rows.index, size=n_remove, replace=False)
        rows_to_drop.extend(drop_idx)

    tmp_df = tmp_df.drop(index=rows_to_drop).reset_index(drop=True)
    print(f"Samples after capping: {len(tmp_df)}")

    # K-fold split
    patient_dict = build_patient_dict(tmp_df)
    folded_df = triple_stratified_fold(patient_dict, tmp_df, num_folds=args.num_folds)

    folds_csv = os.path.join(args.output, "folds.csv")
    folded_df.to_csv(folds_csv, index=False)
    print(f"Folds saved to: {folds_csv}")

    # Preprocess images
    image_paths = [
        os.path.join(args.images, name + ".jpg")
        for name in folded_df["image_name"]
    ]

    if not args.no_parallel:
        results = parallel_preprocess(image_paths, target_size=target_size)
    else:
        results = []
        for i, path in enumerate(image_paths):
            results.append(preprocess_from_path(path, target_size=target_size))
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(image_paths)}")

    # Save
    final_data = []
    for i, result in enumerate(results):
        if result is None:
            continue
        prep_img, metadata = result
        row = folded_df.iloc[i]
        metadata["image_name"] = row["image_name"] + ".jpg"
        metadata["fold_id"] = row["foldID"]
        metadata["true_class"] = row["target_class"]
        metadata["patient_id"] = row["patientID"]

        dest_folder = os.path.join(args.output, f"{row['foldID']}_fold")
        os.makedirs(dest_folder, exist_ok=True)
        cv2.imwrite(os.path.join(dest_folder, f"pre_{metadata['image_name']}"), prep_img)
        final_data.append(metadata)

    json_path = os.path.join(args.output, "metadata.json")
    with open(json_path, "w") as f:
        json.dump(convert_numpy(final_data), f, indent=4)

    print(f"\nDone! {len(final_data)} images preprocessed.")
    print(f"  Images: {args.output}")
    print(f"  Metadata: {json_path}")


if __name__ == "__main__":
    main()
