#!/usr/bin/env python3
"""Add a train/val/test split column to the fused-model metadata CSV.

StratifiedGroupKFold grouped by patient_id, stratified by target, so no patient
(and therefore no lesion — verified: no lesion_id spans two patient_ids) leaks
across splits. Default 13 folds -> 10 train / 2 test / 1 val (~52k/10k/5k on
the 67k dataset). Writes the "split" column back into the CSV in place and
prints leakage / distribution checks.

Usage:
    python scripts/make_split.py --metadata final_metadata.csv
"""

import argparse

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", required=True, help="CSV with patient_id, lesion_id, target columns")
    parser.add_argument("--n-splits", type=int, default=13, help="Number of folds (default: 13)")
    parser.add_argument("--val-folds", default="0", help="Comma-separated folds -> val (default: 0)")
    parser.add_argument("--test-folds", default="1,2", help="Comma-separated folds -> test (default: 1,2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    val_folds = {int(x) for x in args.val_folds.split(",")}
    test_folds = {int(x) for x in args.test_folds.split(",")}

    df = pd.read_csv(args.metadata)

    sgkf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_of = pd.Series(index=df.index, dtype=int)
    for fold, (_, val_idx) in enumerate(sgkf.split(df, df["target"], groups=df["patient_id"])):
        fold_of.iloc[val_idx] = fold

    split = pd.Series("train", index=df.index)
    split[fold_of.isin(val_folds)] = "val"
    split[fold_of.isin(test_folds)] = "test"
    df["split"] = split

    df.to_csv(args.metadata, index=False)
    print(f"Saved: {args.metadata}\n")

    print("Split sizes and malignancy rates:")
    print(df.groupby("split").agg(n=("target", "size"), malignant_rate=("target", "mean")))

    print("\npatient_id leakage across splits:")
    patient_splits = df.groupby("patient_id")["split"].nunique()
    print(f"  patient_ids in >1 split: {(patient_splits > 1).sum()}")

    print("\nlesion_id leakage across splits:")
    lesion_splits = df.dropna(subset=["lesion_id"]).groupby("lesion_id")["split"].nunique()
    print(f"  lesion_ids in >1 split: {(lesion_splits > 1).sum()}")

    if "dataset_source" in df.columns:
        print("\ndataset_source distribution per split:")
        print(pd.crosstab(df["dataset_source"], df["split"]))


if __name__ == "__main__":
    main()
