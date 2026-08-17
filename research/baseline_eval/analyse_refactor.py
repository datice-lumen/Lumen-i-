"""Compute per-dataset and global metrics for the refactored predict.py output.

Joins predictions_refactor.csv (image_name, target, conf) with all_GT.csv
(im_name_original, new_im_name, dataset, true_label, ...) and prints the
same metric blocks as the original analyse.py.

Usage:
    python analyse_refactor.py
    python analyse_refactor.py --predictions /path/to/predictions.csv
"""

import argparse

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)


def evaluate_subset(name, subset):
    y_true = subset["true_label"]
    y_pred = subset["predicted_label"]
    y_prob = subset["conf"]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    total = len(subset)

    print(f"\n==================== {name} ====================")
    print(f"Samples: {total}")

    print("\n--- METRICS ---")
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f} ({prec*100:.2f}%)")
    print(f"Recall:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"AUC:       {auc:.4f} ({auc*100:.2f}%)")
    print(f"TPR:       {tpr:.4f} ({tpr*100:.2f}%)")
    print(f"FPR:       {fpr:.4f} ({fpr*100:.2f}%)")

    print("\n--- CONFUSION MATRIX (counts) ---")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"TN: {tn}")
    print(f"FN: {fn}")

    print("\n--- CONFUSION MATRIX (%) ---")
    print(f"TP: {tp/total:.4f} ({tp/total*100:.2f}%)")
    print(f"FP: {fp/total:.4f} ({fp/total*100:.2f}%)")
    print(f"TN: {tn/total:.4f} ({tn/total*100:.2f}%)")
    print(f"FN: {fn/total:.4f} ({fn/total*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Analyse refactor predictions")
    parser.add_argument(
        "--predictions",
        default="/home/datice/data/baseline_eval/predictions_refactor.csv",
        help="Path to refactor predictions CSV (image_name, target, conf)",
    )
    parser.add_argument(
        "--ground-truth",
        default="/home/datice/data/baseline_eval/all_GT.csv",
        help="Path to all_GT.csv",
    )
    args = parser.parse_args()

    pred = pd.read_csv(args.predictions)
    gt = pd.read_csv(args.ground_truth)

    # all_GT already contains slavica's predicted_label/conf from the original run;
    # drop them so they don't collide with the refactor's columns on merge.
    gt = gt.drop(columns=[c for c in ("predicted_label", "conf") if c in gt.columns])

    # all_GT keeps .jpg in new_im_name; refactor strips it from image_name
    gt["image_name"] = gt["new_im_name"].str.replace(".jpg", "", regex=False)

    df = gt.merge(pred[["image_name", "target", "conf"]], on="image_name", how="inner")
    df = df.rename(columns={"target": "predicted_label"})

    df = df.dropna(subset=["predicted_label", "conf"])
    df["predicted_label"] = df["predicted_label"].astype(int)
    df["true_label"] = df["true_label"].astype(int)

    print(f"Predictions: {len(pred)}  |  Ground truth: {len(gt)}  |  Merged: {len(df)}")
    missing = len(gt) - len(df)
    if missing:
        print(f"Note: {missing} GT rows had no matching prediction (skipped from analysis).")

    for dataset_name in df["dataset"].unique():
        subset = df[df["dataset"] == dataset_name]
        evaluate_subset(str(dataset_name), subset)

    evaluate_subset("ALL DATASETS", df)


if __name__ == "__main__":
    main()
