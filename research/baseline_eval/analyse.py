# /home/datice/data/baseline_eval/all_GT.csv

# im_name_original,new_im_name,dataset,true_label,predicted_label,conf
# ISIC_0000000,2019_ISIC_0000000.jpg,2019,0,1.0,0.9062871336936952
# ISIC_0000001,2019_ISIC_0000001.jpg,2019,0,1.0,0.5274428129196167
# ISIC_0000002,2019_ISIC_0000002.jpg,2019,1,1.0,0.9477034211158752
# ISIC_0000003,2019_ISIC_0000003.jpg,2019,0,1.0,0.9060420989990234


# I need you to analyse this, and write me summary

# first per dataset (2019, 2020, MILK10k), and at the end combined

# I want accuracy, precision, recall, AUC, TPR, FPR  (both in numbers and percentage)
# Conf matrix (both in numbers and percentage)


import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)

GT_PATH = "/home/datice/Lumen-i-/training/predictions_v4.csv"

df = pd.read_csv(GT_PATH)

# osiguraj tipove
df = df.dropna(subset=["predicted_label", "conf"])
df["predicted_label"] = df["predicted_label"].astype(int)
df["true_label"] = df["true_label"].astype(int)


def evaluate_subset(name, subset):
    y_true = subset["true_label"]
    y_pred = subset["predicted_label"]
    y_prob = subset["conf"]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # TPR / FPR
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


# ===== PER DATASET =====
for dataset_name in df["dataset"].unique():
    subset = df[df["dataset"] == dataset_name]
    evaluate_subset(dataset_name, subset)

# ===== GLOBAL =====
evaluate_subset("ALL DATASETS", df)
