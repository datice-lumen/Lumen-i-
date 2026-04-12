"""Evaluation functions for model performance and fairness metrics."""

from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_auc_score,
)


def evaluate_loss(val_loader, model, criterion, device):
    """Compute average loss over a DataLoader.

    Returns:
        Average loss (float).
    """
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, labels, fitz_groups, augm_list in tqdm(val_loader, desc="Eval loss"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(
                outputs.squeeze(),
                labels.float(),
                fitz_groups=fitz_groups.to(device),
                augm_list=augm_list.to(device),
            )
            val_loss += loss.item() * imgs.size(0)
    return val_loss / len(val_loader.dataset)


def detailed_evaluation(val_loader, model, device, threshold=0.5):
    """Compute classification report and AUC.

    Returns:
        (report_dict, auc_score) — sklearn classification_report as dict, and ROC AUC.
    """
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for imgs, labels, _, _ in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).int()

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0).astype(int)
    all_labels = np.concatenate(all_labels, axis=0).astype(int)

    print(classification_report(all_labels, all_preds))
    report = classification_report(all_labels, all_preds, output_dict=True)
    auc = roc_auc_score(all_labels, all_probs)
    return report, auc


def evaluate_fairness(loader, model, criterion, device):
    """Compute per-Fitzpatrick-group precision, recall, F1, accuracy, and FPR.

    Returns:
        Dict mapping group_id -> {precision, recall, f1, accuracy, fpr}.
    """
    model.eval()
    all_probs, all_targets, all_preds, all_fitz = [], [], [], []

    with torch.no_grad():
        for images, labels, fitz_groups, augm_list in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_fitz.extend(fitz_groups.cpu().numpy() if isinstance(fitz_groups, torch.Tensor)
                            else fitz_groups)

    fitz_metrics = {}
    for group in sorted(set(all_fitz)):
        indices = [i for i, fg in enumerate(all_fitz) if fg == group]
        if not indices:
            continue
        t = [all_targets[i] for i in indices]
        p = [all_preds[i] for i in indices]

        # FPR
        tn = sum(1 for ti, pi in zip(t, p) if ti == 0 and pi == 0)
        fp = sum(1 for ti, pi in zip(t, p) if ti == 0 and pi == 1)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        fitz_metrics[group] = {
            "precision": precision_score(t, p, zero_division=0),
            "recall": recall_score(t, p, zero_division=0),
            "f1": f1_score(t, p, zero_division=0),
            "accuracy": accuracy_score(t, p),
            "fpr": fpr,
        }
    return fitz_metrics


def evaluate_thresholds(model, val_loader, device="cuda"):
    """Sweep thresholds to find the one maximizing weighted F1.

    Plots confusion matrix, precision/recall/F1 curves, and the optimal
    confusion matrix.

    Returns:
        (best_threshold, best_f1).
    """
    model.eval()
    all_targets, all_probs = [], []

    with torch.no_grad():
        for inputs, targets, _, _ in tqdm(val_loader, desc="Threshold sweep"):
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits).squeeze()
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Confusion matrix at default threshold
    default_preds = (all_probs >= 0.5).astype(int)
    cm = confusion_matrix(all_targets, default_preds)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (threshold=0.5)")
    plt.show()

    # Sweep
    thresholds = np.linspace(0.2, 0.8, 151)
    f1_scores, precision_scores, recall_scores = [], [], []

    for t in thresholds:
        preds = (all_probs >= t).astype(int)
        precision_scores.append(precision_score(all_targets, preds, average="weighted", zero_division=0))
        recall_scores.append(recall_score(all_targets, preds, average="weighted"))
        f1_scores.append(f1_score(all_targets, preds, average="weighted"))

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"Best Threshold: {best_threshold:.3f} | Best F1: {best_f1:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, label="F1 Score", color="blue")
    plt.plot(thresholds, precision_scores, label="Precision", color="green")
    plt.plot(thresholds, recall_scores, label="Recall", color="orange")
    plt.axvline(best_threshold, color="red", linestyle="--", label=f"Best = {best_threshold:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision, Recall, and F1 vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Confusion matrix at best threshold
    best_preds = (all_probs >= best_threshold).astype(int)
    cm = confusion_matrix(all_targets, best_preds)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (threshold={best_threshold:.3f})")
    plt.show()

    return best_threshold, best_f1


def plot_metrics(history):
    """Plot training history: loss, AUC, F1, and class-1 recall."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0, 0].plot(history["train_loss"], label="Train", color="blue")
    axs[0, 0].plot(history["val_loss"], label="Val", color="red")
    axs[0, 0].set_title("Loss")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].legend()

    axs[0, 1].plot(history["train_auc"], label="Train", color="blue")
    axs[0, 1].plot(history["val_auc"], label="Val", color="red")
    axs[0, 1].set_title("AUC")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].legend()

    axs[1, 0].plot(history["train_f1_global"], label="Train", color="blue")
    axs[1, 0].plot(history["val_f1_global"], label="Val", color="red")
    axs[1, 0].set_title("Global Weighted F1")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].legend()

    axs[1, 1].plot(history["train_class1_recall"], label="Train", color="blue")
    axs[1, 1].plot(history["val_class1_recall"], label="Val", color="red")
    axs[1, 1].set_title("Class 1 Recall")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()
