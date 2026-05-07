"""Training loop and k-fold orchestration."""

import copy
import gc

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from lumen.model import CustomCNN
from lumen.training.loss import FairnessAwareLoss
from lumen.training.dataset import SkinImageDataset
from lumen.training.evaluation import (
    evaluate_loss,
    detailed_evaluation,
    evaluate_fairness,
    evaluate_thresholds,
    plot_metrics,
)


def train_epoch(train_loader, model, criterion, optimizer, device):
    """Run one training epoch.

    Returns:
        (average_loss, all_predictions, all_labels) as numpy arrays.
    """
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels, fitz_groups, augm_list in tqdm(train_loader, desc="Training"):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(
            outputs.squeeze(),
            labels.float(),
            fitz_groups=fitz_groups.to(device),
            augm_list=augm_list.to(device),
        )
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        all_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

        del imgs, labels, outputs, loss
        torch.cuda.empty_cache()

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    avg_loss = running_loss / len(train_loader.dataset)
    return avg_loss, all_preds, all_labels


def k_fold_training(
    metadata,
    base_dir,
    train_folds,
    val_folds,
    augm_needed,
    num_epochs=35,
    batch_size=128,
    lr=3e-5,
    early_patience=5,
    lr_patience=2,
    class_weights_override=None,
    save_path=None,
):
    """Train a CustomCNN model on specified folds with early stopping.

    Args:
        metadata: List of dicts with image metadata (from preprocessing JSON).
        base_dir: Root directory containing fold subfolders.
        train_folds: List of fold IDs for training.
        val_folds: List of fold IDs for validation.
        augm_needed: Number of augmented copies per class-1 image.
        num_epochs: Maximum training epochs.
        batch_size: Batch size for DataLoaders.
        lr: Initial learning rate for AdamW.
        early_patience: Epochs without improvement before stopping.
        lr_patience: Epochs without improvement before halving LR.
        class_weights_override: Optional (2,) tensor. If None, computed automatically.
        save_path: If provided, save best model weights to this path.

    Returns:
        (model, history) — trained model with best weights, and training history dict.
    """
    img_size = (224, 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load datasets
    train_dataset = SkinImageDataset(
        metadata, train_folds, base_dir, img_size, mode="train", augm_needed=augm_needed
    )
    val_dataset = SkinImageDataset(
        metadata, val_folds, base_dir, img_size, mode="val"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=2
    )

    # Class weights
    if class_weights_override is not None:
        class_weights_tensor = class_weights_override.to(device)
    else:
        all_labels = []
        for _, labels, _, _ in train_loader:
            all_labels.append(labels.numpy())
        all_labels = np.concatenate(all_labels, axis=0)
        cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=all_labels)
        class_weights_tensor = torch.tensor(cw, dtype=torch.float).to(device)

    # Model, loss, optimizer
    model = CustomCNN().to(device)
    criterion = FairnessAwareLoss(class_weights=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    history = {
        "train_loss": [], "val_loss": [],
        "train_auc": [], "val_auc": [],
        "train_f1_global": [], "val_f1_global": [],
        "train_class1_recall": [], "val_class1_recall": [],
    }

    best_val_loss = np.inf
    best_model_weights = copy.deepcopy(model.state_dict())
    early_counter = 0
    lr_counter = 0
    current_lr = lr

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, _, _ = train_epoch(train_loader, model, criterion, optimizer, device)
        val_loss = evaluate_loss(val_loader, model, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Detailed metrics
        print("\n-- Validation --")
        val_report, val_auc = detailed_evaluation(val_loader, model, device)
        val_fairness = evaluate_fairness(val_loader, model, criterion, device)
        for group, stats in val_fairness.items():
            print(f"  Group {group}: P={stats['precision']:.4f} R={stats['recall']:.4f} "
                  f"F1={stats['f1']:.4f} Acc={stats['accuracy']:.4f}")

        print("\n-- Training --")
        train_report, train_auc = detailed_evaluation(train_loader, model, device)

        # Track history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auc"].append(train_auc)
        history["val_auc"].append(val_auc)
        history["train_f1_global"].append(train_report["weighted avg"]["f1-score"])
        history["val_f1_global"].append(val_report["weighted avg"]["f1-score"])
        history["train_class1_recall"].append(train_report["1"]["recall"])
        history["val_class1_recall"].append(val_report["1"]["recall"])

        # Early stopping + LR reduction
        if val_loss < best_val_loss * 0.97:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            early_counter = 0
            lr_counter = 0
            print("New best model saved.")
        else:
            early_counter += 1
            lr_counter += 1
            print(f"No improvement. Early: {early_counter}/{early_patience}, "
                  f"LR: {lr_counter}/{lr_patience}")

            if lr_counter >= lr_patience:
                current_lr *= 0.5
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr
                print(f"LR reduced to {current_lr:.10f}")
                lr_counter = 0

            if early_counter >= early_patience:
                print("Early stopping triggered.")
                break

    # Restore best weights
    model.load_state_dict(best_model_weights)
    plot_metrics(history)

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    return model, history
