"""Fairness-aware loss function for melanoma classification.

Custom weighted BCE loss with:
- Per-class weighting to handle imbalance
- Reduced weight for augmented samples
- Recall-balance regularization (penalizes poor per-class recall)
- Equalized Odds fairness regularization across skin tone groups
"""

import torch
import torch.nn as nn


class FairnessAwareLoss(nn.Module):
    """Weighted BCE with recall-balance and equalized-odds regularization.

    Args:
        class_weights: Tensor of shape (2,) with weights for class 0 and 1.
        augm_weight: Weight multiplier for augmented samples (< 1 reduces their
                     contribution to the gradient).
        f1_threshold: Threshold used for computing recall statistics in the
                      regularization term.
    """

    def __init__(self, class_weights=None, augm_weight=0.5, f1_threshold=0.5):
        super().__init__()
        self.class_weights = class_weights
        self.augm_weight = augm_weight
        self.f1_threshold = f1_threshold

    def forward(self, logits, targets, fitz_groups, augm_list):
        targets = targets.float()
        probs = torch.sigmoid(logits)
        eps = 1e-7

        # Basic BCE
        loss_pos = -targets * torch.log(probs + eps)
        loss_neg = -(1 - targets) * torch.log(1 - probs + eps)
        loss = loss_pos + loss_neg

        # Class weights
        if self.class_weights is not None:
            weights = self.class_weights[targets.long()]
            loss = loss * weights

        # Reduced weight for augmented samples
        augm_weights = torch.ones_like(loss)
        augm_weights[augm_list] = self.augm_weight
        loss = loss * augm_weights

        final_loss = loss.mean()

        # 1) Recall-balance regularization
        preds = (probs > self.f1_threshold).float()
        tp = torch.sum((preds == 1) & (targets == 1)).float()
        fn = torch.sum((preds == 0) & (targets == 1)).float()
        tn = torch.sum((preds == 0) & (targets == 0)).float()
        fp = torch.sum((preds == 1) & (targets == 0)).float()

        recall_1 = tp / (tp + fn + eps)
        recall_0 = tn / (tn + fp + eps)

        recall_balance = (recall_0 ** 0.5) * (recall_1 ** 1.3)
        balance_penalty = 1.0 - recall_balance
        final_loss = final_loss * ((1.0 + balance_penalty) ** 2)

        # 2) Equalized Odds fairness regularization
        unique_groups = torch.unique(fitz_groups)
        pred_labels = (probs > 0.5).long()
        true_labels = targets.long()

        group_metrics = {}
        for group in unique_groups:
            mask = fitz_groups == group
            if mask.sum() == 0:
                continue
            g_preds = pred_labels[mask]
            g_targets = true_labels[mask]

            g_tp = ((g_preds == 1) & (g_targets == 1)).sum().float()
            g_fn = ((g_preds == 0) & (g_targets == 1)).sum().float()
            g_tn = ((g_preds == 0) & (g_targets == 0)).sum().float()
            g_fp = ((g_preds == 1) & (g_targets == 0)).sum().float()

            tpr = g_tp / (g_tp + g_fn + eps)
            fpr = g_fp / (g_fp + g_tn + eps)
            group_metrics[group.item()] = {"tpr": tpr, "fpr": fpr}

        tpr_fpr_gaps = []
        groups = list(group_metrics.keys())
        for i, g1 in enumerate(groups):
            for g2 in groups[i + 1:]:
                tpr_gap = torch.abs(group_metrics[g1]["tpr"] - group_metrics[g2]["tpr"])
                fpr_gap = torch.abs(group_metrics[g1]["fpr"] - group_metrics[g2]["fpr"])
                tpr_fpr_gaps.append(tpr_gap + fpr_gap)

        if tpr_fpr_gaps:
            max_gap = torch.max(torch.stack(tpr_fpr_gaps))
            final_loss = final_loss * (1 + max_gap * 0.7)

        return final_loss
