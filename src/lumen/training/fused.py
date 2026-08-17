"""Training components for the fused metadata model (model_10_6 family).

Shared by ``scripts/train_fused.py`` (dermatoscopic base model),
``scripts/train_mobile.py`` (mobile fine-tune) and ``scripts/eval_mobile.py``.
The architecture itself lives in :mod:`lumen.model_meta`; this module adds the
training-only pieces: batch metadata encoding, datasets, the TPR/FPR-aware loss,
metrics, the epoch runner and the checkpoint format.

Ported unchanged (bar path handling) from the original run scripts on slavica
(``model_10_6/train.py``, ``mobile_eval/train_mobile.py``) that produced
``checkpoint_20260610_230527.pt`` and ``checkpoint_mobile_best.pt``, so re-runs
reproduce the shipped models.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import Dataset
from torchvision import transforms

from lumen.model_meta import (
    Classifier,
    FusedMetaModel,
    MetaMLP,
    TinyCNN,
    VisionProj,
    load_dino,
)
from lumen.preprocessing import preprocess_mobile

# Canonical category order — must match the checkpoints' config.
SEX_CATEGORIES = ["male", "female", "unknown"]
SITE_CATEGORIES = ["torso", "lower_extremity", "upper_extremity", "head_neck", "unknown", "palms_soles"]
META_DIM = 1 + 1 + len(SEX_CATEGORIES) + len(SITE_CATEGORIES)  # age_norm + age_missing + sex + site = 11

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def encode_metadata_frame(df, age_mean, age_std):
    """Encode a metadata DataFrame into the model's (N, 11) float32 matrix.

    Batch counterpart of :func:`lumen.model_meta.encode_metadata` (single row):
    age_norm + age_missing flag + one-hot sex + one-hot anatom_site, with the
    category order fixed by SEX_CATEGORIES / SITE_CATEGORIES.
    """
    age_raw = pd.to_numeric(df["age"], errors="coerce")
    age_missing = age_raw.isna().astype(np.float32).to_numpy().reshape(-1, 1)
    age_norm = ((age_raw.fillna(age_mean) - age_mean) / age_std).astype(np.float32).to_numpy().reshape(-1, 1)

    sex_oh = pd.get_dummies(df["sex"]).reindex(columns=SEX_CATEGORIES, fill_value=0).to_numpy(dtype=np.float32)
    site_oh = pd.get_dummies(df["anatom_site"]).reindex(columns=SITE_CATEGORIES, fill_value=0).to_numpy(dtype=np.float32)

    return np.concatenate([age_norm, age_missing, sex_oh, site_oh], axis=1)


class BCEJLoss(nn.Module):
    """BCE + λ · (soft_FPR + tpr_weight · (1 − soft_TPR)).

    Youden-style regularizer on soft (sigmoid) rates: pushes TPR up (weighted
    ``tpr_weight``×) while keeping FPR down, on top of plain BCE.
    """

    def __init__(self, lam=0.9, tpr_weight=2.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lam = lam
        self.tpr_weight = tpr_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        n_pos = targets.sum()
        n_neg = (1.0 - targets).sum()
        soft_tpr = (probs * targets).sum() / n_pos.clamp(min=1)
        soft_fpr = (probs * (1.0 - targets)).sum() / n_neg.clamp(min=1)

        return bce_loss + self.lam * (soft_fpr + self.tpr_weight * (1.0 - soft_tpr))


def _build_transform(resize, augment, mobile=False):
    """Torchvision transform stack; augment set differs between base and mobile."""
    ops = [transforms.Resize((resize, resize))] if resize else []
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
        if mobile:
            ops += [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


class FusedImageDataset(Dataset):
    """Preprocessed ``pre_<image_id>.jpg`` images + metadata for the base model.

    Rows whose image is missing from ``img_dir`` are dropped with a notice.
    Images come from ``scripts/preprocess_fused_dataset.py``.
    """

    def __init__(self, df, img_dir, age_mean, age_std, resize=448, augment=False):
        self.index = {p.stem[4:]: p for p in Path(img_dir).glob("pre_*.jpg")}
        mask = df["image_id"].isin(self.index)
        dropped = int((~mask).sum())
        if dropped > 0:
            print(f"    [Dataset] skipping {dropped} rows without an image")
        self.df = df[mask].reset_index(drop=True)
        self.meta = encode_metadata_frame(self.df, age_mean, age_std)
        self.transform = _build_transform(resize, augment)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.index[row["image_id"]]).convert("RGB")
        meta = torch.tensor(self.meta[idx], dtype=torch.float32)
        return self.transform(img), meta, torch.tensor(float(row["target"]), dtype=torch.float32)


def build_milk_path_index(milk_dir):
    """Map image_id -> path for MILK10k's one-level-nested image folders."""
    import os

    idx = {}
    for entry in os.scandir(milk_dir):
        if entry.is_dir():
            for sub in os.scandir(entry.path):
                if sub.is_file() and not sub.name.endswith(".txt"):
                    idx[os.path.splitext(sub.name)[0]] = sub.path
    return idx


class MobileImageDataset(Dataset):
    """Raw MILK10k mobile close-ups + metadata, preprocessed on the fly.

    Uses :func:`lumen.preprocessing.preprocess_mobile` (center square crop →
    resize, deliberately no hair removal — see docs/training/mobile_findings.md).
    """

    def __init__(self, df, age_mean, age_std, path_index, resize=448, augment=False):
        self.df = df.reset_index(drop=True)
        self.meta = encode_metadata_frame(self.df, age_mean, age_std)
        self.path_index = path_index
        self.resize = resize
        # Images are already resize×resize after preprocess_mobile — no Resize op.
        self.transform = _build_transform(None, augment, mobile=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import cv2

        row = self.df.iloc[idx]
        orig = cv2.imread(self.path_index[row["image_id"]])
        if orig is None:
            raise FileNotFoundError(f"Image not found: {self.path_index[row['image_id']]}")
        rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(preprocess_mobile(rgb, target_size=self.resize))
        meta = torch.tensor(self.meta[idx], dtype=torch.float32)
        return self.transform(img), meta, torch.tensor(float(row["target"]), dtype=torch.float32)


def build_fused_model(device, checkpoint=None):
    """Fresh fused model (random heads) or one initialised from a checkpoint dict.

    Returns the :class:`~lumen.model_meta.FusedMetaModel`; DINOv2-S is loaded
    frozen from torch.hub either way.
    """
    dino = load_dino(device)
    cnn = TinyCNN().to(device)
    vision_proj = VisionProj().to(device)
    meta_mlp = MetaMLP().to(device)
    classifier = Classifier().to(device)
    if checkpoint is not None:
        cnn.load_state_dict(checkpoint["cnn"])
        vision_proj.load_state_dict(checkpoint["vision_proj"])
        meta_mlp.load_state_dict(checkpoint["meta_mlp"])
        classifier.load_state_dict(checkpoint["classifier"])
    return FusedMetaModel(dino, cnn, vision_proj, meta_mlp, classifier).to(device)


def snapshot_state(model):
    """Clone the trainable heads' state_dicts (for best-epoch bookkeeping)."""
    return {
        "cnn": {k: v.clone() for k, v in model.cnn.state_dict().items()},
        "vision_proj": {k: v.clone() for k, v in model.vision_proj.state_dict().items()},
        "meta_mlp": {k: v.clone() for k, v in model.meta_mlp.state_dict().items()},
        "classifier": {k: v.clone() for k, v in model.classifier.state_dict().items()},
    }


def restore_state(model, state):
    """Load a :func:`snapshot_state` snapshot back into the model's heads."""
    model.cnn.load_state_dict(state["cnn"])
    model.vision_proj.load_state_dict(state["vision_proj"])
    model.meta_mlp.load_state_dict(state["meta_mlp"])
    model.classifier.load_state_dict(state["classifier"])


def save_fused_checkpoint(path, state, age_mean, age_std, best_epoch, resize=448):
    """Write the canonical checkpoint dict read by lumen.model_meta.load_fused_model."""
    torch.save(
        {
            **state,
            "age_mean": age_mean,
            "age_std": age_std,
            "best_epoch": best_epoch,
            "config": {
                "DINO_DIM": 384, "CNN_OUT_CHANNELS": 96, "CNN_DIM": 192,
                "VISION_DIM": 576, "VISION_EMB_DIM": 256,
                "META_DIM": META_DIM, "META_EMB_DIM": 16, "FUSED_DIM": 272,
                "SEX_CATEGORIES": SEX_CATEGORIES, "SITE_CATEGORIES": SITE_CATEGORIES,
                "RESIZE": resize,
            },
        },
        path,
    )


def calc_metrics(y_true, y_pred, probs=None):
    """Confusion-matrix metrics; adds AUC when probabilities are given."""
    n = len(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    m = {
        "Acc": (tp + tn) / n,
        "Prec": prec,
        "Rec": rec,
        "F1": 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0,
        "TPR": rec,
        "FPR": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "n": n,
    }
    if probs is not None:
        try:
            m["AUC"] = roc_auc_score(y_true, probs)
        except ValueError:
            m["AUC"] = float("nan")
    return m


def format_metrics_block(m, title):
    """Human-readable metrics block with confusion matrix (as in the run logs)."""
    n = m["n"]
    auc = f"  AUC={m['AUC']:.3f}" if "AUC" in m else ""
    return "\n".join([
        f"{title} (n={n}):",
        f"  Acc={m['Acc']:.3f}  Prec={m['Prec']:.3f}  Rec={m['Rec']:.3f}"
        f"  F1={m['F1']:.3f}  TPR={m['TPR']:.3f}  FPR={m['FPR']:.3f}{auc}",
        "  Confusion matrix (count / %):",
        "                  Pred Benign        Pred Malignant",
        f"  True Benign     {m['TN']:5d} ({m['TN']/n:5.1%})    {m['FP']:5d} ({m['FP']/n:5.1%})",
        f"  True Malignant  {m['FN']:5d} ({m['FN']/n:5.1%})    {m['TP']:5d} ({m['TP']/n:5.1%})",
    ])


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, dino_autocast=False):
    """One train or eval epoch over a fused-model DataLoader.

    Train mode iff ``optimizer`` is given. The DINO backbone always runs under
    ``no_grad``; ``dino_autocast=True`` additionally runs it in fp16 autocast
    (the base-model training configuration on CUDA).
    """
    is_train = optimizer is not None
    if is_train:
        model.train()
        model.dino.eval()
    else:
        model.eval()

    total_loss, preds_all, probs_all, labels_all = 0.0, [], [], []

    for imgs, meta, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            if dino_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    dino_feat = model.dino(imgs)
            else:
                dino_feat = model.dino(imgs)
        dino_feat = dino_feat.float()

        def _forward():
            cnn_feat = model.cnn(imgs)
            vision_emb = model.vision_proj(torch.cat([dino_feat, cnn_feat], dim=1))
            meta_emb = model.meta_mlp(meta)
            fused = torch.cat([vision_emb, meta_emb], dim=1)
            return model.classifier(fused).squeeze(1)

        if is_train:
            optimizer.zero_grad()
            logits = _forward()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = _forward()
                loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        probs = torch.sigmoid(logits.detach()).cpu().numpy()
        preds_all.extend((probs >= 0.5).astype(int).tolist())
        probs_all.extend(probs.tolist())
        labels_all.extend(labels.long().cpu().numpy().tolist())

    if is_train and scheduler is not None:
        scheduler.step()

    m = calc_metrics(np.array(labels_all), np.array(preds_all), np.array(probs_all))
    return total_loss / len(loader.dataset), m


def build_optimizer(model, lr, weight_decay, warmup_epochs, t_max):
    """AdamW over the trainable heads + LinearWarmup→CosineAnnealing schedule."""
    trainable_params = (
        list(model.cnn.parameters())
        + list(model.vision_proj.parameters())
        + list(model.meta_mlp.parameters())
        + list(model.classifier.parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )
    return optimizer, scheduler
