"""
Train/Fine-tune the DINOv2-S + TinyCNN + MetaMLP model on MILK10k mobile close-ups.
Loads pretrained weights from model_10_6/checkpoint_20260610_230527.pt.
Fine-tunes the CNN, VisionProj, MetaMLP, and Classifier components on the mobile domain.
Uses data augmentation, BCEJLoss (TPR/FPR aware), and early stopping.
"""

import os
import time
import datetime
import warnings
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, roc_auc_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WORKSPACE_DIR = "/home/datice" if os.path.exists("/home/datice") else "/Users/jj/Desktop/datice"
CKPT_PRETRAINED = f"{WORKSPACE_DIR}/model_10_6/checkpoint_20260610_230527.pt"
EVAL_CSV = f"{WORKSPACE_DIR}/mobile_eval/mobile_eval.csv"
MILK_DIR = f"{WORKSPACE_DIR}/data/original_data/MILK10k/MILK10k_Training_Input"
OUTPUT_DIR = f"{WORKSPACE_DIR}/mobile_eval"

RESIZE = 448
BATCH_SIZE = 64
MAX_EPOCHS = 40
PATIENCE = 5
LR = 1e-4  # slightly lower LR for fine-tuning
WEIGHT_DECAY = 1e-2
WARMUP_EPOCHS = 2
T_MAX = 40
LAM = 0.9
TPR_WEIGHT = 2.5
NUM_WORKERS = 12  # Optimal for high-core GPU servers

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

SEX_CATEGORIES = ["male", "female", "unknown"]
SITE_CATEGORIES = ["torso", "lower_extremity", "upper_extremity", "head_neck", "unknown", "palms_soles"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DINO_DIM, CNN_OUT = 384, 96
CNN_DIM = CNN_OUT * 2
VISION_DIM = DINO_DIM + CNN_DIM
VISION_EMB_DIM, META_EMB_DIM = 256, 16
META_DIM = 1 + 1 + len(SEX_CATEGORIES) + len(SITE_CATEGORIES)
FUSED_DIM = VISION_EMB_DIM + META_EMB_DIM

# ---------------------------------------------------------------------------
# Metadata Encoding
# ---------------------------------------------------------------------------
def encode_metadata(df, age_mean, age_std):
    age_raw = pd.to_numeric(df["age"], errors="coerce")
    age_missing = age_raw.isna().astype(np.float32).to_numpy().reshape(-1, 1)
    age_norm = ((age_raw.fillna(age_mean) - age_mean) / age_std).astype(np.float32).to_numpy().reshape(-1, 1)
    sex_oh = pd.get_dummies(df["sex"]).reindex(columns=SEX_CATEGORIES, fill_value=0).to_numpy(dtype=np.float32)
    site_oh = pd.get_dummies(df["anatom_site"]).reindex(columns=SITE_CATEGORIES, fill_value=0).to_numpy(dtype=np.float32)
    return np.concatenate([age_norm, age_missing, sex_oh, site_oh], axis=1)

# ---------------------------------------------------------------------------
# Preprocessing (NO hair removal)
# ---------------------------------------------------------------------------
def preprocess_image(path, target=RESIZE):
    orig = cv2.imread(path)
    if orig is None:
        raise FileNotFoundError(f"Image not found: {path}")
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    h, w, _ = orig.shape
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = orig[y0:y0+side, x0:x0+side]
    return cv2.resize(crop, (target, target), interpolation=cv2.INTER_LANCZOS4)

def build_path_index():
    idx = {}
    for entry in os.scandir(MILK_DIR):
        if entry.is_dir():
            for sub in os.scandir(entry.path):
                if sub.is_file() and not sub.name.endswith(".txt"):
                    idx[os.path.splitext(sub.name)[0]] = sub.path
    return idx

# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class MobileDataset(Dataset):
    def __init__(self, df, age_mean, age_std, path_index, augment=False):
        self.df = df.reset_index(drop=True)
        self.meta = encode_metadata(self.df, age_mean, age_std)
        self.path_index = path_index
        self.augment = augment

        ops = []
        if augment:
            ops += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            ]
        ops += [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
        self.tf = transforms.Compose(ops)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_id = row["image_id"]
        img_path = self.path_index[img_id]
        
        arr = preprocess_image(img_path, target=RESIZE)
        img = Image.fromarray(arr)
        
        meta = torch.tensor(self.meta[i], dtype=torch.float32)
        return self.tf(img), meta, torch.tensor(float(row["target"]), dtype=torch.float32)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class TinyCNN(nn.Module):
    def __init__(self, out_channels=CNN_OUT):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 96, 3, 2, 1), nn.BatchNorm2d(96), nn.ReLU(True),
            nn.Conv2d(96, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, out_channels, 3, 2, 1), nn.BatchNorm2d(out_channels), nn.ReLU(True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        f = self.features(x)
        return self.dropout(torch.cat([self.avgpool(f).flatten(1), self.maxpool(f).flatten(1)], 1))

class VisionProj(nn.Module):
    def __init__(self, in_dim=VISION_DIM, hidden=384, out_dim=VISION_EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, out_dim), nn.ReLU(), nn.Dropout(0.3)
        )
    def forward(self, x):
        return self.net(x)

class MetaMLP(nn.Module):
    def __init__(self, in_dim=META_DIM, hidden=32, out_dim=META_EMB_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, out_dim), nn.ReLU()
        )
    def forward(self, x):
        return self.mlp(x)

class Classifier(nn.Module):
    def __init__(self, in_dim=FUSED_DIM, h1=256, h2=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h2, 1)
        )
    def forward(self, x):
        return self.net(x)

class LateFusionModel(nn.Module):
    def __init__(self, dino, cnn, vision_proj, meta_mlp, classifier):
        super().__init__()
        self.dino = dino
        self.cnn = cnn
        self.vision_proj = vision_proj
        self.meta_mlp = meta_mlp
        self.classifier = classifier

    def forward(self, imgs, meta):
        with torch.no_grad():
            dino_feat = self.dino(imgs).float()
        cnn_feat = self.cnn(imgs)
        vision_emb = self.vision_proj(torch.cat([dino_feat, cnn_feat], dim=1))
        meta_emb = self.meta_mlp(meta)
        fused = torch.cat([vision_emb, meta_emb], dim=1)
        return self.classifier(fused).squeeze(1)

def load_dino():
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg", verbose=False, trust_repo=True)
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval()

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class BCEJLoss(nn.Module):
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

# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------
def calc_metrics(y_true, y_pred, probs):
    n = len(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    try:
        auc = roc_auc_score(y_true, probs)
    except Exception:
        auc = float("nan")
    return {
        "Acc": (tp + tn) / n,
        "Prec": prec,
        "Rec": rec,
        "F1": 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0,
        "TPR": rec,
        "FPR": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "AUC": auc,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)
    }

def format_block(m, n, title):
    lines = [
        f"{title} (n={n}):",
        f"  Acc={m['Acc']:.3f}  Prec={m['Prec']:.3f}  Rec={m['Rec']:.3f}"
        f"  F1={m['F1']:.3f}  TPR={m['TPR']:.3f}  FPR={m['FPR']:.3f}  AUC={m['AUC']:.3f}",
        f"  Confusion matrix (count / %):",
        f"                  Pred Benign        Pred Malignant",
        f"  True Benign     {m['TN']:5d} ({m['TN']/n:5.1%})    {m['FP']:5d} ({m['FP']/n:5.1%})",
        f"  True Malignant  {m['FN']:5d} ({m['FN']/n:5.1%})    {m['TP']:5d} ({m['TP']/n:5.1%})",
    ]
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Training Step
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer=None, scheduler=None, is_train=True):
    if is_train:
        model.train()
        model.dino.eval()
    else:
        model.eval()

    total_loss = 0.0
    preds_all = []
    labels_all = []
    probs_all = []

    for imgs, meta, labels in loader:
        imgs = imgs.to(DEVICE)
        meta = meta.to(DEVICE)
        labels = labels.to(DEVICE)

        if is_train:
            optimizer.zero_grad()
            logits = model(imgs, meta)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(imgs, meta)
                loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        probs = torch.sigmoid(logits.detach()).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        
        preds_all.extend(preds.tolist())
        probs_all.extend(probs.tolist())
        labels_all.extend(labels.cpu().numpy().tolist())

    if is_train and scheduler is not None:
        scheduler.step()

    m = calc_metrics(np.array(labels_all), np.array(preds_all), np.array(probs_all))
    return total_loss / len(loader.dataset), m

# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    df = pd.read_csv(EVAL_CSV)
    path_index = build_path_index()

    ck = torch.load(CKPT_PRETRAINED, map_location="cpu", weights_only=True)
    age_mean = ck["age_mean"]
    age_std = ck["age_std"]

    valid_mask = df["image_id"].isin(path_index)
    dropped = (~valid_mask).sum()
    if dropped > 0:
        print(f"Warning: {dropped} images could not be located in directory. Filtering them out.")
        df = df[valid_mask].reset_index(drop=True)

    train_df = df[df["twin_split"] == "train"].reset_index(drop=True)
    val_df = df[df["twin_split"] == "val"].reset_index(drop=True)
    test_df = df[df["twin_split"] == "test"].reset_index(drop=True)

    print(f"Dataset split: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    train_ds = MobileDataset(train_df, age_mean, age_std, path_index, augment=True)
    val_ds = MobileDataset(val_df, age_mean, age_std, path_index, augment=False)
    test_ds = MobileDataset(test_df, age_mean, age_std, path_index, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    dino = load_dino().to(DEVICE)
    cnn = TinyCNN().to(DEVICE)
    vision_proj = VisionProj().to(DEVICE)
    meta_mlp = MetaMLP().to(DEVICE)
    classifier = Classifier().to(DEVICE)

    cnn.load_state_dict(ck["cnn"])
    vision_proj.load_state_dict(ck["vision_proj"])
    meta_mlp.load_state_dict(ck["meta_mlp"])
    classifier.load_state_dict(ck["classifier"])

    model = LateFusionModel(dino, cnn, vision_proj, meta_mlp, classifier).to(DEVICE)

    trainable_params = (
        list(model.cnn.parameters())
        + list(model.vision_proj.parameters())
        + list(model.meta_mlp.parameters())
        + list(model.classifier.parameters())
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])

    criterion = BCEJLoss(lam=LAM, tpr_weight=TPR_WEIGHT)

    best_val_loss = float("inf")
    best_state = None
    patience_cnt = 0
    best_epoch = 0

    print("\n" + "="*70)
    print("STARTING MOBILE FINE-TUNING")
    print("="*70)
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_m = run_epoch(model, train_loader, criterion, optimizer, scheduler, is_train=True)
        val_loss, val_m = run_epoch(model, val_loader, criterion, is_train=False)

        print(f"Epoch {epoch:02d}/{MAX_EPOCHS} | Train Loss={tr_loss:.4f} Val Loss={val_loss:.4f} "
              f"Val F1={val_m['F1']:.3f} Val TPR={val_m['TPR']:.3f} Val FPR={val_m['FPR']:.3f} "
              f"Time={int(time.time() - t0)}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                "cnn": {k: v.clone() for k, v in model.cnn.state_dict().items()},
                "vision_proj": {k: v.clone() for k, v in model.vision_proj.state_dict().items()},
                "meta_mlp": {k: v.clone() for k, v in model.meta_mlp.state_dict().items()},
                "classifier": {k: v.clone() for k, v in model.classifier.state_dict().items()},
            }
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}! Best epoch: {best_epoch} (Val Loss: {best_val_loss:.4f})")
                break

    model.cnn.load_state_dict(best_state["cnn"])
    model.vision_proj.load_state_dict(best_state["vision_proj"])
    model.meta_mlp.load_state_dict(best_state["meta_mlp"])
    model.classifier.load_state_dict(best_state["classifier"])

    print("\n" + "="*70)
    print("FINAL POST-TRAINING PERFORMANCE")
    print("="*70)
    _, final_val_m = run_epoch(model, val_loader, criterion, is_train=False)
    print(format_block(final_val_m, len(val_ds), "Validation Split"))
    _, final_test_m = run_epoch(model, test_loader, criterion, is_train=False)
    print(format_block(final_test_m, len(test_ds), "Test Split"))

    ckpt_save_path = f"{OUTPUT_DIR}/checkpoint_mobile_best.pt"
    print(f"\nSaving best checkpoint to {ckpt_save_path}...")
    torch.save({
        "cnn": best_state["cnn"],
        "vision_proj": best_state["vision_proj"],
        "meta_mlp": best_state["meta_mlp"],
        "classifier": best_state["classifier"],
        "age_mean": age_mean,
        "age_std": age_std,
        "best_epoch": best_epoch,
        "config": {
            "DINO_DIM": DINO_DIM, "CNN_OUT_CHANNELS": CNN_OUT, "CNN_DIM": CNN_DIM,
            "VISION_DIM": VISION_DIM, "VISION_EMB_DIM": VISION_EMB_DIM,
            "META_DIM": META_DIM, "META_EMB_DIM": META_EMB_DIM, "FUSED_DIM": FUSED_DIM,
            "SEX_CATEGORIES": SEX_CATEGORIES, "SITE_CATEGORIES": SITE_CATEGORIES,
            "RESIZE": RESIZE,
        },
    }, ckpt_save_path)
    print("Checkpoint saved successfully.")

if __name__ == "__main__":
    main()
