"""
Glavni trening: DINOv2-S (frozen, 384) ++ TinyCNN (trenirajući, AvgPool+MaxPool 192)
  -> VisionProj (576->384->256, BN samo na skrivenom sloju) = vision_emb (256)
Metapodaci (age, sex, anatom_site, 11-dim) -> MetaMLP (11->32->16, bez BN) = meta_emb (16)
Late fusion: concat(vision_emb 256, meta_emb 16) = 272 -> Classifier (272->256->128->1, BN post-fusion)

448px, 67k slika (train ~52k / val ~5.2k / test ~10.3k iz final_metadata.csv "split" stupca).
AdamW + warmup(2ep) + cosine(T_max=40), early stop na eval(val) lossu, BCEJLoss(λ=0.9, tpr_weight=2.5).
"""

import time
import warnings
import datetime
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore", message="xFormers")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CSV_PATH   = "/home/datice/model_10_6/final_metadata.csv"
IMG_DIR    = "/home/datice/model_10_6/preprocessed448_67k"
RUN_DIR    = "/home/datice/model_10_6"
RESIZE     = 448

TIMESTAMP  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_PATH     = f"{RUN_DIR}/training_log_{TIMESTAMP}.txt"
RESULTS_PATH = f"{RUN_DIR}/results_{TIMESTAMP}.txt"
CKPT_PATH    = f"{RUN_DIR}/checkpoint_{TIMESTAMP}.pt"

DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE    = 64
MAX_EPOCHS    = 150
PATIENCE      = 4
LR            = 3e-4
WEIGHT_DECAY  = 1e-2
WARMUP_EPOCHS = 2
T_MAX         = 40
LAM           = 0.9
TPR_WEIGHT    = 2.5
NUM_WORKERS   = 12

DINO_DIM = 384
CNN_OUT_CHANNELS = 96
CNN_DIM  = CNN_OUT_CHANNELS * 2  # avg + max pool concat = 192
VISION_DIM = DINO_DIM + CNN_DIM  # 576

VISION_EMB_DIM = 256
META_EMB_DIM   = 16
FUSED_DIM      = VISION_EMB_DIM + META_EMB_DIM  # 272

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Metapodaci: age (numeric + missing flag), sex (one-hot), anatom_site (one-hot)
# ---------------------------------------------------------------------------
SEX_CATEGORIES  = ["male", "female", "unknown"]
SITE_CATEGORIES = ["torso", "lower_extremity", "upper_extremity", "head_neck", "unknown", "palms_soles"]
META_DIM = 1 + 1 + len(SEX_CATEGORIES) + len(SITE_CATEGORIES)  # age_norm + age_missing + sex + site = 11

def encode_metadata(df, age_mean, age_std):
    age_raw     = pd.to_numeric(df["age"], errors="coerce")
    age_missing = age_raw.isna().astype(np.float32).to_numpy().reshape(-1, 1)
    age_norm    = ((age_raw.fillna(age_mean) - age_mean) / age_std).astype(np.float32).to_numpy().reshape(-1, 1)

    sex_oh  = pd.get_dummies(df["sex"]).reindex(columns=SEX_CATEGORIES, fill_value=0).to_numpy(dtype=np.float32)
    site_oh = pd.get_dummies(df["anatom_site"]).reindex(columns=SITE_CATEGORIES, fill_value=0).to_numpy(dtype=np.float32)

    return np.concatenate([age_norm, age_missing, sex_oh, site_oh], axis=1)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SkinDataset(Dataset):
    def __init__(self, df, age_mean, age_std, augment=False):
        self.index = {
            p.stem[4:]: p
            for p in Path(IMG_DIR).glob("pre_*.jpg")
        }
        mask = df["image_id"].isin(self.index)
        dropped = (~mask).sum()
        if dropped > 0:
            print(f"    [Dataset] preskačem {dropped} redaka bez slike")
        self.df = df[mask].reset_index(drop=True)
        self.meta = encode_metadata(self.df, age_mean, age_std)

        ops = [transforms.Resize((RESIZE, RESIZE))]
        if augment:
            ops += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ]
        ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        img  = Image.open(self.index[row["image_id"]]).convert("RGB")
        meta = torch.tensor(self.meta[idx], dtype=torch.float32)
        return self.transform(img), meta, torch.tensor(float(row["target"]), dtype=torch.float32)

# ---------------------------------------------------------------------------
# Modeli
# ---------------------------------------------------------------------------
def load_dino():
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg", verbose=False)
    for p in m.parameters():
        p.requires_grad_(False)
    m.eval()
    return m

class TinyCNN(nn.Module):
    """Trenirajući konv. feature extractor — komplementira frozen DINOv2 lokalnim
    teksturama/mikro-uzorcima. Sekundaran: 192-dim (DINO=384). AvgPool+MaxPool
    concat hvata i 'prosječnu' i 'najjaču' lokalnu aktivaciju."""
    def __init__(self, out_channels=CNN_OUT_CHANNELS):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),               # 448 -> 224
            nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),              # 224 -> 112
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),              # 112 -> 56
            nn.BatchNorm2d(96),  nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=2, padding=1),             # 56  -> 28
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, 3, stride=2, padding=1),   # 28  -> 14
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        feat = self.features(x)
        avg  = self.avgpool(feat).flatten(1)
        mx   = self.maxpool(feat).flatten(1)
        return self.dropout(torch.cat([avg, mx], dim=1))

class VisionProj(nn.Module):
    """Fuzija DINO+TinyCNN (576) -> vision_emb (256). BN samo na skrivenom sloju
    — izlaz (vision_emb) NE smije imati BN jer ide u kasniju fuziju s metapodacima."""
    def __init__(self, in_dim=VISION_DIM, hidden=384, out_dim=VISION_EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        return self.net(x)

class MetaMLP(nn.Module):
    """Mali MLP za metapodatke (age, sex, anatom_site). Bez BN — izlaz (meta_emb)
    ide u kasniju fuziju s vizijskom granom."""
    def __init__(self, in_dim=META_DIM, hidden=32, out_dim=META_EMB_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)

class Classifier(nn.Module):
    """Post-fusion klasifikator (vision_emb ++ meta_emb = 272). BN dozvoljen jer
    djeluje na već spojenoj/fuzioniranoj reprezentaciji."""
    def __init__(self, in_dim=FUSED_DIM, hidden1=256, hidden2=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x)

def build_model():
    dino        = load_dino().to(DEVICE)
    cnn         = TinyCNN(CNN_OUT_CHANNELS).to(DEVICE)
    vision_proj = VisionProj(VISION_DIM, 384, VISION_EMB_DIM).to(DEVICE)
    meta_mlp    = MetaMLP(META_DIM, 32, META_EMB_DIM).to(DEVICE)
    classifier  = Classifier(FUSED_DIM, 256, 128).to(DEVICE)
    return dino, cnn, vision_proj, meta_mlp, classifier

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
class BCEJLoss(nn.Module):
    """BCE + λ · (soft_FPR + tpr_weight · (1 - soft_TPR))"""
    def __init__(self, lam=0.9, tpr_weight=2.5):
        super().__init__()
        self.bce        = nn.BCEWithLogitsLoss()
        self.lam        = lam
        self.tpr_weight = tpr_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs    = torch.sigmoid(logits)
        n_pos    = targets.sum()
        n_neg    = (1.0 - targets).sum()
        soft_tpr = (probs * targets).sum() / n_pos.clamp(min=1)
        soft_fpr = (probs * (1.0 - targets)).sum() / n_neg.clamp(min=1)

        return bce_loss + self.lam * (soft_fpr + self.tpr_weight * (1.0 - soft_tpr))

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def calc_metrics(y_true, y_pred):
    n = len(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "Acc":  (tp + tn) / n,
        "Prec": prec,
        "Rec":  rec,
        "F1":   2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0,
        "TPR":  rec,
        "FPR":  fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }

def format_block(m, n, title):
    lines = [
        f"{title} (n={n}):",
        f"  Acc={m['Acc']:.3f}  Prec={m['Prec']:.3f}  Rec={m['Rec']:.3f}"
        f"  F1={m['F1']:.3f}  TPR={m['TPR']:.3f}  FPR={m['FPR']:.3f}",
        f"  Confusion matrix (count / %):",
        f"                  Pred Benign        Pred Malignant",
        f"  True Benign     {m['TN']:5d} ({m['TN']/n:5.1%})    {m['FP']:5d} ({m['FP']/n:5.1%})",
        f"  True Malignant  {m['FN']:5d} ({m['FN']/n:5.1%})    {m['TP']:5d} ({m['TP']/n:5.1%})",
    ]
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Epoch runner
# ---------------------------------------------------------------------------
def run_epoch(dino, cnn, vision_proj, meta_mlp, classifier, loader, criterion,
               optimizer=None, scheduler=None):
    is_train = optimizer is not None
    modules = [cnn, vision_proj, meta_mlp, classifier]
    for m in modules:
        m.train() if is_train else m.eval()

    total_loss, preds_all, labels_all = 0.0, [], []

    for imgs, meta, labels in loader:
        imgs   = imgs.to(DEVICE, non_blocking=True)
        meta   = meta.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            dino_feat = dino(imgs)
        dino_feat = dino_feat.float()

        if is_train:
            optimizer.zero_grad()
            cnn_feat   = cnn(imgs)
            vision_emb = vision_proj(torch.cat([dino_feat, cnn_feat], dim=1))
            meta_emb   = meta_mlp(meta)
            fused      = torch.cat([vision_emb, meta_emb], dim=1)
            logits     = classifier(fused).squeeze(1)
            loss       = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                cnn_feat   = cnn(imgs)
                vision_emb = vision_proj(torch.cat([dino_feat, cnn_feat], dim=1))
                meta_emb   = meta_mlp(meta)
                fused      = torch.cat([vision_emb, meta_emb], dim=1)
                logits     = classifier(fused).squeeze(1)
                loss       = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        preds = (torch.sigmoid(logits.detach()) >= 0.5).long().cpu().numpy()
        preds_all.extend(preds.tolist())
        labels_all.extend(labels.long().cpu().numpy().tolist())

    if is_train and scheduler is not None:
        scheduler.step()

    m = calc_metrics(np.array(labels_all), np.array(preds_all))
    return total_loss / len(loader.dataset), m

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
df       = pd.read_csv(CSV_PATH)
train_df = df[df["split"] == "train"].reset_index(drop=True)
val_df   = df[df["split"] == "val"].reset_index(drop=True)
test_df  = df[df["split"] == "test"].reset_index(drop=True)

n_pos = int((train_df["target"] == 1).sum())
n_neg = int((train_df["target"] == 0).sum())

age_train = pd.to_numeric(train_df["age"], errors="coerce")
AGE_MEAN  = float(age_train.mean())
AGE_STD   = float(age_train.std())

print(f"Device   : {DEVICE}")
print(f"Train    : {len(train_df)}  ({n_pos} mal / {n_neg} ben)")
print(f"Val      : {len(val_df)}")
print(f"Test     : {len(test_df)}")
print(f"Age stats (train): mean={AGE_MEAN:.2f}  std={AGE_STD:.2f}")
print(f"Meta dim : {META_DIM}  (age_norm + age_missing + sex[{len(SEX_CATEGORIES)}] + site[{len(SITE_CATEGORIES)}])\n")

dino, cnn, vision_proj, meta_mlp, classifier = build_model()

n_cnn_params  = sum(p.numel() for p in cnn.parameters())
n_vp_params   = sum(p.numel() for p in vision_proj.parameters())
n_meta_params = sum(p.numel() for p in meta_mlp.parameters())
n_clf_params  = sum(p.numel() for p in classifier.parameters())
n_trainable   = n_cnn_params + n_vp_params + n_meta_params + n_clf_params

header_lines = [
    "=" * 70,
    "DINOv2-S (frozen) ++ TinyCNN(AvgPool+MaxPool, 192) -> VisionProj -> vision_emb(256)",
    "  ++  MetaMLP(11->16)  ->  Classifier(272->256->128->1)  |  448px  |  67k dataset",
    "=" * 70,
    f"Optimizer  : AdamW (lr={LR}, wd={WEIGHT_DECAY})",
    f"Schedule   : LinearWarmup({WARMUP_EPOCHS}ep) + CosineAnnealingLR (T_max={T_MAX})",
    f"Early stop : patience={PATIENCE} na eval(val) lossu  |  Max epochs: {MAX_EPOCHS}",
    f"Batch size : {BATCH_SIZE}",
    f"Loss       : BCEJLoss (λ={LAM}, tpr_weight={TPR_WEIGHT})",
    f"TinyCNN    : {n_cnn_params:,} param  |  VisionProj: {n_vp_params:,}  |  MetaMLP: {n_meta_params:,}  |  Classifier: {n_clf_params:,}",
    f"Trainable ukupno: {n_trainable:,}  (DINOv2-S frozen ~21M)",
    f"Age stats  : mean={AGE_MEAN:.2f}  std={AGE_STD:.2f}  (iz train splita)",
    "",
]
for line in header_lines:
    print(line)

with open(LOG_PATH, "w") as f:
    f.write("\n".join(header_lines) + "\n")

train_ds      = SkinDataset(train_df, AGE_MEAN, AGE_STD, augment=True)
train_eval_ds = SkinDataset(train_df, AGE_MEAN, AGE_STD, augment=False)
val_ds        = SkinDataset(val_df,   AGE_MEAN, AGE_STD, augment=False)
test_ds       = SkinDataset(test_df,  AGE_MEAN, AGE_STD, augment=False)

train_loader      = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
train_eval_loader = DataLoader(train_eval_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
val_loader        = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
test_loader       = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

criterion = BCEJLoss(lam=LAM, tpr_weight=TPR_WEIGHT)
trainable_params = (
    list(cnn.parameters())
    + list(vision_proj.parameters())
    + list(meta_mlp.parameters())
    + list(classifier.parameters())
)
optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
warmup    = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
)
cosine    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS]
)

best_eval_loss = float("inf")
best_state     = None
patience_cnt   = 0
best_epoch     = 0
t0 = time.time()

for epoch in range(1, MAX_EPOCHS + 1):
    tr_loss, tr_m = run_epoch(dino, cnn, vision_proj, meta_mlp, classifier,
                               train_loader, criterion, optimizer, scheduler)
    ev_loss, ev_m = run_epoch(dino, cnn, vision_proj, meta_mlp, classifier,
                               val_loader, criterion)

    line = (f"  ep{epoch:03d}  tr={tr_loss:.4f}  ev={ev_loss:.4f}"
            f"  F1={ev_m['F1']:.3f}  TPR={ev_m['TPR']:.3f}  FPR={ev_m['FPR']:.3f}"
            f"  ({int(time.time()-t0)}s)")
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

    if ev_loss < best_eval_loss:
        best_eval_loss = ev_loss
        best_state = {
            "cnn":         {k: v.clone() for k, v in cnn.state_dict().items()},
            "vision_proj": {k: v.clone() for k, v in vision_proj.state_dict().items()},
            "meta_mlp":    {k: v.clone() for k, v in meta_mlp.state_dict().items()},
            "classifier":  {k: v.clone() for k, v in classifier.state_dict().items()},
        }
        patience_cnt = 0
        best_epoch   = epoch
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            stop_line = f"  >> Early stop (best epoch={best_epoch}, best_val_loss={best_eval_loss:.4f})"
            print(stop_line)
            with open(LOG_PATH, "a") as f:
                f.write(stop_line + "\n")
            break

elapsed = int(time.time() - t0)
cnn.load_state_dict(best_state["cnn"])
vision_proj.load_state_dict(best_state["vision_proj"])
meta_mlp.load_state_dict(best_state["meta_mlp"])
classifier.load_state_dict(best_state["classifier"])

torch.save({
    "cnn":         best_state["cnn"],
    "vision_proj": best_state["vision_proj"],
    "meta_mlp":    best_state["meta_mlp"],
    "classifier":  best_state["classifier"],
    "age_mean":    AGE_MEAN,
    "age_std":     AGE_STD,
    "best_epoch":  best_epoch,
    "config": {
        "DINO_DIM": DINO_DIM, "CNN_OUT_CHANNELS": CNN_OUT_CHANNELS, "CNN_DIM": CNN_DIM,
        "VISION_DIM": VISION_DIM, "VISION_EMB_DIM": VISION_EMB_DIM,
        "META_DIM": META_DIM, "META_EMB_DIM": META_EMB_DIM, "FUSED_DIM": FUSED_DIM,
        "SEX_CATEGORIES": SEX_CATEGORIES, "SITE_CATEGORIES": SITE_CATEGORIES,
        "RESIZE": RESIZE,
    },
}, CKPT_PATH)

_, tr_m_f  = run_epoch(dino, cnn, vision_proj, meta_mlp, classifier, train_eval_loader, criterion)
_, val_m_f = run_epoch(dino, cnn, vision_proj, meta_mlp, classifier, val_loader,        criterion)
_, te_m_f  = run_epoch(dino, cnn, vision_proj, meta_mlp, classifier, test_loader,       criterion)

results_lines = header_lines + [
    "=" * 70,
    f"RUN: train (67k)  |  best_epoch={best_epoch}  |  best_val_loss={best_eval_loss:.4f}  |  {elapsed}s",
    f"Checkpoint: {CKPT_PATH}",
    "=" * 70,
    format_block(tr_m_f,  len(train_eval_ds), "TRAIN"),
    "",
    format_block(val_m_f, len(val_ds),  "VAL"),
    "",
    format_block(te_m_f,  len(test_ds), "TEST"),
]
for line in results_lines[len(header_lines):]:
    print(line)

with open(RESULTS_PATH, "w") as f:
    f.write("\n".join(results_lines))

print(f"\nLog       : {LOG_PATH}")
print(f"Results   : {RESULTS_PATH}")
print(f"Checkpoint: {CKPT_PATH}")