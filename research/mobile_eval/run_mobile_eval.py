"""
Evaluate the newest model (model_10_6/checkpoint_20260610_230527.pt) on the
5,240 MILK10k MOBILE ("clinical: close-up") images — a held-out domain
(only the dermoscopic twins were in the 67k training set).

Labels + metadata are taken from each mobile image's dermoscopic twin row in
final_metadata.csv (same lesion -> identical age/sex/site/target), so the ONLY
variable vs training is image modality.

Runs twice: WITH hair removal (matches training pipeline) and WITHOUT (D-14:
hair removal is tuned for dermoscopy and mis-fires on phone photos).
"""
import os, time, warnings
import cv2, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, roc_auc_score

warnings.filterwarnings("ignore")

CKPT     = "/home/datice/model_10_6/checkpoint_20260610_230527.pt"
EVAL_CSV = "/home/datice/mobile_eval/mobile_eval.csv"
MILK_DIR = "/home/datice/data/original_data/MILK10k/MILK10k_Training_Input"
RESIZE   = 448
BATCH    = 64
NUM_WORKERS = 12
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEX_CATEGORIES  = ["male", "female", "unknown"]
SITE_CATEGORIES = ["torso", "lower_extremity", "upper_extremity", "head_neck", "unknown", "palms_soles"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DINO_DIM, CNN_OUT = 384, 96
CNN_DIM = CNN_OUT * 2
VISION_DIM = DINO_DIM + CNN_DIM
VISION_EMB_DIM, META_EMB_DIM = 256, 16
META_DIM = 1 + 1 + len(SEX_CATEGORIES) + len(SITE_CATEGORIES)
FUSED_DIM = VISION_EMB_DIM + META_EMB_DIM

# ---------------- metadata encoding (identical to train.py) ----------------
def encode_metadata(df, age_mean, age_std):
    age_raw     = pd.to_numeric(df["age"], errors="coerce")
    age_missing = age_raw.isna().astype(np.float32).to_numpy().reshape(-1, 1)
    age_norm    = ((age_raw.fillna(age_mean) - age_mean) / age_std).astype(np.float32).to_numpy().reshape(-1, 1)
    sex_oh  = pd.get_dummies(df["sex"]).reindex(columns=SEX_CATEGORIES, fill_value=0).to_numpy(dtype=np.float32)
    site_oh = pd.get_dummies(df["anatom_site"]).reindex(columns=SITE_CATEGORIES, fill_value=0).to_numpy(dtype=np.float32)
    return np.concatenate([age_norm, age_missing, sex_oh, site_oh], axis=1)

# ---------------- preprocessing (identical to preprocess.py) ----------------
def remove_hair(img, kernel_size=25):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 20, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(img, mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)

def preprocess_image(path, hair=True, target=RESIZE):
    orig = cv2.imread(path)
    if orig is None:
        return None
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    h, w, _ = orig.shape
    side = min(h, w); y0 = (h - side)//2; x0 = (w - side)//2
    crop = orig[y0:y0+side, x0:x0+side]
    intermed = max(800, target*2)
    rk = round(25*intermed/800); hk = rk if rk % 2 == 1 else rk + 1
    crop = cv2.resize(crop, (intermed, intermed), interpolation=cv2.INTER_AREA)
    if hair:
        crop = remove_hair(crop, kernel_size=hk)
    return cv2.resize(crop, (target, target), interpolation=cv2.INTER_LANCZOS4)

def build_path_index():
    idx = {}
    for entry in os.scandir(MILK_DIR):
        if entry.is_dir():
            for sub in os.scandir(entry.path):
                if sub.is_file() and not sub.name.endswith(".txt"):
                    idx[os.path.splitext(sub.name)[0]] = sub.path
    return idx

class MobileDataset(Dataset):
    def __init__(self, df, age_mean, age_std, hair=True):
        self.df = df.reset_index(drop=True)
        self.meta = encode_metadata(self.df, age_mean, age_std)
        self.hair = hair
        self.index = build_path_index()
        self.tf = transforms.Compose([
            transforms.Resize((RESIZE, RESIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        arr = preprocess_image(self.index[row["image_id"]], hair=self.hair)
        img = Image.fromarray(arr)
        meta = torch.tensor(self.meta[i], dtype=torch.float32)
        return self.tf(img), meta, torch.tensor(float(row["target"]), dtype=torch.float32)

# ---------------- model (identical to train.py) ----------------
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
        self.avgpool = nn.AdaptiveAvgPool2d(1); self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        f = self.features(x)
        return self.dropout(torch.cat([self.avgpool(f).flatten(1), self.maxpool(f).flatten(1)], 1))

class VisionProj(nn.Module):
    def __init__(self, in_dim=VISION_DIM, hidden=384, out_dim=VISION_EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, out_dim), nn.ReLU(), nn.Dropout(0.3))
    def forward(self, x): return self.net(x)

class MetaMLP(nn.Module):
    def __init__(self, in_dim=META_DIM, hidden=32, out_dim=META_EMB_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, out_dim), nn.ReLU())
    def forward(self, x): return self.mlp(x)

class Classifier(nn.Module):
    def __init__(self, in_dim=FUSED_DIM, h1=256, h2=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h2, 1))
    def forward(self, x): return self.net(x)

def load_dino():
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg", verbose=False, trust_repo=True)
    for p in m.parameters(): p.requires_grad_(False)
    return m.eval()

# ---------------- metrics ----------------
def metrics(y, p, prob):
    n = len(y)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0,1]).ravel()
    prec = tp/(tp+fp) if tp+fp else 0.0
    rec  = tp/(tp+fn) if tp+fn else 0.0
    try: auc = roc_auc_score(y, prob)
    except Exception: auc = float("nan")
    return dict(n=n, Acc=(tp+tn)/n, Prec=prec, Rec=rec,
                F1=2*prec*rec/(prec+rec) if prec+rec else 0.0,
                TPR=rec, FPR=fp/(fp+tn) if fp+tn else 0.0, AUC=auc,
                TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn))

def block(m, title):
    n=m["n"]
    return "\n".join([
        f"{title} (n={n}):",
        f"  Acc={m['Acc']:.3f}  Prec={m['Prec']:.3f}  Rec={m['Rec']:.3f}  F1={m['F1']:.3f}"
        f"  TPR={m['TPR']:.3f}  FPR={m['FPR']:.3f}  AUC={m['AUC']:.3f}",
        f"  Confusion (count / %):",
        f"                  Pred Benign        Pred Malignant",
        f"  True Benign     {m['TN']:5d} ({m['TN']/n:5.1%})    {m['FP']:5d} ({m['FP']/n:5.1%})",
        f"  True Malignant  {m['FN']:5d} ({m['FN']/n:5.1%})    {m['TP']:5d} ({m['TP']/n:5.1%})",
    ])

@torch.no_grad()
def infer(df, dino, cnn, vp, mm, clf, age_mean, age_std, hair):
    ds = MobileDataset(df, age_mean, age_std, hair=hair)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    probs, labels = [], []
    for imgs, meta, y in loader:
        imgs, meta = imgs.to(DEVICE), meta.to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            dfeat = dino(imgs).float()
        vemb = vp(torch.cat([dfeat, cnn(imgs)], 1))
        memb = mm(meta)
        logits = clf(torch.cat([vemb, memb], 1)).squeeze(1)
        probs.extend(torch.sigmoid(logits).float().cpu().numpy().tolist())
        labels.extend(y.numpy().tolist())
    return np.array(labels), np.array(probs)

def main():
    print(f"Device: {DEVICE}")
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    age_mean, age_std = ck["age_mean"], ck["age_std"]
    print(f"Checkpoint best_epoch={ck['best_epoch']}  age_mean={age_mean:.2f} age_std={age_std:.2f}")

    dino = load_dino().to(DEVICE)
    cnn = TinyCNN().to(DEVICE); cnn.load_state_dict(ck["cnn"]); cnn.eval()
    vp = VisionProj().to(DEVICE); vp.load_state_dict(ck["vision_proj"]); vp.eval()
    mm = MetaMLP().to(DEVICE); mm.load_state_dict(ck["meta_mlp"]); mm.eval()
    clf = Classifier().to(DEVICE); clf.load_state_dict(ck["classifier"]); clf.eval()

    df = pd.read_csv(EVAL_CSV)
    print(f"Mobile eval rows: {len(df)}  malignant frac={(df['target']==1).mean():.3f}\n")

    for hair in (True, False):
        tag = "WITH hair removal (matches training)" if hair else "WITHOUT hair removal (mobile-appropriate)"
        print("="*70); print(f"MOBILE EVAL — {tag}"); print("="*70)
        t0=time.time()
        y, prob = infer(df, dino, cnn, vp, mm, clf, age_mean, age_std, hair)
        pred = (prob >= 0.5).astype(int)
        print(f"  (inference {int(time.time()-t0)}s)  mean_prob={prob.mean():.3f}\n")
        print(block(metrics(y, pred, prob), "ALL MOBILE"))
        for sp in ["test", "val", "train"]:
            mask = (df["twin_split"]==sp).to_numpy()
            lab = f"mobile whose derm-twin was in {sp.upper()}"
            print(); print(block(metrics(y[mask], pred[mask], prob[mask]), lab))
        print()

if __name__ == "__main__":
    main()
