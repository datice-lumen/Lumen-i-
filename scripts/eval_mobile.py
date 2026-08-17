#!/usr/bin/env python3
"""Evaluate a fused-model checkpoint on MILK10k mobile close-ups (out-of-domain).

The OOD evaluation behind docs/training/mobile_findings.md: the dermoscopy-trained
model is run on the 5,240 MILK10k mobile ("clinical: close-up") images, whose
dermoscopic twins were in the base training set. Labels + metadata come from
each mobile image's twin row (same lesion -> identical age/sex/site/target), so
the ONLY variable vs training is image modality.

Runs twice: WITH hair removal (matches the training pipeline) and WITHOUT
(hair removal is tuned for dermoscopy and mis-fires on phone photos). Both
variants keep the training pipeline's intermediate resize so that hair removal
is the only difference between them.

Usage:
    python scripts/eval_mobile.py \
        --checkpoint checkpoint_20260610_230527.pt \
        --eval-csv mobile_eval.csv \
        --images MILK10k/MILK10k_Training_Input
"""

import argparse
import time
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from lumen.model_meta import load_fused_model
from lumen.preprocessing import remove_hair, square_crop
from lumen.training.fused import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_milk_path_index,
    calc_metrics,
    encode_metadata_frame,
    format_metrics_block,
)

warnings.filterwarnings("ignore")


def preprocess_eval(path, hair, target):
    """Training pipeline with hair removal toggleable (intermediate resize kept)."""
    orig = cv2.imread(path)
    if orig is None:
        raise FileNotFoundError(f"Image not found: {path}")
    crop = square_crop(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    intermed = max(800, target * 2)
    raw_kernel = round(25 * intermed / 800)
    hair_kernel = raw_kernel if raw_kernel % 2 == 1 else raw_kernel + 1
    crop = cv2.resize(crop, (intermed, intermed), interpolation=cv2.INTER_AREA)
    if hair:
        _, crop = remove_hair(crop, kernel_size=hair_kernel)
    return cv2.resize(crop, (target, target), interpolation=cv2.INTER_LANCZOS4)


class MobileEvalDataset(Dataset):
    def __init__(self, df, age_mean, age_std, path_index, resize, hair):
        self.df = df.reset_index(drop=True)
        self.meta = encode_metadata_frame(self.df, age_mean, age_std)
        self.path_index = path_index
        self.resize = resize
        self.hair = hair
        self.tf = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        arr = preprocess_eval(self.path_index[row["image_id"]], self.hair, self.resize)
        meta = torch.tensor(self.meta[i], dtype=torch.float32)
        return self.tf(Image.fromarray(arr)), meta, torch.tensor(float(row["target"]), dtype=torch.float32)


@torch.no_grad()
def infer(df, model, age_mean, age_std, path_index, args, device, hair):
    ds = MobileEvalDataset(df, age_mean, age_std, path_index, args.resize, hair)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    probs, labels = [], []
    for imgs, meta, y in loader:
        imgs, meta = imgs.to(device), meta.to(device)
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                dino_feat = model.dino(imgs)
        else:
            dino_feat = model.dino(imgs)
        dino_feat = dino_feat.float()
        vision_emb = model.vision_proj(torch.cat([dino_feat, model.cnn(imgs)], dim=1))
        meta_emb = model.meta_mlp(meta)
        logits = model.classifier(torch.cat([vision_emb, meta_emb], dim=1)).squeeze(1)
        probs.extend(torch.sigmoid(logits).float().cpu().numpy().tolist())
        labels.extend(y.numpy().tolist())
    return np.array(labels), np.array(probs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Fused-model checkpoint (.pt)")
    parser.add_argument("--eval-csv", required=True, help="Mobile metadata CSV with twin_split column")
    parser.add_argument("--images", required=True, help="MILK10k_Training_Input directory")
    parser.add_argument("--resize", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=12)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, meta_cfg = load_fused_model(args.checkpoint, device=device)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    age_mean, age_std = meta_cfg["age_mean"], meta_cfg["age_std"]
    print(f"Checkpoint best_epoch={ck['best_epoch']}  age_mean={age_mean:.2f} age_std={age_std:.2f}")

    df = pd.read_csv(args.eval_csv)
    path_index = build_milk_path_index(args.images)
    print(f"Mobile eval rows: {len(df)}  malignant frac={(df['target'] == 1).mean():.3f}\n")

    for hair in (True, False):
        tag = "WITH hair removal (matches training)" if hair else "WITHOUT hair removal (mobile-appropriate)"
        print("=" * 70)
        print(f"MOBILE EVAL — {tag}")
        print("=" * 70)
        t0 = time.time()
        y, prob = infer(df, model, age_mean, age_std, path_index, args, device, hair)
        pred = (prob >= 0.5).astype(int)
        print(f"  (inference {int(time.time()-t0)}s)  mean_prob={prob.mean():.3f}\n")
        print(format_metrics_block(calc_metrics(y, pred, prob), "ALL MOBILE"))
        for sp in ["test", "val", "train"]:
            mask = (df["twin_split"] == sp).to_numpy()
            label = f"mobile whose derm-twin was in {sp.upper()}"
            print()
            print(format_metrics_block(calc_metrics(y[mask], pred[mask], prob[mask]), label))
        print()


if __name__ == "__main__":
    main()
