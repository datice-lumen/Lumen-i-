#!/usr/bin/env python3
"""Prepare skin-lesion detection data: copy a skin sample from the mounted
slavica filesystem and download a small, diverse set of non-skin negatives.

Run once; everything downstream (fit/eval) is local and CPU-only.

Usage:
    .venv/bin/python scripts/prepare_gate_data.py            # sensible defaults
    .venv/bin/python scripts/prepare_gate_data.py --help
"""
import argparse
import os
import shutil
import tarfile
import urllib.request
from glob import glob

import numpy as np
import pandas as pd

MOUNT = "/home/hlupek/slavica/remote"
MILK_META = f"{MOUNT}/data/original_data/MILK10k/MILK10k_Training_Metadata.csv"
MILK_IMG_ROOT = f"{MOUNT}/data/original_data/MILK10k/MILK10k_Training_Input"
PREP_DERM = f"{MOUNT}/model_10_6/preprocessed448_67k"
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"


_TYPE_SOURCE = {"clinical: close-up": "milk_clinical", "dermoscopic": "milk_dermoscopic"}


def _walk_milk(meta, need):
    """Single early-stopping walk over the MILK input tree (SSHFS-friendly).

    `need`: {source: count}. Globbing per-id over SSHFS is pathologically slow, so we
    walk the tree once, bucket each .jpg by its metadata image_type, and stop as soon
    as every bucket is full. Returns [(path, source), ...].
    """
    id_to_type = dict(zip(meta["isic_id"].astype(str), meta["image_type"].astype(str)))
    remaining = dict(need)
    rows = []
    for root, _dirs, files in os.walk(MILK_IMG_ROOT):
        for fn in files:
            if not fn.lower().endswith(".jpg"):
                continue
            src = _TYPE_SOURCE.get(id_to_type.get(os.path.splitext(fn)[0]))
            if src and remaining.get(src, 0) > 0:
                rows.append((os.path.join(root, fn), src))
                remaining[src] -= 1
        if all(v <= 0 for v in remaining.values()):
            break
    return rows


def _scan_preprocessed(n):
    """Grab up to n preprocessed dermoscopy files via scandir (early stop, no full listing)."""
    if n <= 0:
        return []
    out = []
    with os.scandir(PREP_DERM) as it:
        for entry in it:
            if entry.name.startswith("pre_") and entry.name.endswith(".jpg"):
                out.append((entry.path, "preprocessed_dermoscopy"))
                if len(out) >= n:
                    break
    return out


def copy_skin(out_dir, n_clinical, n_dermoscopic, n_preprocessed, eval_frac, seed):
    rng = np.random.default_rng(seed)
    meta = pd.read_csv(MILK_META)

    rows = _walk_milk(meta, {"milk_clinical": n_clinical, "milk_dermoscopic": n_dermoscopic})
    rows += _scan_preprocessed(n_preprocessed)

    manifest = []
    for i, (src, source) in enumerate(rows):
        split = "eval" if rng.random() < eval_frac else "fit"
        dst_dir = os.path.join(out_dir, "samples", "skin", split)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, f"{source}_{i:05d}.jpg")
        shutil.copyfile(src, dst)
        manifest.append({"path": dst, "source": source, "split": split})

    df = pd.DataFrame(manifest)
    mpath = os.path.join(out_dir, "samples", "manifest.csv")
    df.to_csv(mpath, index=False)
    print(f"skin: copied {len(df)} images "
          f"(fit={sum(df['split']=='fit')}, eval={sum(df['split']=='eval')}) -> {mpath}")


def download_negatives(out_dir, n):
    neg_dir = os.path.join(out_dir, "negatives")
    os.makedirs(neg_dir, exist_ok=True)
    tgz = os.path.join(out_dir, "imagenette2-160.tgz")
    try:
        if not os.path.exists(tgz):
            print(f"downloading negatives from {IMAGENETTE_URL} ...")
            urllib.request.urlretrieve(IMAGENETTE_URL, tgz)
        with tarfile.open(tgz) as t:
            members = [m for m in t.getmembers() if m.name.lower().endswith(".jpeg")]
            members = members[:n]
            for m in members:
                m.name = os.path.basename(m.name)
                t.extract(m, neg_dir)
        print(f"negatives: extracted {len(glob(os.path.join(neg_dir, '*')))} images -> {neg_dir}")
    except Exception as exc:  # network/egress failure -> manual fallback
        print(f"WARNING: negative download failed ({exc}).")
        print(f"Drop ~{n} non-skin .jpg images into {neg_dir} manually and re-run fit/eval.")


def main():
    ap = argparse.ArgumentParser(description="Prepare detection data (copy skin + download negatives)")
    ap.add_argument("--out", default="data", help="output root (default: data)")
    ap.add_argument("--n-clinical", type=int, default=200)
    ap.add_argument("--n-dermoscopic", type=int, default=120)
    # Off by default: the preprocessed448_67k dir has 67k files and scanning it over
    # SSHFS is slow. MILK clinical close-ups + dermoscopic are the relevant mix.
    ap.add_argument("--n-preprocessed", type=int, default=0)
    ap.add_argument("--n-negatives", type=int, default=120)
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(MILK_META):
        raise SystemExit(f"Mount not found at {MILK_META}. Mount slavica first "
                         f"(~/slavica/mount-slavica.sh).")

    copy_skin(args.out, args.n_clinical, args.n_dermoscopic,
              args.n_preprocessed, args.eval_frac, args.seed)
    download_negatives(args.out, args.n_negatives)


if __name__ == "__main__":
    main()
