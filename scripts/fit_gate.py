#!/usr/bin/env python3
"""Fit the one-class OOD gate on the skin fit-split. Statistics only — no training.

Detection-only: embeds the WHOLE image with frozen DINOv2-S (no SAM 2 / cropping),
then fits mean + shrunk covariance + percentile threshold.

Usage:
    .venv/bin/python scripts/fit_gate.py            # uses data/samples/manifest.csv
    .venv/bin/python scripts/fit_gate.py --help
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from lumen.gating.dino_embed import embed
from lumen.gating.ood_gate import OODGate


def _features_for(paths, cache_path):
    if cache_path and os.path.exists(cache_path):
        print(f"loading cached embeddings from {cache_path}")
        return np.load(cache_path)
    feats = []
    for p in tqdm(paths, desc="embedding fit skin"):
        rgb = np.array(Image.open(p).convert("RGB"))
        feats.append(embed(rgb))
    feats = np.stack(feats).astype(np.float32)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, feats)
    return feats


def main():
    ap = argparse.ArgumentParser(description="Fit the OOD detection gate")
    ap.add_argument("--manifest", default="data/samples/manifest.csv")
    ap.add_argument("--out", default="data/gate.npz")
    ap.add_argument("--cache", default="data/cache/fit_embeddings.npy")
    ap.add_argument("--percentile", type=float, default=99.0)
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the percentile-derived Mahalanobis cutoff with an "
                         "absolute value (e.g. picked from the eval score gap)")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    fit_paths = df[df["split"] == "fit"]["path"].tolist()
    if not fit_paths:
        raise SystemExit("no fit-split images in manifest; run prepare_gate_data.py first")

    feats = _features_for(fit_paths, args.cache)
    gate = OODGate.fit(feats, percentile=args.percentile)
    if args.threshold is not None:
        gate.threshold = float(args.threshold)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    gate.save(args.out)
    print(f"fit on {len(feats)} images; threshold={gate.threshold:.3f}; saved -> {args.out}")


if __name__ == "__main__":
    main()
