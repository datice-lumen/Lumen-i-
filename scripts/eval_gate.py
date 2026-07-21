#!/usr/bin/env python3
"""Evaluate the skin-lesion detection gate: skin should pass, non-skin rejected.

Detection-only: uses detect_lesion (whole-image embed → gate), no SAM 2 / crop.

Usage:
    .venv/bin/python scripts/eval_gate.py
    .venv/bin/python scripts/eval_gate.py --help
"""
import argparse
import os
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from lumen.gating.ood_gate import OODGate
from lumen.gating.detect import detect_lesion


def _run(paths, gate, desc):
    results = []
    for p in tqdm(paths, desc=desc):
        results.append((p, detect_lesion(p, gate)))
    return results


def main():
    ap = argparse.ArgumentParser(description="Evaluate the skin-lesion detection gate")
    ap.add_argument("--gate", default="data/gate.npz")
    ap.add_argument("--manifest", default="data/samples/manifest.csv")
    ap.add_argument("--negatives", default="data/negatives")
    ap.add_argument("--reports", default="reports")
    args = ap.parse_args()

    gate = OODGate.load(args.gate)
    df = pd.read_csv(args.manifest)
    skin_paths = df[df["split"] == "eval"]["path"].tolist()
    neg_paths = sorted(glob(os.path.join(args.negatives, "*")))

    skin = _run(skin_paths, gate, "eval skin")
    negs = _run(neg_paths, gate, "eval negatives")

    skin_pass = np.mean([o["status"] == "skin" for _, o in skin]) if skin else float("nan")
    neg_reject = np.mean([o["status"] == "unclassified" for _, o in negs]) if negs else float("nan")
    print(f"skin-pass rate:    {skin_pass:.1%}  (n={len(skin)})")
    print(f"negative-reject:   {neg_reject:.1%}  (n={len(negs)})")
    print(f"threshold:         {gate.threshold:.3f}")

    os.makedirs(args.reports, exist_ok=True)

    # distance histogram
    skin_scores = [o["score"] for _, o in skin if o["score"] is not None]
    neg_scores = [o["score"] for _, o in negs if o["score"] is not None]

    # Save raw scores so any candidate threshold can be evaluated without re-embedding.
    np.savez(os.path.join(args.reports, "gate_scores.npz"),
             skin=np.array(skin_scores), neg=np.array(neg_scores))
    if skin_scores and neg_scores:
        p95, p99, p995 = np.percentile(skin_scores, [95, 99, 99.5])
        print(f"skin p95/p99/p99.5: {p95:.1f}/{p99:.1f}/{p995:.1f}; "
              f"skin max {max(skin_scores):.1f}; non-skin min {min(neg_scores):.1f}")
    plt.figure(figsize=(7, 4))
    plt.hist(skin_scores, bins=30, alpha=0.6, label="skin")
    plt.hist(neg_scores, bins=30, alpha=0.6, label="non-skin")
    plt.axvline(gate.threshold, color="k", ls="--", label="threshold")
    plt.xlabel("Mahalanobis distance"); plt.ylabel("count"); plt.legend()
    plt.title("Skin-lesion detection: score distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(args.reports, "gate_hist.png"), dpi=120)
    plt.close()

    # contact sheet: 4 skin + 4 negatives (original -> verdict)
    examples = skin[:4] + negs[:4]
    n = len(examples)
    if n:
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2))
        if n == 1:
            axes = [axes]
        for ax, (p, out) in zip(axes, examples):
            ax.imshow(Image.open(p).convert("RGB"))
            ax.set_title(f"{out['status']}\nd={out['score']:.1f}" if out["score"] is not None
                         else out["status"], fontsize=9)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(args.reports, "gate_contact_sheet.png"), dpi=120)
        plt.close()
    print(f"saved reports -> {args.reports}/gate_hist.png, {args.reports}/gate_contact_sheet.png")


if __name__ == "__main__":
    main()
