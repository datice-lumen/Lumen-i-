#!/usr/bin/env python3
"""Preprocess the fused-model dataset (67k dermatoscopic images) to 448x448.

Pipeline per image (lumen.preprocessing.preprocess_fused): central square crop
-> resize to max(800, 2*target) -> hair removal (black-hat + inpaint,
proportional odd kernel) -> resize to target (LANCZOS4).

Output: <output>/pre_<image_id>.jpg (the "pre_" prefix is what
lumen.training.fused.FusedImageDataset expects).

Usage:
    python scripts/preprocess_fused_dataset.py \
        --metadata final_metadata.csv \
        --images data/2019/ISIC_2019_Training_Input \
        --images data/2020/train \
        --images data/MILK10k/MILK10k_Training_Input \
        --output preprocessed448
"""

import argparse
import multiprocessing
import os
import time

import cv2
import pandas as pd

from lumen.preprocessing import preprocess_fused


def build_path_index(image_dirs):
    """Map image_id -> full path across all given dirs (one nesting level deep)."""
    index = {}
    for d in image_dirs:
        for entry in os.scandir(d):
            if entry.is_dir():
                for sub in os.scandir(entry.path):
                    if sub.is_file() and not sub.name.endswith(".txt"):
                        index[os.path.splitext(sub.name)[0]] = sub.path
            elif entry.is_file() and not entry.name.endswith(".txt"):
                index[os.path.splitext(entry.name)[0]] = entry.path
    return index


_PATH_INDEX = None
_OUT_DIR = None
_TARGET = None


def _init_worker(path_index, out_dir, target):
    global _PATH_INDEX, _OUT_DIR, _TARGET
    _PATH_INDEX = path_index
    _OUT_DIR = out_dir
    _TARGET = target


def _process_one(image_id):
    src_path = _PATH_INDEX.get(image_id)
    if src_path is None:
        return (image_id, "missing_path")

    out_path = os.path.join(_OUT_DIR, f"pre_{image_id}.jpg")
    if os.path.exists(out_path):
        return (image_id, "skip_exists")

    orig = cv2.imread(src_path)
    if orig is None:
        return (image_id, "read_failed")
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    final, _, _ = preprocess_fused(rgb, target_size=_TARGET)

    cv2.imwrite(out_path, cv2.cvtColor(final, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return (image_id, "ok")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", required=True, help="CSV with an image_id column (final_metadata.csv)")
    parser.add_argument("--images", action="append", required=True,
                        help="Directory with raw images; repeat for multiple datasets")
    parser.add_argument("--output", default="preprocessed448", help="Output folder (default: preprocessed448)")
    parser.add_argument("--target-size", type=int, default=448, help="Output image side (default: 448)")
    parser.add_argument("--processes", type=int, default=16, help="Worker processes (default: 16)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Building path index...")
    t0 = time.time()
    path_index = build_path_index(args.images)
    print(f"  {len(path_index):,} images indexed in {time.time()-t0:.1f}s\n")

    df = pd.read_csv(args.metadata)
    image_ids = df["image_id"].tolist()
    print(f"To process: {len(image_ids):,} images")
    print(f"Output dir: {args.output}")
    print(f"Processes : {args.processes}\n")

    counts = {"ok": 0, "skip_exists": 0, "missing_path": 0, "read_failed": 0}
    failed_ids = []

    t0 = time.time()
    with multiprocessing.Pool(
        processes=args.processes, initializer=_init_worker,
        initargs=(path_index, args.output, args.target_size),
    ) as pool:
        for i, (image_id, status) in enumerate(pool.imap_unordered(_process_one, image_ids, chunksize=32), 1):
            counts[status] = counts.get(status, 0) + 1
            if status not in ("ok", "skip_exists"):
                failed_ids.append((image_id, status))

            if i % 2000 == 0 or i == len(image_ids):
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (len(image_ids) - i) / rate if rate > 0 else 0
                print(f"  {i:6d}/{len(image_ids)}  ({rate:5.1f} img/s, ETA {eta/60:5.1f} min)  "
                      f"ok={counts['ok']} skip={counts['skip_exists']} fail={counts['missing_path']+counts['read_failed']}")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
    for k in ("ok", "skip_exists", "missing_path", "read_failed"):
        print(f"  {k:13s}: {counts.get(k, 0)}")

    if failed_ids:
        fail_path = os.path.join(args.output, "..", "preprocess_failed.csv")
        pd.DataFrame(failed_ids, columns=["image_id", "reason"]).to_csv(fail_path, index=False)
        print(f"\nFailures written to: {os.path.abspath(fail_path)}")


if __name__ == "__main__":
    main()
