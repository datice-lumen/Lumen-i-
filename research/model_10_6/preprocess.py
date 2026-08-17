"""
Preprocessing 67k slika (final_metadata.csv) -> 448x448, isti pipeline kao
/home/datice/data/baseline_eval/evaluate.py:
  central square crop -> resize na 896 -> hair removal (blackhat+inpaint,
  proporcionalni kernel) -> resize na 448 (LANCZOS4)

Output: /home/datice/model_10_6/preprocessed448_67k/pre_{image_id}.jpg
(naziv "pre_..." zbog kompatibilnosti s postojećim SkinDataset loaderom)
"""

import os
import time
import multiprocessing
import cv2
import numpy as np
import pandas as pd

CSV_PATH    = "/home/datice/model_10_6/final_metadata.csv"
OUT_DIR     = "/home/datice/model_10_6/preprocessed448_67k"
TARGET_SIZE = (448, 448)
NUM_PROCESSES = 16

DATA_DIRS = {
    "ISIC_2019": [
        "/home/datice/data/original_data/2019/ISIC_2019_Training_Input",
        "/home/datice/data/original_data/2019/ISIC_2019_Test_Input",
    ],
    "ISIC_2020": [
        "/home/datice/data/original_data/2020/train",
    ],
    "MILK10k": [
        "/home/datice/data/original_data/MILK10k/MILK10k_Training_Input",
    ],
}

# ---------------------------------------------------------------------------
# Path index: image_id -> full path (uključujući ugnijezdene MILK10k foldere)
# ---------------------------------------------------------------------------
def build_path_index():
    index = {}
    for source, dirs in DATA_DIRS.items():
        for d in dirs:
            for entry in os.scandir(d):
                if entry.is_dir():
                    for sub in os.scandir(entry.path):
                        if sub.is_file() and not sub.name.endswith(".txt"):
                            index[os.path.splitext(sub.name)[0]] = sub.path
                elif entry.is_file() and not entry.name.endswith(".txt"):
                    index[os.path.splitext(entry.name)[0]] = entry.path
    return index

# ---------------------------------------------------------------------------
# Preprocessing (isti pipeline kao evaluate.py)
# ---------------------------------------------------------------------------
def remove_hair(in_image, kernel_size=25):
    gray = cv2.cvtColor(in_image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 20, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(in_image, hair_mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
    return inpainted

def preprocess_image(image_path, target_size=TARGET_SIZE):
    orig = cv2.imread(image_path)
    if orig is None:
        return None
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    h, w, _ = orig.shape
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    cropped = orig[y0:y0 + side, x0:x0 + side]

    intermed = max(800, target_size[0] * 2)
    raw_kernel = round(25 * intermed / 800)
    hair_kernel = raw_kernel if raw_kernel % 2 == 1 else raw_kernel + 1
    cropped = cv2.resize(cropped, (intermed, intermed), interpolation=cv2.INTER_AREA)

    hairless = remove_hair(cropped, kernel_size=hair_kernel)
    final = cv2.resize(hairless, target_size, interpolation=cv2.INTER_LANCZOS4)
    return final

# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
_PATH_INDEX = None

def _init_worker(path_index):
    global _PATH_INDEX
    _PATH_INDEX = path_index

def _process_one(image_id):
    src_path = _PATH_INDEX.get(image_id)
    if src_path is None:
        return (image_id, "missing_path")

    out_path = os.path.join(OUT_DIR, f"pre_{image_id}.jpg")
    if os.path.exists(out_path):
        return (image_id, "skip_exists")

    img = preprocess_image(src_path)
    if img is None:
        return (image_id, "read_failed")

    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return (image_id, "ok")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Gradim path index...")
    t0 = time.time()
    path_index = build_path_index()
    print(f"  {len(path_index):,} slika indeksirano za {time.time()-t0:.1f}s\n")

    df = pd.read_csv(CSV_PATH)
    image_ids = df["image_id"].tolist()
    print(f"Ukupno za obraditi: {len(image_ids):,} slika")
    print(f"Output dir: {OUT_DIR}")
    print(f"Procesa: {NUM_PROCESSES}\n")

    counts = {"ok": 0, "skip_exists": 0, "missing_path": 0, "read_failed": 0}
    failed_ids = []

    t0 = time.time()
    with multiprocessing.Pool(processes=NUM_PROCESSES, initializer=_init_worker, initargs=(path_index,)) as pool:
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

    elapsed = time.time() - t0
    print(f"\nGotovo za {elapsed/60:.1f} min")
    print(f"  ok           : {counts.get('ok', 0)}")
    print(f"  skip_exists  : {counts.get('skip_exists', 0)}")
    print(f"  missing_path : {counts.get('missing_path', 0)}")
    print(f"  read_failed  : {counts.get('read_failed', 0)}")

    if failed_ids:
        fail_path = os.path.join(OUT_DIR, "..", "preprocess_failed.csv")
        pd.DataFrame(failed_ids, columns=["image_id", "reason"]).to_csv(fail_path, index=False)
        print(f"\nNeuspjeli zapisani u: {os.path.abspath(fail_path)}")