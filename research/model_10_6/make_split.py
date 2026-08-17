"""
Train/val/test split za final_metadata.csv (67,085 redaka).

Cilj: train ~52k, test ~10k, val ~5k (omjer 10:2:1 -> 13 dijelova).
StratifiedGroupKFold(n_splits=13) po patient_id, stratificirano po target:
  - 1 fold  -> val   (~5.2k)
  - 2 folda -> test  (~10.3k)
  - 10 fold -> train (~51.6k)

Grupiranje po patient_id osigurava da su i isti lesion_id uvijek u istom
splitu (provjereno: nijedan lesion_id ne razapinje vise od jednog patient_id).
"""

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

CSV_PATH = "/home/datice/model_10_6/final_metadata.csv"
N_SPLITS = 13
VAL_FOLDS  = {0}
TEST_FOLDS = {1, 2}
# ostalih 10 foldova -> train

df = pd.read_csv(CSV_PATH)

sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
fold_of = pd.Series(index=df.index, dtype=int)
for fold, (_, val_idx) in enumerate(sgkf.split(df, df["target"], groups=df["patient_id"])):
    fold_of.iloc[val_idx] = fold

split = pd.Series("train", index=df.index)
split[fold_of.isin(VAL_FOLDS)]  = "val"
split[fold_of.isin(TEST_FOLDS)] = "test"
df["split"] = split

df.to_csv(CSV_PATH, index=False)
print(f"Spremljeno: {CSV_PATH}\n")

# ---------------------------------------------------------------------------
# Provjere
# ---------------------------------------------------------------------------
print("Veličine i omjeri po splitu:")
print(df.groupby("split").agg(n=("target", "size"), malignant_rate=("target", "mean")))

print("\nPatient_id curenje između splitova:")
patient_splits = df.groupby("patient_id")["split"].nunique()
print(f"  patient_id-jevi u >1 splitu: {(patient_splits > 1).sum()}")

print("\nLesion_id curenje između splitova:")
lesion_splits = df.dropna(subset=["lesion_id"]).groupby("lesion_id")["split"].nunique()
print(f"  lesion_id-jevi u >1 splitu: {(lesion_splits > 1).sum()}")

print("\ndataset_source distribucija po splitu:")
print(pd.crosstab(df["dataset_source"], df["split"]))