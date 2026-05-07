# Slavica Changes To Port

Uncommitted state of `~/Lumen-i-/` on slavica (`/home/datice/Lumen-i-/`,
branch `main` at `0ba32b8`), assessed against `feature/refactor` of this
working tree on 2026-05-06.

---

## Status legend

- ✅ **In refactor** — change is already present (often via a different mechanism, e.g. CLI flag instead of hardcode).
- ⚠️ **Missing** — change is real, slavica relies on it, refactor will misbehave without porting.
- 🤷 **Skipped** — appears intentionally abandoned in the refactor (different design choice).
- 🗑️ **Artifact** — generated output, not source.

---

## Modified files

### `configs/default.yaml`

```diff
 dataset:
   base_dir: "data/"
   train_dir: "data/train/"
   val_dir: "data/val/"
+  image_size: [384, 384]

 ...

 training:
-  batch_size: 32
+  batch_size: 16
```

| Change | Status | Notes |
|---|---|---|
| `image_size: [384, 384]` | 🤷 Skipped | Refactor uses `model.input_size: [224, 224]` and a different config schema. Looks like an experiment with larger inputs, not the canonical setting. |
| `batch_size: 32 → 16` | 🤷 Skipped | Refactor sets `batch_size: 128`. Slavica's lower value pairs with the 384×384 experiment (VRAM constraint). |

**Action: none, unless you want to revisit the 384×384 experiment.**

---

### `preprocessing/preprocess.py`

| # | Slavica change | Status | Notes |
|---|---|---|---|
| 1 | `import matplotlib; matplotlib.use('Agg')` | ✅ N/A | Refactor removed plotting from core preprocessing entirely. |
| 2 | `final_folder_name = "preprocesirani_podaci_800"` (hardcoded) | ✅ Better | Refactor uses `--output` CLI flag in `scripts/preprocess_dataset.py`. |
| 3 | `TARGET_SIZE: (200, 200) → (800, 800)` | ⚠️ **Missing** | Refactor default is 200. If `lumen_model.pth` was trained on 800×800-preprocessed data, you must pass `--target-size 800` when reproducing preprocessing. |
| 4 | `BASE_DIR = "/home/datice/data/original_data/2020"` (hardcoded) | ✅ Better | Refactor takes `--images`. |
| 5 | `duplicates_csv = "ISIC_2020_Training_Duplicates.csv"` | ✅ Better | Refactor takes `--duplicates`. |
| 6 | `duplicates_set = set(duplicates_df['image_name_2'].tolist())` (was `'ISIC_id_paired'`) | ⚠️ **Missing** | `scripts/preprocess_dataset.py:56` still hardcodes `'ISIC_id_paired'`. If slavica's duplicates CSV uses the new column name, the refactor will throw `KeyError`. |
| 7 | `final_folder_path = "/home/datice/data/preprocesirani_podaci_800"` (hardcoded) | ✅ Better | Same as #2. |
| 8 | `plt.show() → plt.savefig() + plt.close()` (multiple sites) | ✅ N/A | No `plt.show()` calls remain in refactor's preprocessing path. |

**Action — port these two:**

1. **Duplicates column name:** make `scripts/preprocess_dataset.py` either accept a `--duplicates-column` flag (default `'ISIC_id_paired'`, override with `'image_name_2'`) or auto-detect.
2. **Document `--target-size 800`** in the README/USAGE if the production model was trained on 800-preprocessed data. Or add a config preset.

---

### `training/TRAIN_melanoma.ipynb`

Notebook diff was not analysed cell-by-cell. The refactor reorganises training into:

- `src/lumen/training/loss.py`
- `src/lumen/training/dataset.py`
- `src/lumen/training/trainer.py`
- `src/lumen/training/augmentation.py`
- `src/lumen/training/evaluation.py`
- `scripts/train.py`

**Action — verify before any retraining run:** open the notebook on slavica and walk each modified cell, confirming the logic exists in the corresponding refactor module. Particular attention to: loss function (fairness/equalized-odds term), augmentation pipeline, optimizer/scheduler setup, class-weight computation.

---

## Untracked files on slavica

| File | Type | Status |
|---|---|---|
| `preprocessing/patient_distribution.png` | 🗑️ Artifact | Plot output. Don't port. |
| `training/plots/` | 🗑️ Artifact | Plot output dir. Don't port. |
| `training/predictions_new.csv` | 🗑️ Artifact | Output of `evaluate_new.py`. Keep on slavica as a reference. |
| `training/predictions_v3.csv` | 🗑️ Artifact | Earlier evaluation output. Keep on slavica. |
| `training/requirements_TRAIN.txt` | ❓ Unknown | Pinned training deps. Refactor has `requirements.txt`; might be redundant or might pin different versions. **Diff the two and merge if anything's exclusive to the slavica file.** |
| `training/evaluate_new.py` | ❓ Unknown | "New" evaluator. Compare against `src/lumen/training/evaluation.py` and `scripts/predict.py` to see if its logic is captured. |
| `training/TRAIN_melanoma.py` | ❓ Unknown | Notebook converted to script. Likely overlaps with `scripts/train.py`. Diff before deleting. |
| `training/train_melanoma.py` | ❓ Unknown | Similar to above. |

---

## Concrete porting checklist

If you decide to retrain or re-run dataset preprocessing on slavica:

- [ ] Make `scripts/preprocess_dataset.py:56` flexible to either `'ISIC_id_paired'` or `'image_name_2'` column name in the duplicates CSV.
- [ ] Document or default `--target-size 800` if the deployed model was trained on 800-preprocessed data.
- [ ] Diff `training/requirements_TRAIN.txt` (slavica) against `requirements.txt` (refactor); reconcile pins.
- [ ] Diff `training/TRAIN_melanoma.ipynb` (slavica, modified) and `training/evaluate_new.py` (slavica, untracked) against `src/lumen/training/*` modules to confirm full port.
- [ ] Verify the loss/augmentation/optimizer logic from the modified notebook is present in `src/lumen/training/`.

For the **inference run** currently in flight (`predict.py` against `all_imgs/`): none of the above matters — none of the missing items are on the inference code path.
