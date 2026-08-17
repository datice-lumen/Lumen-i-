# patches — uncommitted working-tree changes from server clones

## `legacy-448-migration.patch`

Recovered from the dirty working tree of `/home/datice/lumen_refactor/Lumen-i-`
(branch `feature/refactor` @ 792c950) on slavica. Migrates the **legacy
CustomCNN path** from 224×224 to 448×448:

- `configs/default.yaml` — input/target/inference size 448, batch 32
- `src/lumen/model.py` — CustomCNN classifier `128*14*14` → `128*28*28`
- `src/lumen/preprocessing.py` — resolution-aware hair removal (intermediate
  `max(800, 2×target)`, proportional odd kernel)
- `src/lumen/{inference,training/dataset,training/trainer}.py`,
  `web_app/backend/router.py`, `scripts/preprocess_dataset.py` — 448 defaults
- `scripts/preprocess_dataset.py` — duplicates-CSV column fallback
  (`ISIC_id_paired` → `image_name_2`)

**Status: partially superseded.** The resolution-aware hair removal idea landed
on `main` as `preprocess_fused` / `preprocess_mobile`, and the production model
is now the fused metadata model — but the legacy CustomCNN path on `main` is
still 224px. Kept as a patch rather than applied, since applying it would change
the behavior of the legacy path (and the tracked 224px legacy weights would no
longer match the architecture). The duplicates-CSV column fallback is the one
hunk that may still be worth cherry-picking into `main`.

Apply with: `git apply research/patches/legacy-448-migration.patch`
(against `feature/refactor`; needs 3-way merge against current `main`).
