# Mobile-domain evaluation of the newest model (model_10_6)

**Date:** 2026-07-15
**Model:** `/home/datice/model_10_6/checkpoint_20260610_230527.pt`
(DINOv2-S frozen + TinyCNN + MetaMLP → Classifier, 448px, best_epoch=7, val_loss=1.4265)
**Eval set:** all 5,240 MILK10k **mobile** ("clinical: close-up") images
**Scripts/artifacts:** `/home/datice/mobile_eval/run_mobile_eval.py`, `mobile_eval.csv`, `eval.log`

---

## Question
Does the newest (dermoscopy-trained) model give good-enough predictions on
MILK10k mobile / phone close-up photos, off the shelf?

## Answer
**No.** Sensitivity collapses from 0.91 (dermoscopy) to 0.56–0.79 (mobile), and
the discriminative ceiling on mobile is only AUC ~0.73–0.78 (vs ~0.94 in-domain).
Not screening-grade as-is.

---

## Setup / methodology
- The 5,240 MILK10k images in the 67k training set were the **dermoscopic** twins;
  the 5,240 **mobile** images were held out entirely → this is a genuine
  out-of-domain test.
- **Labels + metadata** for each mobile image were taken from its dermoscopic
  twin's row in `final_metadata.csv` (same `lesion_id` → identical age/sex/site/
  target and identical encoding). So the ONLY variable vs training is image modality.
- Malignant fraction of the mobile set: **0.717** (matches the known ~72%).
- Preprocessing replicates the training pipeline (central square crop → 896 →
  [hair removal] → 448 LANCZOS4). Run **twice**: with hair removal (matches
  training) and without (mobile-appropriate; hair removal is tuned for dermoscopy).
- Threshold 0.5 on the sigmoid (same as training). AUC is threshold-independent.
- Run on slavica (RTX A6000), DATICE conda env, torch 2.6.0+cu124.

## Results

| Metric            | Dermoscopic (in-domain test) | Mobile — WITH hair removal | Mobile — WITHOUT hair removal |
|-------------------|------------------------------|----------------------------|-------------------------------|
| n                 | 10,326                       | 5,240                      | 5,240                         |
| Accuracy          | **0.876**                    | 0.612                      | 0.727                         |
| **TPR (sens.)**   | **0.912**                    | 0.559                      | 0.787                         |
| FPR               | 0.136                        | 0.253                      | 0.425                         |
| Precision         | 0.687                        | 0.848                      | 0.824                         |
| F1                | 0.784                        | 0.674                      | 0.805                         |
| AUC               | ~0.94 (implied)              | 0.728                      | 0.748                         |
| mean pred. prob   | —                            | 0.460                      | 0.642                         |

Clean held-out subset (836 mobile images whose dermoscopic twin was in the TEST split):
- WITH hair removal:    Acc 0.614, TPR 0.556, FPR 0.231, AUC 0.744
- WITHOUT hair removal:  Acc 0.736, TPR 0.773, FPR 0.364, AUC 0.775

## Key findings
1. **Sensitivity collapses.** With the training pipeline (hair removal on), the
   model misses ~44% of malignant lesions on mobile (1,657 FN of 3,757).
2. **Low discriminative ceiling.** AUC ~0.73–0.78 on mobile regardless of
   preprocessing (vs ~0.94 in-domain) — a genuine feature-separability problem,
   not just a threshold issue.
3. **Hair removal actively hurts on mobile (confirms D-14).** Disabling it lifts
   TPR 0.56 → 0.77, but mostly by shifting the operating point (mean prob
   0.46 → 0.64); AUC barely moves (0.728 → 0.748). It trades false negatives for
   false positives (FPR 0.25 → 0.43).
4. **No leakage advantage.** Mobile images whose dermoscopic twin was *trained on*
   (TPR 0.793) perform the same as fully held-out ones (0.773) → seeing the
   dermoscopic view gives ~zero help on the phone photo. The domain gap dominates;
   the held-out estimate is trustworthy.
5. **Miscalibration.** On a 72%-malignant set the mean predicted probability is
   0.46 (hair removal on) — the model systematically under-calls malignancy on
   this domain.

## Recommendation
- Off-the-shelf inference is not screening-grade on mobile.
- Cheap immediate step: **disable hair removal for mobile inference** (free win).
- Real fix: **fine-tune / transfer-learn on the mobile domain** (D-19/D-20 —
  paired-image distillation is well-suited since every lesion has both views),
  plus threshold recalibration. Even the best clean number (AUC 0.775) doesn't
  clear the bar.

---

## Raw run log
(verbatim `eval.log` appended below)

```
Device: cuda
Checkpoint best_epoch=7  age_mean=52.43 age_std=16.54
Mobile eval rows: 5240  malignant frac=0.717

======================================================================
MOBILE EVAL — WITH hair removal (matches training)
======================================================================
  (inference 645s)  mean_prob=0.460

ALL MOBILE (n=5240):
  Acc=0.612  Prec=0.848  Rec=0.559  F1=0.674  TPR=0.559  FPR=0.253  AUC=0.728
  Confusion (count / %):
                  Pred Benign        Pred Malignant
  True Benign      1108 (21.1%)      375 ( 7.2%)
  True Malignant   1657 (31.6%)     2100 (40.1%)

mobile whose derm-twin was in TEST (n=836):
  Acc=0.614  Prec=0.867  Rec=0.556  F1=0.678  TPR=0.556  FPR=0.231  AUC=0.744
  Confusion (count / %):
                  Pred Benign        Pred Malignant
  True Benign       173 (20.7%)       52 ( 6.2%)
  True Malignant    271 (32.4%)      340 (40.7%)

mobile whose derm-twin was in VAL (n=391):
  Acc=0.609  Prec=0.869  Rec=0.552  F1=0.675  TPR=0.552  FPR=0.233  AUC=0.720
  Confusion (count / %):
                  Pred Benign        Pred Malignant
  True Benign        79 (20.2%)       24 ( 6.1%)
  True Malignant    129 (33.0%)      159 (40.7%)

mobile whose derm-twin was in TRAIN (n=4013):
  Acc=0.612  Prec=0.843  Rec=0.560  F1=0.673  TPR=0.560  FPR=0.259  AUC=0.725
  Confusion (count / %):
                  Pred Benign        Pred Malignant
  True Benign       856 (21.3%)      299 ( 7.5%)
  True Malignant   1257 (31.3%)     1601 (39.9%)

======================================================================
MOBILE EVAL — WITHOUT hair removal (mobile-appropriate)
======================================================================
  (inference 15s)  mean_prob=0.642

ALL MOBILE (n=5240):
  Acc=0.727  Prec=0.824  Rec=0.787  F1=0.805  TPR=0.787  FPR=0.425  AUC=0.748
  Confusion (count / %):
                  Pred Benign        Pred Malignant
  True Benign       853 (16.3%)      630 (12.0%)
  True Malignant    799 (15.2%)     2958 (56.5%)

mobile whose derm-twin was in TEST (n=836):
  Acc=0.736  Prec=0.852  Rec=0.773  F1=0.810  TPR=0.773  FPR=0.364  AUC=0.775
  Confusion (count / %):
                  Pred Benign        Pred Malignant
  True Benign       143 (17.1%)       82 ( 9.8%)
  True Malignant    139 (16.6%)      472 (56.5%)

mobile whose derm-twin was in VAL (n=391):
  Acc=0.708  Prec=0.827  Rec=0.764  F1=0.794  TPR=0.764  FPR=0.447  AUC=0.733
  Confusion (count / %):
                  Pred Benign        Pred Malignant
  True Benign        57 (14.6%)       46 (11.8%)
  True Malignant     68 (17.4%)      220 (56.3%)

mobile whose derm-twin was in TRAIN (n=4013):
  Acc=0.727  Prec=0.819  Rec=0.793  F1=0.806  TPR=0.793  FPR=0.435  AUC=0.745
  Confusion (count / %):
                  Pred Benign        Pred Malignant
  True Benign       653 (16.3%)      502 (12.5%)
  True Malignant    592 (14.8%)     2266 (56.5%)

```
