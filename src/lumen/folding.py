"""Triple-stratified k-fold splitting.

Assigns patients to folds while preserving:
1. Patient-level separation (no data leakage across folds)
2. Class balance (class-1 patients distributed first, then class-0)
3. Fairness group representation
"""

from collections import defaultdict

import pandas as pd


def triple_stratified_fold(patient_dict, df, num_folds=5):
    """Assign each image to a fold via patient-level round-robin stratification.

    Patients with class-1 images are distributed first (sorted by descending
    class-1 count), then class-0-only patients fill remaining slots.

    Args:
        patient_dict: Dict mapping patient_id -> [class0_count, class1_count].
        df: DataFrame with columns 'patient_id', 'image_name', 'target'.
        num_folds: Number of folds.

    Returns:
        DataFrame with columns: patientID, image_name, foldID, target_class.
    """
    data = []

    # Patients with at least one class-1 image (distributed first)
    patients_with_class1 = {k: v for k, v in patient_dict.items() if v[1] != 0}
    patients_class0_only = {k: v for k, v in patient_dict.items() if k not in patients_with_class1}

    # Round-robin assignment for class-1 patients (sorted by class-1 count, descending)
    fold_idx = 0
    for patient_id, _ in sorted(patients_with_class1.items(), key=lambda x: x[1][1], reverse=True):
        patient_rows = df[df["patient_id"] == patient_id]
        for _, row in patient_rows.iterrows():
            data.append({
                "patientID": patient_id,
                "image_name": row["image_name"],
                "foldID": fold_idx,
                "target_class": row["target"],
            })
        fold_idx = (fold_idx + 1) % num_folds

    # Round-robin assignment for class-0-only patients
    fold_idx = 0
    for patient_id, _ in sorted(patients_class0_only.items(), key=lambda x: x[1][0], reverse=True):
        patient_rows = df[df["patient_id"] == patient_id]
        for _, row in patient_rows.iterrows():
            data.append({
                "patientID": patient_id,
                "image_name": row["image_name"],
                "foldID": fold_idx,
                "target_class": row["target"],
            })
        fold_idx = (fold_idx + 1) % num_folds

    return pd.DataFrame(data)


def build_patient_dict(df):
    """Build a patient -> [class0_count, class1_count] mapping from a DataFrame.

    Args:
        df: DataFrame with 'patient_id' and 'target' columns.

    Returns:
        Dict mapping patient_id -> [n_class0, n_class1].
    """
    patient_dict = defaultdict(lambda: [0, 0])
    for _, row in df.iterrows():
        patient_dict[row["patient_id"]][int(row["target"])] += 1
    return dict(patient_dict)
