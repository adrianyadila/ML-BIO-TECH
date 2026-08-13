
"""
Dataset builder module

This module is responsible for:
1. Building EC feature matrix (X)
2. Building KO feature matrix (X)
3. Building KEGG pathway feature matrix (X)
4. Building Functional Aromatic Metabolism targets (MES)
5. Safely merging X and y by biosample
6. Producing the final ML-ready dataset

All builder functions iterate over study-level directories
(sty-*) internally, since biosample dirs (nmdc_bsm-*) are
nested one level below the root.

IMPORTANT:
- No normalization is performed here
- No model training is performed here
- No data leakage is introduced
- This module only builds the dataset

Author: Adriany Adila
"""

import os
import pandas as pd

from ec_processing import build_ec_feature_matrix
from ko_processing import (
    build_ko_feature_matrix,
    build_ko_long_table
)
from kegg_pathway_processing import build_kegg_pathway_feature_matrix
from gcms_processing import build_gcms_targets

# UTILITY — STUDY DISCOVERY

def _discover_studies(root: str) -> list:
    """Returns sorted list of sty-* directories under root."""
    if not os.path.exists(root):
        return []
    return sorted([
        d for d in os.listdir(root)
        if d.startswith("sty-")
        and os.path.isdir(os.path.join(root, d))
    ])


# DATASET BUILDER — EC only

def build_ec_only_dataset(
    ec_root: str,
    gcms_root: str,
    drop_missing: bool = True
) -> pd.DataFrame:

    all_datasets = []

    for study in _discover_studies(ec_root):
        ec_study = os.path.join(ec_root, study)
        gcms_study = os.path.join(gcms_root, study)

        if not os.path.exists(gcms_study):
            continue

        try:
            X_ec = build_ec_feature_matrix(ec_study)
            y = build_gcms_targets(gcms_study)

            dataset = X_ec.merge(y, on="biosample", how="inner")
            if not dataset.empty:
                dataset["study"] = study
                all_datasets.append(dataset)
        except Exception as e:
            print(f"  EC-only error in {study}: {e}")
            continue

    if not all_datasets:
        return pd.DataFrame()

    result = (
        pd.concat(all_datasets, ignore_index=True)
        .drop_duplicates(subset="biosample")
    )

    if drop_missing:
        result = result.dropna().reset_index(drop=True)

    return result

# DATASET BUILDER — KO only

def build_ko_only_dataset(
    ko_root: str,
    gcms_root: str,
    drop_missing: bool = True
) -> pd.DataFrame:

    all_datasets = []

    for study in _discover_studies(ko_root):
        ko_study = os.path.join(ko_root, study)
        gcms_study = os.path.join(gcms_root, study)

        if not os.path.exists(gcms_study):
            continue

        try:
            X_ko = build_ko_feature_matrix(ko_study)
            y = build_gcms_targets(gcms_study)

            dataset = X_ko.merge(y, on="biosample", how="inner")
            if not dataset.empty:
                dataset["study"] = study
                all_datasets.append(dataset)
        except Exception as e:
            print(f"  KO-only error in {study}: {e}")
            continue

    if not all_datasets:
        return pd.DataFrame()

    result = (
        pd.concat(all_datasets, ignore_index=True)
        .drop_duplicates(subset="biosample")
    )

    if drop_missing:
        result = result.dropna().reset_index(drop=True)

    return result

# DATASET BUILDER — EC + KO

def build_ec_ko_dataset(
    ec_root: str,
    ko_root: str,
    gcms_root: str,
    drop_missing: bool = True
) -> pd.DataFrame:

    all_datasets = []

    for study in _discover_studies(ko_root):
        ec_study = os.path.join(ec_root, study)
        ko_study = os.path.join(ko_root, study)
        gcms_study = os.path.join(gcms_root, study)

        if not os.path.exists(ec_study) or not os.path.exists(gcms_study):
            continue

        try:
            X_ec = build_ec_feature_matrix(ec_study)
            X_ko = build_ko_feature_matrix(ko_study)

            X = X_ec.merge(X_ko, on="biosample", how="inner")

            y = build_gcms_targets(gcms_study)

            dataset = X.merge(y, on="biosample", how="inner")
            if not dataset.empty:
                dataset["study"] = study
                all_datasets.append(dataset)
        except Exception as e:
            print(f"  EC+KO error in {study}: {e}")
            continue

    if not all_datasets:
        return pd.DataFrame()

    result = (
        pd.concat(all_datasets, ignore_index=True)
        .drop_duplicates(subset="biosample")
    )

    if drop_missing:
        result = result.dropna().reset_index(drop=True)

    return result

# DATASET BUILDER — KEGG PATHWAYS ONLY

def build_kegg_only_dataset(
    ko_root: str,
    gcms_root: str,
    ko_pathway_mapping: str,
    drop_missing: bool = True
) -> pd.DataFrame:

    all_datasets = []

    for study in _discover_studies(ko_root):
        ko_study = os.path.join(ko_root, study)
        gcms_study = os.path.join(gcms_root, study)

        if not os.path.exists(gcms_study):
            continue

        try:
            ko_long = build_ko_long_table(ko_study)
            X_kegg = build_kegg_pathway_feature_matrix(
                ko_long, ko_pathway_mapping
            )

            y = build_gcms_targets(gcms_study)

            dataset = X_kegg.merge(y, on="biosample", how="inner")
            if not dataset.empty:
                dataset["study"] = study
                all_datasets.append(dataset)
        except Exception as e:
            print(f"  KEGG-only error in {study}: {e}")
            continue

    if not all_datasets:
        return pd.DataFrame()

    result = (
        pd.concat(all_datasets, ignore_index=True)
        .drop_duplicates(subset="biosample")
    )

    if drop_missing:
        result = result.dropna().reset_index(drop=True)

    return result

# DATASET BUILDER — FINAL (EC + KO + KEGG)

def build_final_dataset(
    ec_root: str,
    ko_root: str,
    gcms_root: str,
    ko_pathway_mapping: str | None = None,
    drop_missing: bool = True
) -> pd.DataFrame:

    all_datasets = []

    for study in _discover_studies(ko_root):
        print(f"\nProcessing study: {study}")

        ec_study = os.path.join(ec_root, study)
        ko_study = os.path.join(ko_root, study)
        gcms_study = os.path.join(gcms_root, study)

        missing = []
        if not os.path.exists(ec_study):   missing.append("EC")
        if not os.path.exists(ko_study):   missing.append("KO")
        if not os.path.exists(gcms_study): missing.append("GCMS")
        if missing:
            print(f"  Skipping — missing: {missing}")
            continue

        try:
            X_ec = build_ec_feature_matrix(ec_study)
            X_ko = build_ko_feature_matrix(ko_study)

            X = X_ec.merge(X_ko, on="biosample", how="inner")

            if ko_pathway_mapping is not None:
                ko_long = build_ko_long_table(ko_study)
                X_kegg = build_kegg_pathway_feature_matrix(
                    ko_long_table=ko_long,
                    mapping_file=ko_pathway_mapping
                )
                X = X.merge(X_kegg, on="biosample", how="inner")

            y = build_gcms_targets(gcms_study)

            dataset = X.merge(y, on="biosample", how="inner")

            if not dataset.empty:
                dataset["study"] = study
                all_datasets.append(dataset)
                print(f"  OK — {len(dataset)} samples")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    if not all_datasets:
        return pd.DataFrame()

    final = (
        pd.concat(all_datasets, ignore_index=True)
        .drop_duplicates(subset="biosample")
        .reset_index(drop=True)
    )

    if drop_missing:
        final = final.dropna().reset_index(drop=True)

    return final

# OPTIONAL CLI-LIKE USAGE

if __name__ == "__main__":

    PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/content/ML-BIO-TECH")
    DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "parquet_output_samples")

    EC_ROOT = os.path.join(DATA_ROOT, "Annotation_Enzyme_Commission")
    KO_ROOT = os.path.join(DATA_ROOT, "Annotation_KEGG_Orthology")
    GCMS_ROOT = os.path.join(DATA_ROOT, "GC_MS_Metabolomics_Results")
    KO_PATHWAY_MAPPING = os.path.join(PROJECT_ROOT, "model", "ko_to_kegg_pathway.tsv")

    df = build_final_dataset(
        ec_root=EC_ROOT,
        ko_root=KO_ROOT,
        gcms_root=GCMS_ROOT,
        ko_pathway_mapping=KO_PATHWAY_MAPPING,
    )

    print("Dataset shape:", df.shape)
    print("Unique biosamples:", df["biosample"].nunique())
