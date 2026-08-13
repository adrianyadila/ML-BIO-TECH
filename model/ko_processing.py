
"""
KO processing module

This module is responsible for:
1. Loading KEGG Orthology (KO) annotation parquets by biosample
2. Cleaning and standardizing KO annotations
3. Aggregating KO information into ML-ready features
4. Building a KO feature matrix (X)

IMPORTANT:
- This module DOES NOT use GC-MS data
- This module DOES NOT build targets (y)
- This module DOES NOT train models
- It only builds FEATURES (X)

Author: Adriany Adila
"""

import os
import re
import pandas as pd

# GLOBAL CONFIGURATION

RE_BIOSAMPLE = re.compile(r"(nmdc_bsm-\d+-[a-z0-9]+)", re.IGNORECASE)
RE_KO = re.compile(r"(K\d{5})")

# UTILITIES

def extract_biosample(path: str, df: pd.DataFrame | None = None) -> str:
    """
    Extracts biosample identifier.

    Priority:
    1. From path (nmdc_bsm-...)
    2. From DataFrame column 'sample' (fallback)

    Raises error if biosample cannot be uniquely identified.
    """
    match = RE_BIOSAMPLE.search(path)
    if match:
        return match.group(1)

    if df is not None and "sample" in df.columns:
        samples = df["sample"].dropna().unique()
        if len(samples) == 1:
            return samples[0]
        elif len(samples) > 1:
            raise ValueError(
                f"Multiple biosamples found in parquet: {samples}"
            )

    raise ValueError(f"Biosample not found in path or dataframe: {path}")


def extract_ko_code(value: str) -> str | None:
    """
    Extracts a valid KEGG Orthology code (Kxxxxx) from a string.
    Returns None if no KO code is found.
    """
    if not isinstance(value, str):
        return None

    match = RE_KO.search(value)
    if match is None:
        return None

    return match.group(1)


# STEP 1 — LOAD KO PARQUET FILES

def load_ko_biosample_parquets(ko_root: str) -> dict[str, pd.DataFrame]:
    """
    Loads KO parquet files organized by biosample.

    ko_root can be:
    - Annotation_KEGG_Orthology/
    - Annotation_KEGG_Orthology/sty-XXXX/
    """

    biosample_dirs = [
        d for d in os.listdir(ko_root)
        if d.startswith("nmdc_bsm")
        and os.path.isdir(os.path.join(ko_root, d))
    ]

    if not biosample_dirs:
        raise RuntimeError(
            f"No KO biosamples found under {ko_root}. "
            "Expected directories starting with nmdc_bsm."
        )

    biosample_data = {}

    for biosample in biosample_dirs:
        biosample_path = os.path.join(ko_root, biosample)

        parts = [
            os.path.join(biosample_path, f)
            for f in os.listdir(biosample_path)
            if f.endswith(".parquet")
        ]

        if not parts:
            continue

        dfs = [pd.read_parquet(p) for p in parts]
        biosample_data[biosample] = pd.concat(dfs, ignore_index=True)

    return biosample_data

# STEP 2 — CLEAN KO ANNOTATIONS

def clean_ko_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the KO annotation DataFrame.

    Operations:
    - Extracts valid KO codes
    - Drops rows without KO
    """

    # In NMDC KO parquets, KO code is usually in the 3rd column
    ko_col = df.columns[2]

    df = df.copy()
    df["KO"] = df[ko_col].apply(extract_ko_code)

    df = df.dropna(subset=["KO"])

    return df

# STEP 3 — AGGREGATION BY BIOSAMPLE

def aggregate_ko_features(df: pd.DataFrame) -> pd.Series:
    """
    Aggregates KO annotations into numerical features.

    Current features:
    - total_ko_count
    - unique_ko_count
    """

    ko_codes = df["KO"]

    total_ko_count = len(ko_codes)
    unique_ko_count = ko_codes.nunique()

    return pd.Series({
        "total_ko_count": total_ko_count,
        "unique_ko_count": unique_ko_count,
    })



def build_ko_long_table(ko_root: str) -> pd.DataFrame:
    """
    Builds a long-format KO table:
    biosample | KO
    """

    biosample_data = load_ko_biosample_parquets(ko_root)

    records = []

    for biosample, df in biosample_data.items():
        df = clean_ko_dataframe(df)

        for ko in df["KO"].dropna().unique():
            records.append({
                "biosample": biosample,
                "KO": ko
            })

    return pd.DataFrame(records)

# PIPELINE — SINGLE BIOSAMPLE

def process_single_biosample(df: pd.DataFrame) -> pd.Series:
    """
    Executes the complete KO processing pipeline
    for a single biosample.
    """

    df = clean_ko_dataframe(df)
    features = aggregate_ko_features(df)

    return features

# PIPELINE — ALL BIOSAMPLES

def build_ko_feature_matrix(ko_root: str) -> pd.DataFrame:
    """
    Processes all KO biosamples and builds
    the final KO feature matrix (X).

    Returns a DataFrame with:
    - biosample
    - KO-derived numerical features
    """

    biosample_data = load_ko_biosample_parquets(ko_root)

    records = []

    for biosample, df in biosample_data.items():
        features = process_single_biosample(df)

        record = {"biosample": biosample}
        record.update(features.to_dict())

        records.append(record)

    feature_matrix = (
        pd.DataFrame(records)
          .sort_values("biosample")
          .reset_index(drop=True)
    )

    return feature_matrix
