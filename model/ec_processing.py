
"""
EC processing module

This module is responsible for:
1. Loading Enzyme Commission (EC) annotation parquets by biosample
2. Cleaning and standardizing EC annotations
3. Aggregating EC information into ML-ready features
4. Building an EC feature matrix (X)

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
RE_EC = re.compile(r"(EC:\d+\.\d+\.\d+\.\d+)")

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


def extract_ec_code(value: str) -> str | None:
    """
    Extracts a valid EC code (EC:x.x.x.x) from a string.
    Returns None if no EC code is found.
    """
    if not isinstance(value, str):
        return None

    match = RE_EC.search(value)
    if match is None:
        return None

    return match.group(1)

# STEP 1 — LOAD EC PARQUET FILES

def load_ec_biosample_parquets(root_dir: str) -> dict:
    """
    Loads EC annotation parquet files and organizes them by biosample.

    Supports biosample-level and dataset-level layouts.
    """
    biosample_dfs = {}

    for root, _, files in os.walk(root_dir):
        parquet_files = [f for f in files if f.endswith(".parquet")]
        if not parquet_files:
            continue

        dfs = []
        biosample = None

        for fn in parquet_files:
            path = os.path.join(root, fn)
            df = pd.read_parquet(path)

            if biosample is None:
                biosample = extract_biosample(root, df)

            dfs.append(df)

        if biosample in biosample_dfs:
            biosample_dfs[biosample].append(pd.concat(dfs, ignore_index=True))
        else:
            biosample_dfs[biosample] = pd.concat(dfs, ignore_index=True)

    if not biosample_dfs:
        raise RuntimeError("No EC parquet files found.")

    return biosample_dfs

# STEP 2 — CLEAN EC ANNOTATIONS

def clean_ec_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the EC annotation DataFrame.

    Operations:
    - Extracts valid EC codes
    - Drops rows without EC
    """

    # EC code is typically the 3rd column in NMDC EC parquets
    ec_col = df.columns[2]

    df = df.copy()
    df["EC"] = df[ec_col].apply(extract_ec_code)

    df = df.dropna(subset=["EC"])

    return df

# STEP 3 — AGGREGATION BY BIOSAMPLE

def aggregate_ec_features(df: pd.DataFrame) -> pd.Series:
    """
    Aggregates EC annotations into numerical features.

    Features:
    - total_ec_count
    - unique_ec_count
    - ec_level1_count
    - ec_level2_count
    - ec_level3_count
    """

    ec_codes = df["EC"]

    total_ec_count = len(ec_codes)
    unique_ec_count = ec_codes.nunique()

    ec_level1 = ec_codes.str.extract(r"EC:(\d+)")[0]
    ec_level2 = ec_codes.str.extract(r"EC:(\d+\.\d+)")[0]
    ec_level3 = ec_codes.str.extract(r"EC:(\d+\.\d+\.\d+)")[0]

    ec_level1_count = ec_level1.nunique()
    ec_level2_count = ec_level2.nunique()
    ec_level3_count = ec_level3.nunique()

    return pd.Series({
        "total_ec_count": total_ec_count,
        "unique_ec_count": unique_ec_count,
        "ec_level1_count": ec_level1_count,
        "ec_level2_count": ec_level2_count,
        "ec_level3_count": ec_level3_count,
    })

# PIPELINE — SINGLE BIOSAMPLE

def process_single_biosample(df: pd.DataFrame) -> pd.Series:
    """
    Executes the complete EC processing pipeline
    for a single biosample.
    """
    df = clean_ec_dataframe(df)
    features = aggregate_ec_features(df)
    return features

# PIPELINE — ALL BIOSAMPLES

def build_ec_feature_matrix(ec_root: str) -> pd.DataFrame:
    """
    Processes all EC biosamples and builds
    the final EC feature matrix (X).

    Returns a DataFrame with:
    - biosample
    - EC-derived numerical features
    """

    biosample_data = load_ec_biosample_parquets(ec_root)

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
