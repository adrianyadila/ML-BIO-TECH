
"""
GC-MS processing module

This module is responsible for:
1. Loading GC-MS parquet files by biosample
2. Resolving duplicate annotations per chromatographic peak
3. Detecting functional aromatic-degradation markers (curated list)
4. Classifying compounds (functional_aromatic / aromatic / non_aromatic)
5. Building the Metabolic Evidence Score (MES) and MES_label

DEFINITION (revised):
    The MES measures the fraction of total chromatographic peak area
    attributable to FUNCTIONAL aromatic-degradation markers
    (intermediates / products of aromatic degradation pathways).

    MES_s = sum(area of functional markers) / sum(total area)

    General aromaticity is NOT counted as evidence of degradation.
    A generic aromatic compound may be an *accumulated substrate*
    (i.e. NOT degraded), so including it would be conceptually
    inverted relative to the study objective: detecting evidence
    that a microbial community DEGRADES aromatic compounds.

    The 'aromatic' class is still assigned for descriptive/diagnostic
    purposes only and does NOT enter the MES.

IMPORTANT:
- This module DOES NOT use EC data
- This module DOES NOT perform any global normalization
- This module DOES NOT train models
- It only builds the TARGET (y)

Author: Adriany Adila
"""

import os
import re
import numpy as np
import pandas as pd

# MES CLASSIFICATION THRESHOLD
#
# NOTE: with the revised MES (functional markers only), values are
# expected to be SMALLER than under the previous hybrid formula,
# because general aromatic area no longer inflates the numerator.
# Do NOT inherit the old threshold blindly — inspect the empirical
# MES distribution (printed by build_gcms_targets) and choose the
# threshold from the quantiles / class balance you want.

STUDY_QUANTILE = 0.40

RE_BIOSAMPLE = re.compile(r"(nmdc_bsm-\d+-[a-z0-9]+)", re.IGNORECASE)

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

# STEP 1 — LOAD GC-MS PARQUET FILES

def load_gcms_biosample_parquets(root_dir: str) -> dict:
    """
    Loads GC-MS parquet files and organizes them by biosample.

    Supports both layouts:
    - biosample-level directories (nmdc_bsm-*)
    - dataset-level directories (dgms), using dataframe fallback

    Returns:
        dict[biosample] = concatenated DataFrame
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
        raise RuntimeError("No GC-MS parquet files found.")

    return biosample_dfs

# STEP 2 — RESOLVE DUPLICATE PEAK ANNOTATIONS

def resolve_peak_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each chromatographic peak (Peak Index),
    keeps only the compound with the highest
    Spectral Similarity Score.
    """
    required_cols = ["Peak Index", "Spectral Similarity Score"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")

    df = (
        df.sort_values("Spectral Similarity Score", ascending=False)
          .groupby("Peak Index", as_index=False)
          .first()
    )

    return df

# STEP 3 — FUNCTIONAL AROMATIC METABOLISM MARKERS (CURATED)
# These are SPECIFIC markers of functional aromatic degradation:
# intermediates and products of aromatic-ring degradation pathways.
# Only these compounds count as evidence of degradation in the MES.
#
# NOTE: gentisate / gentisic acid / homogentisate removed at the
FUNCTIONAL_AROMATIC_MARKERS = {
    # --- Central ring-cleavage substrates (dihydroxylated aromatics) ---
    "catechol",
    "protocatechuate",
    "protocatechuic acid",
    "pyrocatechol",
    "hydroquinone",
    "homoprotocatechuate",
    "resorcinol",
    "phloroglucinol",

    # --- Ring-cleavage products (muconate family / ortho pathway) ---
    "muconic acid",
    "cis cis muconic acid",
    "muconolactone",
    "muconate",
    "beta ketoadipate",          # β-ketoadipate — hub of the ortho pathway
    "3 oxoadipate",              # synonym of β-ketoadipate

    # --- Meta-cleavage pathway intermediates ---
    "2 hydroxymuconate",
    "2 hydroxymuconic semialdehyde",
    "hydroxymuconic semialdehyde",

    # --- Hydroxybenzoates (upper-pathway funneling substrates) ---
    "hydroxybenzoate",
    "4 hydroxybenzoate",
    "3 hydroxybenzoic acid",
    "2 hydroxybenzoic acid",
    "p hydroxybenzoic acid",
    "salicylic acid",
    "salicylate",

    # --- Methoxylated / lignin-derived aromatic acids and aldehydes ---
    # (relevant given the planned lignin-degradation focus)
    "vanillate",
    "vanillic acid",
    "vanillin",
    "syringate",
    "syringic acid",
    "syringaldehyde",
    "ferulate",
    "ferulic acid",
    "coumarate",
    "p coumaric acid",
    "coumaric acid",
    "guaiacol",
    "gallate",
    "gallic acid",
    "sinapic acid",

    # --- Benzoate / phenylpropanoid-derived acids ---
    "benzoate",
    "benzoic acid",
    "phenylacetate",
    "phenylacetic acid",
    "cinnamate",
    "cinnamic acid",
    "phenylpropionate",

    # --- Phenolic monomers commonly seen as degradation intermediates ---
    "phenol",
    "cresol",
    "hydroxyphenylacetate",
    "4 hydroxyphenylacetate",
    "homogentisic acid",

}

# STEP 4 — COMPOUND CLASSIFICATION
#
# Aromaticity via SMILES is kept ONLY to assign the descriptive
# 'aromatic' class (for diagnostics / descriptive statistics).
# It does NOT enter the MES computation.

def is_aromatic_smiles(smiles: str) -> bool:
    """
    Detects basic aromaticity from SMILES.
    Returns True if an aromatic ring is present.

    NOTE: substring matching is a coarse heuristic and is used here
    only for the descriptive 'aromatic' class, which does NOT affect
    the MES. Kept for backward-compatible diagnostics.
    """
    if not isinstance(smiles, str):
        return False

    s = smiles.lower()

    aromatic_patterns = [
        "c1", "c2", "c3", "c4", "c5", "c6", "[c",
    ]

    return any(pattern in s for pattern in aromatic_patterns)


def is_functional_aromatic_marker(compound_name: str) -> bool:
    """
    Checks whether a compound name matches a curated functional marker.
    """
    if not isinstance(compound_name, str):
        return False

    name_lower = compound_name.lower().strip()
    name_clean = name_lower.replace("-", " ").replace("_", " ").strip()

    # Partial match to capture variations
    # (e.g. "4-hydroxybenzoate" -> "hydroxybenzoate")
    for marker in FUNCTIONAL_AROMATIC_MARKERS:
        if marker in name_clean:
            return True

    return False


def get_compound_class(row: pd.Series) -> str:
    """
    Assigns a compound class:
    - functional_aromatic: curated degradation marker (counts in MES)
    - aromatic: generic aromatic by SMILES (descriptive ONLY, not in MES)
    - non_aromatic: everything else (descriptive ONLY, not in MES)
    """
    compound_name = row.get("Compound Name", "")
    smiles = row.get("SMILES", "")

    # PRIORITY 1: specific functional marker
    if is_functional_aromatic_marker(compound_name):
        return "functional_aromatic"

    # PRIORITY 2: generic aromatic (descriptive label only)
    if is_aromatic_smiles(smiles):
        return "aromatic"

    # PRIORITY 3: non aromatic
    return "non_aromatic"


# STEP 5 — MES COMPUTATION

def compute_mes(df: pd.DataFrame) -> dict:
    """
    Computes the continuous Metabolic Evidence Score (MES).

        MES = (total peak area of functional degradation markers)
              / (total peak area of all compounds)

    The binary label is NOT assigned here — it depends on the
    per-study distribution and is assigned in build_gcms_targets.
    """
    df = df.copy()
    df["compound_class"] = df.apply(get_compound_class, axis=1)

    functional_mask = df["compound_class"] == "functional_aromatic"
    aromatic_mask = df["compound_class"] == "aromatic"

    functional_area = df.loc[functional_mask, "Peak Area"].sum()
    aromatic_area = df.loc[aromatic_mask, "Peak Area"].sum()  # descriptive only
    total_area = df["Peak Area"].sum()

    if total_area <= 0:
        return {
            "MES": 0.0,
            "_functional_area": 0.0,
            "_aromatic_area": 0.0,
            "_total_area": 0.0,
        }

    MES = functional_area / total_area

    return {
        "MES": MES,
        "_functional_area": functional_area,
        "_aromatic_area": aromatic_area,
        "_total_area": total_area,
    }


# STEP 6 — Process single biosample

def process_single_biosample(df: pd.DataFrame) -> tuple:
    """Processes a single biosample (continuous MES only)."""
    df = resolve_peak_duplicates(df)
    scores = compute_mes(df)
    return df, scores


# STEP 7 — Build GC-MS targets (per-study threshold applied here)

def build_gcms_targets(
    gcms_root: str,
    study_quantile: float = STUDY_QUANTILE,
) -> pd.DataFrame:
    """
    Builds GC-MS targets for all biosamples in ONE study.

    IMPORTANT: this function is called once per study by the dataset
    builder, so `targets` below contains exactly one study's biosamples.
    The MES_label is assigned by thresholding at the given quantile of
    THIS study's MES distribution — i.e. a per-study threshold.
    """
    biosample_data = load_gcms_biosample_parquets(gcms_root)

    records = []
    for biosample, df in biosample_data.items():
        _, scores = process_single_biosample(df)
        records.append({
            "biosample": biosample,
            "MES": scores["MES"],
        })

    targets = (
        pd.DataFrame(records)
        .sort_values("biosample")
        .reset_index(drop=True)
    )

    # --- Per-study threshold ---
    # threshold = quantile of THIS study's MES distribution
    study_threshold = targets["MES"].quantile(study_quantile)

    # >= keeps the upper part as positive; ties at the quantile go positive
    targets["MES_label"] = (targets["MES"] >= study_threshold).astype(int)

    # DEBUG
    print("\nMES distribution (this study)")
    print(targets["MES"].describe())

    print(f"\nPer-study quantile used : {study_quantile}")
    print(f"Per-study threshold     : {study_threshold:.6f}")

    print("\nMES_label distribution (this study)")
    print(targets["MES_label"].value_counts())
    print(f"Proportion positive     : {targets['MES_label'].mean():.3f}")

    n_zero = int((targets["MES"] == 0).sum())
    print(f"Biosamples with MES == 0: {n_zero}")

    return targets

# AUXILIARY — DEBUG class distribution

def debug_class_distribution(df: pd.DataFrame) -> None:
    """Shows the compound-class distribution and examples per class."""
    df = df.copy()
    df["compound_class"] = df.apply(get_compound_class, axis=1)

    print("\nCompound class distribution:")
    print(df["compound_class"].value_counts())

    for class_name in ["functional_aromatic", "aromatic", "non_aromatic"]:
        samples = df[df["compound_class"] == class_name]["Compound Name"].head(5).tolist()
        print(f"\n{class_name} examples: {samples}")
