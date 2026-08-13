
"""
KEGG pathway processing module

This module is responsible for:
1. Mapping KO annotations to KEGG pathways
2. Selecting aromatic degradation pathways
3. Computing pathway-level features per biosample
4. Producing ML-ready pathway features (X)

Features produced (11 total):
- 10 coverage features (one per aromatic pathway)
- 1 aggregate: mean_aromatic_coverage

Excluded pathways:
- map00641 (styrene): no KOs mapped in current KEGG version
- map00363 (bisphenol): only 2 KOs — high false-positive coverage risk

Removed in v2 (redundant with coverage):
- 12 presence features (derivable: int(coverage > 0))
- aromatic_pathway_count (derivable: sum of presences)

IMPORTANT:
- Input: KO annotations per biosample
- Output: Pathway-level features
- No GC-MS data is used here
- No model training is performed

Author: Adriany Adila
"""

import os
import pandas as pd
from collections import defaultdict

# CONFIGURATION

# Aromatic degradation pathways (KEGG maps)
# Styrene (map00641) excluded: 0 KOs mapped — always zero coverage
# Bisphenol (map00363) excluded: only 2 KOs — coverage dominated by noise
AROMATIC_PATHWAYS = {
    # Central hubs
    "map00362": "benzoate_degradation",
    "map00401": "benzene_degradation",

    # BTEX
    "map00623": "toluene_degradation",
    "map00627": "xylene_degradation",
    "map00642": "ethylbenzene_degradation",

    # Complex aromatics
    "map00624": "pah_degradation",

    # Halogenated / industrial
    "map00361": "chlorobenzene_degradation",
    "map00364": "fluorobenzoate_degradation",
    "map00633": "nitrotoluene_degradation",

    # Anaerobic
    "map00622": "anaerobic_aromatic_degradation",
}

# STEP 1 — LOAD KO → PATHWAY MAPPING

def load_ko_to_pathway_mapping(mapping_file: str) -> pd.DataFrame:
    """
    Loads a KO → KEGG pathway mapping table.

    Expected columns:
    - KO
    - pathway_id (e.g. map00362)
    """
    df = pd.read_csv(mapping_file, sep="\t")

    required = {"ko", "pathway"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Mapping file must contain columns: {required}"
        )

    df = df.rename(columns={
        "ko": "KO",
        "pathway": "pathway_id"
    })

    # Keep only KEGG pathway maps
    df = df[df["pathway_id"].str.startswith("map")]

    return df


# STEP 2 — BUILD PATHWAY KO SETS

def build_pathway_ko_sets(mapping_df: pd.DataFrame) -> dict:
    """
    Builds a dictionary mapping each pathway
    to the set of KOs involved.
    """

    pathway_kos = defaultdict(set)

    for _, row in mapping_df.iterrows():
        pathway_kos[row["pathway_id"]].add(row["KO"])

    return pathway_kos


# STEP 3 — PATHWAY FEATURES FOR ONE BIOSAMPLE

def compute_pathway_features_for_biosample(
    ko_df: pd.DataFrame,
    pathway_ko_sets: dict
) -> pd.Series:
    """
    Computes aromatic pathway coverage features
    for a single biosample.

    Output: 10 coverage features + 1 aggregate
    """

    sample_kos = set(ko_df["KO"].dropna().unique())

    features = {}
    coverages = []

    for pathway_id, pathway_name in AROMATIC_PATHWAYS.items():

        kos_in_pathway = pathway_ko_sets.get(pathway_id, set())

        if not kos_in_pathway:
            coverage = 0.0
        else:
            coverage = len(sample_kos & kos_in_pathway) / len(kos_in_pathway)

        features[f"{pathway_name}_coverage"] = coverage
        coverages.append(coverage)

    # Single aggregate metric
    features["mean_aromatic_coverage"] = (
        sum(coverages) / len(coverages) if coverages else 0.0
    )

    return pd.Series(features)

# STEP 4 — BUILD KEGG PATHWAY FEATURE MATRIX

def build_kegg_pathway_feature_matrix(
    ko_long_table: pd.DataFrame,
    mapping_file: str
) -> pd.DataFrame:
    """
    Builds KEGG aromatic pathway features for all biosamples.

    Parameters
    ----------
    ko_long_table : pd.DataFrame
        Long-format KO table with columns:
        - biosample
        - KO

    mapping_file : str
        Path to KO → KEGG pathway mapping file

    Returns
    -------
    pd.DataFrame
        Pathway-level feature matrix (X)
        11 columns: 10 coverages + mean_aromatic_coverage
    """

    # Sanity checks
    required_cols = {"biosample", "KO"}
    if not required_cols.issubset(ko_long_table.columns):
        raise ValueError(
            f"ko_long_table must contain columns {required_cols}"
        )

    # Load KO → pathway mapping
    mapping_df = load_ko_to_pathway_mapping(mapping_file)
    pathway_ko_sets = build_pathway_ko_sets(mapping_df)

    records = []

    # Compute features per biosample
    for biosample, df_sample in ko_long_table.groupby("biosample"):
        features = compute_pathway_features_for_biosample(
            ko_df=df_sample,
            pathway_ko_sets=pathway_ko_sets
        )

        record = {"biosample": biosample}
        record.update(features.to_dict())
        records.append(record)

    feature_matrix = (
        pd.DataFrame(records)
        .sort_values("biosample")
        .reset_index(drop=True)
    )

    return feature_matrix
