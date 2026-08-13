
"""
Model training module

This module is responsible for:
1. Loading the final dataset
2. Splitting data into train / test
3. Performing normalization without leakage
4. Training baseline ML models
5. Evaluating performance

Models included:
- Random Forest Classifier
- Neural Network (MLP)
- Support Vector Machine
- XGBoost

Author: Adriany Adila
"""

import os
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    cross_val_predict
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from dataset_builder import build_kegg_only_dataset

# STEP 1 - CONFIGURATION

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/content/ML-BIO-TECH")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "parquet_output_samples")

EC_ROOT = os.path.join(DATA_ROOT, "Annotation_Enzyme_Commission")
GCMS_ROOT = os.path.join(DATA_ROOT, "GC_MS_Metabolomics_Results")
KO_ROOT = os.path.join(DATA_ROOT, "Annotation_KEGG_Orthology")
KO_PATHWAY_MAPPING = os.path.join(PROJECT_ROOT, "model", "ko_to_kegg_pathway.tsv")

RANDOM_STATE = 42

CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

# STEP 2 — Dataset loading

def load_dataset() -> pd.DataFrame:

    df = build_kegg_only_dataset(
        ko_root=KO_ROOT,
        gcms_root=GCMS_ROOT,
        ko_pathway_mapping=KO_PATHWAY_MAPPING,
    )

    return df


# STEP 3 — Feature / Target Split

def split_features_targets(df: pd.DataFrame):

    target_col = "MES_label"

    feature_cols = [
        c for c in df.columns
        if c not in [
            "biosample",
            "study",
            "MES",
            "MES_label",
        ]
    ]

    X = df[feature_cols]
    y = df[target_col]

    return X, y

# STEP 4 — Dataset Diagnostics

def dataset_diagnostics(df):

    print("\nDataset diagnostics")
    print("-" * 50)

    print(f"Samples : {len(df)}")
    print(f"Features: {df.shape[1]-2}")   # remove biosample and target

    print("\nClass distribution")

    print(df["MES_label"].value_counts())

    print("\nClass proportions")

    print(
        df["MES_label"]
        .value_counts(normalize=True)
        .round(3)
    )


def check_class_balance(y):
    """Check if both classes are present."""
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        warnings.warn(
            f"Only one class present in target: {unique_classes[0]}. "
            "ROC-AUC will be NaN and metrics may be meaningless."
        )
        return False
    return True


# STEP 5 — Evaluation

def evaluate_classifier(
    y_true,
    y_pred,
    y_prob=None
):
    """
    Final evaluation using cross_val_predict.
    """
    # Check if we have both classes
    has_both_classes = check_class_balance(y_true)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    # Only compute ROC-AUC if both classes exist and we have probabilities
    if has_both_classes and y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = np.nan

    print("\nClassification Report")
    print(classification_report(
        y_true,
        y_pred,
        digits=3,
        zero_division=0
    ))

    print("\nConfusion Matrix")
    print(confusion_matrix(
        y_true,
        y_pred
    ))

    print("\nMetrics")
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")
    print(f"ROC-AUC  : {auc:.3f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc
    }


def evaluate_cross_validation(
    model,
    X,
    y,
    model_name,
):
    """
    Performs Stratified 5-Fold Cross Validation.
    Prints mean ± std for all metrics.
    Also returns cross-validated predictions.
    """
    # Check if we have both classes
    has_both_classes = check_class_balance(y)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    # Only add roc_auc if both classes are present
    if has_both_classes:
        scoring["roc_auc"] = "roc_auc"

    scores = cross_validate(
        model,
        X,
        y,
        cv=CV,
        scoring=scoring,
        n_jobs=-1
    )

    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)

    metrics = {}

    for metric in scoring:

        mean = scores[f"test_{metric}"].mean()
        std = scores[f"test_{metric}"].std()

        print(f"{metric:<10}: {mean:.3f} ± {std:.3f}")

        metrics[metric] = {
            "mean": mean,
            "std": std
        }

    # Also store roc_auc as NaN if not available
    if "roc_auc" not in metrics:
        metrics["roc_auc"] = {"mean": np.nan, "std": np.nan}

    # Cross-validated predictions
    y_pred = cross_val_predict(
        model,
        X,
        y,
        cv=CV,
        method="predict"
    )

    # Only get probabilities if both classes exist
    if has_both_classes:
        try:
            y_prob = cross_val_predict(
                model,
                X,
                y,
                cv=CV,
                method="predict_proba"
            )
            # Check if we got a 2D array with at least 2 columns
            if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
                y_prob = y_prob[:, 1]
            else:
                # If only one column, use it as the probability for the positive class
                if y_prob.ndim == 2 and y_prob.shape[1] == 1:
                    y_prob = y_prob[:, 0]
                else:
                    y_prob = None
                    warnings.warn(
                        "predict_proba returned unexpected shape. "
                        f"Shape: {y_prob.shape if hasattr(y_prob, 'shape') else 'unknown'}"
                    )
        except Exception as e:
            y_prob = None
            warnings.warn(f"Could not get predict_proba: {e}")
    else:
        y_prob = None

    evaluate_classifier(
        y,
        y_pred,
        y_prob
    )

    return metrics

# TRAINING — RANDOM FOREST

def train_classifier_rf(df: pd.DataFrame):

    X, y = split_features_targets(df)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

    metrics = evaluate_cross_validation(
        pipeline,
        X,
        y,
        "RANDOM FOREST"
    )

    pipeline.fit(X, y)

    return {
        "model": pipeline,
        "metrics": {k: v["mean"] for k, v in metrics.items()}
    }


# TRAINING — NEURAL NETWORK

def train_classifier_nn(df: pd.DataFrame):

    X, y = split_features_targets(df)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(16,),
            activation="relu",
            solver="adam",
            alpha=1e-2,
            learning_rate_init=1e-3,
            max_iter=1000,
            early_stopping=True,
            random_state=RANDOM_STATE
        ))
    ])

    metrics = evaluate_cross_validation(
        pipeline,
        X,
        y,
        "NEURAL NETWORK"
    )

    pipeline.fit(X, y)

    return {
        "model": pipeline,
        "metrics": {k: v["mean"] for k, v in metrics.items()}
    }

# TRAINING — SVM

def train_classifier_svm(df: pd.DataFrame):

    X, y = split_features_targets(df)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])

    metrics = evaluate_cross_validation(
        pipeline,
        X,
        y,
        "SVM"
    )

    pipeline.fit(X, y)

    return {
        "model": pipeline,
        "metrics": {k: v["mean"] for k, v in metrics.items()}
    }

# TRAINING — XGBOOST

def train_classifier_xgb(df: pd.DataFrame):

    X, y = split_features_targets(df)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0
        ))
    ])

    metrics = evaluate_cross_validation(
        pipeline,
        X,
        y,
        "XGBOOST"
    )

    pipeline.fit(X, y)

    return {
        "model": pipeline,
        "metrics": {k: v["mean"] for k, v in metrics.items()}
    }

# MAIN

if __name__ == "__main__":

    df = load_dataset()

    print("DATASET")

    print(f"Total samples      : {df.shape[0]}")
    print(f"Unique biosamples  : {df['biosample'].nunique()}")
    print(f"Dataset shape      : {df.shape}")

    print("\nStudies")
    print(df.groupby("study").size())

    dataset_diagnostics(df)

    print("\nTarget distribution")
    print(df["MES_label"].value_counts())
    print(df["MES_label"].value_counts(normalize=True).round(3))

    print("\nTRAINING MODELS")

    print("\nRandom Forest")
    rf_model = train_classifier_rf(df)

    print("\nNeural Network")
    nn_model = train_classifier_nn(df)

    print("\nSupport Vector Machine")
    svm_model = train_classifier_svm(df)

    print("\nXGBoost")
    xgb_model = train_classifier_xgb(df)

    print("\nMEAN 5-FOLD CROSS-VALIDATION F1")

    print(f"Random Forest : {rf_model['metrics']['f1']:.3f}")
    print(f"Neural Network: {nn_model['metrics']['f1']:.3f}")
    print(f"SVM           : {svm_model['metrics']['f1']:.3f}")
    print(f"XGBoost       : {xgb_model['metrics']['f1']:.3f}")
