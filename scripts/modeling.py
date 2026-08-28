"""Shared fitting and scoring primitives for the stacked fraud ensemble."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from scripts.config import FEAT_M2, FEAT_M3, FEAT_M4


def score_base_models(frame, bundle):
    p2 = bundle["model_2"].predict_proba(frame[FEAT_M2].values)[:, 1]
    p3 = bundle["model_3"].predict_proba(frame[FEAT_M3].values)[:, 1]
    scaled = bundle["scaler_4"].transform(frame[FEAT_M4].values)
    p4 = bundle["model_4"].predict_proba(scaled)[:, 1]
    return p2, p3, p4


def score_ensemble(frame, bundle):
    p2, p3, p4 = score_base_models(frame, bundle)
    meta = bundle["meta_model"].predict_proba(
        np.column_stack([p2, p3, p4]))[:, 1]
    return meta, p2, p3, p4


def fit_stacked_ensemble(base_df, blend_df, fast=False):
    """Fit base learners on an earlier window and the stacker on a later window.

    ``fast`` preserves the algorithms and contracts with fewer trees for rolling
    validation or fixture-scale checks. Final artifacts always use the full setting.
    """
    y_base = base_df["is_fraud"].to_numpy()
    y_blend = blend_df["is_fraud"].to_numpy()
    if np.unique(y_base).size < 2 or np.unique(y_blend).size < 2:
        raise ValueError("Both base and blend windows must contain fraud and non-fraud rows.")

    model_2 = RandomForestClassifier(
        n_estimators=80 if fast else 400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model_2.fit(base_df[FEAT_M2].values, y_base)

    ratio = (len(y_base) - y_base.sum()) / (y_base.sum() + 1e-6)
    model_3 = XGBClassifier(
        n_estimators=200 if fast else 1400,
        learning_rate=0.05 if fast else 0.03,
        max_depth=7,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.85,
        gamma=0.3,
        reg_alpha=1.0,
        reg_lambda=2.0,
        scale_pos_weight=ratio * 1.5,
        max_delta_step=2,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=42,
    )
    model_3.fit(base_df[FEAT_M3].values, y_base)

    scaler_4 = StandardScaler()
    x4 = scaler_4.fit_transform(base_df[FEAT_M4].values)
    model_4 = RandomForestClassifier(
        n_estimators=60 if fast else 300,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    model_4.fit(x4, y_base)

    partial = {
        "model_2": model_2,
        "model_3": model_3,
        "model_4": model_4,
        "scaler_4": scaler_4,
    }
    blend_matrix = np.column_stack(score_base_models(blend_df, partial))
    meta_model = LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1)
    meta_model.fit(blend_matrix, y_blend)
    partial["meta_model"] = meta_model
    return partial
