"""Shared fitting and scoring primitives for the stacked fraud ensemble."""

import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from scripts.config import FEAT_M2, FEAT_M3, FEAT_M4


def freeze_target_encodings(history, scoring, smoothing=20.0):
    """Apply label maps fitted only on history to a later scoring window.

    The mart's training encodings are prequential. Reusing them for a simulated
    holdout would update the map with that holdout's own earlier labels. A fold
    must instead freeze the maps at its fit boundary, just like real serving.
    Raw category/state/merchant keys are deliberately required so a global-rate
    fallback cannot hide a missing join.
    """
    keys = {"category_risk": "category", "state_risk": "state", "merchant_risk": "merchant"}
    required = {"trans_date_trans_time", *keys.values()}
    if required - set(history) or required - set(scoring) or "is_fraud" not in history:
        raise ValueError("Frozen encodings require timestamps, categorical keys, and history labels")
    if history.empty or scoring.empty or not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("Frozen encodings require nonempty windows and positive smoothing")
    labels = history["is_fraud"].to_numpy(dtype=float)
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("History labels must be binary and finite")
    if (history["trans_date_trans_time"].isna().any()
            or scoring["trans_date_trans_time"].isna().any()
            or history["trans_date_trans_time"].max() >= scoring["trans_date_trans_time"].min()):
        raise ValueError("Encoding history must be strictly earlier than the scoring window")
    transformed = scoring.copy()
    global_rate = float(labels.mean())
    for feature, key in keys.items():
        counts = history.groupby(key, dropna=False)["is_fraud"].agg(["sum", "count"])
        mapping = (counts["sum"] + smoothing * global_rate) / (counts["count"] + smoothing)
        transformed[feature] = scoring[key].map(mapping).fillna(global_rate)
    transformed["amt_x_catRisk"] = transformed["amt"] * transformed["category_risk"]
    return transformed


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
    jobs = int(os.environ.get("AEGIS_TRAINING_JOBS", "-1"))
    if jobs == 0 or jobs < -1:
        raise ValueError("AEGIS_TRAINING_JOBS must be -1 or a positive integer")
    y_base = base_df["is_fraud"].to_numpy()
    y_blend = blend_df["is_fraud"].to_numpy()
    if np.unique(y_base).size < 2 or np.unique(y_blend).size < 2:
        raise ValueError("Both base and blend windows must contain fraud and non-fraud rows.")

    model_2 = RandomForestClassifier(
        n_estimators=80 if fast else 400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=jobs,
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
        n_jobs=jobs,
        random_state=42,
    )
    model_3.fit(base_df[FEAT_M3].values, y_base)

    scaler_4 = StandardScaler()
    x4 = scaler_4.fit_transform(base_df[FEAT_M4].values)
    model_4 = RandomForestClassifier(
        n_estimators=60 if fast else 300,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=jobs,
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
    meta_model = LogisticRegression(max_iter=2000, random_state=42, n_jobs=jobs)
    meta_model.fit(blend_matrix, y_blend)
    partial["meta_model"] = meta_model
    return partial
