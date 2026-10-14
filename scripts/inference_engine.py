"""Scoring path for the stacked fraud ensemble."""
import argparse
import os
import sys

import duckdb
import numpy as np
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import (ARTIFACTS_DIR, DB_PATH, FEAT_M2, FEAT_M3, FEAT_M4, MODEL_REGISTRY_PATH,
                            resolve_model_dir)

# Union of every column the three base models consume, so validation covers what
# is actually fed to the estimators rather than an unrelated subset.
SCORED_FEATURES = sorted(set(FEAT_M2) | set(FEAT_M3) | set(FEAT_M4))


class TransactionFeatures(BaseModel):
    """Runtime contract for a single scoreable row."""
    trans_num: str = Field(..., min_length=1)
    amt: float = Field(..., ge=0)
    distance_km: float = Field(..., ge=0)
    txns_24h: int = Field(..., ge=0)
    txns_7d: int = Field(..., ge=0)
    category_risk: float = Field(..., ge=0, le=1)
    state_risk: float = Field(..., ge=0, le=1)
    merchant_risk: float = Field(..., ge=0, le=1)
    night: int = Field(..., ge=0, le=1)
    is_online: int = Field(..., ge=0, le=1)
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)


def load_models(artifacts_dir=None):
    load_dir = resolve_model_dir(artifacts_dir)
    # Explicit version paths are used by challenger evaluation and must receive the
    # same integrity checks as the default serving pointer, before unpickling.
    registry_path = (
        MODEL_REGISTRY_PATH if load_dir.parent == ARTIFACTS_DIR.resolve()
        else load_dir.parent / "registry.json"
    )
    if registry_path.exists():
        from scripts.model_registry import load_registry, verify_registered_artifacts

        verify_registered_artifacts(load_registry(registry_path), load_dir.parent, load_dir.name)
    print(f"Loading models from: {load_dir}")

    threshold = _read_threshold(load_dir)

    model_2 = joblib_load(load_dir / "model_2_geo_rf.joblib")
    model_3 = joblib_load(load_dir / "model_3_cat_xgb.joblib")
    model_4 = joblib_load(load_dir / "model_4_vel_rf.joblib")
    scaler_4 = joblib_load(load_dir / "model_4_scaler.joblib")
    meta_model = joblib_load(load_dir / "meta_model.joblib")

    return model_2, model_3, model_4, scaler_4, meta_model, threshold


def _validate_threshold(threshold):
    if isinstance(threshold, (bool, str)) or not np.isscalar(threshold):
        raise ValueError("Decision threshold must be a finite number between zero and one")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Decision threshold must be a finite number between zero and one")
    return float(threshold)


def _read_threshold(directory):
    # A threshold belongs to its model bundle; inventing 0.5 changes alert policy.
    threshold = float((directory / "meta_threshold.txt").read_text(encoding="utf-8").strip())
    return _validate_threshold(threshold)


def joblib_load(path):
    import joblib
    return joblib.load(path)


def validate_frame(df):
    """Validate every scored row using vectorized numeric and domain checks."""
    if df.empty:
        raise ValueError("Input frame must contain at least one transaction")
    if df.columns.duplicated().any():
        raise ValueError("Input frame contains duplicate column names")
    missing = [c for c in SCORED_FEATURES + ['trans_num'] if c not in df.columns]
    if missing:
        raise ValueError(f"Input frame is missing required columns: {missing}")

    if df['trans_num'].isna().any():
        bad = df.index[df['trans_num'].isna()].tolist()[:5]
        raise ValueError(f"Validation failed: trans_num is null at rows {bad}")
    if not df['trans_num'].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Validation failed: trans_num must contain nonempty strings")

    numeric = df[SCORED_FEATURES]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        bad_cols = [c for c in SCORED_FEATURES
                    if not np.isfinite(df[c].to_numpy(dtype=float)).all()]
        raise ValueError(f"Validation failed: non-finite values in {bad_cols}")

    bounds = {
        "amt": (0, np.inf), "distance_km": (0, np.inf),
        "txns_24h": (0, np.inf), "txns_7d": (0, np.inf), "card_txn_cnt": (0, np.inf),
        "card_mean_amt": (0, np.inf), "card_std_amt": (0, np.inf),
        "category_risk": (0, 1), "state_risk": (0, 1), "merchant_risk": (0, 1),
        "night": (0, 1), "is_online": (0, 1), "hour": (0, 23), "day_of_week": (0, 6),
        "hour_sin": (-1, 1), "hour_cos": (-1, 1),
    }
    integer_columns = {"txns_24h", "txns_7d", "card_txn_cnt", "night", "is_online", "hour", "day_of_week"}
    for column, (lower, upper) in bounds.items():
        values = df[column].to_numpy(dtype=float)
        if ((values < lower) | (values > upper)).any():
            raise ValueError(f"Input feature validation failed: {column} is outside its bounds")
        if column in integer_columns and (values != np.floor(values)).any():
            raise ValueError(f"Input feature validation failed: {column} must contain integers")


class NoAlertsInSample(RuntimeError):
    """No transaction in the scored sample breached the decision threshold."""


def select_highest_risk_alert(meta_probs, triggered):
    """Index of the highest-scoring transaction that actually triggered an alert.

    Taking a plain argmax over the sample is wrong: at 0.39% prevalence a few
    hundred rows usually contain no alert at all, so the "riskiest" row is a
    perfectly ordinary transaction. Drafting an investigation narrative for it—as
    this pipeline once did, on a transaction scoring 0.0040 against a 0.6152
    threshold—would fabricate a model alert. The gate establishes alert eligibility
    only; it does not establish RGS or a reporting obligation.
    """
    meta_probs = np.asarray(meta_probs, dtype=float)
    triggered = np.asarray(triggered)
    if meta_probs.ndim != 1 or triggered.shape != meta_probs.shape:
        raise ValueError("Alert scores and flags must be matching one-dimensional arrays")
    if not np.isfinite(meta_probs).all() or ((meta_probs < 0) | (meta_probs > 1)).any():
        raise ValueError("Alert scores must be finite probabilities")
    if triggered.dtype != np.bool_:
        raise ValueError("Alert flags must be booleans")
    alert_idx = np.flatnonzero(triggered)
    if alert_idx.size == 0:
        raise NoAlertsInSample(
            f"No transaction in this sample of {len(meta_probs)} breached the "
            f"decision threshold (max score {meta_probs.max() if meta_probs.size else 0:.4f}). A narrative draft "
            f"is only generated for an alert. Increase the sample size so it contains one."
        )
    return int(alert_idx[np.argmax(meta_probs[alert_idx])])


def run_inference(df, model_2, model_3, model_4, scaler_4, meta_model, threshold=None):
    validate_frame(df)

    if threshold is None:
        raise ValueError("Pass the decision threshold loaded with this model bundle")
    threshold = _validate_threshold(threshold)

    p_m2 = model_2.predict_proba(df[FEAT_M2].values)[:, 1]
    p_m3 = model_3.predict_proba(df[FEAT_M3].values)[:, 1]
    p_m4 = model_4.predict_proba(scaler_4.transform(df[FEAT_M4].values))[:, 1]

    meta_p = meta_model.predict_proba(np.column_stack([p_m2, p_m3, p_m4]))[:, 1]
    for probabilities in (p_m2, p_m3, p_m4, meta_p):
        if probabilities.shape != (len(df),) or not np.isfinite(probabilities).all() or (
            (probabilities < 0) | (probabilities > 1)
        ).any():
            raise ValueError("Model produced invalid probabilities")
    triggered_alert = (meta_p >= threshold).astype(bool)

    return meta_p, p_m2, p_m3, p_m4, triggered_alert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true', help='Run verification mode')
    parser.add_argument('--artifacts-dir', type=str, default=str(ARTIFACTS_DIR))
    parser.add_argument('--limit', type=int, default=100)
    args = parser.parse_args()

    if not args.verify:
        parser.print_help()
        return

    print("=== INFERENCE ENGINE VERIFICATION MODE ===")
    models = load_models(args.artifacts_dir)

    print(f"Connecting to DuckDB at {DB_PATH}...")
    con = duckdb.connect(str(DB_PATH))
    try:
        test_df = con.execute(
            f"SELECT * FROM fct_fraud_features WHERE evaluation_role = 'development_holdout' "
            f"LIMIT {int(args.limit)}").df()
    finally:
        con.close()

    if len(test_df) == 0:
        print("ERROR: 'fct_fraud_features' has 0 test rows. Run dbt run first.")
        return

    print(f"Loaded {len(test_df)} sample transactions.")
    meta_probs, _, _, _, triggered = run_inference(test_df, *models)

    out_of_bounds = [(i, v) for i, v in enumerate(meta_probs) if not 0.0 < v < 1.0]
    if out_of_bounds:
        for i, v in out_of_bounds[:5]:
            print(f"FAIL: Row {i} score {v} is not strictly in (0.0, 1.0)")
    else:
        print("SUCCESS: All meta-classifier outputs strictly within (0.0, 1.0).")

    print(f"Triggered alerts: {np.sum(triggered)} out of {len(triggered)}")
    print("\nProbability Score Distribution Statistics:")
    for label, val in [("Min", np.min(meta_probs)), ("Max", np.max(meta_probs)),
                       ("Mean", np.mean(meta_probs)), ("Median", np.median(meta_probs)),
                       ("Std Dev", np.std(meta_probs))]:
        print(f"  {label:<8}: {val:.6f}")

    print("\nProbability Distribution Histogram:")
    hist, bin_edges = np.histogram(meta_probs, bins=10, range=(0, 1))
    for i in range(10):
        bar = '#' * int(hist[i] / 2) if hist[i] > 0 else ''
        print(f"[{bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}): {hist[i]:3d} | {bar}")


if __name__ == "__main__":
    main()
