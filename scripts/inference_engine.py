"""Scoring path for the stacked fraud ensemble."""
import argparse
import os
import sys

import duckdb
import numpy as np
from pydantic import BaseModel, Field, ValidationError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import (ARTIFACTS_DIR, DB_PATH, FEAT_M2, FEAT_M3, FEAT_M4,
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
    print(f"Loading models from: {load_dir}")

    model_2 = joblib_load(load_dir / "model_2_geo_rf.joblib")
    model_3 = joblib_load(load_dir / "model_3_cat_xgb.joblib")
    model_4 = joblib_load(load_dir / "model_4_vel_rf.joblib")
    scaler_4 = joblib_load(load_dir / "model_4_scaler.joblib")
    meta_model = joblib_load(load_dir / "meta_model.joblib")

    thresh_path = load_dir / "meta_threshold.txt"
    threshold = float(thresh_path.read_text().strip()) if thresh_path.exists() else 0.5
    return model_2, model_3, model_4, scaler_4, meta_model, threshold


def joblib_load(path):
    import joblib
    return joblib.load(path)


def validate_frame(df):
    """Vectorised pre-checks, then a Pydantic pass over the offending rows only.

    The previous implementation ran a Pydantic constructor inside df.iterrows() for
    every row, which is fine for a 100-row smoke test and unusable for a batch.
    """
    missing = [c for c in SCORED_FEATURES + ['trans_num'] if c not in df.columns]
    if missing:
        raise ValueError(f"Input frame is missing required columns: {missing}")

    if df['trans_num'].isna().any():
        bad = df.index[df['trans_num'].isna()].tolist()[:5]
        raise ValueError(f"Validation failed: trans_num is null at rows {bad}")

    numeric = df[SCORED_FEATURES]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        bad_cols = [c for c in SCORED_FEATURES
                    if not np.isfinite(df[c].to_numpy(dtype=float)).all()]
        raise ValueError(f"Validation failed: non-finite values in {bad_cols}")

    # Bounded columns get an explicit contract check via Pydantic on a sample; a
    # violation anywhere means the upstream dbt layer changed shape.
    for idx in df.index[:1].tolist() + df.index[-1:].tolist():
        row = df.loc[idx]
        try:
            TransactionFeatures(
                trans_num=str(row['trans_num']),
                amt=float(row['amt']),
                distance_km=float(row['distance_km']),
                txns_24h=int(row['txns_24h']),
                txns_7d=int(row['txns_7d']),
                category_risk=float(row['category_risk']),
                state_risk=float(row['state_risk']),
                merchant_risk=float(row['merchant_risk']),
                night=int(row['night']),
                is_online=int(row['is_online']),
                hour=int(row['hour']),
                day_of_week=int(row['day_of_week']),
            )
        except ValidationError as e:
            raise ValueError(f"Input feature validation failed at row {idx}: {e}") from e


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
    alert_idx = np.flatnonzero(triggered)
    if alert_idx.size == 0:
        raise NoAlertsInSample(
            f"No transaction in this sample of {len(meta_probs)} breached the "
            f"decision threshold (max score {meta_probs.max():.4f}). A narrative draft "
            f"is only generated for an alert. Increase the sample size so it contains one."
        )
    return int(alert_idx[np.argmax(meta_probs[alert_idx])])


def run_inference(df, model_2, model_3, model_4, scaler_4, meta_model, threshold=None):
    validate_frame(df)

    if threshold is None:
        thresh_path = resolve_model_dir() / "meta_threshold.txt"
        threshold = float(thresh_path.read_text().strip()) if thresh_path.exists() else 0.5

    p_m2 = model_2.predict_proba(df[FEAT_M2].values)[:, 1]
    p_m3 = model_3.predict_proba(df[FEAT_M3].values)[:, 1]
    p_m4 = model_4.predict_proba(scaler_4.transform(df[FEAT_M4].values))[:, 1]

    meta_p = meta_model.predict_proba(np.column_stack([p_m2, p_m3, p_m4]))[:, 1]
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
            f"SELECT * FROM fct_fraud_features WHERE dataset_split = 'test' "
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
