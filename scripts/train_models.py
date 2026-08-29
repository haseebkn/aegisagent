"""Train the stacked fraud ensemble.

Model-fitting split design (chronological, with no random row shuffling):

    fraudTrain.csv   -> 70% base    : fit Models 2/3/4
                     -> 15% blend   : fit the logistic meta-learner on base-model
                                      probabilities the base models have not seen
                     -> 15% calib   : select the demo decision threshold
    fraudTest.csv    -> first 75% development holdout
                     -> final 25% prospectively locked evaluation window

All model features are prior-only for training rows. The final window is protected by
an explicit unlock flag, but it is not historically pristine: earlier project versions
reported aggregate metrics over all of fraudTest.csv. A truly external blind dataset
is still required for an independent generalization claim.

The threshold used to be chosen on the same rows the meta-learner was fitted on,
which made it optimistically biased. The calib split exists solely to break that.
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime

import duckdb
import joblib
import numpy as np
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import ARTIFACTS_DIR, DB_PATH
from scripts.model_registry import import_existing_champion, register_version
from scripts.modeling import fit_stacked_ensemble, score_base_models


def chronological_split(df, fractions=(0.70, 0.15, 0.15)):
    df = df.sort_values('trans_date_trans_time').reset_index(drop=True)
    n = len(df)
    c1 = int(n * fractions[0])
    c2 = int(n * (fractions[0] + fractions[1]))
    return df.iloc[:c1].copy(), df.iloc[c1:c2].copy(), df.iloc[c2:].copy()


def prune_old_versions(artifacts_dir, keep, protected=()):
    """Delete superseded version directories, newest-first.

    Each training run writes ~230 MB of joblib into a new v_<timestamp>/ directory.
    Nothing removed them, so six runs left 1.7 GB on disk -- and because the
    Dockerfile does `COPY models_artifacts/`, every one of those dead versions was
    baked into the image, producing a 4 GB build for a model that needs 244 MB.
    CI never caught it: it trains a single tiny fixture model, so the bloat is
    invisible there by construction.

    Artifacts are reproducible from code plus data, so retention is a convenience
    for rollback rather than a safety net.
    """
    protected = set(protected)
    versions = sorted((d for d in artifacts_dir.glob("v_*") if d.is_dir()),
                      key=lambda d: d.name, reverse=True)
    unprotected = [directory for directory in versions if directory.name not in protected]
    stale = unprotected[keep:]
    if not stale:
        return []
    for d in stale:
        shutil.rmtree(d)
    return [d.name for d in stale]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep", type=int, default=2,
        help="Unprotected version directories to retain after training. The champion "
             "and active candidates are always retained.")
    parser.add_argument(
        "--unlock-final-evaluation", action="store_true",
        help="Score the prospectively locked tail window. Run only after feature and "
             "model decisions are frozen; the result will be written to model metrics.")
    args = parser.parse_args()

    print(f"Connecting to DuckDB at {DB_PATH}...")
    con = duckdb.connect(str(DB_PATH))
    df = con.execute("SELECT * FROM fct_fraud_features").df()
    con.close()
    print(f"Total rows loaded: {len(df):,}")

    train_full_df = df[df['dataset_split'] == 'train'].copy()
    test_df = df[df['evaluation_role'] == 'development_holdout'].copy()
    locked_df = df[df['evaluation_role'] == 'locked_evaluation'].copy()

    base_df, blend_df, calib_df = chronological_split(train_full_df)
    print(f"Chronological split -> base: {len(base_df):,} | blend: {len(blend_df):,} "
          f"| calib: {len(calib_df):,} | development holdout: {len(test_df):,} "
          f"| prospectively locked: {len(locked_df):,}")

    y_base = base_df['is_fraud'].values
    y_blend = blend_df['is_fraud'].values
    y_calib = calib_df['is_fraud'].values
    y_te = test_df['is_fraud'].values
    print(f"Fraud counts -> base: {y_base.sum():,} | blend: {y_blend.sum():,} "
          f"| calib: {y_calib.sum():,} | test: {y_te.sum():,}")

    print("\nTraining base learners and stacked meta-model...")
    bundle = fit_stacked_ensemble(base_df, blend_df)
    rf_geo = bundle["model_2"]
    xgb = bundle["model_3"]
    rf_vel = bundle["model_4"]
    scaler = bundle["scaler_4"]
    meta_model = bundle["meta_model"]

    def base_probs(frame):
        return score_base_models(frame, bundle)

    # ------- THRESHOLD: selected on calib, which the meta-model has never seen -------
    print("Selecting decision threshold on the held-out calibration split...")
    X_meta_calib = np.column_stack(base_probs(calib_df))
    calib_meta_probs = meta_model.predict_proba(X_meta_calib)[:, 1]

    prec, rec, thresholds = precision_recall_curve(y_calib, calib_meta_probs)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = int(np.argmax(f1_scores[:-1])) if len(thresholds) else 0
    best_thresh = float(thresholds[best_idx]) if len(thresholds) else 0.5
    calib_f1 = float(f1_scores[best_idx])
    print(f"Optimal threshold (chosen on calib, F1={calib_f1:.4f}): {best_thresh:.4f}")

    # ---------------- ITERATIVE EVALUATION ON THE DEVELOPMENT HOLDOUT -----------
    print("\nEvaluating on the reused development-holdout split...")
    te_p2, te_p3, te_p4 = base_probs(test_df)
    meta_probs = meta_model.predict_proba(np.column_stack([te_p2, te_p3, te_p4]))[:, 1]

    for name, p in [("Model 2 (Geographic RF)", te_p2),
                    ("Model 3 (Category XGBoost)", te_p3),
                    ("Model 4 (Velocity RF)", te_p4),
                    ("Stacked Meta-Model", meta_probs)]:
        print(f"  {name:<28} ROC AUC={roc_auc_score(y_te, p):.4f}  "
              f"PR AUC={average_precision_score(y_te, p):.4f}")

    meta_preds = (meta_probs >= best_thresh).astype(int)
    print("\nStacked Meta-Model Classification Report (test):")
    print(classification_report(y_te, meta_preds, digits=4))
    cm = confusion_matrix(y_te, meta_preds)
    print("Confusion Matrix:")
    print(cm)

    te_prec, te_rec, te_f1, _ = precision_recall_fscore_support(
        y_te, meta_preds, average='binary', zero_division=0)

    # Alert-volume framing: what this threshold actually costs an investigations team.
    alerts = int(meta_preds.sum())
    days = max((test_df['trans_date_trans_time'].max()
                - test_df['trans_date_trans_time'].min()).days, 1)
    print(f"\nOperational load: {alerts:,} alerts over {days} days "
          f"({alerts / days:.1f}/day) at {te_prec:.1%} precision, "
          f"catching {int(cm[1][1]):,} of {int(y_te.sum()):,} frauds.")

    locked_metrics = None
    if args.unlock_final_evaluation:
        print("\nUnlocking the prospectively held evaluation window exactly once for this run...")
        locked_y = locked_df["is_fraud"].to_numpy()
        locked_p2, locked_p3, locked_p4 = base_probs(locked_df)
        locked_meta = meta_model.predict_proba(
            np.column_stack([locked_p2, locked_p3, locked_p4]))[:, 1]
        locked_pred = locked_meta >= best_thresh
        locked_prec, locked_rec, locked_f1, _ = precision_recall_fscore_support(
            locked_y, locked_pred, average="binary", zero_division=0)
        locked_days = max((locked_df["trans_date_trans_time"].max()
                           - locked_df["trans_date_trans_time"].min()).days, 1)
        locked_metrics = {
            "rows": len(locked_df),
            "positives": int(locked_y.sum()),
            "start": str(locked_df["trans_date_trans_time"].min()),
            "end": str(locked_df["trans_date_trans_time"].max()),
            "pr_auc": float(average_precision_score(locked_y, locked_meta)),
            "roc_auc": float(roc_auc_score(locked_y, locked_meta)),
            "precision": float(locked_prec),
            "recall": float(locked_rec),
            "f1": float(locked_f1),
            "alerts": int(locked_pred.sum()),
            "alerts_per_day": round(float(locked_pred.sum()) / locked_days, 2),
            "historical_caveat": (
                "Prospectively locked at Phase 1, but this public dataset had been "
                "evaluated in aggregate before the lock; not a pristine external test."
            ),
        }
        print(f"Locked window PR AUC={locked_metrics['pr_auc']:.4f}, "
              f"precision={locked_prec:.1%}, recall={locked_rec:.1%}")

    # ---------------- PERSIST ARTIFACTS ----------------
    version_str = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    versioned_dir = ARTIFACTS_DIR / version_str
    versioned_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving versioned model artifacts to {versioned_dir}...")
    joblib.dump(rf_geo, versioned_dir / "model_2_geo_rf.joblib")
    joblib.dump(xgb, versioned_dir / "model_3_cat_xgb.joblib")
    joblib.dump(rf_vel, versioned_dir / "model_4_vel_rf.joblib")
    joblib.dump(scaler, versioned_dir / "model_4_scaler.joblib")
    joblib.dump(meta_model, versioned_dir / "meta_model.joblib")
    (versioned_dir / "meta_threshold.txt").write_text(str(best_thresh))

    metrics_payload = {
        "trained_at": datetime.now().isoformat(),
        "version": version_str,
        "split_rows": {"base": len(base_df), "blend": len(blend_df),
                       "calib": len(calib_df), "development_holdout": len(test_df),
                       "locked_evaluation": len(locked_df)},
        "feature_semantics": "causal_prior_only_90d_v1",
        "evaluation_protocol": {
            "development_role": "development_holdout",
            "locked_role": "locked_evaluation",
            "locked_fraction": 0.25,
            "locked_scored": bool(args.unlock_final_evaluation),
            "historically_pristine": False,
        },
        "threshold_selected_on": "calib (held out from both base and blend fits)",
        "optimal_threshold": best_thresh,
        "calib_f1": calib_f1,
        "calib_auc_meta": float(roc_auc_score(y_calib, calib_meta_probs)),
        "test_auc_m2": float(roc_auc_score(y_te, te_p2)),
        "test_auc_m3": float(roc_auc_score(y_te, te_p3)),
        "test_auc_m4": float(roc_auc_score(y_te, te_p4)),
        "test_auc_meta": float(roc_auc_score(y_te, meta_probs)),
        "test_pr_auc_meta": float(average_precision_score(y_te, meta_probs)),
        "test_precision": float(te_prec),
        "test_recall": float(te_rec),
        "test_f1": float(te_f1),
        "test_alerts": alerts,
        "test_alerts_per_day": round(alerts / days, 2),
        "development_pr_auc_meta": float(average_precision_score(y_te, meta_probs)),
        "development_auc_m2": float(roc_auc_score(y_te, te_p2)),
        "development_auc_m3": float(roc_auc_score(y_te, te_p3)),
        "development_auc_m4": float(roc_auc_score(y_te, te_p4)),
        "development_auc_meta": float(roc_auc_score(y_te, meta_probs)),
        "development_precision": float(te_prec),
        "development_recall": float(te_rec),
        "development_f1": float(te_f1),
        "development_alerts": alerts,
        "development_alerts_per_day": round(alerts / days, 2),
        "locked_evaluation": locked_metrics,
        "confusion_matrix": cm.tolist(),
        "meta_coefficients": {
            "model_2_geo": float(meta_model.coef_[0][0]),
            "model_3_cat": float(meta_model.coef_[0][1]),
            "model_4_vel": float(meta_model.coef_[0][2]),
            "intercept": float(meta_model.intercept_[0]),
        },
    }
    (versioned_dir / "training_metrics.json").write_text(json.dumps(metrics_payload, indent=4))
    print(f"Metrics saved to {versioned_dir / 'training_metrics.json'}")
    registry = import_existing_champion(artifacts_dir=ARTIFACTS_DIR)
    registry = register_version(
        version_str,
        artifacts_dir=ARTIFACTS_DIR,
        bootstrap_champion=registry["champion"] is None,
    )
    if registry["champion"] == version_str:
        print(f"Bootstrapped {version_str} as the first champion.")
    else:
        print(
            f"Registered {version_str} as a challenger; serving remains on "
            f"{registry['champion']} until an authorized promotion."
        )

    protected = {
        version for version, record in registry["versions"].items()
        if record["status"] in {"champion", "candidate"}
    }
    removed = prune_old_versions(ARTIFACTS_DIR, max(args.keep, 0), protected)
    if removed:
        print(f"Pruned {len(removed)} superseded version(s): {', '.join(removed)}")
    kept = sorted(d.name for d in ARTIFACTS_DIR.glob("v_*") if d.is_dir())
    print(f"Retained: {', '.join(kept)}")


if __name__ == "__main__":
    main()
