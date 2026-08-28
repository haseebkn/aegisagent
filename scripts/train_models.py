"""Train the stacked fraud ensemble.

Model-fitting split design (chronological, with no random row shuffling):

    fraudTrain.csv   -> 70% base    : fit Models 2/3/4
                     -> 15% blend   : fit the logistic meta-learner on base-model
                                      probabilities the base models have not seen
                     -> 15% calib   : select the demo decision threshold
    fraudTest.csv    -> later development holdout used for iterative evaluation

Important: this is not end-to-end temporal isolation. Upstream random-fold target
encodings and full-window card statistics expose later training-period information
to earlier rows, and fraudTest.csv has informed feature decisions. The resulting
metrics are development evidence, not final blind-test or pre-deployment estimates.

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import ARTIFACTS_DIR, DB_PATH, FEAT_M2, FEAT_M3, FEAT_M4


def chronological_split(df, fractions=(0.70, 0.15, 0.15)):
    df = df.sort_values('trans_date_trans_time').reset_index(drop=True)
    n = len(df)
    c1 = int(n * fractions[0])
    c2 = int(n * (fractions[0] + fractions[1]))
    return df.iloc[:c1].copy(), df.iloc[c1:c2].copy(), df.iloc[c2:].copy()


def prune_old_versions(artifacts_dir, keep):
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
    versions = sorted((d for d in artifacts_dir.glob("v_*") if d.is_dir()),
                      key=lambda d: d.name, reverse=True)
    stale = versions[keep:]
    if not stale:
        return []
    for d in stale:
        shutil.rmtree(d)
    return [d.name for d in stale]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep", type=int, default=1,
        help="Version directories to retain after training (default 1: the one just "
             "trained). Raise it if you want previous versions available to roll back to.")
    args = parser.parse_args()

    print(f"Connecting to DuckDB at {DB_PATH}...")
    con = duckdb.connect(str(DB_PATH))
    df = con.execute("SELECT * FROM fct_fraud_features").df()
    con.close()
    print(f"Total rows loaded: {len(df):,}")

    train_full_df = df[df['dataset_split'] == 'train'].copy()
    test_df = df[df['dataset_split'] == 'test'].copy()

    base_df, blend_df, calib_df = chronological_split(train_full_df)
    print(f"Chronological split -> base: {len(base_df):,} | blend: {len(blend_df):,} "
          f"| calib: {len(calib_df):,} | development holdout: {len(test_df):,}")

    y_base = base_df['is_fraud'].values
    y_blend = blend_df['is_fraud'].values
    y_calib = calib_df['is_fraud'].values
    y_te = test_df['is_fraud'].values
    print(f"Fraud counts -> base: {y_base.sum():,} | blend: {y_blend.sum():,} "
          f"| calib: {y_calib.sum():,} | test: {y_te.sum():,}")

    # ---------------- MODEL 2: Random Forest (Geographic Focus) ----------------
    print("\nTraining Model 2 (Geographic RF)...")
    rf_geo = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        class_weight='balanced', n_jobs=-1, random_state=42)
    rf_geo.fit(base_df[FEAT_M2].values, y_base)

    # ---------------- MODEL 3: XGBoost (Category Risk Focus) ----------------
    print("Training Model 3 (Category XGBoost)...")
    ratio = (len(y_base) - y_base.sum()) / (y_base.sum() + 1e-6)
    xgb = XGBClassifier(
        n_estimators=1400, learning_rate=0.03, max_depth=7, min_child_weight=2,
        subsample=0.9, colsample_bytree=0.85, gamma=0.3, reg_alpha=1.0,
        reg_lambda=2.0, scale_pos_weight=ratio * 1.5, max_delta_step=2,
        eval_metric='aucpr', n_jobs=-1, random_state=42)
    xgb.fit(base_df[FEAT_M3].values, y_base)

    # ---------------- MODEL 4: Random Forest (Velocity Focus) ----------------
    print("Training Model 4 (Velocity RF with Scaling)...")
    scaler = StandardScaler()
    X4_base_sc = scaler.fit_transform(base_df[FEAT_M4].values)
    rf_vel = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, class_weight='balanced_subsample',
        n_jobs=-1, random_state=42)
    rf_vel.fit(X4_base_sc, y_base)

    def base_probs(frame):
        p2 = rf_geo.predict_proba(frame[FEAT_M2].values)[:, 1]
        p3 = xgb.predict_proba(frame[FEAT_M3].values)[:, 1]
        p4 = rf_vel.predict_proba(scaler.transform(frame[FEAT_M4].values))[:, 1]
        return p2, p3, p4

    # ---------------- META-MODEL: fitted on the blend split ----------------
    print("\nFitting stacked meta-model on the blend split...")
    X_meta_blend = np.column_stack(base_probs(blend_df))
    meta_model = LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1)
    meta_model.fit(X_meta_blend, y_blend)

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
                       "calib": len(calib_df), "test": len(test_df)},
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
        "confusion_matrix": cm.tolist(),
        "meta_coefficients": {
            "model_2_geo": float(meta_model.coef_[0][0]),
            "model_3_cat": float(meta_model.coef_[0][1]),
            "model_4_vel": float(meta_model.coef_[0][2]),
            "intercept": float(meta_model.intercept_[0]),
        },
    }
    (versioned_dir / "training_metrics.json").write_text(json.dumps(metrics_payload, indent=4))
    (ARTIFACTS_DIR / "latest_version.txt").write_text(version_str)

    print(f"Metrics saved to {versioned_dir / 'training_metrics.json'}")
    print(f"latest_version.txt now points at {version_str}")

    removed = prune_old_versions(ARTIFACTS_DIR, max(args.keep, 1))
    if removed:
        print(f"Pruned {len(removed)} superseded version(s): {', '.join(removed)}")
    kept = sorted(d.name for d in ARTIFACTS_DIR.glob("v_*") if d.is_dir())
    print(f"Retained: {', '.join(kept)}")


if __name__ == "__main__":
    main()
