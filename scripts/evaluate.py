"""Decision-focused evaluation for the causal Phase 1 fraud pipeline.

Reports model comparisons, day-block bootstrap intervals, alert capacity and cost,
calibration, time/subgroup slices, and batch scoring latency. The prospectively
locked tail window requires an explicit unlock flag and is still disclosed as
historically exposed at the aggregate dataset level.
"""

import argparse
import json
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import ARTIFACTS_DIR, DB_PATH, resolve_model_dir
from scripts.evaluation_utils import (calibration_summary,
                                      day_block_bootstrap_pr_auc,
                                      subgroup_metrics)
from scripts.inference_engine import load_models, run_inference


DEFAULT_INVESTIGATION_COST = 25.0
BASELINE_FEATURES = [
    "log_amt", "distance_km", "night", "is_online", "txns_24h", "txns_7d"
]


def load_windows(role):
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        train = con.execute("""
            SELECT * FROM fct_fraud_features
            WHERE dataset_split = 'train'
            ORDER BY trans_date_trans_time, trans_num
        """).df()
        evaluation = con.execute("""
            SELECT f.*, t.gender, t.category AS raw_category, t.state AS raw_state
            FROM fct_fraud_features f
            LEFT JOIN stg_transactions_test t USING (trans_num)
            WHERE f.evaluation_role = ?
            ORDER BY f.trans_date_trans_time, f.trans_num
        """, [role]).df()
    finally:
        con.close()
    if evaluation.empty:
        raise ValueError(f"No rows found for evaluation_role={role!r}; rebuild dbt models.")
    return train, evaluation


def score_models(frame):
    meta, p2, p3, p4, _ = run_inference(frame, *load_models(ARTIFACTS_DIR))
    return {"model_2": p2, "model_3": p3, "model_4": p4, "ensemble": meta}


def score_simple_baseline(train, evaluation):
    base = train.iloc[:int(len(train) * 0.70)]
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    )
    pipeline.fit(base[BASELINE_FEATURES], base["is_fraud"])
    return pipeline.predict_proba(evaluation[BASELINE_FEATURES])[:, 1]


def cost_at(y, amounts, probabilities, threshold, investigation_cost):
    pred = probabilities >= threshold
    tp = pred & (y == 1)
    fp = pred & (y == 0)
    fn = (~pred) & (y == 1)
    fraud_loss = float(amounts[fn].sum())
    review_cost = float(pred.sum() * investigation_cost)
    return {
        "threshold": float(threshold),
        "alerts": int(pred.sum()),
        "tp": int(tp.sum()),
        "fp": int(fp.sum()),
        "fn": int(fn.sum()),
        "precision": float(tp.sum() / pred.sum()) if pred.sum() else 0.0,
        "recall": float(tp.sum() / (y == 1).sum()),
        "fraud_loss": fraud_loss,
        "review_cost": review_cost,
        "total_cost": fraud_loss + review_cost,
        "amount_recovered": float(amounts[tp].sum()),
    }


def benchmark_latency(frame, models, runs):
    sample = frame.iloc[:min(len(frame), 5000)]
    durations = []
    for _ in range(runs):
        start = time.perf_counter()
        run_inference(sample, *models)
        durations.append((time.perf_counter() - start) * 1000.0)
    per_row = np.asarray(durations) / len(sample)
    return {
        "sample_rows": len(sample),
        "runs": runs,
        "batch_ms_p50": float(np.median(durations)),
        "batch_ms_p95": float(np.quantile(durations, 0.95)),
        "ms_per_row_p50": float(np.median(per_row)),
        "environment": "local batch benchmark; not an online-service SLA",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", choices=["development_holdout", "locked_evaluation"],
                        default="development_holdout")
    parser.add_argument("--unlock-final-evaluation", action="store_true")
    parser.add_argument("--investigation-cost", type=float,
                        default=DEFAULT_INVESTIGATION_COST)
    parser.add_argument("--alert-budget", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--latency-runs", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.window == "locked_evaluation" and not args.unlock_final_evaluation:
        parser.error("locked_evaluation requires --unlock-final-evaluation")

    train, frame = load_windows(args.window)
    probabilities = score_models(frame)
    probabilities["simple_logistic"] = score_simple_baseline(train, frame)
    y = frame["is_fraud"].to_numpy()
    amounts = frame["amt"].to_numpy()
    timestamps = pd.to_datetime(frame["trans_date_trans_time"])
    span_days = max((timestamps.max() - timestamps.min()).days, 1)
    threshold = float((resolve_model_dir() / "meta_threshold.txt").read_text().strip())

    print("=" * 84)
    print(f"EVALUATION WINDOW: {args.window}  ({len(frame):,} rows; {int(y.sum()):,} frauds)")
    if args.window == "locked_evaluation":
        print("CAVEAT: prospectively locked in Phase 1, but historically exposed in aggregate.")
    print("=" * 84)

    # Calibration is reported per model, not only for the ensemble. Ranking quality
    # and calibration disagree here: the strongest single base learner edges the
    # ensemble on PR AUC while the ensemble is markedly better calibrated. Reporting
    # only the ensemble, and only against the weak logistic baseline, would hide a
    # comparison the ensemble does not win outright.
    ranking = {}
    for name, values in probabilities.items():
        interval = day_block_bootstrap_pr_auc(
            y, values, timestamps.to_numpy(), args.bootstrap_samples)
        model_calibration = calibration_summary(y, values)
        ranking[name] = {
            "pr_auc": float(average_precision_score(y, values)),
            "pr_auc_ci_95_day_block": list(interval),
            "roc_auc": float(roc_auc_score(y, values)),
            "brier": model_calibration["brier"],
            "ece": model_calibration["ece"],
            "mce": model_calibration["mce"],
        }
        print(f"{name:<20} PR AUC={ranking[name]['pr_auc']:.4f} "
              f"95% CI [{interval[0]:.4f}, {interval[1]:.4f}]  "
              f"ROC AUC={ranking[name]['roc_auc']:.4f}  "
              f"Brier={ranking[name]['brier']:.6f}  ECE={ranking[name]['ece']:.5f}")

    best_pr = max(ranking, key=lambda k: ranking[k]["pr_auc"])
    if best_pr != "ensemble":
        gap = ranking[best_pr]["pr_auc"] - ranking["ensemble"]["pr_auc"]
        print(f"\nNOTE: {best_pr} leads the ensemble on PR AUC by {gap:.4f}. "
              f"The ensemble is retained for calibration "
              f"(ECE {ranking['ensemble']['ece']:.5f} vs {ranking[best_pr]['ece']:.5f}), "
              f"not for ranking. See MODEL_CARD.md.")

    ensemble = probabilities["ensemble"]
    current = cost_at(y, amounts, ensemble, threshold, args.investigation_cost)
    candidate_thresholds = np.unique(np.quantile(ensemble, np.linspace(0.90, 0.9999, 400)))
    candidates = [cost_at(y, amounts, ensemble, value, args.investigation_cost)
                  for value in candidate_thresholds]
    cost_minimising = min(candidates, key=lambda row: row["total_cost"])
    target_alerts = args.alert_budget * span_days
    budget = min(candidates, key=lambda row: abs(row["alerts"] - target_alerts))

    print("\nOPERATING POINTS")
    for label, row in [("current", current), ("cost-minimising", cost_minimising),
                       (f"capacity {args.alert_budget}/day", budget)]:
        print(f"{label:<22} threshold={row['threshold']:.4f} "
              f"alerts/day={row['alerts']/span_days:.1f} precision={row['precision']:.1%} "
              f"recall={row['recall']:.1%} total_cost=${row['total_cost']:,.0f}")

    calibration = calibration_summary(y, ensemble)
    alert_mask = ensemble >= threshold
    alert_calibration = (calibration_summary(y[alert_mask], ensemble[alert_mask], bins=5)
                         if alert_mask.any() else None)
    print(f"\nCALIBRATION: Brier={calibration['brier']:.6f} "
          f"ECE={calibration['ece']:.4f} MCE={calibration['mce']:.4f}")
    if alert_mask.any():
        print(f"Alert region: predicted={ensemble[alert_mask].mean():.3f}, "
              f"observed={y[alert_mask].mean():.3f}, "
              f"gap={ensemble[alert_mask].mean()-y[alert_mask].mean():+.3f}")

    frame = frame.copy()
    frame["amount_band"] = pd.qcut(frame["amt"], q=4, duplicates="drop").astype(str)
    frame["calendar_month"] = timestamps.dt.to_period("M").astype(str)
    subgroup = {
        column: subgroup_metrics(frame, y, ensemble, threshold, column)
        for column in ("gender", "raw_category", "amount_band", "calendar_month")
    }
    print("\nSUPPORTED SLICES (descriptive, not a fairness certification)")
    for column, rows in subgroup.items():
        recalls = [row["recall"] for row in rows]
        if recalls:
            print(f"{column:<18} groups={len(rows):>2} recall range "
                  f"{min(recalls):.1%}..{max(recalls):.1%}")

    models = load_models(ARTIFACTS_DIR)
    latency = benchmark_latency(frame, models, args.latency_runs)
    print(f"\nLOCAL BATCH LATENCY: p50={latency['batch_ms_p50']:.1f} ms for "
          f"{latency['sample_rows']:,} rows ({latency['ms_per_row_p50']:.4f} ms/row)")

    payload = {
        "model_version": resolve_model_dir().name,
        "evaluation_role": args.window,
        "historically_pristine": False,
        "rows": len(frame),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "span_days": span_days,
        "ranking": ranking,
        "operating_points": {
            "current": current,
            "cost_minimising": cost_minimising,
            "alert_capacity": budget,
        },
        "calibration": calibration,
        "alert_region_calibration": alert_calibration,
        "subgroups": subgroup,
        "latency": latency,
    }
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
