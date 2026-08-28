"""Threshold and cost analysis for the stacked ensemble.

A fraud model should not be operated at "the F1-optimal threshold" by default. Its
operating point must reflect what an investigations team can actually staff, given what a missed fraud
costs and what an investigator hour costs. This script reports that trade-off
explicitly, following the cost-sensitive framing in Bahnsen et al. (2016).

Cost model per transaction:
    false negative -> the fraud amount is lost
    false positive -> a fixed investigation/customer-friction cost
    true positive  -> investigation cost paid, but the amount is recovered
    true negative  -> nothing

Headline metric is PR AUC, not ROC AUC. At 0.39% prevalence a ROC AUC near 0.99
says very little: the negative class is so large that a model can rank well and
still bury the queue in false positives.
"""
import argparse
import json
import os
import sys

import duckdb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import ARTIFACTS_DIR, DB_PATH, resolve_model_dir
from scripts.inference_engine import load_models, run_inference

DEFAULT_INVESTIGATION_COST = 25.0   # analyst time + customer friction per alert


def score_test_split():
    con = duckdb.connect(str(DB_PATH))
    try:
        df = con.execute(
            "SELECT * FROM fct_fraud_features WHERE dataset_split = 'test'").df()
    finally:
        con.close()

    models = load_models(ARTIFACTS_DIR)
    meta_p, p2, p3, p4, _ = run_inference(df, *models)
    return df, meta_p, p2, p3, p4, models[-1]


def cost_at(y, amounts, probs, threshold, investigation_cost):
    pred = probs >= threshold
    tp = pred & (y == 1)
    fp = pred & (y == 0)
    fn = (~pred) & (y == 1)

    fraud_loss = float(amounts[fn].sum())
    review_cost = float((tp.sum() + fp.sum()) * investigation_cost)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--investigation-cost', type=float,
                        default=DEFAULT_INVESTIGATION_COST,
                        help='Cost of reviewing one alert (analyst time + friction).')
    parser.add_argument('--alert-budget', type=int, default=None,
                        help='Max alerts per day the team can work; reports the '
                             'threshold that fits it.')
    parser.add_argument('--output', type=str, default=None,
                        help='Write the report as JSON to this path.')
    args = parser.parse_args()

    df, meta_p, p2, p3, p4, deployed_threshold = score_test_split()
    y = df['is_fraud'].to_numpy()
    amounts = df['amt'].to_numpy()
    span_days = max((df['trans_date_trans_time'].max()
                     - df['trans_date_trans_time'].min()).days, 1)

    print("=" * 74)
    print("RANKING QUALITY (reused development holdout)")
    print("=" * 74)
    print(f"Transactions: {len(y):,}   Frauds: {int(y.sum()):,} "
          f"({y.sum()/len(y):.3%} prevalence)   Span: {span_days} days")
    print(f"Total fraud exposure: ${amounts[y == 1].sum():,.2f}\n")

    for name, p in [("Model 2 (geographic RF)", p2), ("Model 3 (category XGB)", p3),
                    ("Model 4 (velocity RF)", p4), ("Stacked meta-model", meta_p)]:
        print(f"  {name:<26} PR AUC={average_precision_score(y, p):.4f}   "
              f"ROC AUC={roc_auc_score(y, p):.4f}")
    print("\n  PR AUC is the headline. At this prevalence ROC AUC flatters everything.")

    # ---- Operating points across the threshold range ----
    print("\n" + "=" * 74)
    print(f"OPERATING POINTS  (investigation cost ${args.investigation_cost:.2f}/alert)")
    print("=" * 74)
    print(f"{'thresh':>7} {'alerts':>8} {'/day':>7} {'prec':>7} {'recall':>7} "
          f"{'missed $':>13} {'review $':>11} {'total $':>13}")

    grid = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    rows = [cost_at(y, amounts, meta_p, t, args.investigation_cost) for t in grid]
    for r in rows:
        print(f"{r['threshold']:>7.2f} {r['alerts']:>8,} {r['alerts']/span_days:>7.1f} "
              f"{r['precision']:>7.1%} {r['recall']:>7.1%} "
              f"{r['fraud_loss']:>13,.0f} {r['review_cost']:>11,.0f} "
              f"{r['total_cost']:>13,.0f}")

    # ---- Cost-minimising threshold, searched finely ----
    fine = np.unique(np.quantile(meta_p, np.linspace(0.90, 0.9999, 400)))
    fine_rows = [cost_at(y, amounts, meta_p, t, args.investigation_cost) for t in fine]
    best = min(fine_rows, key=lambda r: r['total_cost'])
    deployed = cost_at(y, amounts, meta_p, deployed_threshold, args.investigation_cost)

    print("\n" + "=" * 74)
    print("COST-MINIMISING vs CURRENT DEMO THRESHOLD")
    print("=" * 74)
    for label, r in [("Current demo (F1-optimal on calib)", deployed),
                     ("Cost-minimising", best)]:
        print(f"\n{label}: threshold={r['threshold']:.4f}")
        print(f"  alerts={r['alerts']:,} ({r['alerts']/span_days:.1f}/day)  "
              f"precision={r['precision']:.1%}  recall={r['recall']:.1%}")
        print(f"  fraud lost=${r['fraud_loss']:,.0f}  review=${r['review_cost']:,.0f}  "
              f"TOTAL=${r['total_cost']:,.0f}")

    delta = deployed['total_cost'] - best['total_cost']
    if abs(delta) > 1:
        direction = "saves" if delta > 0 else "costs"
        print(f"\n  Moving to the cost-minimising threshold {direction} "
              f"${abs(delta):,.0f} over {span_days} days, and changes the queue by "
              f"{best['alerts'] - deployed['alerts']:+,} alerts "
              f"({(best['alerts'] - deployed['alerts'])/span_days:+.1f}/day).")

    # ---- Threshold implied by an alert budget ----
    budget_row = None
    if args.alert_budget:
        target = args.alert_budget * span_days
        budget_row = min(fine_rows, key=lambda r: abs(r['alerts'] - target))
        print("\n" + "=" * 74)
        print(f"ALERT BUDGET: {args.alert_budget}/day ({target:,} over {span_days} days)")
        print("=" * 74)
        print(f"  threshold={budget_row['threshold']:.4f}  "
              f"alerts={budget_row['alerts']:,} "
              f"({budget_row['alerts']/span_days:.1f}/day)")
        print(f"  precision={budget_row['precision']:.1%}  "
              f"recall={budget_row['recall']:.1%}  "
              f"frauds caught={budget_row['tp']:,}/{int(y.sum()):,}")
        print(f"  fraud lost=${budget_row['fraud_loss']:,.0f}")

    if args.output:
        payload = {
            "model_version": resolve_model_dir().name,
            "span_days": span_days,
            "prevalence": float(y.mean()),
            "total_fraud_exposure": float(amounts[y == 1].sum()),
            "investigation_cost": args.investigation_cost,
            "pr_auc": {"m2": float(average_precision_score(y, p2)),
                       "m3": float(average_precision_score(y, p3)),
                       "m4": float(average_precision_score(y, p4)),
                       "meta": float(average_precision_score(y, meta_p))},
            "roc_auc": {"meta": float(roc_auc_score(y, meta_p))},
            "operating_points": rows,
            "deployed": deployed,
            "cost_minimising": best,
            "alert_budget": budget_row,
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
