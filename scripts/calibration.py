"""Calibration measurement for the stacked meta-model.

An earlier version of this project described the meta-score as "highly calibrated"
without ever measuring it. Ranking quality (AUC) and calibration are different
properties: a model can rank perfectly and still be systematically overconfident.

Calibration matters here specifically because the score is shown to an investigator
as a percentage. "97% risk" should mean that roughly 97 of every 100 transactions
scored that way are fraud. If it does not, the number is a ranking dressed up as a
probability.

Reported:
    Brier score  -- mean squared error of the probability. Lower is better; compare
                    against the base-rate-only baseline.
    ECE          -- expected calibration error, the average gap between predicted
                    confidence and observed frequency, weighted by bin population.
    MCE          -- the worst such gap in any populated bin.
    Reliability  -- the per-bin table those summaries come from.
"""
import argparse
import json
import os
import sys

import duckdb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import ARTIFACTS_DIR, DB_PATH, resolve_model_dir
from scripts.evaluation_utils import validate_binary_probabilities
from scripts.inference_engine import load_models, run_inference


def reliability(y, p, bins=10, strategy="quantile"):
    """Per-bin predicted vs observed frequency."""
    y, p = validate_binary_probabilities(y, p)
    if not isinstance(bins, int) or bins < 1 or strategy not in {"quantile", "uniform"}:
        raise ValueError("Use positive bins and either quantile or uniform strategy")
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    else:
        edges = np.linspace(0, 1, bins + 1)
    if len(edges) < 2:
        # A constant score still has calibration error. Returning an empty table
        # incorrectly labelled even a constant 99% fraud score perfectly calibrated.
        edges = np.array([0.0, 1.0])

    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append({
            "bin": b,
            "lower": float(edges[b]),
            "upper": float(edges[b + 1]),
            "count": n,
            "mean_predicted": float(p[mask].mean()),
            "observed_rate": float(y[mask].mean()),
            "gap": float(p[mask].mean() - y[mask].mean()),
        })
    return rows


def ece_mce(rows, total):
    if not rows or total < 1 or sum(r["count"] for r in rows) != total:
        raise ValueError("Reliability bins must cover a positive population")
    ece = sum(r["count"] / total * abs(r["gap"]) for r in rows)
    mce = max(abs(r["gap"]) for r in rows)
    return float(ece), float(mce)


def brier(y, p):
    y, p = validate_binary_probabilities(y, p)
    return float(np.mean((p - y) ** 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bins', type=int, default=10)
    parser.add_argument('--strategy', choices=['quantile', 'uniform'], default='quantile')
    parser.add_argument('--window', choices=['development_holdout', 'locked_evaluation'],
                        default='development_holdout')
    parser.add_argument('--unlock-final-evaluation', action='store_true')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    if args.window == 'locked_evaluation' and not args.unlock_final_evaluation:
        parser.error("locked_evaluation requires --unlock-final-evaluation")

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = con.execute(
            "SELECT * FROM fct_fraud_features WHERE evaluation_role = ?",
            [args.window]).df()
    finally:
        con.close()

    meta_p, _, _, _, _ = run_inference(df, *load_models(ARTIFACTS_DIR))
    y = df['is_fraud'].to_numpy()

    base_rate = float(y.mean())
    b_model = brier(y, meta_p)
    b_base = brier(y, np.full_like(meta_p, base_rate))
    rows = reliability(y, meta_p, args.bins, args.strategy)
    ece, mce = ece_mce(rows, len(y))

    print("=" * 76)
    print(f"META-MODEL CALIBRATION  (model {resolve_model_dir().name})")
    print("=" * 76)
    print(f"Transactions: {len(y):,}   base rate: {base_rate:.5f}\n")
    print(f"  Brier score           {b_model:.6f}")
    print(f"  Brier (base rate)     {b_base:.6f}")
    skill = f"{1 - b_model / b_base:+.2%}" if b_base else "undefined (single-class window)"
    print(f"  Skill vs base rate    {skill}")
    print(f"  ECE                   {ece:.5f}")
    print(f"  MCE                   {mce:.5f}")

    print(f"\nReliability ({args.strategy} bins) -- 'gap' is predicted minus observed;")
    print("positive means overconfident, negative means underconfident.\n")
    print(f"{'range':>26} {'count':>9} {'predicted':>11} {'observed':>11} {'gap':>10}")
    print("-" * 76)
    for r in rows:
        rng = f"[{r['lower']:.4f}, {r['upper']:.4f})"
        print(f"{rng:>26} {r['count']:>9,} {r['mean_predicted']:>11.4f} "
              f"{r['observed_rate']:>11.4f} {r['gap']:>+10.4f}")

    # The alerting region is the only part of the score range an investigator ever
    # sees, and aggregate ECE says almost nothing about it: 99.6% of transactions
    # score near the floor, so overall calibration is dominated by scores no human
    # will ever read. Report the alerting region separately.
    alert_rows = []
    thresh_path = resolve_model_dir() / "meta_threshold.txt"
    if thresh_path.exists():
        t = float(thresh_path.read_text().strip())
        above = meta_p >= t
        if above.sum():
            a_ece, a_mce = ece_mce(
                reliability(y[above], meta_p[above], 5, "uniform"), int(above.sum()))
            print(f"\n{'=' * 76}")
            print(f"ALERTING REGION (score >= {t:.4f}) -- what an investigator sees")
            print("=" * 76)
            print(f"  alerts                {int(above.sum()):,}")
            print(f"  mean predicted risk   {meta_p[above].mean():.4f}")
            print(f"  observed fraud rate   {y[above].mean():.4f}")
            print(f"  gap                   {meta_p[above].mean() - y[above].mean():+.4f}"
                  f"  ({'overconfident' if meta_p[above].mean() > y[above].mean() else 'underconfident'})")
            print(f"  ECE within region     {a_ece:.4f}")

            alert_rows = reliability(y[above], meta_p[above], 5, "uniform")
            print(f"\n{'range':>22} {'alerts':>9} {'predicted':>11} {'observed':>11} {'gap':>10}")
            print("-" * 76)
            for r in alert_rows:
                rng = f"[{r['lower']:.3f}, {r['upper']:.3f})"
                print(f"{rng:>22} {r['count']:>9,} {r['mean_predicted']:>11.4f} "
                      f"{r['observed_rate']:>11.4f} {r['gap']:>+10.4f}")
            print("\n  A displayed '93% risk' that is fraud 86% of the time is a ranking")
            print("  presented as a probability. Aggregate ECE hides this completely.")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"model_version": resolve_model_dir().name,
                       "evaluation_role": args.window,
                       "historically_pristine": False,
                       "base_rate": base_rate, "brier": b_model,
                       "brier_base_rate": b_base, "ece": ece, "mce": mce,
                       "reliability": rows,
                       "alerting_region_reliability": alert_rows}, f, indent=2, allow_nan=False)
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
