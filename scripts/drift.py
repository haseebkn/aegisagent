"""Feature drift monitoring between the training window and a scoring window.

This is the control that would have caught the velocity bug in production. When
`txns_24h` averaged 4.88 in training and 0.014 at scoring time, PSI on that feature
would have been enormous and the pipeline would have refused to score. Null checks
and range checks both passed, because zero is neither null nor out of range.

Two complementary statistics per feature:

  PSI (Population Stability Index) -- distributional shift over fixed bins.
      < 0.10  stable
      < 0.25  moderate shift, investigate
      >= 0.25 significant shift, treat the model as unreliable

  KS (Kolmogorov-Smirnov) -- maximum CDF gap, sensitive to shape changes PSI can
      smear across bins.

Bin edges come from the training distribution and are reused for the scoring
window; recomputing them per-window would hide exactly the shift being looked for.
"""
import argparse
import json
import os
import sys

import duckdb
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import DB_PATH, FEAT_M2, FEAT_M3, FEAT_M4

MONITORED = sorted(set(FEAT_M2) | set(FEAT_M3) | set(FEAT_M4))

PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25


def psi(expected, actual, bins=10, epsilon=1e-6, discrete_max_levels=20):
    """PSI between two samples, using bins fixed on the expected distribution.

    Low-cardinality features (binary flags, small integer codes) are compared over
    their actual levels. Forcing them through quantile bins collapses the edges and
    produces spurious infinities -- `night` and `is_online` are identical across
    splits here, and a quantile-only implementation reported both as significant.
    """
    levels = np.unique(expected)
    if len(levels) <= discrete_max_levels:
        all_levels = np.union1d(levels, np.unique(actual))
        e_pct = np.array([(expected == v).mean() for v in all_levels])
        a_pct = np.array([(actual == v).mean() for v in all_levels])
        e_pct = np.clip(e_pct, epsilon, None)
        a_pct = np.clip(a_pct, epsilon, None)
        return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0

    edges[0], edges[-1] = -np.inf, np.inf
    e_pct = np.histogram(expected, bins=edges)[0] / len(expected)
    a_pct = np.histogram(actual, bins=edges)[0] / len(actual)
    e_pct = np.clip(e_pct, epsilon, None)
    a_pct = np.clip(a_pct, epsilon, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def classify(value):
    if value >= PSI_SIGNIFICANT:
        return "SIGNIFICANT"
    if value >= PSI_MODERATE:
        return "MODERATE"
    return "stable"


def compare(reference_df, current_df, features=None):
    features = features or MONITORED
    results = []
    for f in features:
        if f not in reference_df.columns or f not in current_df.columns:
            continue
        ref = reference_df[f].to_numpy(dtype=float)
        cur = current_df[f].to_numpy(dtype=float)
        ref = ref[np.isfinite(ref)]
        cur = cur[np.isfinite(cur)]
        if len(ref) == 0 or len(cur) == 0:
            continue

        p = psi(ref, cur)
        ks_stat, ks_p = stats.ks_2samp(
            ref if len(ref) <= 50000 else np.random.default_rng(0).choice(ref, 50000, replace=False),
            cur if len(cur) <= 50000 else np.random.default_rng(0).choice(cur, 50000, replace=False))
        results.append({
            "feature": f,
            "psi": p,
            "status": classify(p),
            "ks_stat": float(ks_stat),
            "ks_pvalue": float(ks_p),
            "ref_mean": float(ref.mean()),
            "cur_mean": float(cur.mean()),
            "mean_ratio": float(cur.mean() / ref.mean()) if ref.mean() else float('inf'),
        })
    results.sort(key=lambda r: -r['psi'])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', default='train',
                        help="dataset_split used as the training reference.")
    parser.add_argument('--current', default='test',
                        help="dataset_split treated as the scoring window.")
    parser.add_argument('--fail-on-significant', action='store_true',
                        help="Exit non-zero if any feature shows significant drift.")
    parser.add_argument('--output', default=None, help="Write JSON report here.")
    args = parser.parse_args()

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        cols = ", ".join(MONITORED)
        ref = con.execute(
            f"SELECT {cols} FROM fct_fraud_features WHERE dataset_split = '{args.reference}'").df()
        cur = con.execute(
            f"SELECT {cols} FROM fct_fraud_features WHERE dataset_split = '{args.current}'").df()
    finally:
        con.close()

    results = compare(ref, cur)

    print("=" * 78)
    print(f"FEATURE DRIFT: reference='{args.reference}' ({len(ref):,} rows)  "
          f"vs current='{args.current}' ({len(cur):,} rows)")
    print("=" * 78)
    print(f"{'feature':<28} {'PSI':>8} {'KS':>7} {'ref mean':>12} {'cur mean':>12}  status")
    print("-" * 78)
    for r in results:
        print(f"{r['feature']:<28} {r['psi']:>8.4f} {r['ks_stat']:>7.3f} "
              f"{r['ref_mean']:>12.3f} {r['cur_mean']:>12.3f}  {r['status']}")

    significant = [r for r in results if r['status'] == "SIGNIFICANT"]
    moderate = [r for r in results if r['status'] == "MODERATE"]
    print("-" * 78)
    print(f"{len(significant)} significant, {len(moderate)} moderate, "
          f"{len(results) - len(significant) - len(moderate)} stable "
          f"(of {len(results)} monitored)")

    if significant:
        print("\nSignificant drift means the scoring distribution no longer matches what")
        print("the model was fitted on. Investigate the feature pipeline before trusting")
        print("any score produced in this window:")
        for r in significant:
            print(f"  - {r['feature']}: mean {r['ref_mean']:.3f} -> {r['cur_mean']:.3f} "
                  f"({r['mean_ratio']:.2f}x), PSI {r['psi']:.3f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"reference": args.reference, "current": args.current,
                       "results": results}, f, indent=2)
        print(f"\nReport written to {args.output}")

    if args.fail_on_significant and significant:
        sys.exit(1)


if __name__ == "__main__":
    main()
