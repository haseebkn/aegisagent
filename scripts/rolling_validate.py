"""Expanding-window validation for the complete stacked ensemble."""

import argparse
import json
import os
import sys

import duckdb
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import DB_PATH
from scripts.evaluation_utils import rolling_origin_splits
from scripts.modeling import fit_stacked_ensemble, score_ensemble


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--fast", action="store_true",
                        help="Use fewer trees while preserving model structure.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        frame = con.execute("""
            SELECT * FROM fct_fraud_features
            WHERE dataset_split = 'train'
            ORDER BY trans_date_trans_time, trans_num
        """).df()
    finally:
        con.close()

    results = []
    for fold, (train_idx, validation_idx) in enumerate(
            rolling_origin_splits(len(frame), args.folds), start=1):
        fit_window = frame.iloc[train_idx]
        cut = int(len(fit_window) * 0.80)
        base = fit_window.iloc[:cut]
        blend = fit_window.iloc[cut:]
        validation = frame.iloc[validation_idx]
        bundle = fit_stacked_ensemble(base, blend, fast=args.fast)
        probabilities, _, _, _ = score_ensemble(validation, bundle)
        y = validation["is_fraud"].to_numpy()
        row = {
            "fold": fold,
            "train_end": str(fit_window["trans_date_trans_time"].max()),
            "validation_start": str(validation["trans_date_trans_time"].min()),
            "validation_end": str(validation["trans_date_trans_time"].max()),
            "train_rows": len(fit_window),
            "validation_rows": len(validation),
            "validation_positives": int(y.sum()),
            "pr_auc": float(average_precision_score(y, probabilities)),
            "roc_auc": float(roc_auc_score(y, probabilities)),
        }
        results.append(row)
        print(f"Fold {fold}: PR AUC={row['pr_auc']:.4f} ROC AUC={row['roc_auc']:.4f} "
              f"({row['validation_start']} -> {row['validation_end']})")

    summary = {
        "folds": results,
        "mean_pr_auc": sum(r["pr_auc"] for r in results) / len(results),
        "min_pr_auc": min(r["pr_auc"] for r in results),
        "fast_mode": args.fast,
    }
    print(f"Mean PR AUC={summary['mean_pr_auc']:.4f}; "
          f"worst fold={summary['min_pr_auc']:.4f}")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
