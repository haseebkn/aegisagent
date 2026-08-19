"""Measure whether the card <-> merchant graph features carry usable signal.

Adding a graph layer is only worthwhile if the graph has structure. This script
quantifies that instead of assuming it, by reporting univariate discriminative power
and the density of the bipartite graph itself.
"""
import os
import sys

import duckdb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import DB_PATH

GRAPH_FEATURES = [
    'merchant_card_degree', 'card_merchant_degree', 'merchant_fraud_card_cnt',
    'merchant_fraud_card_ratio', 'card_2hop_fraud_cards',
]
REFERENCE_FEATURES = ['amt', 'distance_km', 'merchant_risk', 'category_risk', 'txns_24h']


def main():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = con.execute(
            f"SELECT is_fraud, {', '.join(GRAPH_FEATURES + REFERENCE_FEATURES)} "
            "FROM fct_fraud_features WHERE dataset_split = 'test'").df()
        density = con.execute("""
            SELECT COUNT(DISTINCT cc_num) AS cards,
                   COUNT(DISTINCT merchant) AS merchants,
                   COUNT(DISTINCT (cc_num, merchant)) AS edges
            FROM stg_transactions_train""").fetchone()
    finally:
        con.close()

    cards, merchants, edges = density
    print("=" * 70)
    print("BIPARTITE GRAPH STRUCTURE (train split)")
    print("=" * 70)
    print(f"  cards={cards:,}  merchants={merchants:,}  edges={edges:,}")
    print(f"  density = {edges / (cards * merchants):.3f} "
          f"(1.000 means every card touched every merchant)")
    print("  A near-complete bipartite graph has no community structure to exploit:")
    print("  every entity is a neighbour of every other entity.\n")

    y = df['is_fraud'].to_numpy()
    print("=" * 70)
    print("UNIVARIATE DISCRIMINATIVE POWER (held-out test split)")
    print("=" * 70)
    print(f"{'feature':<30} {'ROC AUC':>9} {'PR AUC':>9}   verdict")
    print("-" * 70)

    baseline_pr = y.mean()
    for group, feats in (("graph", GRAPH_FEATURES), ("reference", REFERENCE_FEATURES)):
        for f in feats:
            x = df[f].to_numpy(dtype=float)
            if np.allclose(x, x[0]):
                print(f"{f:<30} {'constant':>9} {'-':>9}   no signal (constant)")
                continue
            auc = roc_auc_score(y, x)
            pr = average_precision_score(y, x)
            lift = pr / baseline_pr
            verdict = ("no signal" if abs(auc - 0.5) < 0.02
                       else "weak" if abs(auc - 0.5) < 0.10
                       else "useful")
            print(f"{f:<30} {auc:>9.4f} {pr:>9.4f}   {verdict} ({lift:.1f}x base rate)")
        if group == "graph":
            print("-" * 70)

    print(f"\nBase rate (PR AUC of a random ranker): {baseline_pr:.4f}")


if __name__ == "__main__":
    main()
