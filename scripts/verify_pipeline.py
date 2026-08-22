"""End-to-end pipeline verification.

Every stage runs against real artifacts and real data. In particular, stage 4 no
longer hands the STR agent hardcoded ensemble scores: it pulls the highest-risk
transaction the live models actually produce and reports on that, so the
DB -> dbt features -> ensemble -> Bedrock -> compliance-log path is exercised
end to end.
"""
import os
import re
import subprocess
import sys

import duckdb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import ARTIFACTS_DIR, DB_PATH, PROJECT_ROOT
from scripts.inference_engine import load_models, run_inference
from scripts.sar_agent import generate_sar_narrative, save_sar_report

EXPECTED_ROWS = 1852394


def run_dbt_tests():
    print("--- 1. RUNNING DBT TESTS ---")
    result = subprocess.run(
        ["dbt", "test", "--profiles-dir", str(PROJECT_ROOT)],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print("dbt test failed!")
        print(result.stdout)
        print(result.stderr)
        return False
    print("All dbt tests passed successfully (return code 0).")
    return True


def verify_duckdb_schema():
    print("\n--- 2. VERIFYING DUCKDB DATA & SCHEMA ---")
    con = duckdb.connect(str(DB_PATH))
    try:
        row_count = con.execute("SELECT COUNT(*) FROM fct_fraud_features").fetchone()[0]
        print(f"Total row count in fct_fraud_features: {row_count:,}")
        if row_count != EXPECTED_ROWS:
            print(f"ERROR: Row count {row_count} does not match expected {EXPECTED_ROWS}!")
            return False
        print("Row count matches expected value exactly.")

        null_count = con.execute("""
            SELECT COUNT(*) FROM fct_fraud_features
            WHERE distance_km IS NULL OR txns_24h IS NULL OR txns_7d IS NULL
        """).fetchone()[0]
        print(f"Rows with NULL in distance_km, txns_24h, or txns_7d: {null_count}")
        if null_count != 0:
            print("ERROR: Found null values in critical feature columns!")
            return False
        print("Schema verification passed: Zero nulls in distance or velocity columns.")

        # Train/serve skew guard, mirroring the dbt singular test. A feature whose
        # mean differs wildly between splits is not being computed the same way at
        # training and scoring time.
        skew = con.execute("""
            SELECT dataset_split, AVG(txns_24h), AVG(txns_7d)
            FROM fct_fraud_features GROUP BY dataset_split ORDER BY dataset_split
        """).fetchall()
        stats = {r[0]: (r[1], r[2]) for r in skew}
        print("Velocity means by split: " + ", ".join(
            f"{k}=({v[0]:.2f}, {v[1]:.2f})" for k, v in stats.items()))
        if 'train' in stats and 'test' in stats:
            for i, name in enumerate(("txns_24h", "txns_7d")):
                tr, te = stats['train'][i], stats['test'][i]
                ratio = te / tr if tr else 0.0
                if not 0.5 <= ratio <= 2.0:
                    print(f"ERROR: {name} train/serve skew -- test/train mean ratio "
                          f"{ratio:.4f} outside [0.5, 2.0].")
                    return False
        print("Train/serve skew check passed for velocity features.")
        return True
    finally:
        con.close()


def _load_scored_test_sample(limit=500):
    """Score a real slice of the test split with the production artifacts."""
    con = duckdb.connect(str(DB_PATH))
    try:
        df = con.execute(f"""
            SELECT f.*, t.first, t.last, t.gender, t.street, t.city, t.state, t.zip,
                   t.lat, t.long, t.merchant, t.category, t.merch_lat, t.merch_long, t.job
            FROM fct_fraud_features f
            LEFT JOIN stg_transactions_test t ON f.trans_num = t.trans_num
            WHERE f.dataset_split = 'test'
            LIMIT {limit}
        """).df()
    finally:
        con.close()

    models = load_models(ARTIFACTS_DIR)
    meta_probs, p_m2, p_m3, p_m4, triggered = run_inference(df, *models)
    return df, meta_probs, p_m2, p_m3, p_m4, triggered


def verify_inference_bounds():
    print("\n--- 3. VERIFYING INFERENCE SCORING BOUNDARIES ---")
    df, meta_probs, _, _, _, triggered = _load_scored_test_sample(limit=100)

    out_of_bounds = [(i, v) for i, v in enumerate(meta_probs) if not 0.0 < v < 1.0]
    if out_of_bounds:
        for i, v in out_of_bounds[:5]:
            print(f"ERROR: Inference score at index {i} is {v} (out of bounds (0.0, 1.0))")
        return False

    print(f"Inference verification passed: all {len(meta_probs)} probabilities "
          f"strictly in (0.0, 1.0).")
    print(f"Score range: min={meta_probs.min():.6f} max={meta_probs.max():.6f} "
          f"mean={meta_probs.mean():.6f}")
    print(f"Alerts triggered in sample: {int(np.sum(triggered))}/{len(triggered)}")
    return True


def verify_sar_agent():
    print("\n--- 4. VERIFYING AGENTIC STR GENERATION (FINTRAC) ---")
    print("Selecting the highest-risk real transaction from the test split...")
    df, meta_probs, p_m2, p_m3, p_m4, _ = _load_scored_test_sample(limit=500)

    idx = int(np.argmax(meta_probs))
    txn = df.iloc[idx].to_dict()
    meta_score = float(meta_probs[idx])
    print(f"Selected {txn['trans_num']} with live meta-score {meta_score:.4f} "
          f"(base models: {p_m2[idx]:.4f} / {p_m3[idx]:.4f} / {p_m4[idx]:.4f}); "
          f"ground-truth is_fraud={int(txn.get('is_fraud', -1))}")

    narrative = generate_sar_narrative(
        txn, float(p_m2[idx]), float(p_m3[idx]), float(p_m4[idx]), meta_score)
    if not narrative:
        print("ERROR: Failed to generate STR narrative.")
        return False

    print("\nGenerated STR Narrative Preview:")
    print("-" * 50)
    print(narrative[:600] + "...")
    print("-" * 50)

    missing = [s for s in ("WHO", "WHAT", "WHEN", "WHERE", "WHY", "HOW") if s not in narrative]
    if missing:
        print(f"ERROR: Missing required 5W+H sections in narrative: {missing}")
        return False
    print("Narrative contains all required 5W+H section headers.")

    lowered = narrative.lower()
    if not ("fintrac" in lowered or "pcmltfa" in lowered):
        print("ERROR: Narrative does not cite FINTRAC guidelines or PCMLTFA.")
        return False
    print("Narrative successfully references FINTRAC / Canadian regulatory standards.")

    if not ("reasonable grounds to suspect" in lowered or "rgs" in lowered):
        print("ERROR: Narrative does not reference the 'Reasonable Grounds to Suspect' threshold.")
        return False
    print("Narrative WHY section successfully frames findings around the RGS threshold.")

    if re.search(r"\b\d{12,19}\b", narrative):
        print("ERROR: Narrative appears to contain an unmasked card number.")
        return False
    print("No unmasked PAN detected in narrative.")

    try:
        file_path = save_sar_report(
            txn, float(p_m2[idx]), float(p_m3[idx]), float(p_m4[idx]), meta_score, narrative)
    except ValueError as e:
        print(f"COMPLIANCE_ERROR: {e}")
        return False

    if os.path.exists(file_path):
        print(f"STR report verified: File exists at {file_path}")
        return True
    return False


def main():
    print("=" * 70)
    print("               AEGISAGENT PIPELINE VERIFICATION")
    print("=" * 70)

    stages = [
        ("dbt tests", run_dbt_tests),
        ("DuckDB checks", verify_duckdb_schema),
        ("Inference bounds", verify_inference_bounds),
        ("STR Agent", verify_sar_agent),
    ]
    for i, (name, fn) in enumerate(stages, start=1):
        if not fn():
            print(f"\nPipeline verification FAILED at Step {i} ({name}).")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("        ALL STAGES SUCCESSFUL! PIPELINE VERIFIED AND COMPLIANT.")
    print("=" * 70)


if __name__ == "__main__":
    main()
