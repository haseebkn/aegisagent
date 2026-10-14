"""End-to-end pipeline verification.

The default path checks dbt, feature data, actual model scores and an isolated
case/evidence workflow without cloud calls. --with-narrative additionally invokes
paid Bedrock drafting using an actual model alert, never a hardcoded score.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

import duckdb
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config import ARTIFACTS_DIR, DB_PATH, PROJECT_ROOT, resolve_model_dir
from scripts.case_management import CaseStore, ReviewerRole
from scripts.evidence import EvidenceStore
from scripts.inference_engine import (NoAlertsInSample, load_models, run_inference,
                                      select_highest_risk_alert)
from scripts.sar_agent import generate_sar_narrative, save_sar_report
from scripts.security import ALERT_INGESTOR, INVESTIGATOR, local_development_principal

EXPECTED_ROWS = 1852394


def run_dbt_tests(expected_rows=EXPECTED_ROWS):
    print("--- 1. RUNNING DBT TESTS ---")
    result = subprocess.run(
        [sys.executable, "-c",
         "from dbt.cli.main import cli; cli()", "test",
         "--profiles-dir", str(PROJECT_ROOT), "--vars",
         json.dumps({"expected_row_count": expected_rows})],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print("dbt test failed!")
        print(result.stdout)
        print(result.stderr)
        return False
    print("All dbt tests passed successfully (return code 0).")
    return True


def verify_duckdb_schema(expected_rows=EXPECTED_ROWS):
    print("\n--- 2. VERIFYING DUCKDB DATA & SCHEMA ---")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        row_count = con.execute("SELECT COUNT(*) FROM fct_fraud_features").fetchone()[0]
        print(f"Total row count in fct_fraud_features: {row_count:,}")
        if row_count != expected_rows:
            print(f"ERROR: Row count {row_count} does not match expected {expected_rows}!")
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


def _load_scored_test_sample(limit=10000):
    """Score a real slice of the development holdout with the demo artifacts."""
    if type(limit) is not int or limit < 1:
        raise ValueError("Sample limit must be a positive integer")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = con.execute("""
            SELECT f.*, t.merchant, t.category
            FROM fct_fraud_features f
            LEFT JOIN stg_transactions_test t ON f.trans_num = t.trans_num
            WHERE f.evaluation_role = 'development_holdout'
            ORDER BY f.trans_date_trans_time, f.trans_num
            LIMIT ?
        """, [limit]).df()
    finally:
        con.close()

    if df.empty:
        raise ValueError("No development-holdout rows available for verification")
    model_dir = resolve_model_dir(ARTIFACTS_DIR)
    models = load_models(model_dir)
    df.attrs["model_version"] = model_dir.name
    df.attrs["alert_threshold"] = float(models[-1])
    meta_probs, p_m2, p_m3, p_m4, triggered = run_inference(df, *models)
    return df, meta_probs, p_m2, p_m3, p_m4, triggered


def verify_inference_bounds():
    print("\n--- 3. VERIFYING INFERENCE SCORING BOUNDARIES ---")
    df, meta_probs, _, _, _, triggered = _load_scored_test_sample(limit=100)

    out_of_bounds = [(i, v) for i, v in enumerate(meta_probs) if not np.isfinite(v) or not 0.0 <= v <= 1.0]
    if out_of_bounds:
        for i, v in out_of_bounds[:5]:
            print(f"ERROR: Inference score at index {i} is {v} (outside [0.0, 1.0])")
        return False

    print(f"Inference verification passed: all {len(meta_probs)} probabilities "
          f"finite and in [0.0, 1.0].")
    print(f"Score range: min={meta_probs.min():.6f} max={meta_probs.max():.6f} "
          f"mean={meta_probs.mean():.6f}")
    print(f"Alerts triggered in sample: {int(np.sum(triggered))}/{len(triggered)}")
    return True


def verify_sar_agent():
    print("\n--- 4. VERIFYING INVESTIGATION NARRATIVE DRAFTING ---")
    print("Selecting the highest-risk ALERT from the development holdout...")
    df, meta_probs, p_m2, p_m3, p_m4, triggered = _load_scored_test_sample()

    try:
        idx = select_highest_risk_alert(meta_probs, triggered)
    except NoAlertsInSample as e:
        print(f"ERROR: {e}")
        return False

    txn = df.iloc[idx].to_dict()
    meta_score = float(meta_probs[idx])
    print(f"{int(triggered.sum())} alert(s) in a sample of {len(df):,}. "
          f"Selected {txn['trans_num']} with live meta-score {meta_score:.4f} "
          f"(base models: {p_m2[idx]:.4f} / {p_m3[idx]:.4f} / {p_m4[idx]:.4f}); "
          f"ground-truth is_fraud={int(txn.get('is_fraud', -1))}")

    narrative = generate_sar_narrative(
        txn, float(p_m2[idx]), float(p_m3[idx]), float(p_m4[idx]), meta_score)
    if not narrative:
        print("ERROR: Failed to generate an investigation narrative draft.")
        return False

    print("\nGenerated Investigation Narrative Draft Preview (not filed):")
    print("-" * 50)
    print("Narrative generated; content suppressed from verification logs.")
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
        with tempfile.TemporaryDirectory() as temp_dir:
            case_store = CaseStore(os.path.join(temp_dir, "cases.sqlite3"))
            case = case_store.create_alert_case(
                trans_num=txn["trans_num"],
                model_score=meta_score,
                threshold=df.attrs["alert_threshold"],
                model_version=df.attrs["model_version"],
                actor="pipeline-verifier",
            )
            case = case_store.start_review(
                case.case_id,
                actor="pipeline-verifier",
                actor_role=ReviewerRole.INVESTIGATOR,
                rationale="Temporary human-review case for end-to-end pipeline verification.",
                expected_version=case.version,
            )
            file_path = save_sar_report(
                txn,
                float(p_m2[idx]),
                float(p_m3[idx]),
                float(p_m4[idx]),
                meta_score,
                narrative,
                model_version=case.model_version,
                case_id=case.case_id,
                actor="pipeline-verifier",
                actor_role=ReviewerRole.INVESTIGATOR,
                expected_case_version=case.version,
                case_store=case_store,
                evidence_store=EvidenceStore(os.path.join(temp_dir, "evidence")),
                archive_bucket="",
            )
            integrity = case_store.verify_integrity(case.case_id)
            if not integrity["ok"]:
                print(f"ERROR: Evidence integrity verification failed: {integrity['issues']}")
                return False
            if os.path.exists(file_path):
                print(
                    f"Narrative evidence verified: {integrity['events_verified']} events, "
                    f"{integrity['evidence_verified']} artifact, chain {integrity['chain_head']}"
                )
                return True
    except ValueError as e:
        print(f"DRAFT_VALIDATION_ERROR: {e}")
        return False
    return False


def verify_case_workflow():
    """Exercise local case creation, review, evidence and integrity without cloud calls."""
    df, meta, _, _, _, triggered = _load_scored_test_sample()
    idx = select_highest_risk_alert(meta, triggered)
    identity = local_development_principal(
        subject="pipeline-verifier", roles=[INVESTIGATOR, ALERT_INGESTOR]
    )
    with tempfile.TemporaryDirectory() as temporary:
        cases = CaseStore(os.path.join(temporary, "cases.sqlite3"), principal=identity)
        case = cases.create_alert_case(
            trans_num=str(df.iloc[idx]["trans_num"]),
            model_score=float(meta[idx]),
            threshold=df.attrs["alert_threshold"],
            model_version=df.attrs["model_version"],
        )
        case = cases.start_review(
            case.case_id,
            rationale="Scripted synthetic-data review for the local verification workflow.",
            expected_version=case.version,
        )
        receipt = EvidenceStore(os.path.join(temporary, "evidence")).preserve_text(
            case_id=case.case_id, evidence_type="model_observation",
            content=json.dumps({
                "purpose": "synthetic pipeline verification, not an RGS finding",
                "model_score": case.model_score, "model_version": case.model_version,
            }, sort_keys=True),
        )
        cases.attach_evidence(case.case_id, receipt=receipt, expected_version=case.version)
        result = cases.verify_integrity(case.case_id)
        if not result["ok"]:
            raise ValueError("Case/evidence integrity verification failed")
        print(f"Case workflow verified: {result['events_verified']} events, "
              f"{result['evidence_verified']} evidence artifact; no RGS decision recorded.")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    parser.add_argument("--with-narrative", action="store_true",
                        help="Also invoke paid AWS Bedrock narrative drafting")
    args = parser.parse_args()
    if args.expected_rows < 1:
        parser.error("--expected-rows must be positive")
    print("=" * 70)
    print("               AEGISAGENT PIPELINE VERIFICATION")
    print("=" * 70)

    stages = [
        ("dbt tests", lambda: run_dbt_tests(args.expected_rows)),
        ("DuckDB checks", lambda: verify_duckdb_schema(args.expected_rows)),
        ("Inference bounds", verify_inference_bounds),
        ("Case and evidence workflow", verify_case_workflow),
    ]
    if args.with_narrative:
        stages.append(("Optional Bedrock narrative drafting", verify_sar_agent))
    for i, (name, fn) in enumerate(stages, start=1):
        if not fn():
            print(f"\nPipeline verification FAILED at Step {i} ({name}).")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("        ALL STAGES SUCCESSFUL! DEMO PIPELINE VERIFIED.")
    print("=" * 70)


if __name__ == "__main__":
    main()
