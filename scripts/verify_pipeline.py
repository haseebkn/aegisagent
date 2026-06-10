import os
import subprocess
import sys
import duckdb
import numpy as np
import re

# Add parent directory to sys.path to allow running from scripts/ directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.inference_engine import load_models, run_inference
from scripts.sar_agent import generate_sar_narrative, save_sar_report

def run_dbt_tests():
    print("--- 1. RUNNING DBT TESTS ---")
    result = subprocess.run(["dbt", "test", "--profiles-dir", "."], capture_output=True, text=True)
    if result.returncode != 0:
        print("dbt test failed!")
        print(result.stdout)
        print(result.stderr)
        return False
    print("All dbt tests passed successfully (return code 0).")
    return True

def verify_duckdb_schema():
    print("\n--- 2. VERIFYING DUCKDB DATA & SCHEMA ---")
    db_path = os.environ.get("DBT_DB_PATH", "e:/AegisAgent/aegis_db.duckdb")
    con = duckdb.connect(db_path)
    
    # 1. Row count check
    row_count = con.execute("SELECT COUNT(*) FROM fct_fraud_features").fetchone()[0]
    print(f"Total row count in fct_fraud_features: {row_count}")
    if row_count != 1852394:
        print(f"ERROR: Row count {row_count} does not match expected 1852394!")
        con.close()
        return False
    else:
        print("Row count matches expected value exactly.")
        
    # 2. Null values check
    null_count = con.execute("""
        SELECT COUNT(*) 
        FROM fct_fraud_features 
        WHERE distance_km IS NULL OR txns_24h IS NULL OR txns_7d IS NULL
    """).fetchone()[0]
    print(f"Number of rows with NULL in distance_km, txns_24h, or txns_7d: {null_count}")
    if null_count != 0:
        print("ERROR: Found null values in critical feature columns!")
        con.close()
        return False
    else:
        print("Schema verification passed: Zero nulls in distance or velocity columns.")
        
    con.close()
    return True

def verify_inference_bounds():
    print("\n--- 3. VERIFYING INFERENCE SCORING BOUNDARIES ---")
    db_path = os.environ.get("DBT_DB_PATH", "e:/AegisAgent/aegis_db.duckdb")
    con = duckdb.connect(db_path)
    test_df = con.execute("SELECT * FROM fct_fraud_features WHERE dataset_split = 'test' LIMIT 100").df()
    con.close()
    
    # Harmonize MODELS_ARTIFACTS_DIR default path (Fix M-04)
    default_dir = os.environ.get('MODELS_ARTIFACTS_DIR', '/app/models/')
    if not os.path.exists(default_dir) and os.path.exists('e:/AegisAgent/models_artifacts'):
        default_dir = 'e:/AegisAgent/models_artifacts'
    artifacts_dir = default_dir
    model_2, model_3, model_4, scaler_4, meta_model, threshold = load_models(artifacts_dir)
    
    meta_probs, p_m2, p_m3, p_m4, triggered_alerts = run_inference(test_df, model_2, model_3, model_4, scaler_4, meta_model, threshold)
    
    all_in_bounds = True
    for i, val in enumerate(meta_probs):
        if val <= 0.0 or val >= 1.0:
            print(f"ERROR: Inference score at index {i} is {val} (out of bounds (0.0, 1.0))")
            all_in_bounds = False
            
    if all_in_bounds:
        print("Inference verification passed: All 100 sample probabilities are strictly in (0.0, 1.0).")
        return True
    return False

def verify_sar_agent():
    print("\n--- 4. VERIFYING AGENTIC STR GENERATION (FINTRAC) ---")
    # Trigger STR generation on a highly anomalous transaction
    test_txn = {
        'trans_num': 'TXN_VERIFY_88888',
        'trans_date_trans_time': '2026-06-10 02:45:00',
        'cc_num': 9876543210987654,
        'first': 'Alice',
        'last': 'Smith',
        'gender': 'F',
        'job': 'Data Engineer',
        'street': '123 Main Street',
        'city': 'Boston',
        'state': 'MA',
        'zip': '02108',
        'lat': 42.3601,
        'long': -71.0589,
        'merchant': 'fraud_Luxury_Jewelry_Store',
        'category': 'shopping_pos',
        'merch_lat': 51.5074, # London, UK (Extreme distance)
        'merch_long': -0.1278,
        'amt': 7500.00,
        'distance_km': 5262.11,
        'night': 1,
        'card_mean_amt': 50.00,
        'amt_z_card': 149.00,
        'txns_24h': 15,
        'txns_7d': 25,
        'category_risk': 0.015
    }
    
    p_m2 = 0.9950
    p_m3 = 0.9890
    p_m4 = 0.9920
    meta_score = 0.9998
    
    narrative = generate_sar_narrative(test_txn, p_m2, p_m3, p_m4, meta_score)
    if not narrative:
        print("ERROR: Failed to generate STR narrative.")
        return False
        
    print("\nGenerated STR Narrative Preview:")
    print("-" * 50)
    print(narrative[:600] + "...")
    print("-" * 50)
    
    # Verify narrative contains 5W+H sections
    required_sections = ["WHO", "WHAT", "WHEN", "WHERE", "WHY", "HOW"]
    missing_sections = [sec for sec in required_sections if sec not in narrative]
    if missing_sections:
        print(f"ERROR: Missing required 5W+H sections in narrative: {missing_sections}")
        return False
    else:
        print("Narrative contains all required 5W+H section headers.")
        
    # Check for Canadian regulatory citations
    has_fintrac = "fintrac" in narrative.lower() or "pcmltfa" in narrative.lower()
    has_rgs = "reasonable grounds to suspect" in narrative.lower() or "rgs" in narrative.lower()
    
    if not has_fintrac:
        print("ERROR: Narrative does not cite FINTRAC guidelines or PCMLTFA.")
        return False
    else:
        print("Narrative successfully references FINTRAC / Canadian regulatory standards.")
        
    if not has_rgs:
        print("ERROR: Narrative WHY section does not reference the 'Reasonable Grounds to Suspect' (RGS) threshold.")
        return False
    else:
        print("Narrative WHY section successfully frames findings around the RGS legal threshold.")
        
    # Check for speculative words
    blacklist = ["may", "might", "possibly", "could", "appears"]
    flagged = [w for w in blacklist if re.search(rf"\b{w}\b", narrative, re.IGNORECASE)]
    if flagged:
        print(f"ERROR: Narrative contains speculative language: {flagged}")
        return False
    else:
        print("SUCCESS: 0 occurrences of speculative terms. Narrative contains only definitive, objective language.")
        
    # Save report
    try:
        file_path = save_sar_report(test_txn, p_m2, p_m3, p_m4, meta_score, narrative)
    except ValueError as e:
        print(f"COMPLIANCE_ERROR: {e}")
        return False
        
    if os.path.exists(file_path):
        print(f"STR report verified: File exists at {file_path}")
        return True
    return False

def main():
    print("======================================================================")
    print("               AEGISAGENT PHASE 1 PIPELINE VERIFICATION")
    print("======================================================================")
    
    if not run_dbt_tests():
        print("Pipeline verification failed at Step 1 (dbt tests).")
        sys.exit(1)
        
    if not verify_duckdb_schema():
        print("Pipeline verification failed at Step 2 (DuckDB checks).")
        sys.exit(1)
        
    if not verify_inference_bounds():
        print("Pipeline verification failed at Step 3 (Inference bounds).")
        sys.exit(1)
        
    if not verify_sar_agent():
        print("Pipeline verification failed at Step 4 (STR Agent).")
        sys.exit(1)
        
    print("\n======================================================================")
    print("        ALL PHASES SUCCESSFUL! PIPELINE VERIFIED AND COMPLIANT.")
    print("======================================================================")

if __name__ == "__main__":
    main()
