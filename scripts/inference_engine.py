import os
import argparse
import duckdb
import pandas as pd
import numpy as np
import joblib
from pydantic import BaseModel, Field

# Define features for base models
FEAT_M2 = ['amt', 'distance_km', 'night', 'hour_sin', 'hour_cos', 'category_risk']
FEAT_M3 = ['amt', 'log_amt', 'distance_km', 'night', 'hour', 'day_of_week', 
           'hour_sin', 'hour_cos', 'category_risk', 'state_risk', 
           'card_txn_cnt', 'card_mean_amt', 'card_std_amt']
FEAT_M4 = ['amt', 'log_amt', 'distance_km', 'night', 'hour_sin', 'hour_cos', 'day_of_week', 
           'is_online', 'category_risk', 'state_risk', 'merchant_risk', 
           'card_txn_cnt', 'card_mean_amt', 'card_std_amt', 'txns_24h', 'txns_7d', 
           'amt_x_catRisk', 'dist_x_online']

# Rigid Pydantic model for feature validation (Fix I-01)
class TransactionFeatures(BaseModel):
    distance_km: float = Field(..., ge=0)
    txns_24h: int = Field(..., ge=0)
    txns_7d: int = Field(..., ge=0)
    amt_z_card: float
    category_risk: float = Field(..., ge=0, le=1)
    trans_num: str = Field(..., min_length=1)

def load_models(artifacts_dir):
    # Check for versioned subdirectory using pointer (Fix M-01)
    latest_file = os.path.join(artifacts_dir, "latest_version.txt")
    if os.path.exists(latest_file):
        with open(latest_file, "r") as f:
            version_str = f.read().strip()
        load_dir = os.path.join(artifacts_dir, version_str)
        print(f"Loading versioned models from: {load_dir}")
    else:
        load_dir = artifacts_dir
        print(f"latest_version.txt not found. Loading models from root: {load_dir}")

    model_2 = joblib.load(os.path.join(load_dir, "model_2_geo_rf.joblib"))
    model_3 = joblib.load(os.path.join(load_dir, "model_3_cat_xgb.joblib"))
    model_4 = joblib.load(os.path.join(load_dir, "model_4_vel_rf.joblib"))
    scaler_4 = joblib.load(os.path.join(load_dir, "model_4_scaler.joblib"))
    meta_model = joblib.load(os.path.join(load_dir, "meta_model.joblib"))
    
    thresh_path = os.path.join(load_dir, "meta_threshold.txt")
    if os.path.exists(thresh_path):
        with open(thresh_path, "r") as f:
            threshold = float(f.read().strip())
    else:
        threshold = 0.5
        
    return model_2, model_3, model_4, scaler_4, meta_model, threshold

def run_inference(df, model_2, model_3, model_4, scaler_4, meta_model, threshold=None):
    # 1. Pydantic input feature validation (Fix I-01)
    for idx, row in df.iterrows():
        t_num = row.get('trans_num')
        if pd.isna(t_num) or t_num is None:
            raise ValueError(f"Validation failed: trans_num is null or NaN at row {idx}")
        
        try:
            TransactionFeatures(
                distance_km=float(row['distance_km']),
                txns_24h=int(row['txns_24h']),
                txns_7d=int(row['txns_7d']),
                amt_z_card=float(row['amt_z_card']),
                category_risk=float(row['category_risk']),
                trans_num=str(t_num)
            )
        except Exception as e:
            raise ValueError(f"Input feature validation failed at row {idx}: {e}")

    # 2. Read threshold dynamically from file if not explicitly passed (Fix I-02)
    if threshold is None:
        artifacts_dir = os.environ.get("MODELS_ARTIFACTS_DIR", "/app/models/")
        if not os.path.exists(artifacts_dir) and os.path.exists("e:/AegisAgent/models_artifacts"):
            artifacts_dir = "e:/AegisAgent/models_artifacts"
        
        latest_file = os.path.join(artifacts_dir, "latest_version.txt")
        if os.path.exists(latest_file):
            with open(latest_file, "r") as f:
                version_str = f.read().strip()
            load_dir = os.path.join(artifacts_dir, version_str)
        else:
            load_dir = artifacts_dir
            
        thresh_path = os.path.join(load_dir, "meta_threshold.txt")
        if os.path.exists(thresh_path):
            with open(thresh_path, "r") as f:
                threshold = float(f.read().strip())
        else:
            threshold = 0.5

    # Ensure correct types/shapes
    X_m2 = df[FEAT_M2].values
    X_m3 = df[FEAT_M3].values
    X_m4 = df[FEAT_M4].values
    
    # Predict probabilities for base models
    p_m2 = model_2.predict_proba(X_m2)[:, 1]
    p_m3 = model_3.predict_proba(X_m3)[:, 1]
    
    X_m4_sc = scaler_4.transform(X_m4)
    p_m4 = model_4.predict_proba(X_m4_sc)[:, 1]
    
    # Stacking
    X_meta = np.column_stack([p_m2, p_m3, p_m4])
    meta_p = meta_model.predict_proba(X_meta)[:, 1]
    
    # Apply threshold to determine binary alerts (Fix I-02)
    triggered_alert = (meta_p >= threshold).astype(bool)
    
    return meta_p, p_m2, p_m3, p_m4, triggered_alert

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true', help='Run verification mode')
    
    # Harmonize MODELS_ARTIFACTS_DIR path default (Fix M-04)
    default_dir = os.environ.get('MODELS_ARTIFACTS_DIR', '/app/models/')
    if not os.path.exists(default_dir) and os.path.exists('e:/AegisAgent/models_artifacts'):
        default_dir = 'e:/AegisAgent/models_artifacts'
        
    parser.add_argument('--artifacts-dir', type=str, default=default_dir)
    args = parser.parse_args()
    
    if args.verify:
        print("=== INFERENCE ENGINES VERIFICATION MODE ===")
        print(f"Loading models from {args.artifacts_dir}...")
        model_2, model_3, model_4, scaler_4, meta_model, threshold = load_models(args.artifacts_dir)
        
        print("Connecting to DuckDB and loading test sample...")
        db_path = os.environ.get("DBT_DB_PATH", "e:/AegisAgent/aegis_db.duckdb")
        con = duckdb.connect(db_path)
        # Select first 100 test rows
        test_df = con.execute("SELECT * FROM fct_fraud_features WHERE dataset_split = 'test' LIMIT 100").df()
        con.close()
        
        if len(test_df) == 0:
            print("ERROR: Marts table 'fct_fraud_features' contains 0 test split rows. Run dbt run first.")
            return
            
        print(f"Loaded {len(test_df)} sample transactions.")
        
        # Run inference
        meta_probs, p_m2, p_m3, p_m4, triggered = run_inference(test_df, model_2, model_3, model_4, scaler_4, meta_model, threshold)
        
        # Validation checks
        all_in_bounds = True
        for i, val in enumerate(meta_probs):
            if val <= 0.0 or val >= 1.0:
                print(f"FAIL: Row {i} score {val} is not strictly in (0.0, 1.0)")
                all_in_bounds = False
                
        if all_in_bounds:
            print("SUCCESS: All meta-classifier outputs are strictly within (0.0, 1.0) boundaries.")
        else:
            print("WARNING: Some outputs violated boundary constraints.")
            
        # Distribution stats
        print(f"Triggered alerts: {np.sum(triggered)} out of {len(triggered)}")
        print("\nProbability Score Distribution Statistics:")
        print(f"Min probability score:    {np.min(meta_probs):.6f}")
        print(f"Max probability score:    {np.max(meta_probs):.6f}")
        print(f"Mean probability score:   {np.mean(meta_probs):.6f}")
        print(f"Median probability score: {np.median(meta_probs):.6f}")
        print(f"Std Dev of score:         {np.std(meta_probs):.6f}")
        
        # Simple text histogram
        print("\nProbability Distribution Histogram:")
        hist, bin_edges = np.histogram(meta_probs, bins=10, range=(0, 1))
        for i in range(10):
            bar = '#' * int(hist[i] / 2) if hist[i] > 0 else ''
            print(f"[{bin_edges[i]:.1f} - {bin_edges[i+1]:.1f}): {hist[i]:3d} | {bar}")
            
if __name__ == "__main__":
    main()
