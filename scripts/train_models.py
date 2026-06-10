import os
import duckdb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from xgboost import XGBClassifier
import joblib

def main():
    print("Connecting to DuckDB database...")
    db_path = os.environ.get("DBT_DB_PATH", "e:/AegisAgent/aegis_db.duckdb")
    con = duckdb.connect(db_path)
    
    print("Loading fct_fraud_features from DuckDB...")
    df = con.execute("SELECT * FROM fct_fraud_features").df()
    con.close()
    
    print(f"Total rows loaded: {len(df)}")
    
    # Split into train and test datasets
    train_full_df = df[df['dataset_split'] == 'train'].copy()
    test_df = df[df['dataset_split'] == 'test'].copy()
    
    print(f"Train rows: {len(train_full_df)}")
    print(f"Test rows: {len(test_df)}")
    
    # Sort train dataset chronologically for 80/20 train/validation split
    train_full_df = train_full_df.sort_values('trans_date_trans_time').reset_index(drop=True)
    cutoff = int(len(train_full_df) * 0.8)
    
    train_df = train_full_df.iloc[:cutoff].copy()
    val_df = train_full_df.iloc[cutoff:].copy()
    
    print(f"Time-based split: {len(train_df)} train rows, {len(val_df)} validation rows")
    
    # Define features for each model
    FEAT_M2 = ['amt', 'distance_km', 'night', 'hour_sin', 'hour_cos', 'category_risk']
    FEAT_M3 = ['amt', 'log_amt', 'distance_km', 'night', 'hour', 'day_of_week', 
               'hour_sin', 'hour_cos', 'category_risk', 'state_risk', 
               'card_txn_cnt', 'card_mean_amt', 'card_std_amt']
    FEAT_M4 = ['amt', 'log_amt', 'distance_km', 'night', 'hour_sin', 'hour_cos', 'day_of_week', 
               'is_online', 'category_risk', 'state_risk', 'merchant_risk', 
               'card_txn_cnt', 'card_mean_amt', 'card_std_amt', 'txns_24h', 'txns_7d', 
               'amt_x_catRisk', 'dist_x_online']
    
    y_tr = train_df['is_fraud'].values
    y_val = val_df['is_fraud'].values
    y_te = test_df['is_fraud'].values
    
    # ------------------ MODEL 2: Random Forest (Geographic Focus) ------------------
    print("\nTraining Model 2 (Geographic RF)...")
    rf_geo = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    rf_geo.fit(train_df[FEAT_M2].values, y_tr)
    
    # ------------------ MODEL 3: XGBoost (Category Risk Focus) ------------------
    print("Training Model 3 (Category XGBoost)...")
    ratio = (len(y_tr) - y_tr.sum()) / (y_tr.sum() + 1e-6)
    xgb = XGBClassifier(
        n_estimators=1400,
        learning_rate=0.03,
        max_depth=7,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.85,
        gamma=0.3,
        reg_alpha=1.0,
        reg_lambda=2.0,
        scale_pos_weight=ratio * 1.5,
        max_delta_step=2,
        eval_metric='aucpr',
        n_jobs=-1,
        random_state=42
    )
    xgb.fit(train_df[FEAT_M3].values, y_tr)
    
    # ------------------ MODEL 4: Random Forest (Velocity Focus) ------------------
    print("Training Model 4 (Velocity RF with Scaling)...")
    scaler = StandardScaler()
    X4_tr_sc = scaler.fit_transform(train_df[FEAT_M4].values)
    
    rf_vel = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=42
    )
    rf_vel.fit(X4_tr_sc, y_tr)
    
    # ------------------ META-MODEL: Logistic Regression Stacking ------------------
    print("Fitting Stacked Meta-Model (Logistic Regression) on validation set probabilities...")
    # Base models predict on validation set
    val_m2_probs = rf_geo.predict_proba(val_df[FEAT_M2].values)[:, 1]
    val_m3_probs = xgb.predict_proba(val_df[FEAT_M3].values)[:, 1]
    
    X4_val_sc = scaler.transform(val_df[FEAT_M4].values)
    val_m4_probs = rf_vel.predict_proba(X4_val_sc)[:, 1]
    
    # Stack features for meta-learner
    X_meta_val = np.column_stack([val_m2_probs, val_m3_probs, val_m4_probs])
    
    # Train Logistic Regression meta-model
    meta_model = LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1)
    meta_model.fit(X_meta_val, y_val)
    
    # ------------------ EVALUATION ON TEST SET ------------------
    print("\nEvaluating models on the held-out test split...")
    test_m2_probs = rf_geo.predict_proba(test_df[FEAT_M2].values)[:, 1]
    test_m3_probs = xgb.predict_proba(test_df[FEAT_M3].values)[:, 1]
    
    X4_te_sc = scaler.transform(test_df[FEAT_M4].values)
    test_m4_probs = rf_vel.predict_proba(X4_te_sc)[:, 1]
    
    X_meta_test = np.column_stack([test_m2_probs, test_m3_probs, test_m4_probs])
    meta_probs = meta_model.predict_proba(X_meta_test)[:, 1]
    
    # Evaluate Base Model 2
    print(f"\nModel 2 (Geographic RF) Test ROC AUC: {roc_auc_score(y_te, test_m2_probs):.4f}")
    
    # Evaluate Base Model 3
    print(f"Model 3 (Category XGBoost) Test ROC AUC: {roc_auc_score(y_te, test_m3_probs):.4f}")
    
    # Evaluate Base Model 4
    print(f"Model 4 (Velocity RF) Test ROC AUC: {roc_auc_score(y_te, test_m4_probs):.4f}")
    
    # Evaluate Stacked Meta-model
    print(f"\nStacked Meta-Model Test ROC AUC: {roc_auc_score(y_te, meta_probs):.4f}")
    
    # Choose optimal threshold on validation set for stacked model
    val_meta_probs = meta_model.predict_proba(X_meta_val)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y_val, val_meta_probs)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    print(f"Optimal Meta-Model threshold selected from validation set: {best_thresh:.4f}")
    
    meta_preds = (meta_probs >= best_thresh).astype(int)
    print("\nStacked Meta-Model Classification Report:")
    print(classification_report(y_te, meta_preds, digits=4))
    
    print("Stacked Meta-Model Confusion Matrix:")
    print(confusion_matrix(y_te, meta_preds))
    
    # Save artifacts (Fix M-01)
    # Harmonize default artifacts dir
    default_artifacts_dir = os.environ.get("MODELS_ARTIFACTS_DIR", "/app/models/")
    if not os.path.exists(default_artifacts_dir) and os.path.exists("e:/AegisAgent/models_artifacts"):
        default_artifacts_dir = "e:/AegisAgent/models_artifacts"
    artifacts_dir = default_artifacts_dir
    
    from datetime import datetime
    import json
    
    version_str = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    versioned_dir = os.path.join(artifacts_dir, version_str)
    os.makedirs(versioned_dir, exist_ok=True)
    
    print(f"\nSaving versioned model artifacts to {versioned_dir}...")
    joblib.dump(rf_geo, os.path.join(versioned_dir, "model_2_geo_rf.joblib"))
    joblib.dump(xgb, os.path.join(versioned_dir, "model_3_cat_xgb.joblib"))
    joblib.dump(rf_vel, os.path.join(versioned_dir, "model_4_vel_rf.joblib"))
    joblib.dump(scaler, os.path.join(versioned_dir, "model_4_scaler.joblib"))
    joblib.dump(meta_model, os.path.join(versioned_dir, "meta_model.joblib"))
    
    # Save best threshold to file inside versioned directory for inference time use
    with open(os.path.join(versioned_dir, "meta_threshold.txt"), "w") as f:
        f.write(str(best_thresh))
        
    # Write latest version pointer to parent directory
    latest_file_path = os.path.join(artifacts_dir, "latest_version.txt")
    with open(latest_file_path, "w") as f:
        f.write(version_str)
        
    # Evaluate stacking ensemble and construct telemetry payload (Fix M-02)
    # confusion matrix
    cm = confusion_matrix(y_te, meta_preds)
    
    metrics_payload = {
        "trained_at": datetime.now().isoformat(),
        "version": version_str,
        "val_f1": float(f1_scores[best_idx]) if best_idx < len(f1_scores) else 0.0,
        "val_auc_meta": float(roc_auc_score(y_val, val_meta_probs)),
        "test_auc_m2": float(roc_auc_score(y_te, test_m2_probs)),
        "test_auc_m3": float(roc_auc_score(y_te, test_m3_probs)),
        "test_auc_m4": float(roc_auc_score(y_te, test_m4_probs)),
        "test_auc_meta": float(roc_auc_score(y_te, meta_probs)),
        "optimal_threshold": float(best_thresh),
        "confusion_matrix": cm.tolist()
    }
    
    metrics_file_path = os.path.join(versioned_dir, "training_metrics.json")
    with open(metrics_file_path, "w") as f:
        json.dump(metrics_payload, f, indent=4)
        
    print(f"Metrics saved to {metrics_file_path}")
    print("All model artifacts and telemetry saved successfully!")

if __name__ == "__main__":
    main()
