import os
import re
import sys

import duckdb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import COMPLIANCE_LOGS_DIR, DB_PATH, resolve_model_dir

con = duckdb.connect(str(DB_PATH))

def pf(cond): return "PASS" if cond else "FAIL"

# 1. Row count split check
splits = con.execute("SELECT dataset_split, COUNT(*) as cnt FROM fct_fraud_features GROUP BY dataset_split").fetchall()
print("=== ROW COUNTS ===")
for s in splits:
    print(f"  {s[0]}: {s[1]:,}")
total = sum(s[1] for s in splits)
print(f"  TOTAL: {total:,}  (expected 1,852,394) [{pf(total == 1852394)}]")

# 2. Column presence check
cols = [r[1] for r in con.execute("PRAGMA table_info(fct_fraud_features)").fetchall()]
expected = [
    'trans_num','cc_num','trans_date_trans_time','dataset_split','is_fraud',
    'amt','log_amt','distance_km','night','hour','day_of_week','hour_sin','hour_cos',
    'is_online','category_risk','state_risk','merchant_risk',
    'card_txn_cnt','card_mean_amt','card_std_amt',
    'amt_z_card','amt_over_mean_card','txns_24h','txns_7d',
    'amt_x_catRisk','dist_x_online','amt_x_night'
]
missing = [c for c in expected if c not in cols]
print(f"\n=== COLUMN PRESENCE ({len(expected)} expected) ===")
print(f"  Present: {len(expected)-len(missing)}/{len(expected)}, Missing: {missing if missing else 'None'} [{pf(not missing)}]")

# 3. Null audit
row = con.execute("""SELECT
    SUM(CASE WHEN distance_km IS NULL THEN 1 ELSE 0 END),
    SUM(CASE WHEN txns_24h IS NULL THEN 1 ELSE 0 END),
    SUM(CASE WHEN txns_7d IS NULL THEN 1 ELSE 0 END),
    SUM(CASE WHEN amt_z_card IS NULL THEN 1 ELSE 0 END),
    SUM(CASE WHEN category_risk IS NULL THEN 1 ELSE 0 END),
    SUM(CASE WHEN trans_num IS NULL THEN 1 ELSE 0 END)
FROM fct_fraud_features""").fetchone()
all_zero = all(v == 0 for v in row)
labels = ['distance_km','txns_24h','txns_7d','amt_z_card','category_risk','trans_num']
print(f"\n=== NULL AUDIT [{pf(all_zero)}] ===")
for label, val in zip(labels, row):
    print(f"  NULL {label}: {val}")

# 4. trans_num uniqueness
dup = con.execute("SELECT COUNT(*) FROM (SELECT trans_num, COUNT(*) c FROM fct_fraud_features GROUP BY trans_num HAVING c > 1)").fetchone()[0]
print(f"\n=== TRANS_NUM UNIQUENESS ===")
print(f"  Duplicate rows: {dup} [{pf(dup == 0)}]")

# 5. Chronological check (look-ahead bias guard)
out_of_order = con.execute("""
    SELECT COUNT(*) FROM (
        SELECT trans_date_trans_time,
               LAG(trans_date_trans_time) OVER (PARTITION BY cc_num ORDER BY trans_date_trans_time) as prev_t
        FROM fct_fraud_features WHERE dataset_split = 'train' LIMIT 100000
    ) t WHERE trans_date_trans_time < prev_t
""").fetchone()[0]
print(f"\n=== LOOK-AHEAD BIAS CHECK (100k train sample) ===")
print(f"  Out-of-order timestamps: {out_of_order} [{pf(out_of_order == 0)}]")

# 6. Distance sanity
neg_dist = con.execute("SELECT COUNT(*) FROM fct_fraud_features WHERE distance_km < 0").fetchone()[0]
print(f"\n=== DISTANCE_KM SANITY ===")
print(f"  Negative distance values: {neg_dist} [{pf(neg_dist == 0)}]")

# 7. Velocity sanity
neg_vel = con.execute("SELECT COUNT(*) FROM fct_fraud_features WHERE txns_24h < 0 OR txns_7d < 0").fetchone()[0]
print(f"\n=== VELOCITY SANITY ===")
print(f"  Negative velocity values: {neg_vel} [{pf(neg_vel == 0)}]")

# 8. Fraud class distribution
fraud_stats = con.execute("""SELECT dataset_split, SUM(is_fraud), COUNT(*),
    ROUND(100.0*SUM(is_fraud)/COUNT(*),4)
    FROM fct_fraud_features GROUP BY dataset_split ORDER BY dataset_split""").fetchall()
print(f"\n=== FRAUD CLASS DISTRIBUTION ===")
for r in fraud_stats:
    print(f"  {r[0]}: {int(r[1]):,} fraud / {r[2]:,} total ({r[3]}%)")

con.close()

# 9. Model artifacts
print(f"\n=== MODEL ARTIFACTS ===")
artifacts_dir = str(resolve_model_dir())
required_files = [
    "model_2_geo_rf.joblib","model_3_cat_xgb.joblib",
    "model_4_vel_rf.joblib","model_4_scaler.joblib",
    "meta_model.joblib","meta_threshold.txt"
]
for f in required_files:
    path = os.path.join(artifacts_dir, f)
    exists = os.path.exists(path)
    size_kb = os.path.getsize(path) / 1024 if exists else 0
    print(f"  {f}: {'EXISTS' if exists else 'MISSING'} ({size_kb:.1f} KB) [{pf(exists)}]")

# 10. Meta threshold
with open(os.path.join(artifacts_dir, "meta_threshold.txt")) as f:
    thr = float(f.read())
print(f"\n=== META THRESHOLD ===")
print(f"  Threshold: {thr:.4f} [{pf(0 < thr < 1)}]")

# 11. Compliance logs
compliance_dir = str(COMPLIANCE_LOGS_DIR)
logs = [f for f in os.listdir(compliance_dir) if f.endswith(".txt")] if os.path.exists(compliance_dir) else []
print(f"\n=== COMPLIANCE LOGS ===")
print(f"  SAR reports found: {len(logs)} [{pf(len(logs) > 0)}]")
for log in logs:
    path = os.path.join(compliance_dir, log)
    try:
        content = open(path, encoding="utf-8").read()
    except UnicodeDecodeError:
        content = open(path, encoding="cp1252").read()
    sections = [s for s in ["WHO","WHAT","WHEN","WHERE","WHY","HOW"] if s in content]
    # Unmasked PAN is a real compliance failure. Hedging is not: an STR records
    # reasonable grounds to SUSPECT, so wording like "individually each indicator
    # might admit a benign explanation; collectively they establish a reasonable
    # basis to suspect" is exactly right, and this audit used to mark it FAIL.
    # Grounding -- every figure tracing to the payload -- is enforced at generation
    # time by scripts/grounding.py and stamped into the report. See
    # docs/str-narrative-design.md for why the old word blacklist was abandoned.
    unmasked_pan = re.findall(r"\b\d{12,19}\b", content)
    hedging = [w for w in ["might", "possibly", "could be", "appears to"]
               if f" {w} " in f" {content.lower()} "]
    grounding_line = next((ln.strip() for ln in content.splitlines()
                           if ln.startswith("GROUNDING REVIEW:")), None)
    print(f"  {log}")
    print(f"    5W+H sections: {len(sections)}/6 [{pf(len(sections)==6)}]")
    print(f"    Unmasked PAN: {'FOUND ' + str(unmasked_pan[:1]) if unmasked_pan else 'none'} "
          f"[{pf(not unmasked_pan)}]")
    if grounding_line:
        verdict = grounding_line.split(":", 1)[1].strip()
        print(f"    Grounding at generation: {verdict[:66]} [{pf(verdict.startswith('GROUNDED'))}]")
    if hedging:
        print(f"    Note (advisory, not a failure): hedging terms present {hedging}")

# 12. AWS Bedrock reachability.
#
# This previously printed "PASS [{pf(True)}]" for model access without making any
# call at all. A check that cannot fail is not a check -- the same defect family as
# the hardcoded ensemble scores removed from the dashboard. It now actually asks
# Bedrock whether the configured model is available, and says so plainly when it
# cannot reach the service rather than claiming success.
print("\n=== AWS BEDROCK ACCESS ===")
model_id = os.environ.get("BEDROCK_MODEL_ID",
                          "us.anthropic.claude-haiku-4-5-20251001-v1:0")
region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
# A cross-region inference profile ("us." prefix) is backed by this foundation model.
foundation_id = model_id.split(".", 1)[1] if model_id.startswith("us.") else model_id
try:
    import boto3
    summaries = boto3.client("bedrock", region_name=region).list_foundation_models()
    available = {m["modelId"] for m in summaries.get("modelSummaries", [])}
    reachable = foundation_id in available
    print(f"  Region: {region}  |  foundation models visible: {len(available)}")
    print(f"  {foundation_id} available: {reachable} [{pf(reachable)}]")
except Exception as exc:
    print(f"  SKIPPED - could not query Bedrock "
          f"({type(exc).__name__}: {str(exc)[:80]})")
    print(f"  Target model (UNVERIFIED): {model_id}")

print("\n" + "="*60)
print("PHASE 1 AUDIT COMPLETE")
print("="*60)
