import argparse
import json
import os
import re
import sys
from datetime import datetime

import boto3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import COMPLIANCE_LOGS_DIR, FEAT_M2, FEAT_M3, FEAT_M4  # noqa: F401
from scripts.grounding import check_narrative, correction_prompt
from scripts.pii import mask_name, mask_pan

BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
BEDROCK_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

# Retained for reporting only -- vague hedging is a style observation, not grounds to
# reject a filing. Note the deliberate absence of a bare "may", which collides with the
# month name and used to discard valid reports mentioning a date in May.
SPECULATIVE_TERMS = ["might", "possibly", "could", "appears", "may be", "may have", "may indicate"]

def generate_sar_narrative(txn, p_m2, p_m3, p_m4, meta_score, client=None):
    """
    Constructs the prompt and queries Amazon Bedrock to generate a definitive 5W+H STR narrative.
    """
    if client is None or isinstance(client, str):
        client = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)
    # 1. Voting Breakdown
    votes = []
    if p_m2 > 0.5: votes.append("Model 2 (Geographic Focus)")
    if p_m3 > 0.5: votes.append("Model 3 (Category Focus)")
    if p_m4 > 0.5: votes.append("Model 4 (Velocity Focus)")
    voting_str = ", ".join(votes) if votes else "None (flagged by meta-model ensembling)"
    
    # 2. System and User Prompt
    system_instruction = (
        "You are an expert FINTRAC compliance officer drafting a Suspicious Transaction "
        "Report (STR) under the Proceeds of Crime (Money Laundering) and Terrorist "
        "Financing Act (PCMLTFA) and FINTRAC guidelines.\n"
        "Your narrative must follow the 5W+H framework (Who, What, When, Where, Why, How).\n"
        "The WHY section must frame the analysis around the Canadian legal threshold "
        "'Reasonable Grounds to Suspect' (RGS), citing the PCMLTFA and FINTRAC guidelines.\n"
        "GROUNDING RULES -- these govern everything else:\n"
        "1. Every figure you state must appear in the transaction payload below, or be "
        "arithmetic derived from figures in it. Never invent a number.\n"
        "2. Never assert anything about data you were not given. You have no prior "
        "transaction, no previous location, no travel time, no device or IP telemetry, "
        "no linked accounts, no KYC record and no case history. Do not reference them.\n"
        "3. Do not compare measurements taken over different time windows as though they "
        "were the same quantity (a 7-day count is not a rate relative to a 24-hour count).\n"
        "4. Separate observation from inference. State the observed facts, then state what "
        "they ground a suspicion of. An STR records reasonable grounds to SUSPECT; it does "
        "not assert proven conclusions, and overstating certainty is a defect, not a "
        "virtue. Write plainly and avoid vague hedging, but never claim more than the data "
        "supports.\n"
        "Output exactly the sections WHO, WHAT, WHEN, WHERE, WHY, HOW, with no preamble "
        "or postscript."
    )
    
    prompt = f"""Transaction Data:
- Transaction Number: {txn.get('trans_num', 'N/A')}
- Date/Time: {str(txn.get('trans_date_trans_time', 'N/A'))}
- Credit Card (masked): {mask_pan(txn.get('cc_num'))}
- Customer: {mask_name(txn.get('first'), txn.get('last'))} (Gender: {txn.get('gender', 'N/A')}, Job: {txn.get('job', 'N/A')})
- Customer Location: {txn.get('city', 'N/A')}, {txn.get('state', 'N/A')} {txn.get('zip', 'N/A')} (Lat/Long: {txn.get('lat', 'N/A')}/{txn.get('long', 'N/A')})
- Merchant: {txn.get('merchant', 'N/A')} (Category: {txn.get('category', 'N/A')}, Lat/Long: {txn.get('merch_lat', 'N/A')}/{txn.get('merch_long', 'N/A')})
- Amount: ${txn.get('amt', 0.0):.2f}

Engineered Signals:
- Distance from home: {txn.get('distance_km', 0.0):.2f} km
- Night Transaction: {'Yes' if txn.get('night', 0) == 1 else 'No'}
- Card Mean Transaction: ${txn.get('card_mean_amt', 0.0):.2f}
- Transaction Amount Z-score: {txn.get('amt_z_card', 0.0):.2f}
- Card 24h Transaction Velocity: {txn.get('txns_24h', 0)}
- Card 7d Transaction Velocity: {txn.get('txns_7d', 0)}
- Transaction Category Risk: {txn.get('category_risk', 0.0):.6f}

Stacked Ensemble Results:
- Model 2 (Geographic Focus) Score: {p_m2:.4f}
- Model 3 (Category Focus) Score: {p_m3:.4f}
- Model 4 (Velocity Focus) Score: {p_m4:.4f}
- Stacked Meta-Model Score: {meta_score:.4f}
- Flagging Models: {voting_str}

Generate the FINTRAC STR 5W+H narrative now. Use only the figures above. The WHY section must cite the PCMLTFA and FINTRAC guidelines and explain which of these observations ground a reasonable suspicion, and of what."""
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]
    
    max_retries = 3
    
    for attempt in range(max_retries + 1):
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "system": system_instruction,
            "messages": messages,
            "temperature": 0.1
        })
        
        print(f"Sending request to AWS Bedrock us-east-1 (Attempt {attempt + 1}/{max_retries + 1})...")
        try:
            response = client.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                body=body
            )
            response_body = json.loads(response.get('body').read())
            narrative = response_body['content'][0]['text']
            
            # Factual grounding review, not a vocabulary filter. See scripts/grounding.py
            # and docs/str-narrative-design.md.
            report = check_narrative(narrative, txn, p_m2, p_m3, p_m4, meta_score)

            if report.ok:
                print(f"SUCCESS: narrative passed grounding review on attempt "
                      f"{attempt + 1}. {report.summary()}")
                return narrative

            print(f"WARNING: attempt {attempt + 1} failed grounding review. {report.summary()}")
            if attempt < max_retries:
                messages.append({
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": narrative
                        }
                    ]
                })
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": correction_prompt(report)
                        }
                    ]
                })
            else:
                print("Max retries reached; returning last narrative for quarantine.")
                return narrative
        except Exception as e:
            print(f"ERROR calling AWS Bedrock API on attempt {attempt + 1}: {e}")
            if attempt == max_retries:
                return None

def save_sar_report(txn, p_m2, p_m3, p_m4, meta_score, narrative, output_dir=None):
    if output_dir is None:
        output_dir = COMPLIANCE_LOGS_DIR
    os.makedirs(output_dir, exist_ok=True)

    # 1. Factual grounding review.
    #
    # A narrative that fails is quarantined rather than discarded. Silently dropping
    # it meant an alert that reached the reporting stage left no trace at all, which
    # is a worse compliance posture than an imperfect draft in a review queue.
    report = check_narrative(narrative, txn, p_m2, p_m3, p_m4, meta_score)
    if not report.ok:
        quarantine_dir = os.path.join(output_dir, "quarantine")
        os.makedirs(quarantine_dir, exist_ok=True)
        q_name = (f"QUARANTINE_{txn.get('trans_num', 'unknown')}_"
                  f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        q_path = os.path.join(quarantine_dir, q_name)
        with open(q_path, "w", encoding="utf-8") as f:
            f.write(
                "REJECTED BY GROUNDING REVIEW\n"
                f"{report.summary()}\n"
                f"Transaction: {txn.get('trans_num', 'unknown')}\n"
                f"Meta-score: {meta_score:.4f}\n"
                f"{'=' * 70}\n{narrative}\n")
        print(f"QUARANTINED: narrative held for review at {q_path}")
        raise ValueError(
            f"Grounding review failed: {report.summary()} "
            f"Narrative quarantined at {q_path} for manual review.")


    trans_num = txn.get('trans_num', 'unknown')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"STR_{trans_num}_{timestamp}.txt"
    file_path = os.path.join(output_dir, file_name)
    
    report_content = f"""======================================================================
SUSPICIOUS TRANSACTION REPORT (STR) -- CONFIDENTIAL (FINTRAC)
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
======================================================================
METADATA:
Transaction Number:     {trans_num}
Credit Card (masked):   {mask_pan(txn.get('cc_num'))}
Customer Name:          {mask_name(txn.get('first'), txn.get('last'))}
Transaction Amount:     ${txn.get('amt', 0.0):.2f}
Merchant Name:          {txn.get('merchant', 'N/A')}
Merchant Category:      {txn.get('category', 'N/A')}
Distance from Home:     {txn.get('distance_km', 0.0):.2f} km

GROUNDING REVIEW:       {report.summary()}

ENSEMBLE INFERENCE SIGNALS:
Model 2 (Geographic):   {p_m2:.4f}
Model 3 (Category):     {p_m3:.4f}
Model 4 (Velocity):     {p_m4:.4f}
Stacked Meta-Score:     {meta_score:.4f}
======================================================================
NARRATIVE:
{narrative}
======================================================================
"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"STR report written to: {file_path}")

    # S3 Upload logic
    s3_bucket = os.environ.get("COMPLIANCE_S3_BUCKET")
    if s3_bucket:
        print(f"COMPLIANCE_S3_BUCKET is configured. Uploading STR to S3 bucket: {s3_bucket}...")
        try:
            s3_client = boto3.client('s3')
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=file_name,
                Body=report_content.encode('utf-8')
            )
            print(f"SUCCESS: STR report uploaded to S3 bucket '{s3_bucket}' with key '{file_name}'.")
        except Exception as e:
            print(f"ERROR: Failed to upload STR report to S3: {e}")
            
    return file_path

def _highest_risk_test_transaction(sample_size):
    """Score a slice of the real test split and return its riskiest row.

    The agent is always driven by genuine model output -- there is no mode in which
    a narrative is written from invented ensemble scores.
    """
    import duckdb
    import numpy as np

    from scripts.config import ARTIFACTS_DIR, DB_PATH
    from scripts.inference_engine import load_models, run_inference

    con = duckdb.connect(str(DB_PATH))
    try:
        df = con.execute(f"""
            SELECT f.*, t.first, t.last, t.gender, t.street, t.city, t.state, t.zip,
                   t.lat, t.long, t.merchant, t.category, t.merch_lat, t.merch_long, t.job
            FROM fct_fraud_features f
            LEFT JOIN stg_transactions_test t ON f.trans_num = t.trans_num
            WHERE f.dataset_split = 'test'
            LIMIT {int(sample_size)}
        """).df()
    finally:
        con.close()

    meta_probs, p_m2, p_m3, p_m4, _ = run_inference(df, *load_models(ARTIFACTS_DIR))
    i = int(np.argmax(meta_probs))
    return (df.iloc[i].to_dict(), float(p_m2[i]), float(p_m3[i]),
            float(p_m4[i]), float(meta_probs[i]))


def main():
    parser = argparse.ArgumentParser(
        description="Generate a FINTRAC STR narrative for the riskiest scored transaction.")
    parser.add_argument('--sample-size', type=int, default=500,
                        help='How many test-split rows to score before picking the riskiest.')
    parser.add_argument('--output-dir', type=str, default=str(COMPLIANCE_LOGS_DIR))
    args = parser.parse_args()

    print("=== STR AGENT ===")
    print(f"Scoring {args.sample_size} test transactions with the production ensemble...")
    txn, p_m2, p_m3, p_m4, meta_score = _highest_risk_test_transaction(args.sample_size)
    print(f"Riskiest transaction: {txn['trans_num']}  meta-score={meta_score:.4f}  "
          f"(base: {p_m2:.4f} / {p_m3:.4f} / {p_m4:.4f})  "
          f"ground-truth is_fraud={int(txn.get('is_fraud', -1))}")

    narrative = generate_sar_narrative(txn, p_m2, p_m3, p_m4, meta_score)
    if not narrative:
        print("Failed to generate narrative. Verify AWS credentials and Bedrock access.")
        return

    print("\nGenerated STR Narrative:\n")
    print(narrative)
    print("=" * 60)

    try:
        save_sar_report(txn, p_m2, p_m3, p_m4, meta_score, narrative, args.output_dir)
    except ValueError as e:
        print(f"COMPLIANCE_ERROR: {e}")


if __name__ == "__main__":
    main()
