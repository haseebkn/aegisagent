import os
import argparse
import requests
import json
import re
import boto3
from datetime import datetime
import numpy as np

# Align features list
FEAT_M2 = ['amt', 'distance_km', 'night', 'hour_sin', 'hour_cos', 'category_risk']
FEAT_M3 = ['amt', 'log_amt', 'distance_km', 'night', 'hour', 'day_of_week', 
           'hour_sin', 'hour_cos', 'category_risk', 'state_risk', 
           'card_txn_cnt', 'card_mean_amt', 'card_std_amt']
FEAT_M4 = ['amt', 'log_amt', 'distance_km', 'night', 'hour_sin', 'hour_cos', 'day_of_week', 
           'is_online', 'category_risk', 'state_risk', 'merchant_risk', 
           'card_txn_cnt', 'card_mean_amt', 'card_std_amt', 'txns_24h', 'txns_7d', 
           'amt_x_catRisk', 'dist_x_online']

def generate_sar_narrative(txn, p_m2, p_m3, p_m4, meta_score, client=None):
    """
    Constructs the prompt and queries Amazon Bedrock to generate a definitive 5W+H STR narrative.
    """
    if client is None or isinstance(client, str):
        client = boto3.client('bedrock-runtime', region_name='us-east-1')
    # 1. Voting Breakdown
    votes = []
    if p_m2 > 0.5: votes.append("Model 2 (Geographic Focus)")
    if p_m3 > 0.5: votes.append("Model 3 (Category Focus)")
    if p_m4 > 0.5: votes.append("Model 4 (Velocity Focus)")
    voting_str = ", ".join(votes) if votes else "None (flagged by meta-model ensembling)"
    
    # 2. System and User Prompt
    system_instruction = (
        "You are an expert FINTRAC compliance officer writing a definitive, legally compliant Suspicious Transaction Report (STR) under the Proceeds of Crime (Money Laundering) and Terrorist Financing Act (PCMLTFA) and FINTRAC guidelines.\n"
        "Your narrative must follow the 5W+H framework (Who, What, When, Where, Why, How).\n"
        "The WHY section must frame the analysis explicitly around the Canadian legal threshold: 'Reasonable Grounds to Suspect' (RGS), citing the Proceeds of Crime (Money Laundering) and Terrorist Financing Act (PCMLTFA) and FINTRAC guidelines. Focus purely on objective, factual anomalies (velocity spikes, geographic impossibility) with zero speculative language.\n"
        "Crucial Constraint: Use objective, direct, and definitive language. Do NOT use speculative or hedging words (such as 'may', 'might', 'possibly', 'could', 'appears'). State the findings as direct facts.\n"
        "You must output exactly the sections: WHO, WHAT, WHEN, WHERE, WHY, HOW. Do not add any conversational preamble or postscript."
    )
    
    prompt = f"""Transaction Data:
- Transaction Number: {txn.get('trans_num', 'N/A')}
- Date/Time: {str(txn.get('trans_date_trans_time', 'N/A'))}
- Credit Card: {txn.get('cc_num', 'N/A')}
- Customer: {txn.get('first', 'N/A')} {txn.get('last', 'N/A')} (Gender: {txn.get('gender', 'N/A')}, Job: {txn.get('job', 'N/A')})
- Customer Location: {txn.get('street', 'N/A')}, {txn.get('city', 'N/A')}, {txn.get('state', 'N/A')} {txn.get('zip', 'N/A')} (Lat/Long: {txn.get('lat', 'N/A')}/{txn.get('long', 'N/A')})
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

Generate the FINTRAC-compliant STR 5W+H narrative now, ensuring the WHY section explicitly cites the PCMLTFA and FINTRAC guidelines alongside the Reasonable Grounds to Suspect (RGS) threshold with objective factual anomalies only."""
    
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
    
    blacklist = ["may", "might", "possibly", "could", "appears"]
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
                modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
                body=body
            )
            response_body = json.loads(response.get('body').read())
            narrative = response_body['content'][0]['text']
            
            # Programmatic check using word boundaries
            flagged = []
            for word in blacklist:
                if re.search(rf"\b{word}\b", narrative, re.IGNORECASE):
                    flagged.append(word)
            
            if not flagged:
                print(f"SUCCESS: Narrative passed speculative language check on attempt {attempt + 1}.")
                return narrative
            
            print(f"WARNING: Attempt {attempt + 1} generated speculative language: {flagged}")
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
                            "text": f"Error: The previous response contained forbidden speculative language: {flagged}.\nRewrite the narrative focusing ONLY on definitive, objective facts. Do NOT use speculative or hedging words (such as 'may', 'might', 'possibly', 'could', 'appears'). State all findings as established facts."
                        }
                    ]
                })
            else:
                print("Max retries reached. Returning the last generated narrative.")
                return narrative
        except Exception as e:
            print(f"ERROR calling AWS Bedrock API on attempt {attempt + 1}: {e}")
            if attempt == max_retries:
                return None

def save_sar_report(txn, p_m2, p_m3, p_m4, meta_score, narrative, output_dir=None):
    # 1. Speculative language verification (Fix N-03)
    blacklist = ["may", "might", "possibly", "could", "appears"]
    flagged = []
    for word in blacklist:
        if re.search(rf"\b{word}\b", narrative, re.IGNORECASE):
            flagged.append(word)
    if flagged:
        raise ValueError(f"Compliance validation failed: Speculative language detected in narrative: {flagged}. Aborting save routine.")

    if output_dir is None:
        output_dir = os.environ.get("COMPLIANCE_LOGS_DIR", "e:/AegisAgent/compliance_logs")
    os.makedirs(output_dir, exist_ok=True)
    
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
Credit Card:            {txn.get('cc_num', 'N/A')}
Customer Name:          {txn.get('first', 'N/A')} {txn.get('last', 'N/A')}
Transaction Amount:     ${txn.get('amt', 0.0):.2f}
Merchant Name:          {txn.get('merchant', 'N/A')}
Merchant Category:      {txn.get('category', 'N/A')}
Distance from Home:     {txn.get('distance_km', 0.0):.2f} km

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='Inject a test anomalous transaction')
    parser.add_argument('--ollama-url', type=str, default=os.environ.get('OLLAMA_URL', 'http://localhost:11434'))
    parser.add_argument('--output-dir', type=str, default=os.environ.get('COMPLIANCE_LOGS_DIR', 'e:/AegisAgent/compliance_logs'))
    args = parser.parse_args()
    
    if args.test:
        print("=== STR AGENT TEST MODE ===")
        # Build a known anomalous transaction dictionary
        test_txn = {
            'trans_num': 'TXN_TEST_99999',
            'trans_date_trans_time': datetime.now(),
            'cc_num': 1234567890123456,
            'first': 'John',
            'last': 'Doe',
            'gender': 'M',
            'job': 'Compliance Investigator',
            'street': '742 Evergreen Terrace',
            'city': 'Springfield',
            'state': 'IL',
            'zip': '62704',
            'lat': 39.7817,
            'long': -89.6501,
            'merchant': 'fraud_Luxury_Watch_Boutique_POS',
            'category': 'shopping_pos',
            'merch_lat': 48.8566, # Paris, France (Extreme distance from Springfield IL)
            'merch_long': 2.3522,
            'amt': 4999.99, # Extreme amount
            'distance_km': 6645.21, # Distance IL to Paris
            'night': 1, # Night-time
            'card_mean_amt': 45.50, # Usual spend is tiny
            'amt_z_card': 108.88, # Massive Z-score
            'txns_24h': 12, # High velocity
            'txns_7d': 18,
            'category_risk': 0.0125
        }
        
        # Base probabilities representing high anomaly consensus
        p_m2 = 0.9921
        p_m3 = 0.9850
        p_m4 = 0.9912
        meta_score = 0.9995
        
        narrative = generate_sar_narrative(test_txn, p_m2, p_m3, p_m4, meta_score, args.ollama_url)
        if narrative:
            print("\nGenerated STR Narrative:\n")
            print(narrative)
            print("="*60)
            
            # Check for speculative language in the output
            blacklist = ["may", "might", "possibly", "could", "appears"]
            flagged_words = [w for w in blacklist if re.search(rf"\b{w}\b", narrative, re.IGNORECASE)]
            if flagged_words:
                print(f"WARNING: Speculative language detected: {flagged_words}")
            else:
                print("SUCCESS: Checked narrative for speculative terms, none found.")
                
            try:
                save_sar_report(test_txn, p_m2, p_m3, p_m4, meta_score, narrative, args.output_dir)
            except ValueError as e:
                print(f"COMPLIANCE_ERROR: {e}")
        else:
            print("Failed to generate narrative. Make sure AWS credentials are set and Bedrock access is enabled.")

if __name__ == "__main__":
    main()
