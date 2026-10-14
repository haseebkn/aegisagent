import argparse
import json
import math
import os
import sys
from datetime import datetime

import boto3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.case_management import CaseStore, ReviewerRole
from scripts.config import COMPLIANCE_LOGS_DIR, EVIDENCE_DIR
from scripts.evidence import EvidenceStore
from scripts.grounding import check_narrative, correction_prompt
from scripts.inference_engine import NoAlertsInSample
from scripts.pii import mask_pan
from scripts.privacy import minimize_for_narrative, redact_sensitive_text, safe_error_message
from scripts.security import SecurityPrincipal
from scripts.security import AuthenticationRequired, local_development_principal

BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
BEDROCK_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

def generate_sar_narrative(txn, p_m2, p_m3, p_m4, meta_score, client=None):
    """
    Generate a grounded 5W+H investigation narrative draft for human review.

    The legacy function name is retained for compatibility. Its output is neither an
    RGS determination nor a completed or submitted FINTRAC STR.
    """
    if client is None or isinstance(client, str):
        client = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)
    # Strict minimization happens before string interpolation so prohibited fields
    # cannot enter the external model request accidentally.
    txn = minimize_for_narrative(txn)

    # 1. Voting Breakdown
    votes = []
    if p_m2 > 0.5: votes.append("Model 2 (Geographic Focus)")
    if p_m3 > 0.5: votes.append("Model 3 (Category Focus)")
    if p_m4 > 0.5: votes.append("Model 4 (Velocity Focus)")
    voting_str = ", ".join(votes) if votes else "None (flagged by meta-model ensembling)"
    
    # 2. System and User Prompt
    system_instruction = (
        "You are assisting a Canadian financial-crime investigator by drafting a 5W+H "
        "investigation narrative from the supplied transaction and model signals. This "
        "is a decision-support draft, not a completed or submitted FINTRAC STR.\n"
        "Your narrative must follow the 5W+H framework (Who, What, When, Where, Why, How).\n"
        "The WHY section must identify which observations merit human assessment under "
        "the Canadian reasonable-grounds-to-suspect (RGS) framework. Do not state that "
        "RGS has been reached and do not recommend or claim that a filing is required.\n"
        "GROUNDING RULES -- these govern everything else:\n"
        "1. Every figure you state must appear in the transaction payload below, or be "
        "arithmetic derived from figures in it. Never invent a number.\n"
        "2. Never assert anything about data you were not given. You have no prior "
        "transaction, no previous location, no travel time, no device or IP telemetry, "
        "no linked accounts, no KYC record and no case history. Do not reference them.\n"
        "3. Do not compare measurements taken over different time windows as though they "
        "were the same quantity (a 7-day count is not a rate relative to a 24-hour count).\n"
        "4. Separate observation from inference. State the observed facts, then state what "
        "they may support further assessment of. The reporting entity's authorized human "
        "reviewer—not this model—determines whether RGS is reached. Do not assert proven "
        "conclusions, and remember that overstating certainty is a defect, not a "
        "virtue. Write plainly and avoid vague hedging, but never claim more than the data "
        "supports.\n"
        "Output exactly the sections WHO, WHAT, WHEN, WHERE, WHY, HOW, with no preamble "
        "or postscript."
    )
    
    prompt = f"""Transaction Data:
- Transaction Number: {txn.get('trans_num', 'N/A')}
- Date/Time: {str(txn.get('trans_date_trans_time', 'N/A'))}
- Credit Card (masked): {mask_pan(txn.get('cc_num'))}
- Customer reference: cardholder associated with {mask_pan(txn.get('cc_num'))}
- Merchant: {txn.get('merchant', 'N/A')} (Category: {txn.get('category', 'N/A')})
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

Generate the 5W+H investigation narrative draft now. Use only the figures above. The WHY section may identify observations relevant to an investigator's RGS assessment, but it must not claim that RGS has been reached or that an STR must be filed."""
    
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
            narrative = redact_sensitive_text(response_body['content'][0]['text'])
            
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
            print(
                f"ERROR calling AWS Bedrock API on attempt {attempt + 1}: "
                f"{safe_error_message(e)}"
            )
            if attempt == max_retries:
                return None

def save_sar_report(
    txn,
    p_m2,
    p_m3,
    p_m4,
    meta_score,
    narrative,
    output_dir=None,
    *,
    case_id,
    model_version,
    actor=None,
    actor_role=None,
    principal: SecurityPrincipal | None = None,
    expected_case_version,
    case_store=None,
    evidence_store=None,
    s3_client=None,
    archive_bucket=None,
):
    """Preserve and link a grounded draft (or quarantine evidence) to an active case."""
    case_store = case_store or CaseStore()
    case, identity = case_store.authorize_evidence_attachment(
        case_id,
        principal=principal,
        actor=actor,
        actor_role=actor_role,
        expected_version=expected_case_version,
    )
    if case.trans_num != str(txn.get("trans_num", "")):
        raise ValueError(
            f"Case {case_id} belongs to transaction {case.trans_num}, not "
            f"{txn.get('trans_num', 'unknown')}"
        )
    if model_version != case.model_version or not math.isclose(
        meta_score, case.model_score, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError("Narrative model version and score must match the case alert")
    narrative = redact_sensitive_text(narrative)
    if evidence_store is None:
        evidence_root = EVIDENCE_DIR if output_dir is None else os.path.join(output_dir, "evidence")
        evidence_store = EvidenceStore(evidence_root)
    report = check_narrative(narrative, txn, p_m2, p_m3, p_m4, meta_score)
    trans_num = txn.get("trans_num", "unknown")
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")

    if not report.ok:
        evidence_type = "narrative_quarantine"
        report_content = (
            "REJECTED BY GROUNDING REVIEW\n"
            f"Preserved for human review: {generated_at}\n"
            f"{report.summary()}\n"
            f"Transaction: {trans_num}\n"
            f"Meta-score: {meta_score:.4f}\n"
            f"{'=' * 70}\n{narrative}\n"
        )
    else:
        evidence_type = "narrative_draft"
        report_content = f"""======================================================================
INVESTIGATION NARRATIVE DRAFT -- NOT A FINTRAC FILING
Generated for human review: {generated_at}
Status: DRAFT / NOT APPROVED / NOT SUBMITTED
======================================================================
METADATA:
Transaction Number:     {trans_num}
Case Model Version:     {case.model_version}
Case Alert Threshold:   {case.threshold:.4f}
Credit Card (masked):   {mask_pan(txn.get('cc_num'))}
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
    s3_bucket = os.environ.get("COMPLIANCE_S3_BUCKET") if archive_bucket is None else archive_bucket
    receipt = evidence_store.preserve_text(
        case_id=case_id,
        evidence_type=evidence_type,
        content=report_content,
        archive_bucket=s3_bucket,
        s3_client=s3_client,
    )
    case_store.attach_evidence(
        case_id,
        receipt=receipt,
        actor=actor,
        actor_role=actor_role,
        principal=identity,
        expected_version=expected_case_version,
    )
    archive = receipt.archive_receipt
    if archive and archive.get("verified"):
        print(
            f"VERIFIED ARCHIVE: s3://{archive['bucket']}/{archive['key']} "
            f"version={archive['version_id']} checksum={receipt.sha256}"
        )
    elif archive:
        print(
            "UNVERIFIED ARCHIVE ATTEMPT: "
            f"{archive['error_type']}: {safe_error_message(archive['error'])}"
        )
    print(
        f"Evidence {receipt.evidence_id} linked to {case_id}: {receipt.local_path} "
        f"sha256={receipt.sha256}"
    )
    if not report.ok:
        raise ValueError(
            f"Grounding review failed: {report.summary()} Narrative preserved as "
            f"evidence {receipt.evidence_id} at {receipt.local_path}."
        )
    return receipt.local_path

def _highest_risk_alert(sample_size):
    """Score a slice of the development holdout and return its riskiest alert.

    The agent is always driven by genuine model output, and only ever reports on a
    transaction the model actually flagged -- there is no mode in which a narrative
    is written from invented scores, or from a transaction below the threshold.
    """
    import duckdb

    from scripts.config import ARTIFACTS_DIR, DB_PATH, resolve_model_dir
    from scripts.inference_engine import (load_models, run_inference,
                                          select_highest_risk_alert)

    con = duckdb.connect(str(DB_PATH))
    try:
        df = con.execute(f"""
            SELECT f.*, t.merchant, t.category
            FROM fct_fraud_features f
            LEFT JOIN stg_transactions_test t ON f.trans_num = t.trans_num
            WHERE f.evaluation_role = 'development_holdout'
            LIMIT {int(sample_size)}
        """).df()
    finally:
        con.close()

    model_dir = resolve_model_dir(ARTIFACTS_DIR)
    meta_probs, p_m2, p_m3, p_m4, triggered = run_inference(df, *load_models(model_dir))
    i = select_highest_risk_alert(meta_probs, triggered)
    return (df.iloc[i].to_dict(), float(p_m2[i]), float(p_m3[i]),
            float(p_m4[i]), float(meta_probs[i]), int(triggered.sum()), len(df), model_dir.name)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a 5W+H investigation narrative draft for the riskiest alert.")
    parser.add_argument('--sample-size', type=int, default=10000,
                        help='Test-split rows to score before picking the riskiest alert. '
                             'At 0.39%% prevalence a few hundred rows usually contain no '
                             'alert at all.')
    parser.add_argument('--output-dir', type=str, default=str(COMPLIANCE_LOGS_DIR))
    parser.add_argument('--case-id', required=True,
                        help='Existing under-review case for the selected alert.')
    parser.add_argument('--actor', required=True, help='Identified reviewer preserving evidence.')
    parser.add_argument('--role', choices=[role.value for role in ReviewerRole], required=True)
    parser.add_argument('--expected-case-version', type=int, required=True)
    args = parser.parse_args()

    try:
        principal = local_development_principal(
            subject=args.actor,
            roles=[args.role],
        )
    except AuthenticationRequired as exc:
        raise SystemExit(f"AUTHENTICATION_REQUIRED: {exc}") from exc

    print("=== INVESTIGATION NARRATIVE DRAFTING ASSISTANT ===")
    print(f"Scoring {args.sample_size} development-holdout transactions with demo artifacts...")
    try:
        txn, p_m2, p_m3, p_m4, meta_score, n_alerts, n_scored, model_version = _highest_risk_alert(
            args.sample_size)
    except NoAlertsInSample as e:
        print(f"No narrative draft generated: {e}")
        raise SystemExit(1)
    print(f"{n_alerts} alert(s) in {n_scored:,} scored transactions.")
    print(f"Riskiest alert: {txn['trans_num']}  meta-score={meta_score:.4f}  "
          f"(base: {p_m2:.4f} / {p_m3:.4f} / {p_m4:.4f})  "
          f"ground-truth is_fraud={int(txn.get('is_fraud', -1))}")

    narrative = generate_sar_narrative(txn, p_m2, p_m3, p_m4, meta_score)
    if not narrative:
        print("Failed to generate narrative. Verify AWS credentials and Bedrock access.")
        return

    print(
        "Narrative generated in memory; content is not written to stdout. "
        "Proceeding to grounding review and protected evidence storage."
    )

    try:
        save_sar_report(
            txn,
            p_m2,
            p_m3,
            p_m4,
            meta_score,
            narrative,
            args.output_dir,
            case_id=args.case_id,
            model_version=model_version,
            principal=principal,
            expected_case_version=args.expected_case_version,
            case_store=CaseStore(principal=principal),
        )
    except ValueError as e:
        print(f"DRAFT_VALIDATION_ERROR: {e}")


if __name__ == "__main__":
    main()
