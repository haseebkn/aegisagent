# AegisAgent model card

**Version:** 0.3.0

**Status:** portfolio/research baseline

**Last updated:** 2026-08-28

## Model overview

AegisAgent is a stacked binary classifier for the fraud label in the public Sparkov
synthetic credit-card transaction dataset. Three base models—a geographic random
forest, a category-focused XGBoost model, and a velocity random forest—feed a logistic
regression meta-model. The current calibration-selected demo threshold is 0.7868.

## Intended use

- Demonstrate causal tabular feature engineering, temporal validation, evaluation
  under class imbalance, drift/calibration analysis, and investigation support.
- Rank historical synthetic transactions and create demo investigation alerts.
- Support portfolio discussion about model governance and system limitations.

## Out-of-scope and prohibited interpretations

- Real payment authorization, account blocking, adverse action, or autonomous case
  disposition.
- A determination that fraud, money laundering, terrorist financing, or sanctions
  evasion occurred.
- A reasonable-grounds-to-suspect determination or automated FINTRAC reporting.
- Claims of production readiness, regulatory compliance, or performance on real
  Canadian financial-institution data.

## Data and temporal protocol

The model uses the Kaggle-hosted Sparkov synthetic dataset: 1,296,675 training rows
and 555,719 later-period rows. Model fitting divides the training file chronologically
into 70% base-model, 15% meta-model, and 15% threshold-selection windows.

The later file is split chronologically into 416,789 development rows and a final
138,930-row evaluation tail. The tail was prospectively locked in Phase 1, but is not
historically pristine because earlier project versions reported aggregate metrics
over the entire file. A new external time-forward dataset remains necessary.

## Feature availability

- Target encodings use only earlier training labels, with smoothing toward an
  earlier-only global prior.
- Card count, mean, and standard deviation use the strictly preceding 90 days.
- Transaction velocity uses strictly preceding 24-hour and 7-day windows.
- Amount z-score is neutral until five prior transactions exist.
- Scoring target encodings are frozen from the training period and never consume
  scoring labels.

All 15 full-data dbt tests pass, including independent reconstruction of temporal
encodings and card history.

## Evaluation results

| Metric | Development | Prospectively locked tail |
|---|---:|---:|
| Rows / positives | 416,789 / 1,891 | 138,930 / 254 |
| Ensemble PR AUC | 0.7766 | 0.6711 |
| Day-block bootstrap 95% CI | 0.7511–0.8020 | 0.5847–0.7630 |
| ROC AUC | 0.9905 | 0.9882 |
| Precision at 0.7868 | 85.9% | 78.0% |
| Recall at 0.7868 | 62.3% | 55.9% |
| Alerts/day | 8.5 | 6.1 |
| Alert-region calibration gap | +10.1 points | +17.0 points |

Three expanding-window validations produce PR AUC 0.7655, 0.7215, and 0.8373
(mean 0.7748; worst 0.7215). A simple logistic baseline reaches PR AUC 0.3140 on
development and 0.1636 on the locked tail. Reports are in [reports/](reports/).

## Operating-point limitations

The 0.7868 threshold maximizes F1 on a dedicated calibration window; it is not
automatically the business-optimal threshold. On development:

- Current threshold: 8.5 alerts/day, 85.9% precision, 62.3% recall.
- Ten-alert/day capacity point: threshold 0.5261, 78.0% precision, 66.3% recall.
- Modeled cost minimum: threshold 0.0206, 18.3 alerts/day, 51.9% precision,
  81.3% recall.

The cost model is illustrative and assumes a $25 review cost and fraud amount as the
false-negative loss. It is not institution-specific economics.

## Known methodological limitations

- Synthetic generator artifacts limit external validity and likely make the task
  easier than real-world fraud detection.
- The locked tail is prospective only from Phase 1, not historically untouched.
- The score is materially overconfident where investigators see it and must be
  treated as a ranking, not a probability.
- Supported subgroup slices show varying recall, including zero-recall categories
  with few positives. They are diagnostic, not a fairness certification.
- Drift has zero significant, one moderate, and 18 stable features. The moderate
  feature is 90-day card transaction count (PSI 0.1705).
- No external validation, analyst feedback loop, champion/challenger operation, or
  causal estimate of intervention benefit exists.

## Regulatory and human-oversight limitations

The target is card fraud, not money laundering or terrorist financing. A threshold
breach creates a model alert only. An authorized reporting-entity reviewer must
evaluate facts, context, and relevant indicators and decide whether RGS is reached.
See [docs/regulatory-scope.md](docs/regulatory-scope.md).

The Phase 2 demo persists threshold-breaching alerts as cases and enforces the path
`alert_open → under_review → rgs_not_reached / rgs_reached`. Starting review requires
an identified investigator and rationale; only the `authorized_rgs_reviewer` role can
record a terminal disposition. Every transition records an actor, role, rationale,
timestamp, and case version. An RGS-reached disposition does not create or submit a
report—it hands off to an approved reporting workflow outside this project.

The optional LLM output is an unapproved narrative draft available during active
human review. Numeric grounding reduces
some hallucinations but does not prove entailment or completeness. The project does
not implement a complete FINTRAC form, reporting approval, submission, receipt
tracking, or amendment workflow.

## Privacy, security, and deployment limitations

Names and PANs are masked before the Bedrock request, but location, occupation,
gender, and other quasi-identifiers remain. The demo has no authentication or
production RBAC: its reviewer identity and role are self-attested UI/CLI inputs. Its
SQLite history is append-only through the application API, not tamper-evident or a
compliant record archive. HTML rendering, encryption, backup, retention, access
control, and archival behavior require hardening. It must use synthetic data only in
its current state.

Nothing serves production traffic. Streamlit runs locally. The Docker image runs a
verifier and exits. Terraform defines an AWS reference scaffold with an ECS desired
count of zero; there is no scoring API, load balancer, operational SLA, or verified
incident-response process.

## Next validation gate

Acquire or generate a genuinely external later-period dataset, freeze the entire
pipeline before access, recalibrate on a separate window, define subgroup acceptance
criteria with domain owners, and document the institution-specific alert-capacity
and loss model before making any operational claim.
