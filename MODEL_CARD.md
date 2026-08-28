# AegisAgent model card

**Version:** 0.1.0  
**Status:** portfolio/research baseline  
**Last updated:** 2026-08-27

## Model overview

AegisAgent is a stacked binary classifier for the fraud label in the public Sparkov
synthetic credit-card transaction dataset. Three base models—a geographic random
forest, a category-focused XGBoost model, and a velocity random forest—feed a logistic
regression meta-model. The selected demo threshold is 0.6152.

## Intended use

- Demonstrate tabular fraud modeling, dbt feature engineering, evaluation under
  class imbalance, drift/calibration analysis, and an investigation-support workflow.
- Rank historical synthetic transactions and create demo investigation alerts.
- Support portfolio discussion about model governance and system limitations.

## Out-of-scope and prohibited interpretations

- Real payment authorization, account blocking, adverse action, or autonomous case
  disposition.
- A determination that fraud, money laundering, terrorist financing, or sanctions
  evasion occurred.
- A reasonable-grounds-to-suspect determination or automated FINTRAC reporting.
- Claims of production readiness, real-time serving, regulatory compliance, or
  performance on real Canadian financial-institution data.

## Data

The model uses the Kaggle-hosted Sparkov synthetic dataset: 1,296,675 training rows
and 555,719 later-period rows. The later split contains 2,145 positives (0.386%).
Synthetic generator artifacts can make the classification task materially easier
than real fraud detection, so absolute metrics are not transferable.

## Evaluation results

On the later **development holdout**:

| Metric | Value |
|---|---:|
| PR AUC | 0.7823 |
| ROC AUC | 0.9912 |
| Precision at threshold 0.6152 | 85.9% |
| Recall at threshold 0.6152 | 67.9% |
| F1 | 0.759 |
| Alerts | 1,696 over 193 days (8.8/day) |

These are not final blind-test estimates. The later split has been inspected and used
to inform feature decisions, including graph-feature rejection and encoding changes.
A new time-forward holdout is required for final evaluation. Confidence intervals
have not yet been reported.

## Known methodological limitations

- Random-fold target encodings use totals from the full training period; earlier rows
  can indirectly incorporate later labels from other folds.
- Per-card count, mean, and standard deviation are full-window training aggregates,
  so earlier rows see later behavior from the same card.
- Velocity windows include the current transaction even though some documentation
  historically described preceding transactions.
- The alert-region score is overconfident by about seven percentage points and should
  be treated as a ranking score, not a calibrated probability.
- No subgroup, stability-over-time, uncertainty, or external-validity study is yet
  sufficient for operational use.

## Regulatory and human-oversight limitations

The training target is card fraud, not money laundering or terrorist financing. A
threshold breach creates a model alert only. An authorized reporting-entity reviewer
must independently evaluate facts, context, and relevant indicators and decide
whether RGS is reached. See [docs/regulatory-scope.md](docs/regulatory-scope.md).

The optional LLM output is an unapproved narrative draft. Numeric grounding reduces
some hallucinations but does not prove entailment or completeness. The project does
not implement a complete FINTRAC form, approval workflow, or submission.

## Privacy and security limitations

Names and PANs are masked before the Bedrock request, but location, occupation,
gender, and other quasi-identifiers remain. The demo has no authentication or RBAC,
and its HTML rendering and archival behavior require hardening. It must use synthetic
data only in its current state.

## Deployment status

Nothing is serving production traffic. Streamlit runs locally. The Docker image runs
an end-to-end verifier and exits. Terraform defines an AWS reference scaffold with an
ECS desired count of zero; there is no public scoring API, load balancer, operational
SLA, or verified incident-response process.

## Next validation gate

Before updating performance claims: rebuild all temporal features causally, freeze
the design, evaluate once on a new blind time-forward period, report confidence
intervals and subgroup/time slices, and document the alert-capacity operating point.
