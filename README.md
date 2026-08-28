# AegisAgent: Fraud Detection & Investigation Narrative Drafting

AegisAgent scores historical card transactions with a stacked ML ensemble, creates
investigation alerts, and can draft a grounded 5W+H narrative for human review. It is
a portfolio/research project built on a public synthetic dataset—not a production
fraud platform, an AML decision engine, or a FINTRAC filing system. A model alert is
not a reasonable-grounds-to-suspect (RGS) determination. See
[Regulatory scope](docs/regulatory-scope.md), [MODEL_CARD.md](MODEL_CARD.md), and
[Scope & limitations](#scope--limitations).

The models originate from an MSc capstone (*Feature-Enhanced Machine Learning Models
for Credit Card Fraud Detection*, Memorial University of Newfoundland); this repo is
the data-engineering, serving and reporting layer built around them.

---

## Current temporal evaluation

Phase 1 divides `fraudTest.csv` chronologically: the first 75% is the development
holdout and the final 25% is a prospectively locked evaluation tail. The tail is not
historically pristine—older revisions reported aggregate metrics over the entire
file—so it is stronger evidence than another tuning split, but not an external test.

| Metric | Development | Locked tail |
|---|---:|---:|
| Rows / frauds | 416,789 / 1,891 | 138,930 / 254 |
| **Ensemble PR AUC** | **0.7766** | **0.6711** |
| Day-block 95% CI | 0.7511–0.8020 | 0.5847–0.7630 |
| ROC AUC | 0.9905 | 0.9882 |
| Precision at 0.7868 | 85.9% | 78.0% |
| Recall at 0.7868 | 62.3% | 55.9% |
| Alerts/day | 8.5 | 6.1 |

**PR AUC is the headline number.** The locked-window degradation is not hidden by
averaging it into development performance. Three expanding-window validations score
0.7655, 0.7215, and 0.8373 PR AUC (mean 0.7748), demonstrating meaningful temporal
variation. A simple six-feature logistic baseline reaches only 0.3140 development
and 0.1636 locked-tail PR AUC. Full machine-readable evidence is in
[`reports/`](reports/).

---

## How it works

```
fraudTrain.csv ─┐
                ├─► dbt + DuckDB ──► feature mart ──► stacked ensemble ──► alert threshold
fraudTest.csv  ─┘                                                        │
                                                                         ▼
                                            alert case ──► human review ──► RGS disposition
                                                               │                  │
                                                               ▼                  ▼
                                                    optional 5W+H draft   reporting workflow
                                                    (not a filing)         (out of scope)
```

**Feature layer (dbt → DuckDB).** Staging views read the raw CSVs; intermediate
models derive Haversine distance from home, cyclical hour encodings, **prequential
smoothed** category / state / merchant target encodings, prior-only 90-day card
statistics, and strictly preceding 24h/7d transaction velocity. The mart
`fct_fraud_features` is the single
contract the models consume. Intermediates are views by measurement, not by default —
materializing them is 33% slower ([docs/materialization.md](docs/materialization.md)).

**Model layer.** Three base learners (geographic RF, category XGBoost, velocity RF)
feed a logistic-regression meta-learner. Model fitting uses a chronological 70% base /
15% blend / 15% threshold-selection split within `fraudTrain.csv`. All model inputs
are available at event time: target encodings use earlier labels only, card histories
use the prior 90 days, and velocity excludes the current event. `fraudTest.csv` was
historically inspected, so even the prospectively locked tail is not described as a
pristine independent test.

**Human-review layer.** A threshold-breaching transaction can create one persisted
case. The enforced workflow is `alert_open → under_review → rgs_not_reached /
rgs_reached`; assignment and disposition require identified actors and rationales,
and only the asserted `authorized_rgs_reviewer` role can record a terminal RGS
decision. Every transition is appended to local case history with optimistic
concurrency protection. No state represents filing or submission. See
[docs/human-review-workflow.md](docs/human-review-workflow.md).

**Narrative layer.** Narrative drafting is available only while a case is under human
review. The threshold creates an investigation alert, not an RGS determination or
filing obligation. Eligible alerts can be sent to AWS
Bedrock (Claude Haiku 4.5) with a 5W+H drafting prompt. A retry loop enforces
**factual grounding**—every quantity in the
narrative must trace to the payload, and claims about data the pipeline never
supplied (prior transactions, travel times, device telemetry, linked accounts) are
rejected. Narratives that still fail are quarantined for review rather than
discarded. Phase 3 atomically preserves both drafts and quarantines as content-addressed,
SHA-256 evidence linked to the active case. Optional S3 archival is considered verified
only when the response includes the requested checksum and an object VersionId; failed
attempts remain visible and make integrity verification fail. See
[docs/evidence-integrity.md](docs/evidence-integrity.md). Passing drafts remain drafts:
the project does not collect the complete
Schedule 1 data, implement approval, or submit anything to FINTRAC. Card numbers and
names are masked before they leave the process, but other personal and location data
remain in the prompt. The
reasoning behind this control, and why the previous hedging-word blacklist was wrong
in both directions, is in [docs/str-narrative-design.md](docs/str-narrative-design.md).

---

## Setup

**Prerequisites:** Python 3.11+, Docker (optional), AWS credentials with Bedrock
access (only for optional narrative drafting).

```bash
git clone https://github.com/haseebkn/aegisagent.git
cd aegisagent
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download `fraudTrain.csv` and `fraudTest.csv` from the
[Kaggle Credit Card Transactions Fraud Detection dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
into the repo root, then:

```bash
dbt run --profiles-dir .
dbt test --profiles-dir .
python scripts/train_models.py
streamlit run app.py
```

Every path is env-overridable (`DBT_DB_PATH`, `MODELS_ARTIFACTS_DIR`,
`COMPLIANCE_LOGS_DIR`, `AEGIS_CASE_DB_PATH`, `AEGIS_EVIDENCE_DIR`,
`AEGIS_RAW_DATA_DIR`) and defaults to a location relative to the repo root — see
`scripts/config.py`.

### Verification container

The image carries the dbt project, scripts and model artifacts, so `models_artifacts/`
must exist before you build. One version is ~244 MB of joblib, deliberately not in git.
`train_models.py` prunes superseded versions (`--keep`, default 1) so the image never
accumulates dead ones:

```bash
dbt run --profiles-dir .
python scripts/train_models.py
docker build -t aegis-app:latest .
docker run --rm -v "$PWD/aegis_db.duckdb:/app/aegis_db.duckdb" aegis-app:latest
```

The DuckDB file and raw CSVs are data and are mounted at runtime. CI runs this whole
sequence against the fixture dataset on every push, which is how it verifies the
image is genuinely self-contained rather than relying on a bind mount. The image's
command runs `scripts/verify_pipeline.py` and exits; it does **not** serve the
Streamlit dashboard or expose a real-time scoring API.

---

## Analysis tooling

| Command | Purpose |
|---|---|
| `python scripts/evaluate.py --alert-budget 10` | Simple-baseline/base/ensemble comparison, day-block PR AUC confidence intervals, cost and capacity operating points, calibration, supported subgroup/time slices, and local batch latency. |
| `python scripts/rolling_validate.py --folds 3 --fast` | Expanding-window retraining and validation of the complete ensemble. `--fast` reduces tree counts but preserves model structure. |
| `python scripts/drift.py --fail-on-significant` | PSI + KS between the training and scoring windows, across all 19 model features. Exits non-zero on significant drift; runs as a gate in CI. Target encodings are compared on their serving-equivalent columns so the gate measures data, not encoding construction ([docs/target-encoding.md](docs/target-encoding.md)). |
| `python scripts/calibration.py` | Brier, ECE/MCE and reliability, reported separately for the alerting region. |
| `python scripts/graph_signal.py` | Bipartite graph density and univariate power of the entity features. |
| `python scripts/case_cli.py --help` | Create, assign, disposition, list, and inspect human-review cases without the dashboard. |
| `python scripts/case_cli.py verify CASE-ID` | Verify the case event chain, projection version, linked files, hashes, sizes, and remote receipt status. Exits 2 on failure. |

### Threshold selection is a business decision

`evaluate.py` costs a false negative at the fraud amount and a false positive at a
configurable investigation cost (default $25). On the reused development split the
F1-selected threshold is **not** cost- or capacity-optimal on development:

| | Threshold | Alerts/day | Precision | Recall | Total cost |
|---|---|---|---|---|---|
| Current demo (F1 on calibration) | 0.7868 | 8.5 | 85.9% | 62.3% | $217,185 |
| Capacity (10/day) | 0.5261 | 9.9 | 78.0% | 66.3% | $181,130 |
| Cost-minimising | 0.0206 | 18.3 | 51.9% | 81.3% | **$120,024** |

The lowest modeled cost requires more than twice the current alert volume. Which
operating point is viable depends on staffing and loss assumptions, not AUC alone.

### Calibration

Aggregate ECE remains deceptively small because almost every transaction scores near
the floor. In the alerting region, development scores average 0.960 against an
observed fraud rate of 0.859 (+10.1 points); the locked tail averages 0.950 against
0.780 (+17.0 points). The score remains a ranking signal, not a calibrated probability.

### Drift monitoring

`drift.py` is the control that would have caught this project's worst bug. The final
Phase 1 mart has zero significant, one moderate, and 18 stable features. The moderate
feature is 90-day `card_txn_cnt` (PSI 0.1705; mean 285.7 → 330.2). A first causal
implementation used lifetime counts and failed badly (PSI 4.98; mean 908.9 → 2,110.2),
which is why the history is bounded. Replaying the original velocity defect still
shows how strong the gate is:

```
txns_24h     PSI=7.608  KS=0.995  mean 4.884 -> 0.014  [SIGNIFICANT]
txns_7d      PSI=5.681  KS=0.966  mean 26.099 -> 0.502 [SIGNIFICANT]
```

Thirty times the significance threshold, on a bug that every null and range check
passed.

### Entity/graph features

A card ↔ merchant bipartite layer is built in
`models/intermediate/int_graph_features.sql` and materialised into the mart, but is
**deliberately not fed to any model**. `card_2hop_fraud_cards` looks strong
univariately (ROC AUC 0.79, better than either risk encoding) yet adding it and its
siblings cut Model 4's PR AUC from 0.797 to 0.180 and lost 419 caught frauds,
because it is a card-level constant derived from training labels. Full measurements
and the mechanism are in [docs/graph-features.md](docs/graph-features.md).

## Testing and CI

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and PR:

| Job | What it does |
|---|---|
| Lint | `ruff check` on correctness rules only (F, E9, W6). Exists because dead code accumulated twice, including a function renamed at its definition but not its call site — which no test could catch, since nothing imports it. |
| Unit tests | 101 pytest cases over PII masking, grounding, alert gates, the human-review state machine, evidence integrity/migration/archive receipts, temporal contracts, rolling splits, uncertainty/calibration helpers, drift statistics, and path resolution. |
| dbt pipeline | Generates a small fixture dataset (`tests/fixtures/make_fixture.py`), runs the **real** dbt models and data tests against it — no 500 MB download — then runs the drift gate. |
| Terraform | `fmt -check`, `init -backend=false`, `validate`. No AWS credentials, never touches remote state. |
| Docker | Trains artifacts from the fixture, builds the image, and asserts it is self-contained — dbt project present, artifacts loadable, **with no bind mounts**. Regression guard: `.dockerignore` once excluded `models/`, `models_artifacts/` and the DuckDB file, and `docker-compose` bind-mounted the repo over `/app`, hiding it. |

Several unit tests are explicit regressions for bugs this project shipped: PSI
returning infinity on binary features, the grounding checker rejecting a masked PAN
suffix or an ISO date, graph features leaking back into the model feature lists, and
the workflow file itself being invalid YAML — which fails a run in zero seconds
without executing a single job.

## Data quality controls

`dbt test` runs 15 checks. Five are worth calling out because they encode bugs
this project actually shipped:

| Test | Guards against |
|---|---|
| `assert_target_encodings_are_temporal` | Recomputes every training encoding from strictly earlier labels; catches random-fold or current/future-label leakage. |
| `assert_card_history_is_prior_only` | Recomputes the bounded 90-day prior count and verifies first-event velocity is zero. |
| `assert_evaluation_window_is_chronologically_locked` | Ensures the locked evaluation tail starts after the development window and training rows cannot enter either role. |
| `assert_velocity_no_train_serve_skew` | Velocity features being computed differently at training and scoring time. An earlier version filtered the window on `dataset_split`, which zeroed `txns_24h`/`txns_7d` for 99.5% of test rows while training saw real counts. Null checks passed throughout — zero is not null. |
| `assert_no_degenerate_feature_scale` | Unstable early-history z-scores and divide-by-near-zero values reaching narrative prompts as facts. |

`scripts/verify_pipeline.py` runs the same skew check plus an end-to-end pass:
dbt tests → schema/row checks → inference bounds → narrative drafting on the
highest-scoring **alert** in the development window. If the sample contains no transaction
above the threshold it fails rather than drafting from an ordinary one. This gate
does not determine whether RGS has been reached.

---

## Scope & limitations

Read this before drawing conclusions from the metrics above.

- **The data is synthetic.** It comes from the Sparkov generator, not real card
  traffic. Simulated fraud has deterministic artefacts that models learn and that do
  not exist in production. The absolute numbers do not transfer.
- **Card fraud is not automatically an AML suspicion.** The model predicts a synthetic
  card-fraud label. That output alone does not establish reasonable grounds to suspect
  money laundering, terrorist financing, or sanctions evasion. Narratives are
  investigation aids, not completed reports or filings.
- **Grounding checks are necessary, not sufficient.** Quantities are verified against
  the payload and a list of unsupported claim types is screened, but no claim-level
  entailment checking is done: a narrative can use only real figures and still draw
  an unsupported inference. Phase 2 requires an active human-review case before a
  narrative can be drafted; Phase 3 preserves both passing drafts and grounding
  failures in the case evidence history.
- **Workflow roles are not authentication.** The dashboard and CLI enforce state and
  role rules, but identities and roles are self-attested. The SQLite history is
  transactional, hash-chained, and append-only through the application API. That
  detects corruption but is not a signature or external trust anchor; local state is
  not access-controlled, backed up, or retention-managed.
- **A verified upload receipt is not a compliance opinion.** A matching S3 checksum
  and VersionId establish the remote object observed by this process. The project does
  not continuously reconcile or restore-test the bucket, and its Terraform retention
  value is illustrative rather than institution-approved.
- **The public test file was historically reused.** Phase 1 prospectively locked its
  final 25%, but earlier versions reported aggregate results over the full file. The
  locked-tail result is disclosed with that caveat; a new external time-forward
  dataset is required for a genuinely blind generalization claim.
- **The meta-score is overconfident in the alerting region** by 10 points on
  development and 17 points on the locked tail. It is
  usable as a ranking; it should not be read as a probability until calibrated.
- **No champion/challenger evaluation and no analyst feedback loop.** Drift is
  monitored and gated in CI, but there is no mechanism to compare a candidate model
  against the incumbent, and no path for investigator dispositions to feed back into
  evaluation. Both are prerequisites for anything operational.
- **Nothing is actually deployed.** CI builds and verifies the image on every push,
  but promotion to AWS is a manual `deploy.sh` run, and the ECS service is defined
  with `desired_count = 0` — the infrastructure is provisioned and exercised, not
  serving traffic.

---

## Infrastructure reference (Terraform)

Defined in `terraform/` and applied manually. This is an infrastructure exercise,
not evidence of a running service: the ECS service has zero desired tasks, the
container exits after verification, and no load balancer or scoring endpoint exists.

- **Versioned S3 evidence bucket reference** with Object Lock in COMPLIANCE mode,
  server-side encryption, full public-access block, an illustrative five-year policy,
  and lifecycle transition to Glacier at 90 days. Configuration alone is not evidence
  of legal suitability or an operating archive.
- **IAM** task role scoped to `bedrock:InvokeModel` and the compliance bucket; no
  static keys in the task definition. The code invokes a cross-region inference
  profile, so the policy grants the profile ARN *and* the underlying foundation model
  in each region the profile can route to — granting only a single-region
  foundation-model ARN produces `AccessDeniedException` at invoke time.
- **VPC** with a public subnet, IGW and an egress-only security group.
- **ECR** with scan-on-push and untagged-image expiry.
- **ECS Fargate** cluster, task definition and service (`desired_count = 0`).
- Remote state in S3 with DynamoDB locking.

---

## Repository layout

```
models/              dbt project (staging → intermediate → marts)
macros/              prequential target-encoding macro
docs/
  str-narrative-design.md  why the hedging blacklist was replaced by grounding checks
  human-review-workflow.md Phase 2 states, invariants, interfaces, and honest boundary
  evidence-integrity.md    Phase 3 preservation, receipts, verification, and limits
  graph-features.md        entity/graph layer: built, measured, and rejected
  target-encoding.md       causal encodings, bounded card history, and drift references
  materialization.md       why incremental materialization was measured and rejected
scripts/
  config.py            path + feature-contract resolution shared by every entry point
  pii.py               PAN / name masking
  grounding.py         factual grounding checks for generated narratives
  train_models.py      chronological 3-way split, training, versioned artifacts
  modeling.py          shared ensemble fit/score primitives
  rolling_validate.py  expanding-window ensemble validation
  evaluation_utils.py  uncertainty, calibration, subgroup and split helpers
  inference_engine.py  validation, scoring, alert selection
  case_management.py   persisted human-review/RGS state machine and event history
  case_cli.py          command-line case workflow
  evidence.py          atomic content-addressed files and verified S3 receipts
  sar_agent.py         Bedrock narrative drafting, guardrails, quarantine, optional S3 attempt
  evaluate.py          model comparison, uncertainty, cost, calibration, slices, latency
  calibration.py       Brier / ECE / reliability, incl. the alerting region
  drift.py             PSI + KS between training and scoring windows
  graph_signal.py      bipartite graph density and univariate feature power
  verify_pipeline.py   end-to-end verification
  audit_phase1.py      standalone data/artifact audit
tests/
  *.sql                dbt singular tests (causality, evaluation lock, skew, scale)
  unit/                pytest suite
  fixtures/            generates a small dataset so CI runs the real dbt pipeline
terraform/           AWS infrastructure
.github/workflows/   CI: lint, unit tests, dbt + drift gate, terraform, docker
app.py               Streamlit investigator dashboard
```

## Roadmap status

Phase 0 (`0.1.0`) established honest terminology and scope. Phase 1 (`0.2.0`) added
causal feature construction, bounded card history, an evaluation lock, rolling
validation, confidence intervals, baseline comparison, operating-capacity analysis,
calibration/slice reporting, and latency measurement. Phase 2 (`0.3.0`) adds the
persisted human-review and RGS state machine, role and rationale guards, concurrency
control, case history, dashboard workflow, and CLI. Phase 3 (`0.4.0`) adds atomic
case-linked evidence, event hash chaining, local integrity verification, and explicit
S3 checksum/version receipts. The next priority is authentication, authorization,
secrets and privacy controls, followed by service architecture and
champion/challenger operations.
