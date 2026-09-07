# AegisAgent portfolio brief

## The project in one minute

I extended my Memorial University Master of Data Science capstone into an auditable
fraud-investigation prototype. It combines temporal SQL features, a stacked classifier,
an investigator case workflow, evidence integrity checks, and explicit model promotion.
I evaluated models using precision-recall measures, time windows, calibration, and
analyst-capacity trade-offs, and documented results that challenged the original design.

This is synthetic-data engineering evidence. It does not establish prevented fraud,
financial savings, real-time payment authorization, or regulatory compliance. There is
no affiliation with or endorsement by Nasdaq Verafin.

## Engineering evidence

| Question | Demonstration | Evidence |
|---|---|---|
| Can features avoid future-information shortcuts? | Prior-only velocity/history; frozen encoding maps at internal holdout boundaries | `models/`, `scripts/modeling.py`, dbt tests |
| Is performance reported honestly under imbalance? | PR AUC, best-base comparison, temporal degradation, uncertainty, alert-region calibration | `MODEL_CARD.md`, historical `reports/` |
| Can an investigator explain an alert's history? | Review states, actors, rationales, optimistic concurrency | Case/service tests |
| Can modified evidence or model state be detected? | File manifests, event replay, projection checks, governed comparisons and rollback | Registry/evidence tests |
| Do people and services have different authority? | Tenant scope, write-only alert ingestion, separate RGS and governance permissions | Security tests |
| Can another engineer reproduce the workflow? | Isolated fixture demo, pinned direct dependencies, Docker, dbt, CI | `python -m scripts.demo` |

These are discussion points for data engineering, applied ML, backend, and
financial-crime platform roles. Verafin describes card-fraud analytics and centralized
investigation records as product capabilities; this portfolio explores related
engineering problems at a much smaller scale. Primary sources reviewed in September
2026: [card fraud](https://verafin.com/solution/card-fraud/),
[case management](https://verafin.com/solution/case-management/), and
[careers](https://verafin.com/careers/). Tailor each application to its actual posting.

## Reviewer walkthrough

Use Python 3.11 and a virtual environment. From the repository root:

```bash
python -m pip install -r requirements-dev.txt
python -m scripts.demo --serve
```

Allow a few minutes for the first build. The command uses the real feature SQL and
training code with 12,000 synthetic training rows and 2,500 synthetic test rows. It
writes everything under a new `.demo/run-*` directory. A `demo-summary.json` records
generated paths and completed checks. No AWS or Clerk account is needed. The fixture
has an intentionally easy signal and must not be used to advertise model accuracy.
Omit `--serve` to verify without starting a server.

1. Show the synthetic-data and development-identity notices.
2. Select a threshold-breaching transaction and inspect the component scores.
3. Create a case and start review with a rationale. Explain why a fraud score cannot
   decide reasonable grounds to suspect money laundering.
4. Show event history and integrity verification. The verifier separately exercises
   evidence preservation without generated text or a cloud provider.
5. Explain RGS-review authority separately from model-governance authority.
6. Show a regression for an unauthorized action or changed artifact and the CI result
   for the revision being presented.

Bedrock narration is optional and requires separately configured credentials. Its
output is unapproved text, not an STR or a substitute for evidence.

## Interview points

- The ensemble loses to its strongest base learner on historical PR AUC. Explain the
  trade-off without presenting aggregate ECE as reliable alert probabilities.
- Frozen validation encodings tightened the protocol. Historical results are preserved
  with their provenance, not relabeled as measurements from the revised code.
- A hash chain detects ordinary tampering; a privileged disk owner can rewrite it.
  Distributed operation requires additional controls and independent audit storage.
- RGS dispositions are analyst outcomes, not fraud labels. Explain selection bias
  before proposing a feedback-driven training loop.
- Tenant isolation and fail-closed mode do not replace Clerk verification, managed
  persistence, recovery tests, or institutional approval.

## Application wording

Suggested CV bullet without unmeasured business impact:

> Extended an MDSc fraud-detection capstone into a Python/dbt investigation prototype
> with temporal feature contracts, model comparison and promotion controls, a
> role-scoped FastAPI case workflow, tamper-detecting evidence history, and CI-backed
> synthetic-data verification.

Link this repository and discuss two engineering decisions relevant to the role. Do
not claim production deployment, prevented losses, broad AML detection, or certification.

## Research provenance and remaining work

The thesis is *Feature-Enhanced Machine Learning Models for Credit Card Fraud
Detection*, Muhammad Haseeb Khan, Master of Data Science, Memorial University of
Newfoundland. It evaluates the individual capstone models. The stacked pipeline,
governance, case service, privacy controls, and evidence workflow are later repository
work. The thesis and current code use different protocols.

Clerk verification remains deliberately deferred. Operational use still needs real
label-delay modeling, external later-period validation, approved calibration/capacity
criteria, managed persistence, backups and restore drills, request limits, monitoring,
and independent security/privacy review. Use synthetic data and localhost for the demo.
