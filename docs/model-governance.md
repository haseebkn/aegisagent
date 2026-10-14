# Phase 6 model governance and operational feedback

Phase 6 changes model training from deployment-by-side-effect into an explicit local
champion/challenger workflow. It demonstrates controls expected around a model; it is
not a managed registry, an independent validation process, or a production rollout.

## Lifecycle

1. `train_models.py` writes a complete versioned artifact directory.
2. Every required artifact is SHA-256 manifested and registered. In an empty store the
   first version bootstraps the champion; otherwise the new version is a candidate and
   `latest_version.txt` is unchanged.
3. `model_ops.py compare` scores champion and candidate on the identical chronological
   `development_holdout`. It never unlocks the final evaluation tail.
4. The comparison report and its eligibility are hash-registered in the governance
   event chain.
5. `model_ops.py promote` requires the model-governance role, an identified actor, a
   meaningful rationale, a still-current champion, unchanged candidate artifacts, and
   the exact passing report recorded in step 4.
6. The serving pointer changes atomically. A registry/pointer mismatch causes serving
   to fail closed. The previous champion becomes an archived rollback target.
7. `model_ops.py rollback` requires the same authority and rationale and verifies the
   archived artifact manifest before restoring it.

Training never promotes an established candidate automatically.

## Comparison policy

The default comparison uses a paired calendar-day bootstrap because both models score
the same rows and transaction behavior varies by day. Candidate minus champion PR AUC
must have a 95% interval lower bound of at least `-0.01`. Candidate recall may not fall
more than `0.02`, Brier score may not worsen more than `0.02`, and alert volume may not
exceed 10 per day. These defaults are demonstrative policy inputs, not institutionally
approved limits; the report records their exact values.

Passing means “eligible for human consideration,” not “better,” “production ready,” or
“externally validated.” The locked tail is not repeatedly consulted for promotion.

## Roles and audit trail

Only `model_governance_reviewer` has compare, promote, rollback, and operational-
feedback export permissions. Investigator and RGS-review roles cannot promote models.
Local development identities remain self-attested; Clerk or another verified provider
must supply the role in production.

Authorization is enforced inside the registry domain functions, not only by the CLI.
Every comparison registration, promotion, and rollback receives a
`SecurityPrincipal` and checks its named permission, so importing the Python module
does not bypass the role boundary.

Registry events are hash-chained and include artifact/report digests, actor, rationale,
and previous champion. This detects ordinary modification but is not a signature or an
external trust anchor; a privileged filesystem owner could rewrite both content and
hashes. Production needs a managed registry and independently retained audit logs.

Comparison registration independently checks the current champion and candidate
artifact manifests, the development-holdout designation, the fixed promotion policy,
finite metrics, and every derived eligibility gate before hashing the report. Promotion
then requires that exact registered report and rechecks the candidate artifacts and
current champion. A caller therefore cannot obtain eligibility by supplying a report
whose gates are merely marked `true`.

Release 0.8.0 also independently loads the registered models and re-scores the
development holdout, requiring the complete report to match that evaluation. The
report binds a data digest and bootstrap sample count. Registry state is replayed
from its events, artifact hashes are checked before deserialization, and local
registry mutations are serialized. This does not create distributed transactions:
a crash between pointer and registry writes deliberately leaves serving fail-closed
until the two are recovered consistently.

## Analyst feedback semantics

`operational_feedback.py` aggregates case volume, terminal RGS outcomes, reached rate,
and median review duration by model version within the caller's organization. Cohorts
smaller than five are suppressed by default, and transaction/case identifiers are not
exported.

The report declares `training_label_eligible: false`. Human RGS disposition is not the
synthetic card-fraud target. Cases are selected by the model threshold, analysts use
context absent from the feature mart, and only alerts receive review. Directly training
on these outcomes would conflate regulatory judgment with fraud and introduce severe
selection bias. A future learning loop requires separately governed outcome labels,
sampling of non-alerts, label-quality review, temporal delay handling, and bias checks.

## Remaining production work

This phase has no shadow scoring, canary traffic, online performance telemetry,
automatic rollback, distributed registry locking, signed attestations, independent
validation approval, production feature snapshots, or recovery drill. Those controls,
plus Clerk and managed persistence, are required before operational exposure.
