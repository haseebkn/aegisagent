# Changelog

## 0.8.0 — 2026-09-07

Portfolio audit and reproducible reviewer workflow.

- Validate registry projections against event replay and model files before loading;
  reject unsafe version paths and serialize local registry writes.
- Independently rescore the development holdout before accepting a governed comparison
  and bind its data fingerprint, artifacts, policy, metrics and report digest.
- Validate every inference row, thresholds and output probabilities; reject missing or
  nonfinite drift inputs and fix sparse-distribution PSI and constant-score calibration.
- Freeze target encodings before internal holdouts and rolling validation; correct
  calendar-day exposure, capacity analysis and historical graph self-exclusion.
- Detect case projection/history tampering without silently repairing modern ledgers;
  authorize narrative evidence before I/O and bind it to the case's scored model.
- Tighten principal/request validation, escape dynamic narrative HTML, and reject
  unversioned S3 receipts.
- Add an isolated synthetic demo, optional cloud verification, localhost-only Compose,
  portable Terraform backend settings and explicit reference-provisioning preflight.
- Add behavioral regression coverage, reviewer guidance and honest historical-results
  disclosures. Existing research artifacts and the locked evaluation tail are preserved.
- Close SQLite connections explicitly on every path, include the dbt macros in the
  image, and exercise the complete container demo with networking disabled.
- Patch vulnerable application/test dependencies and add a resolved dependency audit
  to CI; replace deprecated Streamlit layout arguments.

## 0.7.0 — 2026-08-29

Phase 6 adds governed champion/challenger operations and a deliberately non-training
analyst-feedback report.

- Changed training so a new version is registered as a candidate instead of silently
  replacing the serving pointer. The first model bootstraps the registry only when no
  champion exists.
- Added artifact manifests and a hash-chained model-governance event history covering
  registration, paired comparison, human promotion, and controlled rollback.
- Added paired calendar-day bootstrap comparison on the common development window,
  with explicit PR-AUC non-inferiority, recall, calibration, and alert-capacity gates.
- Required an authorized `model_governance_reviewer`, identified actor, rationale,
  passing hash-registered report, unchanged artifacts, and unchanged champion before
  atomic promotion.
- Added fail-closed serving when the registry champion and version pointer disagree.
- Added organization-scoped aggregate analyst-disposition reporting with small-cohort
  suppression and an explicit prohibition on treating RGS decisions as fraud labels.
- Added governance tests for silent replacement, eligibility, artifact/report/registry
  tampering, role separation, rollback, paired degradation, feedback semantics, and
  serving-pointer consistency.
- Moved comparison, promotion, and rollback authorization into the registry domain;
  direct module calls now require an authorized security principal.
- Recompute eligibility from finite metrics, current artifact manifests, the locked
  development-holdout designation, and the fixed promotion policy before registering
  a comparison report.
- Replaced implicit all-permission roles with explicit grants and separated trusted
  model-output ingestion from interactive investigator and RGS-review identities.
- Scoped transaction uniqueness and identifier lookup by organization, including a
  migration for databases created with the earlier global constraint.
- Corrected the Compose AWS credential mount to the fixed non-root runtime user's home.

## 0.6.0 — 2026-08-29

Phase 5 introduces a testable HTTP service boundary around the existing human-review
domain without pretending the application is production-deployed.

- Added a FastAPI service factory with injectable request-scoped identity resolution,
  allowing a later Clerk verifier to supply trusted principals without moving policy
  into route handlers.
- Exposed versioned case creation, listing, review, RGS-decision, history, evidence,
  and integrity endpoints; no filing or regulatory-submission endpoint exists.
- Added strict request schemas, bounded inputs, optimistic-concurrency contracts,
  organization isolation, request correlation IDs, no-store/nosniff headers, and
  stable error responses that do not expose internal exceptions.
- Split liveness from readiness. Production remains not-ready and protected routes
  return service-unavailable until a verified identity adapter is configured.
- Configured the development Compose service to run the API with a container-native
  health check while retaining the image's production-secure default.
- Added service-level regression tests covering lifecycle transitions, RBAC,
  cross-organization access, stale writes, strict parsing, OpenAPI scope, and
  production fail-closed behavior.

## 0.5.0 — 2026-08-28

Phase 4A establishes a vendor-neutral security and privacy boundary without coupling
the application to Clerk before the service architecture is stable.

- Added validated security principals, deny-by-default permissions, organization
  isolation, and token-free identity metadata in the hash-chained case history.
- Disabled the local demonstration identity provider in production mode; Terraform
  explicitly selects production mode, which fails closed until Clerk is integrated.
- Replaced Bedrock payload construction with a strict allowlist that removes names,
  demographics, occupation, address/postal data, and exact coordinates.
- Added secret/PAN redaction at error boundaries, generic dashboard errors, owner-only
  local storage permission requests, explicit S3 encryption, and a TLS-only bucket policy.
- Restricted the CI token to read-only repository contents, upgraded all JavaScript
  actions to Node.js 24 generations, and enabled weekly Dependabot action updates.
- Made the image default to fail-closed production authentication, changed its runtime
  to a fixed non-root user, and excluded credential-shaped files from build context.
- Suppressed generated narrative content from CLI and verification logs.
- Added adversarial tests for role escalation, actor spoofing, cross-organization access,
  production self-attestation, prompt leakage, and secret redaction.

## 0.4.2 — 2026-08-28

Phase 3 CI closure.

- Extended the deterministic CI training fixture from 6,000 to 12,000 rows so its
  90-day card-history feature reaches a representative steady state.
- Fixed a false-positive `card_txn_cnt` drift failure: fixture PSI falls from 0.354
  to 0.055 without changing the production drift thresholds or excluding the feature.
- Added a regression guard tying the fixture warm-up ratio to the CI row-count contract.

## 0.4.1 — 2026-08-28

Phase 3 release closure.

- Applied canonical Terraform formatting to the versioned evidence-bucket reference.
- Re-ran Terraform formatting and configuration validation alongside the complete
  Python regression, lint, compilation, and Git-diff gates.
- Supersedes `0.4.0` as the fully release-verified Phase 3 baseline.

## 0.4.0 — 2026-08-28

Phase 3 links narrative evidence to cases and makes integrity failures observable.

- Replaced timestamp-named draft/quarantine writes with atomic, content-addressed
  evidence artifacts carrying SHA-256 and byte-count receipts.
- Added a case evidence inventory and hash-chained event ledger, including automatic
  migration and backfill for Phase 2 SQLite databases.
- Added end-to-end verification of event links, event contents, evidence files, case
  projection version, and remote archive receipt status.
- Added optional S3 uploads with SHA-256 request checksums; an archive is marked
  verified only when S3 returns the expected checksum and an object VersionId.
- Preserved local, case-linked evidence when remote archival fails and recorded the
  attempt as unverified instead of silently claiming success.
- Added dashboard evidence inventory/integrity controls and CLI `show`/`verify`
  output, plus 15 evidence, migration, archive-receipt, and tamper-detection tests.
- Removed infrastructure language that implied a configured retention period or
  Object Lock mode alone establishes FINTRAC compliance.

## 0.3.0 — 2026-08-28

Phase 2 adds an explicit human-review and RGS decision workflow without crossing the
boundary into regulatory filing.

- Added a persisted `alert_open → under_review → RGS disposition` state machine.
- Enforced threshold eligibility, identified actors, documented rationales, role-gated
  RGS decisions, terminal dispositions, and optimistic concurrency.
- Added a case projection and append-only application event history in a dedicated
  SQLite store, with no filing or submission state.
- Added dashboard case creation, assignment, case history, and disposition controls;
  narrative drafting now requires an active human review.
- Added a CLI for creating, reviewing, deciding, listing, and inspecting cases.
- Added 15 state-machine regression tests and documented the security, durability,
  retention, authentication, and reporting-workflow boundaries.

## 0.2.0 — 2026-08-28

Phase 1 makes the model-development and evaluation path temporal and auditable.

- Replaced random-fold target encodings with smoothed prequential encodings built
  exclusively from earlier labels.
- Rebuilt card statistics over the strictly preceding 90 days and excluded the
  current event from 24-hour and 7-day velocity.
- Added a chronological development/locked-tail protocol with explicit unlock gates.
- Added expanding-window ensemble validation, a simple logistic baseline, day-block
  PR AUC confidence intervals, capacity/cost operating points, calibration, supported
  subgroup/time slices, and local batch latency.
- Added independent dbt reconstruction tests for target encodings and card history,
  an evaluation-lock test, and explicit non-null history contracts.
- Added committed machine-readable reports for development, the locked tail, rolling
  validation, and feature drift.
- Retrained artifacts on the causal mart. Development PR AUC is 0.7766; the locked
  tail is 0.6711 and is disclosed as historically exposed at the aggregate level.

## 0.1.0 — 2026-08-27

Phase 0 establishes an honest portfolio baseline.

- Reframed model-threshold breaches as investigation alerts rather than regulatory
  determinations or automatic filing obligations.
- Relabeled generated output and saved artifacts as unapproved narrative drafts.
- Documented the human RGS decision boundary and the absence of FINTRAC submission.
- Corrected real-time, production deployment, verified archival, and untouched-test
  implications in the dashboard and README.
- Added a model card covering intended use, evaluation evidence, temporal leakage,
  reused-holdout, privacy, regulatory, and deployment limitations.

Known defects are intentionally documented rather than hidden. Causal feature
engineering and a new blind evaluation period are Phase 1 work.
