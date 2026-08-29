# Changelog

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
