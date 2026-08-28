# Changelog

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
