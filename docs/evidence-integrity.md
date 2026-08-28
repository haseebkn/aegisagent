# Case-linked evidence and integrity verification

Phase 3 replaces ordinary draft files with an evidence path that can answer four
basic audit questions: what bytes were preserved, which case they belong to, who
attached them, and whether the bytes and case history still match.

## Preservation flow

1. The grounding result determines whether the artifact type is `narrative_draft` or
   `narrative_quarantine`; both outcomes are preserved.
2. UTF-8 content is written to a temporary file, flushed with `fsync`, and atomically
   moved to `evidence/<case-id>/<type>-<sha256>.txt`.
3. The active case records evidence ID, SHA-256, byte count, media type, local path,
   actor, timestamp, and optional archive receipt.
4. An `evidence_attached` event is appended without changing the review status. Its
   hash commits to the complete event content and the previous event hash.
5. The case version increments, so a disposition submitted from a stale screen is
   rejected and must be reconsidered against the newly attached evidence.

Grounding failure is therefore observable evidence, not an exception that destroys
the attempted draft.

## Remote archive receipt

If `COMPLIANCE_S3_BUCKET` is set, the upload supplies an SHA-256 request checksum and
content metadata. The result is marked verified only when S3 returns the identical
checksum and a non-empty object `VersionId`. The receipt stores bucket, key, version,
ETag, checksum, request ID, and timestamp. A network error, checksum mismatch, or
unversioned response leaves the local artifact linked but records an unverified
archive attempt; case integrity then reports failure until the condition is resolved.

## Verification

Run:

```bash
python scripts/case_cli.py verify CASE-ID
```

The verifier checks:

- every event's content hash and link to the previous event;
- the current case version against the final event sequence;
- every registered local file's existence, byte count, and SHA-256;
- a matching evidence event for every evidence record; and
- that any recorded remote archive receipt is verified.

The command exits with status 2 when any check fails. The dashboard exposes the same
verification and inventory.

## Security boundary

This detects corruption, missing files, accidental edits, stale writes, and ordinary
database tampering where hashes are not also recomputed. It is not a cryptographic
signature, external timestamp, or immutable ledger. An administrator with write
access can replace evidence and recompute the entire local chain. The optional S3
version receipt provides a stronger external reference, but this project does not
continuously reconcile the bucket, test restore procedures, authenticate reviewers,
or establish that the illustrative five-year Terraform setting is the correct legal
retention policy for a particular institution.

Accordingly, Phase 3 demonstrates evidence engineering and explicit failure states;
it does not claim a certified records-management or FINTRAC-compliant archive.
