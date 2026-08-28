# Human review and RGS state machine

Phase 2 turns the regulatory boundary from documentation into executable workflow.
The model creates an investigation alert only when its score meets the configured
threshold. A human then owns every subsequent state transition.

```text
threshold breach
      |
      v
 alert_open -- identified investigator + rationale --> under_review
                                                        |        |
                                      authorized human  |        | authorized human
                                      + rationale       |        | + rationale
                                                        v        v
                                             rgs_not_reached   rgs_reached
                                                                  |
                                                                  v
                                                   approved reporting workflow
                                                           (out of scope)
```

There is deliberately no `filed`, `submitted`, or `report_approved` state. An
`rgs_reached` case is a terminal record of the human decision in this application;
it is not evidence that an STR was prepared or transmitted.

## Enforced invariants

- A case cannot be created below the model threshold.
- One transaction can create at most one case.
- A case cannot skip `under_review` on its way to a disposition.
- Starting review requires an identified actor and a documented rationale.
- Only the `authorized_rgs_reviewer` role can record `rgs_reached` or
  `rgs_not_reached`.
- Both dispositions are terminal and cannot be silently rewritten.
- Every transition appends an event containing actor, asserted role, rationale,
  timestamp, previous/new state, and monotonic case version.
- An expected-version check prevents a stale browser or CLI invocation from
  overwriting a newer transition.

## Persistence and interfaces

`scripts/case_management.py` stores the current case projection and its event history
in `compliance_logs/cases.sqlite3`. Override the path with `AEGIS_CASE_DB_PATH`.
The Streamlit dashboard exposes the normal investigator flow. The same workflow can
be inspected or exercised without the UI:

```bash
python scripts/case_cli.py create --trans-num TXN-1 --score 0.91 --threshold 0.80 \
  --model-version v_demo --actor investigator-17
python scripts/case_cli.py start-review CASE-ID --actor investigator-17 \
  --role investigator --expected-version 1 \
  --rationale "Assigned after reviewing the alert and transaction facts."
python scripts/case_cli.py decide CASE-ID --actor reviewer-4 \
  --role authorized_rgs_reviewer --rgs reached --expected-version 2 \
  --rationale "Documented human assessment of the available facts and indicators."
python scripts/case_cli.py show CASE-ID
```

## Honest boundary

This is workflow logic, not a production case-management control. Reviewer identity
and role are self-attested inputs; there is no login, directory integration, RBAC,
maker-checker policy, or cryptographic signature. SQLite provides transactional local
persistence, but the application event API is only logically append-only—the file is
not tamper-evident, replicated, backed up, retention-managed, or independently
reconciled. Narrative/quarantine files are not yet linked into the case event stream.

Those are durable-evidence and security concerns for later phases. The Phase 2 claim
is narrower: the software now makes the human decision boundary explicit, rejects
invalid or unauthorized transitions, records who asserted each decision and why, and
never turns that decision into an automatic regulatory filing.
