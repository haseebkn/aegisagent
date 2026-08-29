# Phase 5 service architecture

Phase 5 separates transport from the human-review domain. `service.py` is a FastAPI
application factory; `scripts/case_management.py` remains the source of workflow,
authorization, organization-isolation, evidence, and concurrency rules. Streamlit,
the CLI, and HTTP therefore cannot acquire different regulatory policy by accident.

## Identity boundary

`create_app(principal_resolver=...)` accepts a request-scoped resolver that must return
a validated `SecurityPrincipal`. The built-in resolver uses only the configured local
development identity and is rejected in production. It does not trust identity, role,
or organization headers.

A later Clerk adapter belongs at this one resolver seam. It must verify the token,
issuer, audience, expiry, session state, organization membership, and server-owned
role mapping before constructing a principal. Route payloads cannot select actors,
roles, or organizations.

## HTTP contract

All workflow operations are under `/v1`:

| Method and path | Domain operation |
|---|---|
| `POST /v1/cases` | Ingest a threshold-breaching alert from a trusted scoring service |
| `GET /v1/cases` | List the caller's organization-scoped cases |
| `GET /v1/cases/{case_id}` | Read one authorized case |
| `POST /v1/cases/{case_id}/review` | Start human review with rationale and expected version |
| `POST /v1/cases/{case_id}/rgs-decision` | Record an authorized human RGS disposition |
| `GET /v1/cases/{case_id}/history` | Read the hash-chained event history |
| `GET /v1/cases/{case_id}/evidence` | List linked evidence receipts |
| `GET /v1/cases/{case_id}/integrity` | Verify the event/evidence chain |

There is no endpoint to file, submit, approve, or transmit a regulatory report.
Narrative generation is also not exposed in Phase 5; it remains an explicitly invoked
human-review aid while its asynchronous and resource-control boundary is designed.

Requests reject unknown fields, non-finite scores, out-of-range values, oversized
rationales, and coercion of strings into RGS booleans. Mutations require the caller's
last observed case version. A stale version returns HTTP 409 and never overwrites the
newer state.

Alert creation is a machine-to-machine trust boundary. `POST /v1/cases` additionally
requires `alert:ingest_verified_model_output`; ordinary investigator and RGS-review
roles are denied even if they can work an existing case. The local Streamlit and CLI
demonstrations attach this permission at the point where they run the model. A
production identity adapter must map it only from a verified scoring-service identity.
This prevents an interactive caller from choosing arbitrary scores, thresholds, or
model versions and presenting them as model output.

## Health and failure behavior

- `GET /health/live` proves only that the process can answer HTTP.
- `GET /health/ready` checks local case persistence and identity-adapter availability.
- In production without a verified adapter, liveness remains 200, readiness returns
  503, and protected routes return `identity_adapter_unavailable`.
- Every response carries a bounded/generated `X-Request-ID`, `Cache-Control: no-store`,
  and `X-Content-Type-Options: nosniff`.
- Expected domain failures have stable status/code contracts. Unexpected exception
  details are redacted and never returned to clients.

## Deployment boundary and deferrals

Docker Compose runs this API for local demonstration. The image itself still defaults
to the batch verifier used by the existing ECS reference task. Terraform does not
expose the API, and no claim is made that it is deployed.

Before public or institutional use, the service still needs Clerk verification,
managed transactional persistence and migrations, rate/request-size limiting at an
edge gateway, TLS termination, CORS/host policy, structured access/security logging,
metrics and alerting, load and failure testing, secrets management, backup/restore,
retention governance, incident response, and an approved deployment architecture.
