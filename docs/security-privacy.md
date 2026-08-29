# Phase 4A security and privacy foundation

Phase 4A establishes the boundary that a future Clerk integration must satisfy. It
does not provision Clerk, render a production sign-in flow, or claim that the current
Streamlit application is ready for public deployment.

## Identity boundary

`scripts/security.py` defines a provider-neutral `SecurityPrincipal`. A trusted
identity adapter must verify a session before constructing one. The case workflow
never accepts or stores passwords, access tokens, ID tokens, JWTs, Clerk secret keys,
or raw identity-provider claims.

The principal contains only the identity data needed for authorization and audit:

- stable subject identifier;
- granted application roles;
- identity-provider and assurance labels;
- organization identifier; and
- optional session identifier.

Every case mutation checks a named permission and commits the token-free identity
metadata to the hash-chained event. An asserted actor or role cannot override a
provided principal. Case reads, listings, transitions, evidence attachments, and
integrity checks enforce the case organization boundary. Integrity verification also
compares the case projection's organization with the organization committed in the
initial event, making later projection tampering observable.

Databases created before Phase 4A are migrated with the sentinel organization
`legacy-unscoped`. Only the local development provider may inspect those records;
production identities cannot implicitly claim ownership. A real migration must map
each legacy case to a verified organization through an approved administrative process.

Transaction identifiers are unique only within an organization. Reads resolve case
and transaction identifiers inside the principal's organization boundary, so an
identifier used by another tenant neither blocks creation nor reveals that tenant's
record.

## Modes and fail-closed behavior

`AEGIS_SECURITY_MODE=development` enables a conspicuously labelled local identity
provider for tests and portfolio demonstrations. Its identities have the assurance
label `self-attested-development-only`.

`AEGIS_SECURITY_MODE=production` disables that provider. Because the Clerk adapter is
deliberately deferred, the Streamlit UI and CLI reject access in production mode.
Terraform sets production mode explicitly. A deployment therefore cannot silently
turn a development identity into production authentication.

## Authorization policy

The policy is deny by default and each role has an explicit permission set. Adding a
new permission to the application cannot silently grant it to an existing role.

| Operation | Investigator | Authorized RGS reviewer | Alert ingestor | Model governance reviewer |
|---|---:|---:|---:|---:|
| Read organization cases | Yes | Yes | No | Yes |
| Start review / attach evidence | Yes | Yes | No | No |
| Verify case integrity | Yes | Yes | No | No |
| Record RGS disposition | No | Yes | No | No |
| Submit verified model output to the HTTP alert endpoint | No | No | Yes | No |
| Compare, promote, or roll back models | No | No | No | Yes |
| Export aggregate operational feedback | No | No | No | Yes |

The local Streamlit and case CLI paths add the alert-ingestor role only at their
in-process model-output boundary. A production adapter must grant that role solely to
an authenticated scoring service, not to an interactive investigator session.

This is application authorization, not a replacement for identity-provider controls.
The future adapter must map verified Clerk organization roles/permissions to these
internal roles; client-side visibility controls are never sufficient.

## Privacy boundary

The Bedrock request is now constructed from a strict allowlist. It excludes first and
last name, gender, occupation, date of birth, street, city, province/state, postal
code, customer coordinates, and merchant coordinates. The only card reference sent
is a masked PAN. Narrative evidence also omits the customer name.

This is data minimization, not anonymization. Transaction identifiers, merchant
information, timestamps, amounts, model signals, and a masked card reference may
still be personal or commercially sensitive when combined with other data.

Error/log boundaries redact common Clerk keys, AWS access-key identifiers, bearer
tokens, and PAN-like values. Generic UI errors avoid returning backend exception
details. Narrative content is suppressed from CLI and pipeline-verification stdout.
These controls are defense in depth; sensitive payloads should not be logged in the
first place.

Local case databases and evidence directories receive owner-only permission requests
(`0700` directories and `0600` files). Enforcement ultimately depends on the host
filesystem. S3 uploads request AES-256 encryption and the bucket policy denies
non-TLS transport.

The container defaults to production authentication mode and runs as a fixed non-root
user. Docker Compose must explicitly override this to development mode for a local
demo. Credential-shaped files are excluded from the Docker build context.

## Threats covered in Phase 4A

- role selection used to grant an unauthorized RGS disposition;
- arbitrary actor IDs written into case history in production;
- cross-organization case access;
- accidental token/secret retention in case events;
- direct and quasi-identifiers sent to the narrative provider;
- credentials or PANs echoed through ordinary exception messages;
- production accidentally running with the development identity provider; and
- plaintext transport to the evidence bucket.

## Explicitly deferred

- Clerk provisioning, hosted sign-in UI, MFA and account recovery;
- Clerk JWT verification and key rotation;
- organization membership synchronization and verified webhooks;
- session revocation and step-up authentication;
- a production HTTP/API boundary, CSRF/CORS controls and rate limiting;
- managed database encryption, backups and restore testing;
- approved retention/deletion schedules and privacy impact assessment;
- SIEM integration, alerting, penetration testing and incident response exercises.

## Clerk adapter acceptance criteria

After the service architecture is stable, the Clerk adapter must:

1. verify signature, algorithm, issuer, expiry/not-before, audience and authorized
   party before creating a `SecurityPrincipal`;
2. derive subject, organization and roles only from verified server-side claims;
3. reject pending, expired, revoked, malformed and cross-organization sessions;
4. keep Clerk secret keys and session tokens out of logs, events and browser-visible
   configuration;
5. require MFA or recent step-up verification for RGS disposition if the institution's
   policy calls for it; and
6. pass end-to-end tests for login, logout, expiry, role removal, organization switch,
   direct URL access and backend authorization.

Phase 4A is therefore a security architecture milestone, not a production security
certification or legal/privacy compliance opinion.
