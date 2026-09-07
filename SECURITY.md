# Security reporting and supported use

AegisAgent supports local demonstrations using synthetic data. Production identity
verification is deliberately deferred. Do not expose development identities on a
public or shared network.

Report security defects through GitHub private vulnerability reporting when enabled.
If it is unavailable, open an issue requesting a private contact without publishing
exploit details. Share the affected revision and a minimal synthetic reproduction once
a private channel is established. Never post credentials, customer information,
investigation narratives, or local databases in a public issue.

See [security/privacy boundaries](docs/security-privacy.md) for implemented controls
and deferrals. Only load joblib/pickle artifacts from trusted sources. Dependency
advisories, tests, and integrity checks are maintenance evidence, not certification.
Older releases do not automatically receive backported fixes.
