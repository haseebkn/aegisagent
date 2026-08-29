"""Vendor-neutral identity and authorization boundary for AegisAgent.

Phase 4A deliberately does not know how a user signed in.  A future Clerk adapter
will turn verified session claims into :class:`SecurityPrincipal`; the case workflow
only consumes that trusted object.  The local provider exists for portfolio demos
and is rejected whenever ``AEGIS_SECURITY_MODE=production``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping


class SecurityError(RuntimeError):
    """Base error for authentication and authorization failures."""


class AuthenticationRequired(SecurityError):
    pass


class PermissionDenied(SecurityError):
    pass


class AuthMode(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"


class Permission(str, Enum):
    READ_CASE = "case:read"
    CREATE_ALERT = "case:create_alert"
    START_REVIEW = "case:start_review"
    ATTACH_EVIDENCE = "case:attach_evidence"
    RECORD_RGS_DECISION = "case:record_rgs_decision"
    VERIFY_INTEGRITY = "case:verify_integrity"


INVESTIGATOR = "investigator"
AUTHORIZED_RGS_REVIEWER = "authorized_rgs_reviewer"
SUPPORTED_ROLES = frozenset({INVESTIGATOR, AUTHORIZED_RGS_REVIEWER})
LEGACY_UNSCOPED_ORGANIZATION = "legacy-unscoped"

ROLE_PERMISSIONS = {
    INVESTIGATOR: frozenset(
        {
            Permission.READ_CASE,
            Permission.CREATE_ALERT,
            Permission.START_REVIEW,
            Permission.ATTACH_EVIDENCE,
            Permission.VERIFY_INTEGRITY,
        }
    ),
    AUTHORIZED_RGS_REVIEWER: frozenset(Permission),
}

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/-]{1,127}$")


def auth_mode(environ: Mapping[str, str] | None = None) -> AuthMode:
    source = os.environ if environ is None else environ
    raw = source.get(
        "AEGIS_SECURITY_MODE",
        source.get("AEGIS_AUTH_MODE", AuthMode.DEVELOPMENT.value),
    ).strip().lower()
    try:
        return AuthMode(raw)
    except ValueError as exc:
        raise SecurityError(
            "AEGIS_SECURITY_MODE must be 'development' or 'production'; refusing an unknown mode"
        ) from exc


def _identifier(value: str | None, field_name: str) -> str:
    clean = str(value or "").strip()
    if not _IDENTIFIER.fullmatch(clean):
        raise AuthenticationRequired(
            f"{field_name} must be 2-128 characters using safe identifier characters"
        )
    return clean


def _roles(values: Iterable[str]) -> frozenset[str]:
    roles = frozenset(str(value).strip() for value in values if str(value).strip())
    unknown = roles - SUPPORTED_ROLES
    if unknown:
        raise AuthenticationRequired(f"Identity contains unsupported roles: {sorted(unknown)}")
    if not roles:
        raise AuthenticationRequired("Identity must contain at least one supported role")
    return roles


@dataclass(frozen=True)
class SecurityPrincipal:
    """Authenticated identity produced by a trusted adapter.

    No raw token or secret is retained.  ``provider`` and ``assurance`` are safe audit
    labels, while ``session_id`` is optional because non-interactive adapters may not
    issue sessions.
    """

    subject: str
    roles: frozenset[str]
    provider: str
    assurance: str
    organization_id: str | None = None
    session_id: str | None = None
    authenticated: bool = True
    attributes: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject", _identifier(self.subject, "subject"))
        object.__setattr__(self, "roles", _roles(self.roles))
        object.__setattr__(self, "provider", _identifier(self.provider, "provider"))
        object.__setattr__(self, "assurance", _identifier(self.assurance, "assurance"))
        if self.organization_id is not None:
            object.__setattr__(
                self,
                "organization_id",
                _identifier(self.organization_id, "organization_id"),
            )
        if self.session_id is not None:
            object.__setattr__(self, "session_id", _identifier(self.session_id, "session_id"))

    @property
    def permissions(self) -> frozenset[Permission]:
        granted: set[Permission] = set()
        for role in self.roles:
            granted.update(ROLE_PERMISSIONS.get(role, ()))
        return frozenset(granted)

    def audit_metadata(self) -> dict[str, object]:
        """Return token-free identity data safe to commit to the event chain."""
        return {
            "subject": self.subject,
            "provider": self.provider,
            "assurance": self.assurance,
            "organization_id": self.organization_id,
            "session_id": self.session_id,
            "roles": sorted(self.roles),
        }


def principal_from_verified_claims(
    *,
    subject: str,
    roles: Iterable[str],
    provider: str,
    organization_id: str | None = None,
    session_id: str | None = None,
    assurance: str = "verified_session",
) -> SecurityPrincipal:
    """Adapter seam for Clerk or another IdP after it verifies the session token."""
    return SecurityPrincipal(
        subject=subject,
        roles=_roles(roles),
        provider=provider,
        assurance=assurance,
        organization_id=organization_id,
        session_id=session_id,
    )


def local_development_principal(
    *,
    subject: str | None = None,
    roles: Iterable[str] | None = None,
    organization_id: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> SecurityPrincipal:
    """Create an explicitly non-production identity for local demos and tests."""
    source = os.environ if environ is None else environ
    if auth_mode(source) is AuthMode.PRODUCTION:
        raise AuthenticationRequired(
            "No production identity adapter is configured. Integrate Clerk (or another "
            "verified provider) before enabling production mode."
        )
    configured_roles = source.get("AEGIS_DEV_ROLES", INVESTIGATOR).split(",")
    return SecurityPrincipal(
        subject=subject or source.get("AEGIS_DEV_SUBJECT", "local-investigator"),
        roles=_roles(roles if roles is not None else configured_roles),
        provider="local-development",
        assurance="self-attested-development-only",
        organization_id=(
            organization_id or source.get("AEGIS_DEV_ORGANIZATION", "local-demo")
        ),
        session_id=source.get("AEGIS_DEV_SESSION", "local-session"),
    )


def require_permission(
    principal: SecurityPrincipal,
    permission: Permission,
    *,
    resource_organization_id: str | None = None,
) -> None:
    if not principal.authenticated:
        raise AuthenticationRequired("An authenticated principal is required")
    if permission not in principal.permissions:
        raise PermissionDenied(
            f"Identity {principal.subject} lacks required permission {permission.value}"
        )
    if resource_organization_id == LEGACY_UNSCOPED_ORGANIZATION:
        if principal.provider != "local-development":
            raise PermissionDenied(
                "Legacy case has no verified organization owner; production access is denied"
            )
    elif resource_organization_id and principal.organization_id != resource_organization_id:
        raise PermissionDenied("Cross-organization case access is denied")


def role_for_audit(principal: SecurityPrincipal, permission: Permission) -> str:
    """Choose one role that actually grants the authorized operation."""
    preferred = (
        AUTHORIZED_RGS_REVIEWER
        if permission is Permission.RECORD_RGS_DECISION
        else INVESTIGATOR
    )
    if preferred in principal.roles and permission in ROLE_PERMISSIONS[preferred]:
        return preferred
    for role in sorted(principal.roles):
        if permission in ROLE_PERMISSIONS[role]:
            return role
    raise PermissionDenied(f"No principal role grants {permission.value}")
