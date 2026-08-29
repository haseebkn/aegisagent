"""Human-review case state machine for model-generated investigation alerts.

The model may create an alert, but only an explicitly identified human reviewer can
move the resulting case to an RGS disposition.  This module deliberately contains no
filing or submission state: reaching RGS hands the case to an approved reporting
workflow that is outside this portfolio project's scope.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from scripts.config import CASE_DB_PATH
from scripts.evidence import EvidenceReceipt, verify_file
from scripts.security import (
    LEGACY_UNSCOPED_ORGANIZATION,
    Permission,
    SecurityError,
    SecurityPrincipal,
    local_development_principal,
    require_permission,
    role_for_audit,
)


class CaseStatus(str, Enum):
    ALERT_OPEN = "alert_open"
    UNDER_REVIEW = "under_review"
    RGS_NOT_REACHED = "rgs_not_reached"
    RGS_REACHED = "rgs_reached"


class ReviewerRole(str, Enum):
    INVESTIGATOR = "investigator"
    AUTHORIZED_RGS_REVIEWER = "authorized_rgs_reviewer"


TERMINAL_STATUSES = frozenset({CaseStatus.RGS_NOT_REACHED, CaseStatus.RGS_REACHED})


class CaseManagementError(RuntimeError):
    """Base error for rejected case-management operations."""


class InvalidTransition(CaseManagementError):
    pass


class AuthorizationError(CaseManagementError):
    pass


class ConcurrencyError(CaseManagementError):
    pass


class CaseNotFound(CaseManagementError):
    pass


class DuplicateAlert(CaseManagementError):
    pass


class ValidationError(CaseManagementError):
    pass


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    trans_num: str
    model_score: float
    threshold: float
    model_version: str
    organization_id: str | None
    status: str
    assigned_to: str | None
    created_at: str
    updated_at: str
    version: int


@dataclass(frozen=True)
class CaseEvent:
    event_id: str
    case_id: str
    sequence: int
    event_type: str
    from_status: str | None
    to_status: str
    actor: str
    actor_role: str
    rationale: str
    metadata: dict[str, Any]
    occurred_at: str
    previous_event_hash: str
    event_hash: str


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    case_id: str
    evidence_type: str
    sha256: str
    byte_size: int
    media_type: str
    local_path: str
    archive_receipt: dict[str, Any] | None
    created_at: str
    created_by: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _required_text(value: str, field: str, minimum: int = 1) -> str:
    clean = str(value or "").strip()
    if len(clean) < minimum:
        raise ValidationError(f"{field} must contain at least {minimum} characters")
    return clean


def _role(value: ReviewerRole | str) -> ReviewerRole:
    try:
        return ReviewerRole(value)
    except ValueError as exc:
        raise AuthorizationError(f"Unknown reviewer role: {value}") from exc


def _event_digest(
    *,
    event_id: str,
    case_id: str,
    sequence: int,
    event_type: str,
    from_status: str | None,
    to_status: str,
    actor: str,
    actor_role: str,
    rationale: str,
    metadata_json: str,
    occurred_at: str,
    previous_event_hash: str,
) -> str:
    canonical = json.dumps(
        {
            "event_id": event_id,
            "case_id": case_id,
            "sequence": sequence,
            "event_type": event_type,
            "from_status": from_status,
            "to_status": to_status,
            "actor": actor,
            "actor_role": actor_role,
            "rationale": rationale,
            "metadata": json.loads(metadata_json),
            "occurred_at": occurred_at,
            "previous_event_hash": previous_event_hash,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class CaseStore:
    """SQLite-backed case projection with an append-only application event history."""

    def __init__(
        self,
        path: str | Path = CASE_DB_PATH,
        *,
        principal: SecurityPrincipal | None = None,
    ):
        self.path = Path(path)
        self.principal = principal
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        self._initialize()
        os.chmod(self.path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        return con

    def _initialize(self) -> None:
        with self._connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    trans_num TEXT NOT NULL UNIQUE,
                    model_score REAL NOT NULL CHECK(model_score >= 0 AND model_score <= 1),
                    threshold REAL NOT NULL CHECK(threshold >= 0 AND threshold <= 1),
                    model_version TEXT NOT NULL,
                    organization_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN (
                        'alert_open', 'under_review', 'rgs_not_reached', 'rgs_reached'
                    )),
                    assigned_to TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version INTEGER NOT NULL CHECK(version >= 1)
                );

                CREATE TABLE IF NOT EXISTS case_events (
                    event_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES cases(case_id),
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    from_status TEXT,
                    to_status TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    actor_role TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    previous_event_hash TEXT NOT NULL DEFAULT '',
                    event_hash TEXT NOT NULL DEFAULT '',
                    UNIQUE(case_id, sequence)
                );

                CREATE TABLE IF NOT EXISTS case_evidence (
                    evidence_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES cases(case_id),
                    evidence_type TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    byte_size INTEGER NOT NULL CHECK(byte_size >= 0),
                    media_type TEXT NOT NULL,
                    local_path TEXT NOT NULL,
                    archive_receipt_json TEXT,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    UNIQUE(case_id, evidence_type, sha256)
                );

                CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status, updated_at);
                CREATE INDEX IF NOT EXISTS idx_events_case ON case_events(case_id, sequence);
                CREATE INDEX IF NOT EXISTS idx_evidence_case ON case_evidence(case_id, created_at);
                """
            )
            columns = {row[1] for row in con.execute("PRAGMA table_info(case_events)")}
            needs_hash_backfill = False
            if "previous_event_hash" not in columns:
                con.execute(
                    "ALTER TABLE case_events ADD COLUMN previous_event_hash TEXT NOT NULL DEFAULT ''"
                )
                needs_hash_backfill = True
            if "event_hash" not in columns:
                con.execute(
                    "ALTER TABLE case_events ADD COLUMN event_hash TEXT NOT NULL DEFAULT ''"
                )
                needs_hash_backfill = True
            incomplete_hashes = con.execute(
                "SELECT 1 FROM case_events WHERE event_hash = '' LIMIT 1"
            ).fetchone()
            if needs_hash_backfill or incomplete_hashes:
                self._backfill_event_hashes(con)
            case_columns = {row[1] for row in con.execute("PRAGMA table_info(cases)")}
            if "organization_id" not in case_columns:
                con.execute("ALTER TABLE cases ADD COLUMN organization_id TEXT")
            con.execute(
                "UPDATE cases SET organization_id = ? "
                "WHERE organization_id IS NULL OR TRIM(organization_id) = ''",
                (LEGACY_UNSCOPED_ORGANIZATION,),
            )

    def _identity(
        self,
        permission: Permission,
        *,
        principal: SecurityPrincipal | None = None,
        actor: str | None = None,
        actor_role: ReviewerRole | str | None = None,
        resource_organization_id: str | None = None,
    ) -> tuple[SecurityPrincipal, str]:
        """Resolve a trusted principal and enforce policy in one place.

        Actor/role inputs are retained only as a development compatibility seam.  The
        local provider refuses to create them in production mode.
        """
        identity = principal or self.principal
        try:
            if identity is None:
                roles = [ReviewerRole(actor_role).value] if actor_role is not None else None
                identity = local_development_principal(subject=actor, roles=roles)
            elif actor is not None and str(actor).strip() != identity.subject:
                raise AuthorizationError("Actor does not match the authenticated identity")
            elif actor_role is not None and ReviewerRole(actor_role).value not in identity.roles:
                raise AuthorizationError("Asserted role is not granted to the identity")
            require_permission(
                identity,
                permission,
                resource_organization_id=resource_organization_id,
            )
            if identity.organization_id is None:
                raise AuthorizationError("A verified organization identity is required")
            return identity, role_for_audit(identity, permission)
        except (SecurityError, ValueError) as exc:
            if permission is Permission.RECORD_RGS_DECISION:
                raise AuthorizationError(
                    "Only an authorized RGS reviewer may record an RGS disposition"
                ) from exc
            raise AuthorizationError(str(exc)) from exc

    @staticmethod
    def _backfill_event_hashes(con: sqlite3.Connection) -> None:
        case_ids = [row[0] for row in con.execute("SELECT DISTINCT case_id FROM case_events")]
        for case_id in case_ids:
            previous = ""
            rows = con.execute(
                "SELECT * FROM case_events WHERE case_id = ? ORDER BY sequence", (case_id,)
            ).fetchall()
            for row in rows:
                values = dict(row)
                digest = _event_digest(
                    event_id=values["event_id"],
                    case_id=values["case_id"],
                    sequence=values["sequence"],
                    event_type=values["event_type"],
                    from_status=values["from_status"],
                    to_status=values["to_status"],
                    actor=values["actor"],
                    actor_role=values["actor_role"],
                    rationale=values["rationale"],
                    metadata_json=values["metadata_json"],
                    occurred_at=values["occurred_at"],
                    previous_event_hash=previous,
                )
                if values.get("previous_event_hash") != previous or values.get("event_hash") != digest:
                    con.execute(
                        """UPDATE case_events
                           SET previous_event_hash = ?, event_hash = ? WHERE event_id = ?""",
                        (previous, digest, values["event_id"]),
                    )
                previous = digest

    @staticmethod
    def _case(row: sqlite3.Row) -> CaseRecord:
        return CaseRecord(**dict(row))

    @staticmethod
    def _event(row: sqlite3.Row) -> CaseEvent:
        values = dict(row)
        values["metadata"] = json.loads(values.pop("metadata_json"))
        return CaseEvent(**values)

    @staticmethod
    def _evidence(row: sqlite3.Row) -> EvidenceRecord:
        values = dict(row)
        archive_json = values.pop("archive_receipt_json")
        values["archive_receipt"] = json.loads(archive_json) if archive_json else None
        return EvidenceRecord(**values)

    def create_alert_case(
        self,
        *,
        trans_num: str,
        model_score: float,
        threshold: float,
        model_version: str,
        actor: str | None = None,
        actor_role: ReviewerRole | str | None = None,
        principal: SecurityPrincipal | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CaseRecord:
        trans_num = _required_text(trans_num, "trans_num")
        model_version = _required_text(model_version, "model_version")
        identity, role_value = self._identity(
            Permission.CREATE_ALERT,
            principal=principal,
            actor=actor,
            actor_role=actor_role,
        )
        actor = identity.subject
        role = _role(role_value)
        if not math.isfinite(model_score) or not math.isfinite(threshold):
            raise ValidationError("model_score and threshold must be finite")
        if not 0 <= threshold <= 1 or not 0 <= model_score <= 1:
            raise ValidationError("model_score and threshold must be between 0 and 1")
        if model_score < threshold:
            raise ValidationError(
                f"Cannot create an alert case: score {model_score:.4f} is below "
                f"threshold {threshold:.4f}"
            )

        case_id = f"CASE-{uuid.uuid4().hex[:12].upper()}"
        timestamp = _now()
        event_id = str(uuid.uuid4())
        event_metadata = dict(metadata or {})
        event_metadata.update({
            "model_score": model_score,
            "threshold": threshold,
            "model_version": model_version,
            "identity": identity.audit_metadata(),
        })
        rationale = "Model score met the configured investigation-alert threshold."
        metadata_json = json.dumps(event_metadata, sort_keys=True)
        event_hash = _event_digest(
            event_id=event_id,
            case_id=case_id,
            sequence=1,
            event_type="alert_created",
            from_status=None,
            to_status=CaseStatus.ALERT_OPEN.value,
            actor=actor,
            actor_role=role.value,
            rationale=rationale,
            metadata_json=metadata_json,
            occurred_at=timestamp,
            previous_event_hash="",
        )
        try:
            with self._connect() as con:
                con.execute("BEGIN IMMEDIATE")
                con.execute(
                    """INSERT INTO cases
                       (case_id, trans_num, model_score, threshold, model_version,
                        organization_id, status, assigned_to, created_at, updated_at, version)
                       VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, 1)""",
                    (case_id, trans_num, model_score, threshold, model_version,
                     identity.organization_id, CaseStatus.ALERT_OPEN.value, timestamp, timestamp),
                )
                con.execute(
                    """INSERT INTO case_events
                       (event_id, case_id, sequence, event_type, from_status, to_status,
                        actor, actor_role, rationale, metadata_json, occurred_at,
                        previous_event_hash, event_hash)
                       VALUES (?, ?, 1, 'alert_created', NULL, ?, ?, ?, ?, ?, ?, '', ?)""",
                    (event_id, case_id, CaseStatus.ALERT_OPEN.value, actor, role.value,
                     rationale, metadata_json, timestamp, event_hash),
                )
        except sqlite3.IntegrityError as exc:
            if "trans_num" in str(exc) or "UNIQUE constraint failed: cases.trans_num" in str(exc):
                raise DuplicateAlert(f"A case already exists for transaction {trans_num}") from exc
            raise
        return self.get_case(case_id, principal=identity)

    def get_case(
        self, case_id: str, *, principal: SecurityPrincipal | None = None
    ) -> CaseRecord:
        with self._connect() as con:
            row = con.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        if row is None:
            raise CaseNotFound(f"Unknown case: {case_id}")
        case = self._case(row)
        self._identity(
            Permission.READ_CASE,
            principal=principal,
            resource_organization_id=case.organization_id,
        )
        return case

    def find_by_transaction(
        self, trans_num: str, *, principal: SecurityPrincipal | None = None
    ) -> CaseRecord | None:
        with self._connect() as con:
            row = con.execute("SELECT * FROM cases WHERE trans_num = ?", (trans_num,)).fetchone()
        if row is None:
            return None
        case = self._case(row)
        self._identity(
            Permission.READ_CASE,
            principal=principal,
            resource_organization_id=case.organization_id,
        )
        return case

    def list_cases(
        self,
        status: CaseStatus | str | None = None,
        *,
        principal: SecurityPrincipal | None = None,
    ) -> list[CaseRecord]:
        identity, _ = self._identity(Permission.READ_CASE, principal=principal)
        query = "SELECT * FROM cases"
        clauses: list[str] = []
        params: list[str] = []
        if identity.provider == "local-development":
            clauses.append("organization_id IN (?, ?)")
            params.extend([identity.organization_id, LEGACY_UNSCOPED_ORGANIZATION])
        else:
            clauses.append("organization_id = ?")
            params.append(identity.organization_id)
        if status is not None:
            status_value = CaseStatus(status).value
            clauses.append("status = ?")
            params.append(status_value)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC, case_id"
        with self._connect() as con:
            rows = con.execute(query, tuple(params)).fetchall()
        return [self._case(row) for row in rows]

    def history(
        self, case_id: str, *, principal: SecurityPrincipal | None = None
    ) -> list[CaseEvent]:
        self.get_case(case_id, principal=principal)
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM case_events WHERE case_id = ? ORDER BY sequence", (case_id,)
            ).fetchall()
        return [self._event(row) for row in rows]

    def list_evidence(
        self, case_id: str, *, principal: SecurityPrincipal | None = None
    ) -> list[EvidenceRecord]:
        self.get_case(case_id, principal=principal)
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM case_evidence WHERE case_id = ? ORDER BY created_at, evidence_id",
                (case_id,),
            ).fetchall()
        return [self._evidence(row) for row in rows]

    def attach_evidence(
        self,
        case_id: str,
        *,
        receipt: EvidenceReceipt,
        actor: str | None = None,
        actor_role: ReviewerRole | str | None = None,
        principal: SecurityPrincipal | None = None,
        expected_version: int,
    ) -> CaseRecord:
        if receipt.case_id != case_id:
            raise ValidationError(
                f"Evidence belongs to {receipt.case_id}, not requested case {case_id}"
            )
        timestamp = _now()
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
            if row is None:
                raise CaseNotFound(f"Unknown case: {case_id}")
            current = self._case(row)
            identity, role_value = self._identity(
                Permission.ATTACH_EVIDENCE,
                principal=principal,
                actor=actor,
                actor_role=actor_role,
                resource_organization_id=current.organization_id,
            )
            actor = identity.subject
            role = _role(role_value)
            metadata = asdict(receipt)
            metadata["identity"] = identity.audit_metadata()
            metadata_json = json.dumps(metadata, sort_keys=True)
            rationale = (
                f"Preserved and linked {receipt.evidence_type} evidence {receipt.evidence_id}."
            )
            if current.version != expected_version:
                raise ConcurrencyError(
                    f"Case {case_id} changed from version {expected_version} to "
                    f"{current.version}; reload before attaching evidence"
                )
            if current.status != CaseStatus.UNDER_REVIEW.value:
                raise InvalidTransition(
                    f"Evidence may only be attached during active review, not {current.status}"
                )
            next_version = current.version + 1
            previous_hash = con.execute(
                "SELECT event_hash FROM case_events WHERE case_id = ? ORDER BY sequence DESC LIMIT 1",
                (case_id,),
            ).fetchone()[0]
            event_id = str(uuid.uuid4())
            event_hash = _event_digest(
                event_id=event_id,
                case_id=case_id,
                sequence=next_version,
                event_type="evidence_attached",
                from_status=current.status,
                to_status=current.status,
                actor=actor,
                actor_role=role.value,
                rationale=rationale,
                metadata_json=metadata_json,
                occurred_at=timestamp,
                previous_event_hash=previous_hash,
            )
            try:
                con.execute(
                    """INSERT INTO case_evidence
                       (evidence_id, case_id, evidence_type, sha256, byte_size, media_type,
                        local_path, archive_receipt_json, created_at, created_by)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        receipt.evidence_id,
                        case_id,
                        receipt.evidence_type,
                        receipt.sha256,
                        receipt.byte_size,
                        receipt.media_type,
                        receipt.local_path,
                        json.dumps(receipt.archive_receipt, sort_keys=True)
                        if receipt.archive_receipt else None,
                        receipt.created_at,
                        actor,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValidationError(
                    f"Identical {receipt.evidence_type} evidence is already linked to {case_id}"
                ) from exc
            con.execute(
                """UPDATE cases SET updated_at = ?, version = ?
                   WHERE case_id = ? AND version = ?""",
                (timestamp, next_version, case_id, expected_version),
            )
            con.execute(
                """INSERT INTO case_events
                   (event_id, case_id, sequence, event_type, from_status, to_status,
                    actor, actor_role, rationale, metadata_json, occurred_at,
                    previous_event_hash, event_hash)
                   VALUES (?, ?, ?, 'evidence_attached', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event_id,
                    case_id,
                    next_version,
                    current.status,
                    current.status,
                    actor,
                    role.value,
                    rationale,
                    metadata_json,
                    timestamp,
                    previous_hash,
                    event_hash,
                ),
            )
        return self.get_case(case_id, principal=identity)

    def verify_integrity(
        self, case_id: str, *, principal: SecurityPrincipal | None = None
    ) -> dict[str, Any]:
        case = self.get_case(case_id, principal=principal)
        identity, _ = self._identity(
            Permission.VERIFY_INTEGRITY,
            principal=principal,
            resource_organization_id=case.organization_id,
        )
        history = self.history(case_id, principal=identity)
        evidence = self.list_evidence(case_id, principal=identity)
        issues: list[str] = []
        previous = ""
        for event in history:
            metadata_json = json.dumps(event.metadata, sort_keys=True)
            expected = _event_digest(
                event_id=event.event_id,
                case_id=event.case_id,
                sequence=event.sequence,
                event_type=event.event_type,
                from_status=event.from_status,
                to_status=event.to_status,
                actor=event.actor,
                actor_role=event.actor_role,
                rationale=event.rationale,
                metadata_json=metadata_json,
                occurred_at=event.occurred_at,
                previous_event_hash=previous,
            )
            if event.previous_event_hash != previous:
                issues.append(f"Event {event.sequence} has a broken previous-hash link")
            if event.event_hash != expected:
                issues.append(f"Event {event.sequence} content hash does not match")
            previous = event.event_hash
        if history and case.version != history[-1].sequence:
            issues.append(
                f"Case version {case.version} does not match last event {history[-1].sequence}"
            )
        if history and case.organization_id != LEGACY_UNSCOPED_ORGANIZATION:
            committed_organization = (
                history[0].metadata.get("identity", {}).get("organization_id")
            )
            if committed_organization != case.organization_id:
                issues.append(
                    "Case organization does not match the organization committed "
                    "in the initial event"
                )
        evidence_events = {
            event.metadata.get("evidence_id"): event.metadata
            for event in history
            if event.event_type == "evidence_attached"
        }
        registered_ids = {item.evidence_id for item in evidence}
        for evidence_id in evidence_events:
            if evidence_id not in registered_ids:
                issues.append(f"Evidence event {evidence_id} has no matching evidence record")
        for item in evidence:
            issues.extend(verify_file(item.local_path, item.sha256, item.byte_size))
            event_metadata = evidence_events.get(item.evidence_id)
            if event_metadata is None:
                issues.append(f"Evidence {item.evidence_id} has no matching case event")
            else:
                committed = {
                    "evidence_type": item.evidence_type,
                    "sha256": item.sha256,
                    "byte_size": item.byte_size,
                    "media_type": item.media_type,
                    "local_path": item.local_path,
                    "archive_receipt": item.archive_receipt,
                }
                for field, value in committed.items():
                    if event_metadata.get(field) != value:
                        issues.append(
                            f"Evidence {item.evidence_id} field {field} differs from its case event"
                        )
            if item.archive_receipt and not item.archive_receipt.get("verified"):
                issues.append(f"Evidence {item.evidence_id} has an unverified archive receipt")
        return {
            "ok": not issues,
            "case_id": case_id,
            "case_version": case.version,
            "events_verified": len(history),
            "evidence_verified": len(evidence),
            "chain_head": previous,
            "issues": issues,
            "verified_at": _now(),
        }

    def start_review(
        self,
        case_id: str,
        *,
        actor: str | None = None,
        actor_role: ReviewerRole | str | None = None,
        principal: SecurityPrincipal | None = None,
        rationale: str,
        expected_version: int,
    ) -> CaseRecord:
        current = self.get_case(case_id, principal=principal)
        identity, role_value = self._identity(
            Permission.START_REVIEW,
            principal=principal,
            actor=actor,
            actor_role=actor_role,
            resource_organization_id=current.organization_id,
        )
        role = _role(role_value)
        return self._transition(
            case_id,
            expected_from=CaseStatus.ALERT_OPEN,
            to_status=CaseStatus.UNDER_REVIEW,
            event_type="review_started",
            actor=identity.subject,
            actor_role=role,
            identity=identity,
            rationale=rationale,
            expected_version=expected_version,
            assigned_to=identity.subject,
        )

    def record_rgs_decision(
        self,
        case_id: str,
        *,
        reached: bool,
        actor: str | None = None,
        actor_role: ReviewerRole | str | None = None,
        principal: SecurityPrincipal | None = None,
        rationale: str,
        expected_version: int,
    ) -> CaseRecord:
        current = self.get_case(case_id, principal=principal)
        identity, role_value = self._identity(
            Permission.RECORD_RGS_DECISION,
            principal=principal,
            actor=actor,
            actor_role=actor_role,
            resource_organization_id=current.organization_id,
        )
        role = _role(role_value)
        to_status = CaseStatus.RGS_REACHED if reached else CaseStatus.RGS_NOT_REACHED
        return self._transition(
            case_id,
            expected_from=CaseStatus.UNDER_REVIEW,
            to_status=to_status,
            event_type=to_status.value,
            actor=identity.subject,
            actor_role=role,
            identity=identity,
            rationale=rationale,
            expected_version=expected_version,
        )

    def _transition(
        self,
        case_id: str,
        *,
        expected_from: CaseStatus,
        to_status: CaseStatus,
        event_type: str,
        actor: str,
        actor_role: ReviewerRole,
        identity: SecurityPrincipal,
        rationale: str,
        expected_version: int,
        assigned_to: str | None = None,
    ) -> CaseRecord:
        actor = _required_text(actor, "actor", 2)
        rationale = _required_text(rationale, "rationale", 20)
        timestamp = _now()
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
            if row is None:
                raise CaseNotFound(f"Unknown case: {case_id}")
            current = self._case(row)
            if current.version != expected_version:
                raise ConcurrencyError(
                    f"Case {case_id} changed from version {expected_version} to "
                    f"{current.version}; reload before deciding"
                )
            if CaseStatus(current.status) in TERMINAL_STATUSES:
                raise InvalidTransition(f"Case {case_id} is terminal at {current.status}")
            if current.status != expected_from.value:
                raise InvalidTransition(
                    f"Cannot move case {case_id} from {current.status} to {to_status.value}"
                )

            next_version = current.version + 1
            next_assignee = assigned_to if assigned_to is not None else current.assigned_to
            updated = con.execute(
                """UPDATE cases
                   SET status = ?, assigned_to = ?, updated_at = ?, version = ?
                   WHERE case_id = ? AND version = ?""",
                (to_status.value, next_assignee, timestamp, next_version,
                 case_id, expected_version),
            )
            if updated.rowcount != 1:
                raise ConcurrencyError(f"Case {case_id} changed while the transition was saved")
            previous_hash_row = con.execute(
                "SELECT event_hash FROM case_events WHERE case_id = ? ORDER BY sequence DESC LIMIT 1",
                (case_id,),
            ).fetchone()
            previous_hash = previous_hash_row[0] if previous_hash_row else ""
            event_id = str(uuid.uuid4())
            metadata_json = json.dumps({"identity": identity.audit_metadata()}, sort_keys=True)
            event_hash = _event_digest(
                event_id=event_id,
                case_id=case_id,
                sequence=next_version,
                event_type=event_type,
                from_status=current.status,
                to_status=to_status.value,
                actor=actor,
                actor_role=actor_role.value,
                rationale=rationale,
                metadata_json=metadata_json,
                occurred_at=timestamp,
                previous_event_hash=previous_hash,
            )
            con.execute(
                """INSERT INTO case_events
                   (event_id, case_id, sequence, event_type, from_status, to_status,
                    actor, actor_role, rationale, metadata_json, occurred_at,
                    previous_event_hash, event_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (event_id, case_id, next_version, event_type,
                 current.status, to_status.value, actor, actor_role.value,
                 rationale, metadata_json, timestamp, previous_hash, event_hash),
            )
        return self.get_case(case_id, principal=identity)


def case_to_dict(record: CaseRecord) -> dict[str, Any]:
    return asdict(record)


def event_to_dict(event: CaseEvent) -> dict[str, Any]:
    return asdict(event)


def evidence_to_dict(evidence: EvidenceRecord) -> dict[str, Any]:
    return asdict(evidence)
