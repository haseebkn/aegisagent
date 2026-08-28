"""Human-review case state machine for model-generated investigation alerts.

The model may create an alert, but only an explicitly identified human reviewer can
move the resulting case to an RGS disposition.  This module deliberately contains no
filing or submission state: reaching RGS hands the case to an approved reporting
workflow that is outside this portfolio project's scope.
"""

from __future__ import annotations

import json
import math
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from scripts.config import CASE_DB_PATH


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


class CaseStore:
    """SQLite-backed case projection with an append-only application event history."""

    def __init__(self, path: str | Path = CASE_DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

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
                    UNIQUE(case_id, sequence)
                );

                CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status, updated_at);
                CREATE INDEX IF NOT EXISTS idx_events_case ON case_events(case_id, sequence);
                """
            )

    @staticmethod
    def _case(row: sqlite3.Row) -> CaseRecord:
        return CaseRecord(**dict(row))

    @staticmethod
    def _event(row: sqlite3.Row) -> CaseEvent:
        values = dict(row)
        values["metadata"] = json.loads(values.pop("metadata_json"))
        return CaseEvent(**values)

    def create_alert_case(
        self,
        *,
        trans_num: str,
        model_score: float,
        threshold: float,
        model_version: str,
        actor: str,
        actor_role: ReviewerRole | str = ReviewerRole.INVESTIGATOR,
        metadata: dict[str, Any] | None = None,
    ) -> CaseRecord:
        trans_num = _required_text(trans_num, "trans_num")
        model_version = _required_text(model_version, "model_version")
        actor = _required_text(actor, "actor", 2)
        role = _role(actor_role)
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
        })
        try:
            with self._connect() as con:
                con.execute("BEGIN IMMEDIATE")
                con.execute(
                    """INSERT INTO cases
                       (case_id, trans_num, model_score, threshold, model_version, status,
                        assigned_to, created_at, updated_at, version)
                       VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?, 1)""",
                    (case_id, trans_num, model_score, threshold, model_version,
                     CaseStatus.ALERT_OPEN.value, timestamp, timestamp),
                )
                con.execute(
                    """INSERT INTO case_events
                       (event_id, case_id, sequence, event_type, from_status, to_status,
                        actor, actor_role, rationale, metadata_json, occurred_at)
                       VALUES (?, ?, 1, 'alert_created', NULL, ?, ?, ?, ?, ?, ?)""",
                    (event_id, case_id, CaseStatus.ALERT_OPEN.value, actor, role.value,
                     "Model score met the configured investigation-alert threshold.",
                     json.dumps(event_metadata, sort_keys=True), timestamp),
                )
        except sqlite3.IntegrityError as exc:
            if "trans_num" in str(exc) or "UNIQUE constraint failed: cases.trans_num" in str(exc):
                raise DuplicateAlert(f"A case already exists for transaction {trans_num}") from exc
            raise
        return self.get_case(case_id)

    def get_case(self, case_id: str) -> CaseRecord:
        with self._connect() as con:
            row = con.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        if row is None:
            raise CaseNotFound(f"Unknown case: {case_id}")
        return self._case(row)

    def find_by_transaction(self, trans_num: str) -> CaseRecord | None:
        with self._connect() as con:
            row = con.execute("SELECT * FROM cases WHERE trans_num = ?", (trans_num,)).fetchone()
        return self._case(row) if row else None

    def list_cases(self, status: CaseStatus | str | None = None) -> list[CaseRecord]:
        query = "SELECT * FROM cases"
        params: tuple[str, ...] = ()
        if status is not None:
            status_value = CaseStatus(status).value
            query += " WHERE status = ?"
            params = (status_value,)
        query += " ORDER BY updated_at DESC, case_id"
        with self._connect() as con:
            rows = con.execute(query, params).fetchall()
        return [self._case(row) for row in rows]

    def history(self, case_id: str) -> list[CaseEvent]:
        self.get_case(case_id)
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM case_events WHERE case_id = ? ORDER BY sequence", (case_id,)
            ).fetchall()
        return [self._event(row) for row in rows]

    def start_review(
        self,
        case_id: str,
        *,
        actor: str,
        actor_role: ReviewerRole | str,
        rationale: str,
        expected_version: int,
    ) -> CaseRecord:
        role = _role(actor_role)
        if role not in {ReviewerRole.INVESTIGATOR, ReviewerRole.AUTHORIZED_RGS_REVIEWER}:
            raise AuthorizationError(f"Role {role.value} cannot start a review")
        return self._transition(
            case_id,
            expected_from=CaseStatus.ALERT_OPEN,
            to_status=CaseStatus.UNDER_REVIEW,
            event_type="review_started",
            actor=actor,
            actor_role=role,
            rationale=rationale,
            expected_version=expected_version,
            assigned_to=_required_text(actor, "actor", 2),
        )

    def record_rgs_decision(
        self,
        case_id: str,
        *,
        reached: bool,
        actor: str,
        actor_role: ReviewerRole | str,
        rationale: str,
        expected_version: int,
    ) -> CaseRecord:
        role = _role(actor_role)
        if role is not ReviewerRole.AUTHORIZED_RGS_REVIEWER:
            raise AuthorizationError(
                "Only an explicitly authorized RGS reviewer may record an RGS disposition"
            )
        to_status = CaseStatus.RGS_REACHED if reached else CaseStatus.RGS_NOT_REACHED
        return self._transition(
            case_id,
            expected_from=CaseStatus.UNDER_REVIEW,
            to_status=to_status,
            event_type=to_status.value,
            actor=actor,
            actor_role=role,
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
            con.execute(
                """INSERT INTO case_events
                   (event_id, case_id, sequence, event_type, from_status, to_status,
                    actor, actor_role, rationale, metadata_json, occurred_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', ?)""",
                (str(uuid.uuid4()), case_id, next_version, event_type,
                 current.status, to_status.value, actor, actor_role.value,
                 rationale, timestamp),
            )
        return self.get_case(case_id)


def case_to_dict(record: CaseRecord) -> dict[str, Any]:
    return asdict(record)


def event_to_dict(event: CaseEvent) -> dict[str, Any]:
    return asdict(event)
