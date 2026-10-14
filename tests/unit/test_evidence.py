import base64
import sqlite3
from types import SimpleNamespace

import pytest

from scripts.case_management import (
    AuthorizationError,
    CaseStatus,
    CaseStore,
    ConcurrencyError,
    InvalidTransition,
    ReviewerRole,
    ValidationError,
)
from scripts.evidence import EvidenceStore, archive_to_s3, sha256_bytes
from scripts import sar_agent
from scripts.security import MODEL_GOVERNANCE_REVIEWER, principal_from_verified_claims


class S3Stub:
    def __init__(self, *, version_id="version-7", echo_checksum=True):
        self.version_id = version_id
        self.echo_checksum = echo_checksum
        self.request = None

    def put_object(self, **kwargs):
        self.request = kwargs
        return {
            "ChecksumSHA256": kwargs["ChecksumSHA256"] if self.echo_checksum else "wrong",
            "VersionId": self.version_id,
            "ETag": '"etag-value"',
            "ResponseMetadata": {"RequestId": "request-9"},
        }


@pytest.fixture
def active_case(tmp_path):
    store = CaseStore(tmp_path / "cases.sqlite3")
    case = store.create_alert_case(
        trans_num="txn-evidence",
        model_score=0.91,
        threshold=0.80,
        model_version="v-evidence",
        actor="queue-worker",
    )
    case = store.start_review(
        case.case_id,
        actor="investigator-17",
        actor_role=ReviewerRole.INVESTIGATOR,
        rationale="Assigned after reviewing the alert and available transaction facts.",
        expected_version=case.version,
    )
    return store, case, EvidenceStore(tmp_path / "evidence")


def test_atomic_store_is_content_addressed_and_has_no_pending_files(tmp_path):
    store = EvidenceStore(tmp_path / "evidence")
    first = store.preserve_text(
        case_id="CASE-ABC", evidence_type="narrative_draft", content="same content"
    )
    second = store.preserve_text(
        case_id="CASE-ABC", evidence_type="narrative_draft", content="same content"
    )
    assert first.local_path == second.local_path
    assert first.sha256 == sha256_bytes(b"same content")
    assert not list((tmp_path / "evidence").rglob(".pending-*"))


def test_store_rejects_unsafe_path_segments(tmp_path):
    with pytest.raises(Exception, match="case_id"):
        EvidenceStore(tmp_path).preserve_text(
            case_id="../escape", evidence_type="draft", content="payload"
        )


def test_verified_s3_receipt_contains_checksum_version_and_request(tmp_path):
    local = EvidenceStore(tmp_path).preserve_text(
        case_id="CASE-ABC", evidence_type="narrative_draft", content="payload"
    )
    client = S3Stub()
    receipt = archive_to_s3(local, b"payload", "bucket-name", s3_client=client)
    assert receipt["verified"] is True
    assert receipt["version_id"] == "version-7"
    assert receipt["request_id"] == "request-9"
    assert client.request["ChecksumSHA256"] == base64.b64encode(
        bytes.fromhex(local.sha256)
    ).decode("ascii")


@pytest.mark.parametrize(
    "client",
    [S3Stub(echo_checksum=False), S3Stub(version_id=None), S3Stub(version_id="null")],
)
def test_unverifiable_s3_response_is_recorded_as_failed_local_receipt(tmp_path, client):
    receipt = EvidenceStore(tmp_path).preserve_text(
        case_id="CASE-ABC",
        evidence_type="narrative_draft",
        content="payload",
        archive_bucket="bucket-name",
        s3_client=client,
    )
    assert receipt.archive_receipt["verified"] is False
    assert receipt.archive_receipt["error_type"] == "ArchiveVerificationError"
    assert (tmp_path / "CASE-ABC").is_dir()


def test_attachment_links_receipt_and_hash_chained_event(active_case):
    store, case, evidence_store = active_case
    receipt = evidence_store.preserve_text(
        case_id=case.case_id, evidence_type="narrative_draft", content="grounded draft"
    )
    updated = store.attach_evidence(
        case.case_id,
        receipt=receipt,
        actor="investigator-17",
        actor_role=ReviewerRole.INVESTIGATOR,
        expected_version=case.version,
    )
    history = store.history(case.case_id)
    assert updated.status == CaseStatus.UNDER_REVIEW.value
    assert updated.version == case.version + 1
    assert history[-1].event_type == "evidence_attached"
    assert history[-1].previous_event_hash == history[-2].event_hash
    assert history[-1].metadata["sha256"] == receipt.sha256
    assert store.list_evidence(case.case_id)[0].evidence_id == receipt.evidence_id
    assert store.verify_integrity(case.case_id)["ok"] is True


def test_evidence_cannot_be_linked_before_review(active_case):
    active_store, _, evidence_store = active_case
    open_case = active_store.create_alert_case(
        trans_num="txn-open",
        model_score=0.90,
        threshold=0.80,
        model_version="v-evidence",
        actor="queue-worker",
    )
    receipt = evidence_store.preserve_text(
        case_id=open_case.case_id, evidence_type="narrative_draft", content="payload"
    )
    with pytest.raises(InvalidTransition, match="active review"):
        active_store.attach_evidence(
            open_case.case_id,
            receipt=receipt,
            actor="investigator-17",
            actor_role=ReviewerRole.INVESTIGATOR,
            expected_version=open_case.version,
        )


def test_evidence_for_another_case_is_rejected(active_case):
    store, case, evidence_store = active_case
    receipt = evidence_store.preserve_text(
        case_id="CASE-OTHER", evidence_type="narrative_draft", content="payload"
    )
    with pytest.raises(ValidationError, match="not requested case"):
        store.attach_evidence(
            case.case_id,
            receipt=receipt,
            actor="investigator-17",
            actor_role=ReviewerRole.INVESTIGATOR,
            expected_version=case.version,
        )


def test_file_tampering_is_detected(active_case):
    store, case, evidence_store = active_case
    receipt = evidence_store.preserve_text(
        case_id=case.case_id, evidence_type="narrative_draft", content="original"
    )
    store.attach_evidence(
        case.case_id,
        receipt=receipt,
        actor="investigator-17",
        actor_role=ReviewerRole.INVESTIGATOR,
        expected_version=case.version,
    )
    with open(receipt.local_path, "w", encoding="utf-8") as handle:
        handle.write("tampered")
    result = store.verify_integrity(case.case_id)
    assert result["ok"] is False
    assert any("mismatch" in issue for issue in result["issues"])


def test_event_tampering_is_detected(active_case):
    store, case, _ = active_case
    with sqlite3.connect(store.path) as con:
        con.execute(
            "UPDATE case_events SET rationale = 'rewritten' WHERE case_id = ? AND sequence = 2",
            (case.case_id,),
        )
    result = store.verify_integrity(case.case_id)
    assert result["ok"] is False
    assert "Event 2 content hash does not match" in result["issues"]


def test_reopening_store_does_not_resign_events_with_erased_hashes(active_case):
    store, case, _ = active_case
    original_head = store.history(case.case_id)[-1].event_hash
    with sqlite3.connect(store.path) as con:
        con.execute(
            "UPDATE case_events SET rationale = 'rewritten', event_hash = '' "
            "WHERE case_id = ? AND sequence = 1",
            (case.case_id,),
        )
    reopened = CaseStore(store.path)
    assert reopened.history(case.case_id)[0].event_hash == ""
    assert reopened.history(case.case_id)[-1].event_hash == original_head
    assert reopened.verify_integrity(case.case_id)["ok"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_score", 0.99), ("threshold", 0.7), ("model_version", "other-model"),
        ("trans_num", "other-transaction"), ("status", "rgs_reached"),
        ("assigned_to", "different-reviewer"),
        ("created_at", "2020-01-01T00:00:00Z"), ("updated_at", "2020-01-01T00:00:00Z"),
    ],
)
def test_case_projection_tampering_is_detected(active_case, field, value):
    store, case, _ = active_case
    with sqlite3.connect(store.path) as con:
        con.execute(f"UPDATE cases SET {field} = ? WHERE case_id = ?", (value, case.case_id))
    result = store.verify_integrity(case.case_id)
    assert result["ok"] is False
    assert any(f"Case {field} differs" in issue for issue in result["issues"])


def test_deleted_history_is_not_reported_as_verified(active_case):
    store, case, _ = active_case
    with sqlite3.connect(store.path) as con:
        con.execute("DELETE FROM case_events WHERE case_id = ?", (case.case_id,))
    assert "Case has no event history" in store.verify_integrity(case.case_id)["issues"]


def test_tampered_receipt_file_is_rejected_before_linking(active_case):
    store, case, evidence_store = active_case
    receipt = evidence_store.preserve_text(
        case_id=case.case_id, evidence_type="narrative_draft", content="original"
    )
    from pathlib import Path

    Path(receipt.local_path).write_text("changed", encoding="utf-8")
    with pytest.raises(ValidationError, match="preservation receipt"):
        store.attach_evidence(case.case_id, receipt=receipt, expected_version=case.version)
    assert store.list_evidence(case.case_id) == []
    assert store.get_case(case.case_id).version == case.version


@pytest.mark.parametrize("field", ["media_type", "created_at", "created_by"])
def test_evidence_record_tampering_is_detected_against_committed_event(active_case, field):
    store, case, evidence_store = active_case
    receipt = evidence_store.preserve_text(
        case_id=case.case_id, evidence_type="narrative_draft", content="original"
    )
    store.attach_evidence(
        case.case_id,
        receipt=receipt,
        actor="investigator-17",
        actor_role=ReviewerRole.INVESTIGATOR,
        expected_version=case.version,
    )
    with sqlite3.connect(store.path) as con:
        con.execute(
            f"UPDATE case_evidence SET {field} = 'rewritten' WHERE evidence_id = ?",
            (receipt.evidence_id,),
        )
    result = store.verify_integrity(case.case_id)
    assert result["ok"] is False
    assert any(f"{field} differs" in issue for issue in result["issues"])


def test_deleted_evidence_record_leaves_detectable_orphan_event(active_case):
    store, case, evidence_store = active_case
    receipt = evidence_store.preserve_text(
        case_id=case.case_id, evidence_type="narrative_draft", content="original"
    )
    store.attach_evidence(
        case.case_id,
        receipt=receipt,
        actor="investigator-17",
        actor_role=ReviewerRole.INVESTIGATOR,
        expected_version=case.version,
    )
    with sqlite3.connect(store.path) as con:
        con.execute("DELETE FROM case_evidence WHERE evidence_id = ?", (receipt.evidence_id,))
    result = store.verify_integrity(case.case_id)
    assert result["ok"] is False
    assert any("no matching evidence record" in issue for issue in result["issues"])


def test_phase2_database_is_migrated_and_backfilled(tmp_path):
    path = tmp_path / "phase2.sqlite3"
    with sqlite3.connect(path) as con:
        con.executescript(
            """
            CREATE TABLE cases (
                case_id TEXT PRIMARY KEY, trans_num TEXT NOT NULL UNIQUE,
                model_score REAL NOT NULL, threshold REAL NOT NULL, model_version TEXT NOT NULL,
                status TEXT NOT NULL, assigned_to TEXT, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, version INTEGER NOT NULL
            );
            CREATE TABLE case_events (
                event_id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES cases(case_id),
                sequence INTEGER NOT NULL, event_type TEXT NOT NULL, from_status TEXT,
                to_status TEXT NOT NULL, actor TEXT NOT NULL, actor_role TEXT NOT NULL,
                rationale TEXT NOT NULL, metadata_json TEXT NOT NULL, occurred_at TEXT NOT NULL,
                UNIQUE(case_id, sequence)
            );
            INSERT INTO cases VALUES (
                'CASE-OLD', 'txn-old', 0.9, 0.8, 'v-old', 'alert_open', NULL,
                '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', 1
            );
            INSERT INTO case_events VALUES (
                'event-old', 'CASE-OLD', 1, 'alert_created', NULL, 'alert_open',
                'worker', 'investigator', 'Original threshold alert rationale.', '{}',
                '2026-01-01T00:00:00Z'
            );
            """
        )
    store = CaseStore(path)
    history = store.history("CASE-OLD")
    assert len(history[0].event_hash) == 64
    assert store.verify_integrity("CASE-OLD")["ok"] is True


@pytest.mark.parametrize("grounded", [True, False])
def test_narrative_or_quarantine_is_always_linked_to_case(
    active_case, monkeypatch, grounded
):
    store, case, evidence_store = active_case
    monkeypatch.setattr(
        sar_agent,
        "check_narrative",
        lambda *args: SimpleNamespace(ok=grounded, summary=lambda: "test grounding verdict"),
    )
    kwargs = dict(
        txn={"trans_num": case.trans_num, "amt": 10.0},
        p_m2=0.1,
        p_m3=0.2,
        p_m4=0.3,
        meta_score=case.model_score,
        narrative="WHO\nTest narrative",
        case_id=case.case_id,
        model_version=case.model_version,
        actor="investigator-17",
        actor_role=ReviewerRole.INVESTIGATOR,
        expected_case_version=case.version,
        case_store=store,
        evidence_store=evidence_store,
        archive_bucket="",
    )
    if grounded:
        path = sar_agent.save_sar_report(**kwargs)
        assert path == store.list_evidence(case.case_id)[0].local_path
    else:
        with pytest.raises(ValueError, match="preserved as evidence"):
            sar_agent.save_sar_report(**kwargs)
    item = store.list_evidence(case.case_id)[0]
    assert item.evidence_type == (
        "narrative_draft" if grounded else "narrative_quarantine"
    )
    assert store.verify_integrity(case.case_id)["ok"] is True


@pytest.mark.parametrize("rejection", ["role", "stale_version", "model_version", "score"])
def test_narrative_is_authorized_and_bound_to_case_before_any_preservation(
    active_case, monkeypatch, rejection
):
    store, case, evidence_store = active_case
    def forbidden_preservation(**_kwargs):
        pytest.fail("Rejected request performed a local or external evidence write")

    monkeypatch.setattr(evidence_store, "preserve_text", forbidden_preservation)
    kwargs = dict(
        txn={"trans_num": case.trans_num}, p_m2=0.1, p_m3=0.2, p_m4=0.3,
        meta_score=case.model_score, narrative="WHO\nMasked cardholder",
        case_id=case.case_id, model_version=case.model_version,
        expected_case_version=case.version, case_store=store, evidence_store=evidence_store,
    )
    expected_error = ValueError
    if rejection == "role":
        kwargs["principal"] = principal_from_verified_claims(
            subject="model-reviewer", provider="test-idp",
            organization_id="local-demo", roles=[MODEL_GOVERNANCE_REVIEWER],
        )
        expected_error = AuthorizationError
    elif rejection == "stale_version":
        kwargs["expected_case_version"] = 1
        expected_error = ConcurrencyError
    elif rejection == "model_version":
        kwargs["model_version"] = "different-model"
    else:
        kwargs["meta_score"] = 0.99
    with pytest.raises(expected_error):
        sar_agent.save_sar_report(**kwargs)
    assert store.list_evidence(case.case_id) == []


def test_narrative_uses_explicit_principal_for_lookup_and_attachment(tmp_path, monkeypatch):
    identity = principal_from_verified_claims(
        subject="investigator-a", provider="test-idp", organization_id="org-a",
        roles=["investigator"],
    )
    store = CaseStore(tmp_path / "cases.sqlite3")
    case = store.create_alert_case(
        trans_num="txn-explicit", model_score=0.9, threshold=0.8,
        model_version="v-explicit", principal=identity,
    )
    case = store.start_review(
        case.case_id, principal=identity, expected_version=case.version,
        rationale="An identified investigator accepted the model alert for review.",
    )
    monkeypatch.setattr(
        sar_agent, "check_narrative",
        lambda *args: SimpleNamespace(ok=True, summary=lambda: "grounded"),
    )
    sar_agent.save_sar_report(
        {"trans_num": case.trans_num, "amt": 10.0}, 0.1, 0.2, 0.3, case.model_score,
        "WHO\nMasked cardholder", case_id=case.case_id, model_version=case.model_version,
        principal=identity, expected_case_version=case.version, case_store=store,
        evidence_store=EvidenceStore(tmp_path / "evidence"), archive_bucket="",
    )
    assert store.list_evidence(case.case_id, principal=identity)[0].created_by == identity.subject
