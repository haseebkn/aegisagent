import base64
import sqlite3
from types import SimpleNamespace

import pytest

from scripts.case_management import (
    CaseStatus,
    CaseStore,
    InvalidTransition,
    ReviewerRole,
    ValidationError,
)
from scripts.evidence import EvidenceStore, archive_to_s3, sha256_bytes
from scripts import sar_agent


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
    [S3Stub(echo_checksum=False), S3Stub(version_id=None)],
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


def test_evidence_record_tampering_is_detected_against_committed_event(active_case):
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
            "UPDATE case_evidence SET media_type = 'application/rewritten' WHERE evidence_id = ?",
            (receipt.evidence_id,),
        )
    result = store.verify_integrity(case.case_id)
    assert result["ok"] is False
    assert any("media_type differs" in issue for issue in result["issues"])


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
        meta_score=0.9,
        narrative="WHO\nTest narrative",
        case_id=case.case_id,
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
