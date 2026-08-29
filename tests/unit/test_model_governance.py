import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from scripts.case_management import CaseEvent, CaseRecord
from scripts.model_comparison import PROMOTION_POLICY, build_comparison
from scripts.model_registry import (
    REQUIRED_ARTIFACTS,
    RegistryError,
    load_registry,
    promote_candidate,
    record_comparison,
    register_version,
    rollback_champion,
)
from scripts.operational_feedback import build_feedback_report
from scripts.security import (
    INVESTIGATOR,
    MODEL_GOVERNANCE_REVIEWER,
    PermissionDenied,
    principal_from_verified_claims,
)


def artifact_version(root, version, marker):
    directory = root / version
    directory.mkdir()
    for name in REQUIRED_ARTIFACTS:
        (directory / name).write_text(f"{marker}:{name}", encoding="utf-8")
    return directory


def eligible_report(registry, candidate):
    return {
        "schema_version": 1,
        "evaluation_role": "development_holdout",
        "historically_pristine": False,
        "eligible": True,
        "champion_version": registry["champion"],
        "candidate_version": candidate,
        "champion_manifest_sha256": registry["versions"][registry["champion"]][
            "manifest_sha256"
        ],
        "candidate_manifest_sha256": registry["versions"][candidate]["manifest_sha256"],
        "rows": 100,
        "positives": 10,
        "champion": {"recall": 0.8, "brier": 0.1},
        "candidate": {"recall": 0.8, "brier": 0.1, "alerts_per_day": 1.0},
        "paired_pr_auc_delta_ci_95": [0.0, 0.0],
        "policy": PROMOTION_POLICY,
        "gates": {
            "paired_pr_auc_noninferiority": True,
            "recall_noninferiority": True,
            "calibration_noninferiority": True,
            "alert_capacity": True,
        },
    }


def governance_principal(role=MODEL_GOVERNANCE_REVIEWER):
    return principal_from_verified_claims(
        subject="governance-reviewer",
        roles=[role],
        provider="test-idp",
        organization_id="org-a",
    )


def test_training_registration_does_not_silently_replace_champion(tmp_path):
    registry_path = tmp_path / "registry.json"
    artifact_version(tmp_path, "v1", "champion")
    artifact_version(tmp_path, "v2", "candidate")
    register_version(
        "v1",
        artifacts_dir=tmp_path,
        registry_path=registry_path,
        bootstrap_champion=True,
    )
    registry = register_version(
        "v2",
        artifacts_dir=tmp_path,
        registry_path=registry_path,
    )
    assert registry["champion"] == "v1"
    assert registry["versions"]["v2"]["status"] == "candidate"
    assert (tmp_path / "latest_version.txt").read_text() == "v1"


def test_eligible_candidate_requires_human_promotion_and_supports_rollback(tmp_path):
    registry_path = tmp_path / "registry.json"
    artifact_version(tmp_path, "v1", "champion")
    artifact_version(tmp_path, "v2", "candidate")
    register_version(
        "v1",
        artifacts_dir=tmp_path,
        registry_path=registry_path,
        bootstrap_champion=True,
    )
    registry = register_version("v2", artifacts_dir=tmp_path, registry_path=registry_path)
    report_path = tmp_path / "v2" / "promotion_report.json"
    report_path.write_text(json.dumps(eligible_report(registry, "v2")), encoding="utf-8")
    identity = governance_principal()
    record_comparison(
        report_path,
        principal=identity,
        artifacts_dir=tmp_path,
        registry_path=registry_path,
    )
    promoted = promote_candidate(
        "v2",
        principal=identity,
        rationale="Candidate passed the documented non-inferiority and capacity gates.",
        report_path=report_path,
        artifacts_dir=tmp_path,
        registry_path=registry_path,
    )
    assert promoted["champion"] == "v2"
    assert promoted["versions"]["v1"]["status"] == "archived"
    assert (tmp_path / "latest_version.txt").read_text() == "v2"

    rolled_back = rollback_champion(
        "v1",
        principal=identity,
        rationale="Post-promotion monitoring requires a controlled rollback to the prior model.",
        artifacts_dir=tmp_path,
        registry_path=registry_path,
    )
    assert rolled_back["champion"] == "v1"
    assert (tmp_path / "latest_version.txt").read_text() == "v1"
    assert [event["event_type"] for event in rolled_back["events"]] == [
        "version_registered",
        "version_registered",
        "candidate_compared",
        "candidate_promoted",
        "champion_rolled_back",
    ]


def test_artifact_or_registry_tampering_blocks_governance(tmp_path):
    registry_path = tmp_path / "registry.json"
    artifact_version(tmp_path, "v1", "champion")
    artifact_version(tmp_path, "v2", "candidate")
    register_version(
        "v1",
        artifacts_dir=tmp_path,
        registry_path=registry_path,
        bootstrap_champion=True,
    )
    registry = register_version("v2", artifacts_dir=tmp_path, registry_path=registry_path)
    report_path = tmp_path / "v2" / "promotion_report.json"
    report_path.write_text(json.dumps(eligible_report(registry, "v2")), encoding="utf-8")
    identity = governance_principal()
    record_comparison(
        report_path,
        principal=identity,
        artifacts_dir=tmp_path,
        registry_path=registry_path,
    )
    (tmp_path / "v2" / "meta_threshold.txt").write_text("tampered", encoding="utf-8")
    with pytest.raises(RegistryError, match="changed after registration"):
        promote_candidate(
            "v2",
            principal=identity,
            rationale="This promotion must fail because its artifact was modified after evaluation.",
            report_path=report_path,
            artifacts_dir=tmp_path,
            registry_path=registry_path,
        )

    raw = json.loads(registry_path.read_text())
    raw["events"][0]["status"] = "candidate"
    registry_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RegistryError, match="event content was modified"):
        load_registry(registry_path)


def test_governance_domain_rejects_wrong_role_and_fabricated_report(tmp_path):
    registry_path = tmp_path / "registry.json"
    artifact_version(tmp_path, "v1", "champion")
    artifact_version(tmp_path, "v2", "candidate")
    register_version(
        "v1",
        artifacts_dir=tmp_path,
        registry_path=registry_path,
        bootstrap_champion=True,
    )
    registry = register_version("v2", artifacts_dir=tmp_path, registry_path=registry_path)
    report = eligible_report(registry, "v2")
    report_path = tmp_path / "v2" / "promotion_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(PermissionDenied):
        record_comparison(
            report_path,
            principal=governance_principal(INVESTIGATOR),
            artifacts_dir=tmp_path,
            registry_path=registry_path,
        )

    report["champion_manifest_sha256"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RegistryError, match="champion manifest"):
        record_comparison(
            report_path,
            principal=governance_principal(),
            artifacts_dir=tmp_path,
            registry_path=registry_path,
        )

    report["champion_manifest_sha256"] = registry["versions"]["v1"][
        "manifest_sha256"
    ]
    report["candidate"]["recall"] = 0.0
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RegistryError, match="gates do not match"):
        record_comparison(
            report_path,
            principal=governance_principal(),
            artifacts_dir=tmp_path,
            registry_path=registry_path,
        )


def test_paired_comparison_passes_identical_candidate_and_rejects_degraded_one():
    rng = np.random.default_rng(7)
    rows = 400
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="3h")
    y = np.zeros(rows, dtype=int)
    y[rng.choice(rows, size=40, replace=False)] = 1
    champion = np.clip(0.1 + y * 0.7 + rng.normal(0, 0.08, rows), 0.001, 0.999)
    common = dict(
        champion_version="v1",
        candidate_version="v2",
        champion_manifest_sha256="a" * 64,
        candidate_manifest_sha256="b" * 64,
        y=y,
        timestamps=timestamps,
        champion_probabilities=champion,
        champion_threshold=0.5,
        candidate_threshold=0.5,
        bootstrap_samples=100,
        max_alerts_per_day=100,
    )
    identical = build_comparison(candidate_probabilities=champion.copy(), **common)
    assert identical["eligible"] is True
    assert identical["paired_pr_auc_delta_ci_95"] == [0.0, 0.0]

    degraded = build_comparison(candidate_probabilities=1 - champion, **common)
    assert degraded["eligible"] is False
    assert degraded["gates"]["paired_pr_auc_noninferiority"] is False


def case(case_id, model_version, status):
    now = "2026-01-01T00:00:00Z"
    return CaseRecord(
        case_id=case_id,
        trans_num=f"txn-{case_id}",
        model_score=0.9,
        threshold=0.8,
        model_version=model_version,
        organization_id="org-a",
        status=status,
        assigned_to="reviewer",
        created_at=now,
        updated_at=now,
        version=3,
    )


def event(case_id, event_type, at):
    return CaseEvent(
        event_id=f"{case_id}-{event_type}",
        case_id=case_id,
        sequence=1,
        event_type=event_type,
        from_status=None,
        to_status="under_review",
        actor="reviewer",
        actor_role="investigator",
        rationale="A sufficiently detailed analyst rationale for operational feedback.",
        metadata={},
        occurred_at=at.isoformat().replace("+00:00", "Z"),
        previous_event_hash="",
        event_hash="hash",
    )


def test_feedback_is_aggregate_and_explicitly_not_training_ground_truth():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    cases = [
        case("1", "v1", "rgs_reached"),
        case("2", "v1", "rgs_not_reached"),
        case("3", "v2", "rgs_reached"),
    ]
    histories = {
        item.case_id: [
            event(item.case_id, "review_started", start),
            replace(event(item.case_id, item.status, start), occurred_at=(start + timedelta(hours=2)).isoformat()),
        ]
        for item in cases
    }
    report = build_feedback_report(cases, histories, min_cohort_size=2)
    assert report["training_label_eligible"] is False
    assert report["models"]["v1"]["rgs_reached_rate_among_terminal"] == 0.5
    assert report["models"]["v1"]["median_review_hours"] == 2.0
    assert report["models"]["v2"]["suppressed"] is True
    rendered = json.dumps(report)
    assert "txn-" not in rendered
    assert "case_id" not in rendered
