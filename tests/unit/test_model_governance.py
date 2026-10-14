import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
import duckdb
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler

from scripts.case_management import CaseEvent, CaseRecord
from scripts import model_registry
from scripts.config import FEAT_M2, FEAT_M3, FEAT_M4
from scripts.inference_engine import SCORED_FEATURES, load_models
from scripts.model_comparison import build_comparison
from scripts.model_registry import (
    REQUIRED_ARTIFACTS,
    RegistryError,
    evaluate_registered_comparison,
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
    for name, features in [
        ("model_2_geo_rf.joblib", len(FEAT_M2)),
        ("model_3_cat_xgb.joblib", len(FEAT_M3)),
        ("model_4_vel_rf.joblib", len(FEAT_M4)),
        ("meta_model.joblib", 3),
    ]:
        model = DummyClassifier(strategy="prior").fit(np.zeros((10, features)), [1] + [0] * 9)
        joblib.dump(model, directory / name)
    joblib.dump(StandardScaler().fit(np.zeros((10, len(FEAT_M4)))), directory / "model_4_scaler.joblib")
    (directory / "meta_threshold.txt").write_text("0.5", encoding="utf-8")
    (directory / "training_metrics.json").write_text(json.dumps({"marker": marker}), encoding="utf-8")
    assert all((directory / name).exists() for name in REQUIRED_ARTIFACTS)
    return directory


@pytest.fixture(autouse=True)
def governed_holdout(tmp_path, monkeypatch):
    frame = pd.DataFrame({column: np.zeros(100) for column in SCORED_FEATURES})
    frame["trans_num"] = [f"txn-{index:03}" for index in range(100)]
    frame["is_fraud"] = [index % 10 == 0 for index in range(100)]
    frame["trans_date_trans_time"] = pd.date_range("2026-01-01", periods=100, freq="8h")
    frame["evaluation_role"] = "development_holdout"
    database = tmp_path / "holdout.duckdb"
    with duckdb.connect(str(database)) as con:
        con.execute("CREATE TABLE fct_fraud_features AS SELECT * FROM frame")
    monkeypatch.setattr(model_registry, "DB_PATH", database)


def eligible_report(registry, candidate, root):
    return evaluate_registered_comparison(
        candidate, artifacts_dir=root, registry_path=root / "registry.json", bootstrap_samples=100,
    )


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
    report_path.write_text(json.dumps(eligible_report(registry, "v2", tmp_path)), encoding="utf-8")
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
    report_path.write_text(json.dumps(eligible_report(registry, "v2", tmp_path)), encoding="utf-8")
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
    report = eligible_report(registry, "v2", tmp_path)
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
    report["champion"]["recall"] = 1.0
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RegistryError, match="gates do not match"):
        record_comparison(
            report_path,
            principal=governance_principal(),
            artifacts_dir=tmp_path,
            registry_path=registry_path,
        )


def test_consistently_forged_metrics_cannot_be_registered(tmp_path):
    registry_path = tmp_path / "registry.json"
    artifact_version(tmp_path, "v1", "champion")
    artifact_version(tmp_path, "v2", "candidate")
    register_version("v1", artifacts_dir=tmp_path, registry_path=registry_path, bootstrap_champion=True)
    registry = register_version("v2", artifacts_dir=tmp_path, registry_path=registry_path)
    report = eligible_report(registry, "v2", tmp_path)
    report["candidate"]["recall"] = 1.0
    report["candidate"]["pr_auc"] = 1.0
    report_path = tmp_path / "v2" / "promotion_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RegistryError, match="independent holdout"):
        record_comparison(report_path, principal=governance_principal(), artifacts_dir=tmp_path, registry_path=registry_path)
    assert "comparison_report_sha256" not in load_registry(registry_path)["versions"]["v2"]


@pytest.mark.parametrize("field,value", [("status", "champion"), ("comparison_report_sha256", "f" * 64), ("manifest_sha256", "f" * 64)])
def test_materialized_registry_state_cannot_bypass_event_history(tmp_path, field, value):
    registry_path = tmp_path / "registry.json"
    artifact_version(tmp_path, "v1", "champion")
    artifact_version(tmp_path, "v2", "candidate")
    register_version("v1", artifacts_dir=tmp_path, registry_path=registry_path, bootstrap_champion=True)
    registry = register_version("v2", artifacts_dir=tmp_path, registry_path=registry_path)
    registry["versions"]["v2"][field] = value
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(RegistryError):
        load_registry(registry_path)


def test_serving_checks_artifact_integrity_before_deserializing(tmp_path, monkeypatch):
    directory = artifact_version(tmp_path, "v1", "champion")
    register_version("v1", artifacts_dir=tmp_path, registry_path=tmp_path / "registry.json", bootstrap_champion=True)
    (directory / "meta_model.joblib").write_bytes(b"tampered pickle")
    def forbidden_load(path):
        pytest.fail("Tampered pickle reached deserialization")
    monkeypatch.setattr("scripts.inference_engine.joblib_load", forbidden_load)
    for path in (tmp_path, directory):
        with pytest.raises(RegistryError, match="changed after registration"):
            load_models(path)


def test_parallel_registration_preserves_every_version_and_event(tmp_path):
    versions = [f"v{index}" for index in range(8)]
    for version in versions:
        artifact_version(tmp_path, version, version)
    def register(version):
        return register_version(version, artifacts_dir=tmp_path, registry_path=tmp_path / "registry.json")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(register, versions))
    registry = load_registry(tmp_path / "registry.json")
    assert set(registry["versions"]) == set(versions)
    assert len(registry["events"]) == len(versions)


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
