from fastapi.testclient import TestClient

from scripts.security import (
    AUTHORIZED_RGS_REVIEWER,
    INVESTIGATOR,
    principal_from_verified_claims,
)
from service import create_app


def identity(subject="investigator-1", role=INVESTIGATOR, organization="org-a"):
    return principal_from_verified_claims(
        subject=subject,
        roles=[role],
        provider="test-idp",
        organization_id=organization,
        session_id=f"session-{subject}",
    )


def client(tmp_path, principal):
    app = create_app(
        principal_resolver=lambda _request: principal,
        case_db_path=tmp_path / "cases.sqlite3",
    )
    return TestClient(app, raise_server_exceptions=False)


def create_case(api, trans_num="txn-api"):
    response = api.post(
        "/v1/cases",
        json={
            "trans_num": trans_num,
            "model_score": 0.91,
            "threshold": 0.8,
            "model_version": "v-api",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_liveness_is_public_and_sets_security_headers(tmp_path):
    api = client(tmp_path, identity())
    response = api.get("/health/live", headers={"X-Request-ID": "test-request-1"})
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["x-request-id"] == "test-request-1"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_case_lifecycle_uses_request_scoped_principal_and_versions(tmp_path):
    investigator_api = client(tmp_path, identity())
    case = create_case(investigator_api)
    started = investigator_api.post(
        f"/v1/cases/{case['case_id']}/review",
        json={
            "expected_version": case["version"],
            "rationale": "Investigator accepted the alert for documented review.",
        },
    )
    assert started.status_code == 200
    assert started.json()["status"] == "under_review"

    reviewer_api = client(
        tmp_path,
        identity("reviewer-1", AUTHORIZED_RGS_REVIEWER),
    )
    decided = reviewer_api.post(
        f"/v1/cases/{case['case_id']}/rgs-decision",
        json={
            "expected_version": started.json()["version"],
            "reached": False,
            "rationale": "Human review found insufficient grounds after documented analysis.",
        },
    )
    assert decided.status_code == 200
    assert decided.json()["status"] == "rgs_not_reached"

    history = reviewer_api.get(f"/v1/cases/{case['case_id']}/history")
    assert history.status_code == 200
    assert [item["event_type"] for item in history.json()["items"]] == [
        "alert_created",
        "review_started",
        "rgs_not_reached",
    ]


def test_investigator_cannot_record_rgs_decision(tmp_path):
    api = client(tmp_path, identity())
    case = create_case(api)
    started = api.post(
        f"/v1/cases/{case['case_id']}/review",
        json={
            "expected_version": case["version"],
            "rationale": "Investigator accepted the alert for documented review.",
        },
    ).json()
    response = api.post(
        f"/v1/cases/{case['case_id']}/rgs-decision",
        json={
            "expected_version": started["version"],
            "reached": True,
            "rationale": "Attempted unauthorized reasonable grounds determination.",
        },
    )
    assert response.status_code == 403
    assert response.json()["error"] == {
        "code": "permission_denied",
        "message": "Operation denied.",
    }


def test_cross_organization_case_is_not_exposed(tmp_path):
    owner_api = client(tmp_path, identity(organization="org-a"))
    case = create_case(owner_api)
    outsider_api = client(tmp_path, identity("outsider-1", organization="org-b"))
    response = outsider_api.get(f"/v1/cases/{case['case_id']}")
    assert response.status_code == 403
    assert "org-a" not in response.text


def test_stale_expected_version_has_stable_conflict_contract(tmp_path):
    api = client(tmp_path, identity())
    case = create_case(api)
    first = api.post(
        f"/v1/cases/{case['case_id']}/review",
        json={
            "expected_version": case["version"],
            "rationale": "Investigator accepted the alert for documented review.",
        },
    )
    assert first.status_code == 200
    stale = api.post(
        f"/v1/cases/{case['case_id']}/review",
        json={
            "expected_version": case["version"],
            "rationale": "A stale browser attempted the same transition once again.",
        },
    )
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "version_conflict"


def test_unknown_request_fields_and_non_boolean_decision_are_rejected(tmp_path):
    api = client(tmp_path, identity())
    invalid_case = api.post(
        "/v1/cases",
        json={
            "trans_num": "txn-extra",
            "model_score": 0.91,
            "threshold": 0.8,
            "model_version": "v-api",
            "role": "authorized_rgs_reviewer",
        },
    )
    assert invalid_case.status_code == 422
    assert "authorized_rgs_reviewer" not in invalid_case.text

    case = create_case(api, "txn-strict-bool")
    started = api.post(
        f"/v1/cases/{case['case_id']}/review",
        json={
            "expected_version": case["version"],
            "rationale": "Investigator accepted the alert for documented review.",
        },
    ).json()
    invalid_bool = api.post(
        f"/v1/cases/{case['case_id']}/rgs-decision",
        json={
            "expected_version": started["version"],
            "reached": "yes",
            "rationale": "A string must never be coerced into a regulatory decision.",
        },
    )
    assert invalid_bool.status_code == 422


def test_production_without_verified_adapter_is_not_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("AEGIS_SECURITY_MODE", "production")
    api = TestClient(
        create_app(case_db_path=tmp_path / "cases.sqlite3"),
        raise_server_exceptions=False,
    )
    assert api.get("/health/live").status_code == 200
    readiness = api.get("/health/ready")
    assert readiness.status_code == 503
    assert readiness.json()["checks"]["identity_adapter"] is False

    protected = api.get("/v1/cases")
    assert protected.status_code == 503
    assert protected.json()["error"]["code"] == "identity_adapter_unavailable"


def test_openapi_describes_the_case_service_boundary(tmp_path):
    api = client(tmp_path, identity())
    schema = api.get("/openapi.json").json()
    assert schema["info"]["version"] == "0.6.0"
    assert "/v1/cases/{case_id}/rgs-decision" in schema["paths"]
    assert not any("submit" in path or "file" in path for path in schema["paths"])
