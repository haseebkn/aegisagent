import pytest

from scripts.case_management import AuthorizationError, CaseStore
from scripts.security import (
    AUTHORIZED_RGS_REVIEWER,
    INVESTIGATOR,
    AuthenticationRequired,
    Permission,
    PermissionDenied,
    local_development_principal,
    principal_from_verified_claims,
    require_permission,
)


def principal(subject, role=INVESTIGATOR, organization="org-a"):
    return principal_from_verified_claims(
        subject=subject,
        roles=[role],
        provider="test-idp",
        organization_id=organization,
        session_id=f"session-{subject}",
    )


def test_local_identity_provider_is_disabled_in_production():
    with pytest.raises(AuthenticationRequired, match="No production identity adapter"):
        local_development_principal(
            subject="attacker",
            roles=[AUTHORIZED_RGS_REVIEWER],
            environ={"AEGIS_SECURITY_MODE": "production"},
        )


def test_investigator_cannot_escalate_to_rgs_permission():
    identity = principal("investigator-17")
    with pytest.raises(PermissionDenied, match="lacks required permission"):
        require_permission(identity, Permission.RECORD_RGS_DECISION)


def test_case_events_commit_verified_identity_without_tokens(tmp_path):
    identity = principal("investigator-17")
    store = CaseStore(tmp_path / "cases.sqlite3", principal=identity)
    case = store.create_alert_case(
        trans_num="txn-secure",
        model_score=0.9,
        threshold=0.8,
        model_version="v-security",
        principal=identity,
    )
    event_identity = store.history(case.case_id)[0].metadata["identity"]
    assert event_identity == identity.audit_metadata()
    assert "token" not in str(event_identity).lower()
    assert case.organization_id == "org-a"


def test_cross_organization_case_access_is_denied(tmp_path):
    owner = principal("investigator-a", organization="org-a")
    outsider = principal("investigator-b", organization="org-b")
    store = CaseStore(tmp_path / "cases.sqlite3", principal=owner)
    case = store.create_alert_case(
        trans_num="txn-org-a",
        model_score=0.9,
        threshold=0.8,
        model_version="v-security",
        principal=owner,
    )
    with pytest.raises(AuthorizationError, match="Cross-organization"):
        store.get_case(case.case_id, principal=outsider)


def test_case_listing_is_scoped_to_principal_organization(tmp_path):
    store = CaseStore(tmp_path / "cases.sqlite3")
    org_a = principal("investigator-a", organization="org-a")
    org_b = principal("investigator-b", organization="org-b")
    for trans_num, identity in (("txn-a", org_a), ("txn-b", org_b)):
        store.create_alert_case(
            trans_num=trans_num,
            model_score=0.9,
            threshold=0.8,
            model_version="v-security",
            principal=identity,
        )
    assert [case.trans_num for case in store.list_cases(principal=org_a)] == ["txn-a"]
    assert [case.trans_num for case in store.list_cases(principal=org_b)] == ["txn-b"]


def test_asserted_actor_cannot_differ_from_verified_subject(tmp_path):
    identity = principal("investigator-17")
    store = CaseStore(tmp_path / "cases.sqlite3", principal=identity)
    with pytest.raises(AuthorizationError, match="does not match"):
        store.create_alert_case(
            trans_num="txn-spoof",
            model_score=0.9,
            threshold=0.8,
            model_version="v-security",
            principal=identity,
            actor="someone-else",
        )


def test_production_case_write_rejects_raw_self_attested_actor(tmp_path, monkeypatch):
    monkeypatch.setenv("AEGIS_SECURITY_MODE", "production")
    store = CaseStore(tmp_path / "cases.sqlite3")
    with pytest.raises(AuthorizationError, match="No production identity adapter"):
        store.create_alert_case(
            trans_num="txn-production",
            model_score=0.9,
            threshold=0.8,
            model_version="v-security",
            actor="self-asserted-admin",
            actor_role=AUTHORIZED_RGS_REVIEWER,
        )


def test_verified_identity_cannot_claim_a_legacy_unscoped_case(tmp_path):
    identity = principal("investigator-a", organization="org-a")
    store = CaseStore(tmp_path / "cases.sqlite3", principal=identity)
    case = store.create_alert_case(
        trans_num="txn-legacy",
        model_score=0.9,
        threshold=0.8,
        model_version="v-security",
        principal=identity,
    )
    import sqlite3

    with sqlite3.connect(store.path) as con:
        con.execute(
            "UPDATE cases SET organization_id = 'legacy-unscoped' WHERE case_id = ?",
            (case.case_id,),
        )
    with pytest.raises(AuthorizationError, match="no verified organization owner"):
        store.get_case(case.case_id, principal=identity)


def test_integrity_detects_case_organization_tampering(tmp_path):
    identity = principal("investigator-a", organization="org-a")
    store = CaseStore(tmp_path / "cases.sqlite3", principal=identity)
    case = store.create_alert_case(
        trans_num="txn-org-tamper",
        model_score=0.9,
        threshold=0.8,
        model_version="v-security",
        principal=identity,
    )
    import sqlite3

    with sqlite3.connect(store.path) as con:
        con.execute(
            "UPDATE cases SET organization_id = 'org-b' WHERE case_id = ?",
            (case.case_id,),
        )
    org_b = principal("investigator-b", organization="org-b")
    result = store.verify_integrity(case.case_id, principal=org_b)
    assert result["ok"] is False
    assert any("organization" in issue for issue in result["issues"])
