import sqlite3

import pytest

from scripts.case_management import (
    AuthorizationError,
    CaseStatus,
    CaseStore,
    ConcurrencyError,
    DuplicateAlert,
    InvalidTransition,
    ReviewerRole,
    ValidationError,
)


@pytest.fixture
def store(tmp_path):
    return CaseStore(tmp_path / "cases.sqlite3")


def create_case(store, trans_num="txn-1"):
    return store.create_alert_case(
        trans_num=trans_num,
        model_score=0.91,
        threshold=0.80,
        model_version="v-test",
        actor="queue-worker",
        actor_role=ReviewerRole.INVESTIGATOR,
        metadata={"source": "unit-test"},
    )


def start_case(store, case):
    return store.start_review(
        case.case_id,
        actor="investigator-17",
        actor_role=ReviewerRole.INVESTIGATOR,
        rationale="Assigned after reviewing the model alert and transaction facts.",
        expected_version=case.version,
    )


def test_creates_only_a_threshold_breaching_alert(store):
    with pytest.raises(ValidationError, match="below threshold"):
        store.create_alert_case(
            trans_num="ordinary-txn",
            model_score=0.79,
            threshold=0.80,
            model_version="v-test",
            actor="queue-worker",
        )


def test_created_case_starts_open_and_unassigned(store):
    case = create_case(store)
    assert case.status == CaseStatus.ALERT_OPEN.value
    assert case.assigned_to is None
    assert case.version == 1


def test_transaction_can_have_only_one_case(store):
    create_case(store)
    with pytest.raises(DuplicateAlert):
        create_case(store)


def test_start_review_assigns_the_human_and_increments_version(store):
    case = start_case(store, create_case(store))
    assert case.status == CaseStatus.UNDER_REVIEW.value
    assert case.assigned_to == "investigator-17"
    assert case.version == 2


def test_disposition_requires_authorized_rgs_reviewer(store):
    case = start_case(store, create_case(store))
    with pytest.raises(AuthorizationError, match="authorized RGS reviewer"):
        store.record_rgs_decision(
            case.case_id,
            reached=True,
            actor="investigator-17",
            actor_role=ReviewerRole.INVESTIGATOR,
            rationale="The reviewed facts support suspicion under the institution process.",
            expected_version=case.version,
        )


@pytest.mark.parametrize(
    ("reached", "expected"),
    [(True, CaseStatus.RGS_REACHED), (False, CaseStatus.RGS_NOT_REACHED)],
)
def test_authorized_reviewer_can_record_either_terminal_disposition(store, reached, expected):
    case = start_case(store, create_case(store))
    decided = store.record_rgs_decision(
        case.case_id,
        reached=reached,
        actor="rgs-reviewer-4",
        actor_role=ReviewerRole.AUTHORIZED_RGS_REVIEWER,
        rationale="Reviewed the available transaction facts and documented indicators in the case.",
        expected_version=case.version,
    )
    assert decided.status == expected.value
    assert decided.version == 3


def test_cannot_skip_human_review(store):
    case = create_case(store)
    with pytest.raises(InvalidTransition, match="Cannot move"):
        store.record_rgs_decision(
            case.case_id,
            reached=True,
            actor="rgs-reviewer-4",
            actor_role=ReviewerRole.AUTHORIZED_RGS_REVIEWER,
            rationale="This attempted decision incorrectly skipped the investigation review stage.",
            expected_version=case.version,
        )


def test_rationale_is_required_for_every_human_transition(store):
    case = create_case(store)
    with pytest.raises(ValidationError, match="at least 20"):
        store.start_review(
            case.case_id,
            actor="investigator-17",
            actor_role=ReviewerRole.INVESTIGATOR,
            rationale="Looks risky",
            expected_version=case.version,
        )


def test_stale_screen_cannot_overwrite_a_newer_decision(store):
    open_case = create_case(store)
    start_case(store, open_case)
    with pytest.raises(ConcurrencyError, match="reload"):
        store.start_review(
            open_case.case_id,
            actor="investigator-22",
            actor_role=ReviewerRole.INVESTIGATOR,
            rationale="A second browser attempted to start review from a stale case version.",
            expected_version=open_case.version,
        )


def test_terminal_case_cannot_be_changed(store):
    case = start_case(store, create_case(store))
    decided = store.record_rgs_decision(
        case.case_id,
        reached=False,
        actor="rgs-reviewer-4",
        actor_role=ReviewerRole.AUTHORIZED_RGS_REVIEWER,
        rationale="Available facts do not meet the institution's documented suspicion standard.",
        expected_version=case.version,
    )
    with pytest.raises(InvalidTransition, match="terminal"):
        store.record_rgs_decision(
            case.case_id,
            reached=True,
            actor="rgs-reviewer-4",
            actor_role=ReviewerRole.AUTHORIZED_RGS_REVIEWER,
            rationale="An attempted second disposition must not rewrite the original decision.",
            expected_version=decided.version,
        )


def test_event_history_is_ordered_and_contains_actor_rationale_and_model_context(store):
    created = create_case(store)
    reviewed = start_case(store, created)
    store.record_rgs_decision(
        reviewed.case_id,
        reached=True,
        actor="rgs-reviewer-4",
        actor_role=ReviewerRole.AUTHORIZED_RGS_REVIEWER,
        rationale="The reviewed facts meet the institution's documented suspicion standard.",
        expected_version=reviewed.version,
    )
    history = store.history(created.case_id)
    assert [event.sequence for event in history] == [1, 2, 3]
    assert [event.event_type for event in history] == [
        "alert_created", "review_started", "rgs_reached"
    ]
    assert history[0].metadata["model_version"] == "v-test"
    assert history[1].actor == "investigator-17"
    assert "documented suspicion standard" in history[2].rationale


def test_case_and_history_survive_store_reopen(tmp_path):
    path = tmp_path / "cases.sqlite3"
    first = CaseStore(path)
    case = create_case(first)
    second = CaseStore(path)
    assert second.get_case(case.case_id) == case
    assert len(second.history(case.case_id)) == 1


def test_no_filing_or_submission_state_exists():
    values = {status.value for status in CaseStatus}
    assert not any("file" in value or "submit" in value for value in values)


def test_event_table_has_no_update_or_delete_api_and_is_append_only_in_workflow(store):
    case = start_case(store, create_case(store))
    before = store.history(case.case_id)
    assert not hasattr(store, "update_event")
    assert not hasattr(store, "delete_event")
    with sqlite3.connect(store.path) as con:
        assert con.execute(
            "SELECT COUNT(*) FROM case_events WHERE case_id = ?", (case.case_id,)
        ).fetchone()[0] == len(before)
