"""Regression guards for the Phase 0 portfolio and regulatory boundary."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_dashboard_does_not_turn_alert_into_filing_requirement():
    source = read("app.py")
    lowered = source.lower()
    assert "filing is required" not in lowered
    assert "submit & lock" not in lowered
    assert "an alert is not an rgs determination" in lowered
    assert "draft — not filed" in lowered


def test_saved_artifact_is_explicitly_an_unsubmitted_draft():
    source = read("scripts/sar_agent.py")
    assert 'file_name = f"STR_DRAFT_' in source
    assert "INVESTIGATION NARRATIVE DRAFT -- NOT A FINTRAC FILING" in source
    assert "Status: DRAFT / NOT APPROVED / NOT SUBMITTED" in source


def test_prompt_reserves_rgs_decision_for_human_review():
    source = read("scripts/sar_agent.py")
    assert "authorized human" in source
    assert "must not claim that RGS has been reached" in source
    assert "must be filed" in source


def test_release_baseline_and_scope_documents_exist():
    version = read("VERSION").strip()
    assert tuple(map(int, version.split("."))) >= (0, 1, 0)
    model_card = read("MODEL_CARD.md")
    regulatory_scope = read("docs/regulatory-scope.md")
    assert f"**Version:** {version}" in model_card
    assert "prospectively locked tail" in model_card.lower()
    assert "not a legal conclusion" in regulatory_scope
