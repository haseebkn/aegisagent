"""Every evaluated model must report calibration, not just the ensemble.

The evaluation reports originally carried PR/ROC AUC for each model but calibration
only for the ensemble, and compared the ensemble against a weak logistic baseline it
beats. That omitted the comparison it loses: the strongest single base learner leads
the ensemble on PR AUC on both the development and locked windows.

The ensemble is retained for calibration, which is a defensible reason -- but only
because it is measured. These tests keep the evidence in the reports so the claim
stays reproducible rather than asserted.
"""
import inspect
import json
import pathlib

import pytest

REPORTS = pathlib.Path(__file__).resolve().parents[2] / "reports"
WINDOWS = ["development_evaluation", "locked_evaluation"]
CALIBRATION_KEYS = ["brier", "ece", "mce"]


def load(window):
    path = REPORTS / f"{window}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not generated")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("window", WINDOWS)
def test_every_model_reports_calibration(window):
    ranking = load(window)["ranking"]
    assert ranking, "report has no ranking block"
    for name, entry in ranking.items():
        for key in CALIBRATION_KEYS:
            assert key in entry, f"{name} in {window} is missing '{key}'"
            assert isinstance(entry[key], float)


@pytest.mark.parametrize("window", WINDOWS)
def test_ensemble_is_compared_against_a_strong_baseline_not_only_a_weak_one(window):
    """A comparison set containing only the ensemble and a deliberately weak baseline
    would let the ensemble look better than the evidence supports."""
    ranking = load(window)["ranking"]
    assert "ensemble" in ranking
    base_learners = [k for k in ranking if k.startswith("model_")]
    assert len(base_learners) >= 3, (
        "each base learner must be scored alongside the ensemble; without them the "
        "only comparison left is against the weak logistic baseline"
    )


@pytest.mark.parametrize("window", WINDOWS)
def test_ensemble_calibration_advantage_is_real(window):
    """The stated reason for keeping the ensemble is calibration. If that stops being
    true the model card is wrong and must be revisited, so fail loudly here."""
    ranking = load(window)["ranking"]
    best_pr = max(ranking, key=lambda k: ranking[k]["pr_auc"])
    if best_pr == "ensemble":
        return  # ensemble leads outright; the calibration argument is not load-bearing
    assert ranking["ensemble"]["ece"] < ranking[best_pr]["ece"], (
        f"{best_pr} leads the ensemble on PR AUC in {window} AND is at least as well "
        f"calibrated. The ensemble's stated justification no longer holds -- update "
        f"MODEL_CARD.md or change the deployed model."
    )


def test_evaluate_surfaces_the_comparison_at_runtime():
    """A reader of stdout should see the comparison, not only a reader of the JSON."""
    from scripts import evaluate
    src = inspect.getsource(evaluate)
    assert "calibration_summary(y, values)" in src, (
        "per-model calibration must be computed inside the ranking loop"
    )
    assert "leads the ensemble on PR AUC" in src, (
        "evaluate.py must print the comparison when a base learner outranks the ensemble"
    )
