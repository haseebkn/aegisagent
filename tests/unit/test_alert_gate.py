"""An STR may only be drafted for a transaction the model actually alerted on.

Regression for a defect that recurred twice. First the dashboard and both
verification scripts displayed hardcoded ensemble scores. That was replaced with
real scores -- but taken as a plain argmax over a small sample, so at 0.39%
prevalence the "riskiest" row was routinely an ordinary transaction scoring 0.0040
against a 0.6152 threshold. A full FINTRAC STR was written for it and the run
reported itself compliant.
"""
import numpy as np
import pytest

from scripts.inference_engine import NoAlertsInSample, select_highest_risk_alert


def test_picks_highest_scoring_alert():
    probs = np.array([0.10, 0.95, 0.80, 0.02])
    triggered = np.array([False, True, True, False])
    assert select_highest_risk_alert(probs, triggered) == 1


def test_ignores_higher_scoring_non_alert():
    """A non-alert must never win, even if it outranks every alert -- that state
    means the threshold and the trigger flags disagree, and silently reporting on
    the non-alert is the bug this guards."""
    probs = np.array([0.99, 0.70])
    triggered = np.array([False, True])
    assert select_highest_risk_alert(probs, triggered) == 1


def test_raises_when_sample_contains_no_alert():
    probs = np.array([0.0011, 0.0040, 0.0002])
    triggered = np.zeros(3, dtype=bool)
    with pytest.raises(NoAlertsInSample) as exc:
        select_highest_risk_alert(probs, triggered)
    msg = str(exc.value)
    assert "0.0040" in msg          # reports the actual max so the cause is obvious
    assert "sample size" in msg.lower()


def test_single_alert_is_selected():
    probs = np.array([0.01, 0.01, 0.77])
    triggered = np.array([False, False, True])
    assert select_highest_risk_alert(probs, triggered) == 2


def test_all_alerts_returns_global_argmax():
    probs = np.array([0.70, 0.99, 0.85])
    assert select_highest_risk_alert(probs, np.ones(3, dtype=bool)) == 1


def test_str_paths_do_not_use_bare_argmax():
    """Both STR entry points must go through the gate, not np.argmax."""
    import inspect
    from scripts import sar_agent, verify_pipeline
    for mod in (sar_agent, verify_pipeline):
        src = inspect.getsource(mod)
        assert "select_highest_risk_alert" in src, f"{mod.__name__} bypasses the gate"
        assert "argmax(meta_probs)" not in src, f"{mod.__name__} still uses bare argmax"
