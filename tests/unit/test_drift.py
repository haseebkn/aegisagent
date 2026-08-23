"""PSI/KS drift detection.

The headline case is the train/serve skew that shipped in this project: velocity
features averaged 4.88 in training and 0.014 at scoring time while every null and
range check passed. Drift detection is the control that catches that class of bug,
so it is tested against a reconstruction of it.
"""
import numpy as np
import pandas as pd
import pytest

from scripts.drift import PSI_SIGNIFICANT, classify, compare, psi

RNG = np.random.default_rng(0)


def test_identical_distributions_have_near_zero_psi():
    x = RNG.normal(size=20000)
    assert psi(x, x.copy()) < 0.01


def test_shifted_distribution_is_detected():
    a = RNG.normal(0, 1, 20000)
    b = RNG.normal(2, 1, 20000)
    assert psi(a, b) >= PSI_SIGNIFICANT


def test_velocity_regression_is_caught():
    """Training velocity ~4.9/txn vs a scoring window collapsed to ~0."""
    train = RNG.poisson(4.88, 50000)
    served = np.zeros(20000, dtype=int)
    served[:100] = RNG.poisson(1.0, 100)  # the handful of boundary rows
    value = psi(train, served)
    assert value >= PSI_SIGNIFICANT
    assert classify(value) == "SIGNIFICANT"


def test_binary_feature_with_identical_distribution_is_stable():
    """Regression: quantile-only binning collapsed the edges for binary flags and
    reported `night` and `is_online` as infinitely drifted when they were identical."""
    a = RNG.binomial(1, 0.35, 30000)
    b = RNG.binomial(1, 0.35, 15000)
    value = psi(a, b)
    assert np.isfinite(value)
    assert classify(value) == "stable"


def test_binary_feature_with_real_shift_is_still_detected():
    a = RNG.binomial(1, 0.35, 30000)
    b = RNG.binomial(1, 0.90, 15000)
    assert psi(a, b) >= PSI_SIGNIFICANT


def test_constant_feature_does_not_produce_infinity():
    a = np.zeros(10000)
    assert np.isfinite(psi(a, np.zeros(5000)))


@pytest.mark.parametrize("value,expected", [
    (0.0, "stable"), (0.05, "stable"),
    (0.10, "MODERATE"), (0.24, "MODERATE"),
    (0.25, "SIGNIFICANT"), (5.0, "SIGNIFICANT"),
])
def test_classify_thresholds(value, expected):
    assert classify(value) == expected


def test_compare_reports_per_feature_and_sorts_by_severity():
    ref = pd.DataFrame({"stable_feat": RNG.normal(size=5000),
                        "drifted_feat": RNG.normal(size=5000)})
    cur = pd.DataFrame({"stable_feat": RNG.normal(size=2000),
                        "drifted_feat": RNG.normal(5, 1, 2000)})
    results = compare(ref, cur, features=["stable_feat", "drifted_feat"])
    assert len(results) == 2
    assert results[0]["feature"] == "drifted_feat"
    assert results[0]["status"] == "SIGNIFICANT"
    assert results[1]["status"] == "stable"


def test_collapsed_encoding_is_caught_despite_unchanged_mean():
    """A broken target-encoding join gives every scoring row the global rate. The mean
    is unchanged -- on real data 0.005791 before and after -- so any mean-ratio check
    passes it. PSI sees the collapsed distribution.

    This is why the three target encodings are monitored on their serving-equivalent
    columns rather than excluded from the gate: excluded, this break could not have
    failed CI at all.
    """
    rng = np.random.default_rng(7)
    reference = rng.choice([0.002, 0.004, 0.006, 0.009, 0.02], size=20000,
                           p=[0.2, 0.3, 0.25, 0.15, 0.1])
    collapsed = np.full(8000, reference.mean())

    assert np.isclose(collapsed.mean(), reference.mean())
    value = psi(reference, collapsed)
    assert classify(value) == "SIGNIFICANT"


def test_healthy_encoding_comparison_is_stable():
    """Same encoding map applied to two samples of the same population."""
    rng = np.random.default_rng(8)
    levels = [0.002, 0.004, 0.006, 0.009, 0.02]
    p = [0.2, 0.3, 0.25, 0.15, 0.1]
    assert classify(psi(rng.choice(levels, 20000, p=p),
                        rng.choice(levels, 8000, p=p))) == "stable"
