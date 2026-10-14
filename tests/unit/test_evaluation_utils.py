import numpy as np
import pandas as pd
import pytest

from scripts.evaluation_utils import (calibration_summary,
                                      calendar_span_days,
                                      day_block_bootstrap_pr_auc,
                                      rolling_origin_splits,
                                      subgroup_metrics)


def test_rolling_origin_splits_only_validate_on_the_future():
    splits = rolling_origin_splits(100, n_splits=3, min_train_fraction=0.5,
                                   validation_fraction=0.1)
    assert len(splits) == 3
    previous_train_size = 0
    for train, validation in splits:
        assert train[0] == 0
        assert train[-1] < validation[0]
        assert set(train).isdisjoint(validation)
        assert len(train) > previous_train_size
        previous_train_size = len(train)


def test_day_block_bootstrap_is_deterministic_and_finite():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    ts = pd.date_range("2025-01-01", periods=8, freq="D").to_numpy()
    first = day_block_bootstrap_pr_auc(y, p, ts, n_boot=50, seed=7)
    second = day_block_bootstrap_pr_auc(y, p, ts, n_boot=50, seed=7)
    assert first == second
    assert 0 <= first[0] <= first[1] <= 1


def test_perfect_probabilities_have_zero_calibration_error():
    summary = calibration_summary(np.array([0, 0, 1, 1]),
                                  np.array([0.0, 0.0, 1.0, 1.0]))
    assert summary["brier"] == 0
    assert summary["ece"] == 0
    assert summary["mce"] == 0


def test_subgroup_metrics_respect_support_thresholds():
    frame = pd.DataFrame({"group": ["a"] * 6 + ["b"] * 2})
    y = np.array([1, 1, 0, 0, 0, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3, 0.9, 0.1])
    rows = subgroup_metrics(frame, y, p, 0.5, "group", min_rows=4,
                            min_positives=2)
    assert [row["group"] for row in rows] == ["a"]
    assert rows[0]["recall"] == 1.0


def test_calendar_exposure_includes_boundary_days_and_gaps():
    assert calendar_span_days(["2025-01-01 23:59", "2025-01-03 00:01"]) == 3
    assert calendar_span_days(["2025-01-01 23:59"]) == 1


@pytest.mark.parametrize("values", [[np.nan, 0.2], [0.1, np.inf], [-0.1, 0.2], [0.1, 1.1]])
def test_invalid_probabilities_are_rejected(values):
    with pytest.raises(ValueError, match="Probabilities"):
        calibration_summary([0, 1], values)


def test_bootstrap_without_day_support_returns_json_safe_unavailable_interval():
    assert day_block_bootstrap_pr_auc([0, 1], [0.2, 0.8],
                                     ["2025-01-01", "2025-01-01"], n_boot=20) == (None, None)


def test_constant_predictions_have_measured_calibration_error():
    from scripts.calibration import ece_mce, reliability
    rows = reliability([0, 0, 0, 1], [0.9] * 4)
    ece, mce = ece_mce(rows, 4)
    assert len(rows) == 1
    assert ece == pytest.approx(0.65)
    assert mce == pytest.approx(0.65)


def test_retrospective_threshold_frontier_enforces_capacity_with_tied_scores():
    from scripts.evaluate import exploratory_operating_points
    result = exploratory_operating_points([0, 1, 1, 0], [10, 200, 300, 10],
                                         [0.9, 0.9, 0.9, 0.1], 25, max_alerts=2)
    assert result["alert_capacity"]["alerts"] == 0  # cannot split a three-row tied score
    assert result["cost_minimising"]["alerts"] == 3


def test_retrospective_cost_frontier_covers_no_alert_and_all_alert_extremes():
    from scripts.evaluate import exploratory_operating_points
    result = exploratory_operating_points([0, 1], [10, 1], [0.9, 0.1], 25, max_alerts=0)
    assert result["cost_minimising"]["alerts"] == 0
    result = exploratory_operating_points([0, 1], [10, 500], [0.9, 0.1], 25, max_alerts=2)
    assert result["cost_minimising"]["alerts"] == 2


def test_cost_metrics_are_finite_for_a_no_fraud_slice():
    from scripts.evaluate import cost_at
    result = cost_at([0, 0], [10, 20], np.array([0.1, 0.2]), 0.5, 25)
    assert result["recall"] == 0
    assert result["total_cost"] == 0
