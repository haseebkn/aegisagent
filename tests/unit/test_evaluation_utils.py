import numpy as np
import pandas as pd

from scripts.evaluation_utils import (calibration_summary,
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
