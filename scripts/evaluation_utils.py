"""Temporal evaluation, uncertainty, calibration, and subgroup helpers."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def validate_binary_probabilities(y, probabilities):
    """Reject malformed evaluation inputs rather than emitting plausible metrics."""
    y = np.asarray(y, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or probabilities.ndim != 1 or not len(y) or len(y) != len(probabilities):
        raise ValueError("Labels and probabilities must be matching nonempty one-dimensional arrays")
    if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be finite binary values")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    return y, probabilities


def calendar_span_days(timestamps):
    """Inclusive calendar-day exposure, including days with no transactions."""
    days = pd.DatetimeIndex(pd.to_datetime(timestamps)).normalize()
    if not len(days) or days.hasnans:
        raise ValueError("A nonempty set of valid timestamps is required")
    return (days.max().date() - days.min().date()).days + 1


def rolling_origin_splits(n_rows, n_splits=3, min_train_fraction=0.50,
                          validation_fraction=0.10):
    """Return expanding-train, forward-validation positional index pairs."""
    if n_rows < 2 or n_splits < 1:
        raise ValueError("n_rows must be >= 2 and n_splits must be >= 1")
    if not 0 < min_train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Training and validation fractions must be between 0 and 1")
    validation_size = max(1, int(n_rows * validation_fraction))
    first_end = int(n_rows * min_train_fraction)
    last_end = n_rows - validation_size
    if first_end < 1 or first_end > last_end:
        raise ValueError("Fractions leave no room for a forward validation window")
    train_ends = np.linspace(first_end, last_end, n_splits, dtype=int)
    return [
        (np.arange(0, end), np.arange(end, min(end + validation_size, n_rows)))
        for end in np.unique(train_ends)
    ]


def day_block_bootstrap_pr_auc(y, probabilities, timestamps, n_boot=500, seed=42):
    """PR-AUC interval by resampling whole calendar days, not individual rows."""
    y, probabilities = validate_binary_probabilities(y, probabilities)
    if not isinstance(n_boot, int) or n_boot < 20:
        raise ValueError("At least 20 bootstrap samples are required")
    days = np.asarray(timestamps).astype("datetime64[D]")
    if days.ndim != 1 or len(days) != len(y) or np.isnat(days).any():
        raise ValueError("Valid timestamps must match the labels")
    unique_days = np.unique(days)
    if unique_days.size < 2 or np.unique(y).size < 2:
        return (None, None)
    rng = np.random.default_rng(seed)
    values = []
    day_indices = {day: np.flatnonzero(days == day) for day in unique_days}
    for _ in range(n_boot):
        sampled_days = rng.choice(unique_days, size=len(unique_days), replace=True)
        idx = np.concatenate([day_indices[day] for day in sampled_days])
        if np.unique(y[idx]).size == 2:
            values.append(average_precision_score(y[idx], probabilities[idx]))
    if len(values) < max(20, n_boot // 4):
        return (None, None)
    return tuple(float(v) for v in np.quantile(values, [0.025, 0.975]))


def calibration_summary(y, probabilities, bins=10):
    y, probabilities = validate_binary_probabilities(y, probabilities)
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    edges = np.linspace(0.0, 1.0, bins + 1)
    assigned = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)
    weighted_gap = 0.0
    max_gap = 0.0
    for bin_id in range(bins):
        mask = assigned == bin_id
        if not mask.any():
            continue
        gap = abs(float(probabilities[mask].mean() - y[mask].mean()))
        weighted_gap += mask.mean() * gap
        max_gap = max(max_gap, gap)
    return {
        "brier": float(np.mean((probabilities - y) ** 2)),
        "ece": float(weighted_gap),
        "mce": float(max_gap),
    }


def subgroup_metrics(frame, y, probabilities, threshold, column,
                     min_rows=100, min_positives=5):
    """Return support-aware alert metrics without implying a fairness verdict."""
    y, probabilities = validate_binary_probabilities(y, probabilities)
    if len(frame) != len(y) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Frame must match the labels and threshold must be within [0, 1]")
    if min_rows < 1 or min_positives < 1:
        raise ValueError("Minimum subgroup support must be positive")
    rows = []
    for value in sorted(frame[column].fillna("<missing>").astype(str).unique()):
        mask = frame[column].fillna("<missing>").astype(str).to_numpy() == value
        positives = int(y[mask].sum())
        if int(mask.sum()) < min_rows or positives < min_positives:
            continue
        pred = probabilities[mask] >= threshold
        tp = int((pred & (y[mask] == 1)).sum())
        rows.append({
            "group": value,
            "rows": int(mask.sum()),
            "positives": positives,
            "alert_rate": float(pred.mean()),
            "precision": float(tp / pred.sum()) if pred.sum() else 0.0,
            "recall": float(tp / positives),
        })
    return rows
