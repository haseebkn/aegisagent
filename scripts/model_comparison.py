"""Paired champion/challenger evaluation on one chronological holdout window."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from scripts.evaluation_utils import calendar_span_days, validate_binary_probabilities


class ComparisonError(RuntimeError):
    pass


PROMOTION_POLICY = {
    "noninferiority_margin": 0.01,
    "recall_tolerance": 0.02,
    "brier_tolerance": 0.02,
    "max_alerts_per_day": 10.0,
}


def operating_metrics(y, probabilities, threshold, timestamps) -> dict[str, float | int]:
    y, probabilities = validate_binary_probabilities(y, probabilities)
    if isinstance(threshold, bool) or not np.isscalar(threshold) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ComparisonError("Threshold must be a finite probability")
    if len(timestamps) != len(y):
        raise ComparisonError("Timestamps must match evaluation rows")
    predicted = probabilities >= threshold
    positives = y == 1
    true_positive = predicted & positives
    days = calendar_span_days(timestamps)
    return {
        "pr_auc": float(average_precision_score(y, probabilities)),
        "brier": float(brier_score_loss(y, probabilities)),
        "threshold": float(threshold),
        "alerts": int(predicted.sum()),
        "alerts_per_day": float(predicted.sum() / days),
        "precision": float(true_positive.sum() / predicted.sum()) if predicted.any() else 0.0,
        "recall": float(true_positive.sum() / positives.sum()) if positives.any() else 0.0,
    }


def paired_day_block_pr_auc_delta(
    y,
    champion_probabilities,
    candidate_probabilities,
    timestamps,
    *,
    samples: int = 500,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap candidate-minus-champion PR AUC by resampling whole calendar days."""
    if type(samples) is not int or samples < 20:
        raise ComparisonError("At least 20 paired bootstrap samples are required")
    y, champion = validate_binary_probabilities(y, champion_probabilities)
    _, candidate = validate_binary_probabilities(y, candidate_probabilities)
    span = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if len(span) != len(y) or span.hasnans:
        raise ComparisonError("Valid timestamps must match evaluation rows")
    days = span.normalize().to_numpy()
    unique_days = np.unique(days)
    if len(unique_days) < 2:
        raise ComparisonError("Paired comparison requires at least two calendar days")
    day_indices = {day: np.flatnonzero(days == day) for day in unique_days}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(samples):
        selected = rng.choice(unique_days, size=len(unique_days), replace=True)
        indices = np.concatenate([day_indices[day] for day in selected])
        if np.unique(y[indices]).size < 2:
            continue
        deltas.append(
            average_precision_score(y[indices], candidate[indices])
            - average_precision_score(y[indices], champion[indices])
        )
    if len(deltas) < max(20, samples // 4):
        raise ComparisonError("Too few valid paired bootstrap samples; evaluation is inconclusive")
    return tuple(float(value) for value in np.quantile(deltas, [0.025, 0.975]))


def build_comparison(
    *,
    champion_version: str,
    candidate_version: str,
    champion_manifest_sha256: str,
    candidate_manifest_sha256: str,
    y,
    timestamps,
    champion_probabilities,
    candidate_probabilities,
    champion_threshold: float,
    candidate_threshold: float,
    bootstrap_samples: int = 500,
    noninferiority_margin: float = PROMOTION_POLICY["noninferiority_margin"],
    recall_tolerance: float = PROMOTION_POLICY["recall_tolerance"],
    brier_tolerance: float = PROMOTION_POLICY["brier_tolerance"],
    max_alerts_per_day: float = PROMOTION_POLICY["max_alerts_per_day"],
) -> dict[str, Any]:
    policy_values = (noninferiority_margin, recall_tolerance, brier_tolerance, max_alerts_per_day)
    if any(type(value) not in {int, float} or not np.isfinite(value) or value < 0 for value in policy_values):
        raise ComparisonError("Policy limits must be finite nonnegative numbers")
    champion = operating_metrics(y, champion_probabilities, champion_threshold, timestamps)
    candidate = operating_metrics(y, candidate_probabilities, candidate_threshold, timestamps)
    interval = paired_day_block_pr_auc_delta(
        y,
        champion_probabilities,
        candidate_probabilities,
        timestamps,
        samples=bootstrap_samples,
    )
    gates = {
        "paired_pr_auc_noninferiority": interval[0] >= -noninferiority_margin,
        "recall_noninferiority": candidate["recall"] >= champion["recall"] - recall_tolerance,
        "calibration_noninferiority": candidate["brier"] <= champion["brier"] + brier_tolerance,
        "alert_capacity": candidate["alerts_per_day"] <= max_alerts_per_day,
    }
    return {
        "schema_version": 1,
        "evaluation_role": "development_holdout",
        "historically_pristine": False,
        "bootstrap_samples": bootstrap_samples,
        "champion_version": champion_version,
        "candidate_version": candidate_version,
        "champion_manifest_sha256": champion_manifest_sha256,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "rows": len(y),
        "positives": int(np.asarray(y).sum()),
        "champion": champion,
        "candidate": candidate,
        "paired_pr_auc_delta_ci_95": list(interval),
        "policy": {
            "noninferiority_margin": noninferiority_margin,
            "recall_tolerance": recall_tolerance,
            "brier_tolerance": brier_tolerance,
            "max_alerts_per_day": max_alerts_per_day,
        },
        "gates": gates,
        "eligible": all(gates.values()),
        "caveat": (
            "Eligibility is a development-window governance gate, not evidence of "
            "external validity or authorization for automatic promotion."
        ),
    }


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
