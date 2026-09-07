"""Behavioral validation at the scoring boundary, including interior batch rows."""

import numpy as np
import pandas as pd
import pytest

from scripts.inference_engine import SCORED_FEATURES, load_models, run_inference, validate_frame


def scoreable_frame():
    frame = pd.DataFrame({column: np.zeros(3) for column in SCORED_FEATURES})
    frame["trans_num"] = ["txn-1", "txn-2", "txn-3"]
    return frame


@pytest.mark.parametrize("column,value", [
    ("amt", -1.0), ("category_risk", 1.1), ("state_risk", -0.01),
    ("hour", 24.0), ("txns_24h", 1.5), ("night", 0.5),
    ("card_txn_cnt", -1.0), ("hour_sin", 1.01), ("merchant_risk", np.inf),
    ("trans_num", ""), ("trans_num", "   "),
])
def test_every_row_obeys_feature_contract(column, value):
    frame = scoreable_frame()
    frame.loc[1, column] = value
    with pytest.raises(ValueError):
        validate_frame(frame)


def test_duplicate_dataframe_index_is_valid_without_label_based_row_sampling():
    frame = scoreable_frame()
    frame.index = [0, 0, 0]
    validate_frame(frame)


def test_empty_batch_is_rejected_before_estimators_run():
    with pytest.raises(ValueError, match="at least one transaction"):
        run_inference(scoreable_frame().iloc[:0], None, None, None, None, None, 0.5)


@pytest.mark.parametrize("threshold", [None, np.nan, np.inf, -0.1, 1.1, True])
def test_threshold_must_come_from_the_loaded_model_bundle(threshold):
    with pytest.raises(ValueError, match="threshold"):
        run_inference(scoreable_frame(), None, None, None, None, None, threshold)


def test_missing_threshold_does_not_invent_an_alert_policy(tmp_path, monkeypatch):
    def forbidden_load(path):
        pytest.fail("Model should not load without its decision threshold")
    monkeypatch.setattr("scripts.inference_engine.joblib_load", forbidden_load)
    with pytest.raises(FileNotFoundError):
        load_models(tmp_path)


def test_nonfinite_model_outputs_cannot_create_silent_nonalerts():
    class Model:
        def __init__(self, probability):
            self.probability = probability

        def predict_proba(self, values):
            return np.tile([1 - self.probability, self.probability], (len(values), 1))

    class Scaler:
        def transform(self, values):
            return values

    with pytest.raises(ValueError, match="invalid probabilities"):
        run_inference(scoreable_frame(), Model(0.1), Model(0.1), Model(0.1), Scaler(), Model(np.nan), 0.5)
