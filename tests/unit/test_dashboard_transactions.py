import duckdb
import numpy as np
import pandas as pd
import pytest

from scripts.dashboard_transactions import data_fingerprint, load_alert_queue
from scripts.inference_engine import SCORED_FEATURES


class FeatureProbability:
    def predict_proba(self, values):
        probabilities = np.asarray(values)[:, 0]
        return np.column_stack([1 - probabilities, probabilities])


class IdentityScaler:
    def transform(self, values):
        return values


def bundle(threshold=.8):
    predictor = FeatureProbability()
    return predictor, predictor, predictor, IdentityScaler(), predictor, threshold


def database(tmp_path, scores, roles=None, labels=None):
    frame = pd.DataFrame({column: np.zeros(len(scores)) for column in SCORED_FEATURES})
    frame["amt"] = scores
    frame["trans_num"] = [f"txn-{index:06}" for index in range(len(scores))]
    frame["trans_date_trans_time"] = pd.date_range("2020-01-01", periods=len(scores), freq="h")
    frame["evaluation_role"] = roles or ["development_holdout"] * len(scores)
    frame["is_fraud"] = labels or [0] * len(scores)
    path = tmp_path / "features.duckdb"
    with duckdb.connect(str(path)) as connection:
        connection.register("feature_rows", frame)
        connection.execute("CREATE TABLE fct_fraud_features AS SELECT * FROM feature_rows")
    descriptions = frame[["trans_num"]].assign(merchant="Synthetic Merchant", category="shopping_pos")
    descriptions.to_csv(tmp_path / "fraudTest.csv", index=False)
    return path


def test_alert_after_first_200_rows_is_offered_even_when_label_is_zero(tmp_path):
    path = database(tmp_path, [.01] * 250 + [.95], labels=[1] * 250 + [0])
    queue = load_alert_queue(path, bundle(), raw_data_dir=tmp_path, batch_size=37)
    assert queue.scored_rows == 251 and queue.alert_count == 1
    assert [item["trans_num"] for item in queue.transactions] == ["txn-000250"]
    assert queue.transactions[0]["alert_score"] == .95
    assert queue.transactions[0]["merchant"] == "Synthetic Merchant"
    assert "is_fraud" not in queue.transactions[0]


def test_global_top_alerts_are_sorted_and_locked_data_is_excluded(tmp_path):
    path = database(tmp_path, [.85, .9, .8, 1.0, .99], roles=[
        "development_holdout", "development_holdout", "development_holdout",
        "locked_evaluation", "model_development",
    ])
    queue = load_alert_queue(path, bundle(), raw_data_dir=tmp_path, limit=2, batch_size=1)
    assert queue.scored_rows == 3 and queue.alert_count == 3
    assert [item["alert_score"] for item in queue.transactions] == [.9, .85]


def test_empty_alert_queue_does_not_invent_a_demo_alert(tmp_path):
    path = database(tmp_path, [.1, .7])
    queue = load_alert_queue(path, bundle(), raw_data_dir=tmp_path)
    assert queue.transactions == [] and queue.alert_count == 0 and queue.scored_rows == 2


def test_equal_scores_use_stable_transaction_order(tmp_path):
    path = database(tmp_path, [.9, .9, .9])
    queue = load_alert_queue(path, bundle(), raw_data_dir=tmp_path, limit=2, batch_size=1)
    assert [item["trans_num"] for item in queue.transactions] == ["txn-000000", "txn-000001"]


def test_changing_model_threshold_changes_queue(tmp_path):
    path = database(tmp_path, [.81, .9])
    assert load_alert_queue(path, bundle(.8), raw_data_dir=tmp_path).alert_count == 2
    assert load_alert_queue(path, bundle(.85), raw_data_dir=tmp_path).alert_count == 1


def test_missing_development_window_is_a_data_error(tmp_path):
    path = database(tmp_path, [.99], roles=["locked_evaluation"])
    with pytest.raises(ValueError, match="development-holdout"):
        load_alert_queue(path, bundle(), raw_data_dir=tmp_path)


@pytest.mark.parametrize("options", [{"limit": 0}, {"limit": True}, {"batch_size": 0}])
def test_invalid_queue_bounds_are_rejected(tmp_path, options):
    path = database(tmp_path, [.9])
    with pytest.raises(ValueError):
        load_alert_queue(path, bundle(), raw_data_dir=tmp_path, **options)


def test_missing_description_file_fails_without_creating_data(tmp_path):
    path = database(tmp_path, [.9])
    (tmp_path / "fraudTest.csv").unlink()
    with pytest.raises(FileNotFoundError):
        load_alert_queue(path, bundle(), raw_data_dir=tmp_path)
    assert not (tmp_path / "fraudTest.csv").exists()


def test_cache_fingerprint_changes_when_descriptions_are_updated(tmp_path):
    path = database(tmp_path, [.9])
    before = data_fingerprint(path, tmp_path)
    pd.DataFrame({"trans_num": ["txn-000000"], "merchant": ["Changed Merchant Name"],
                  "category": ["shopping_pos"]}).to_csv(tmp_path / "fraudTest.csv", index=False)
    assert data_fingerprint(path, tmp_path) != before
