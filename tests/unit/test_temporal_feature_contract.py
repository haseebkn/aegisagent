"""Static regression guards for temporal SQL contracts.

The dbt singular tests verify values. These checks make the intended construction
obvious and fail quickly in CI before a full fixture build.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def source(path):
    return (ROOT / path).read_text(encoding="utf-8").lower()


def test_target_encoding_uses_prior_rows_not_random_folds():
    feature_sql = source("models/intermediate/int_feature_engineering.sql")
    macro_sql = source("macros/temporal_target_encoding.sql")
    assert "hash(" not in feature_sql
    assert "enc_fold" not in feature_sql
    assert "1 preceding" in macro_sql
    assert "target_encoding_cold_start_prior" in macro_sql


def test_card_statistics_exclude_current_and_future_rows():
    feature_sql = source("models/intermediate/int_feature_engineering.sql")
    assert feature_sql.count("interval '90 days' preceding") == 3
    assert feature_sql.count("interval '1 microsecond' preceding") == 3
    assert "card_rates" not in feature_sql
    assert "card_txn_cnt < 5" in feature_sql


def test_velocity_window_ends_before_current_timestamp():
    velocity_sql = source("models/intermediate/int_velocity_features.sql")
    assert velocity_sql.count("interval '1 microsecond' preceding") == 2
    assert "and current row" not in velocity_sql


def test_mart_has_prospectively_locked_evaluation_role():
    mart_sql = source("models/marts/fct_fraud_features.sql")
    assert "development_holdout" in mart_sql
    assert "locked_evaluation" in mart_sql
