"""Path resolution and the feature contract shared by training and serving."""
import importlib
import os

import pytest

from scripts import config


def test_defaults_are_relative_to_repo_root():
    """Regression: 21 hardcoded 'e:/AegisAgent/...' defaults meant nothing ran
    outside one machine."""
    for path in (config.DB_PATH, config.ARTIFACTS_DIR, config.COMPLIANCE_LOGS_DIR):
        assert config.PROJECT_ROOT in path.parents or path == config.PROJECT_ROOT
        assert "AegisAgent" not in str(path).replace(str(config.PROJECT_ROOT), "")


def test_env_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("DBT_DB_PATH", str(tmp_path / "other.duckdb"))
    reloaded = importlib.reload(config)
    try:
        assert reloaded.DB_PATH == tmp_path / "other.duckdb"
    finally:
        monkeypatch.delenv("DBT_DB_PATH")
        importlib.reload(config)


def test_feature_lists_are_non_empty_and_unique():
    for feats in (config.FEAT_M2, config.FEAT_M3, config.FEAT_M4):
        assert feats
        assert len(feats) == len(set(feats))


def test_graph_features_are_not_wired_into_any_model():
    """They degraded held-out performance badly; see docs/graph-features.md."""
    graph = {"merchant_fraud_card_ratio", "card_2hop_fraud_cards",
             "card_merchant_degree", "merchant_card_degree", "merchant_fraud_card_cnt"}
    for feats in (config.FEAT_M2, config.FEAT_M3, config.FEAT_M4):
        assert not (graph & set(feats))


def test_resolve_model_dir_follows_version_pointer(tmp_path):
    (tmp_path / "v_test").mkdir()
    (tmp_path / "latest_version.txt").write_text("v_test")
    assert config.resolve_model_dir(tmp_path) == tmp_path / "v_test"


def test_resolve_model_dir_falls_back_when_pointer_is_stale(tmp_path):
    (tmp_path / "latest_version.txt").write_text("v_does_not_exist")
    assert config.resolve_model_dir(tmp_path) == tmp_path
