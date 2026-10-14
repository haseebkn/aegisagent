"""Path resolution and the feature contract shared by training and serving."""
import importlib

import pytest

from scripts import config


def test_defaults_are_relative_to_repo_root():
    """Regression: 21 hardcoded 'e:/AegisAgent/...' defaults meant nothing ran
    outside one machine."""
    for path in (config.DB_PATH, config.ARTIFACTS_DIR, config.COMPLIANCE_LOGS_DIR,
                 config.CASE_DB_PATH, config.EVIDENCE_DIR):
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


def test_serving_equivalent_encodings_are_not_model_inputs():
    """The mart emits <encoding>_serving purely as a drift-monitoring reference. On a
    training row it is computed from statistics that include that row, so feeding it to
    a model would reintroduce exactly the leakage temporal encoding removes."""
    serving = {"category_risk_serving", "state_risk_serving", "merchant_risk_serving"}
    for feats in (config.FEAT_M2, config.FEAT_M3, config.FEAT_M4):
        assert not (serving & set(feats))


def test_resolve_model_dir_follows_version_pointer(tmp_path):
    (tmp_path / "v_test").mkdir()
    (tmp_path / "latest_version.txt").write_text("v_test")
    assert config.resolve_model_dir(tmp_path) == tmp_path / "v_test"


def test_resolve_model_dir_raises_on_stale_pointer(tmp_path):
    """Regression: the base dir held a stale pre-versioning artifact set with the
    wrong threshold, so falling back on a broken pointer silently scored with the
    wrong models."""
    (tmp_path / "latest_version.txt").write_text("v_does_not_exist")
    with pytest.raises(FileNotFoundError, match="v_does_not_exist"):
        config.resolve_model_dir(tmp_path)


def test_resolve_model_dir_allows_flat_layout_without_pointer(tmp_path):
    assert config.resolve_model_dir(tmp_path) == tmp_path


def test_resolve_model_dir_fails_closed_when_registry_and_pointer_disagree(tmp_path):
    from scripts.model_registry import REQUIRED_ARTIFACTS, register_version

    (tmp_path / "v1").mkdir()
    (tmp_path / "v2").mkdir()
    for name in REQUIRED_ARTIFACTS:
        (tmp_path / "v2" / name).write_text("artifact", encoding="utf-8")
    register_version("v2", artifacts_dir=tmp_path, registry_path=tmp_path / "registry.json", bootstrap_champion=True)
    (tmp_path / "latest_version.txt").write_text("v1")
    with pytest.raises(RuntimeError, match="disagree"):
        config.resolve_model_dir(tmp_path)


@pytest.mark.parametrize("version", ["", ".", "..", "../escape", "nested/version", "nested\\version", "C:\\outside", "/outside"])
def test_serving_pointer_cannot_escape_artifact_root(tmp_path, version):
    (tmp_path / "latest_version.txt").write_text(version, encoding="utf-8")
    with pytest.raises(ValueError, match="safe directory name"):
        config.resolve_model_dir(tmp_path)


def test_missing_pointer_does_not_bypass_a_governed_registry(tmp_path):
    (tmp_path / "registry.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no serving pointer"):
        config.resolve_model_dir(tmp_path)


def test_registry_defaults_beside_overridden_artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv("MODELS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    reloaded = importlib.reload(config)
    try:
        assert reloaded.MODEL_REGISTRY_PATH == tmp_path / "artifacts" / "registry.json"
    finally:
        monkeypatch.delenv("MODELS_ARTIFACTS_DIR")
        importlib.reload(config)
