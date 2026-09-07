import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.demo import demo_environment
from scripts import verify_pipeline


def test_demo_isolates_data_and_disables_cloud_archival(tmp_path, monkeypatch):
    monkeypatch.setenv("AEGIS_CASE_DB_PATH", "research-cases.sqlite3")
    monkeypatch.setenv("AEGIS_MODEL_REGISTRY_PATH", "research-registry.json")
    monkeypatch.setenv("AEGIS_SECURITY_MODE", "production")
    env = demo_environment(tmp_path)
    assert env["AEGIS_CASE_DB_PATH"] == str(tmp_path / "cases" / "cases.sqlite3")
    assert env["AEGIS_MODEL_REGISTRY_PATH"] == str(tmp_path / "models_artifacts" / "registry.json")
    assert env["AEGIS_SECURITY_MODE"] == "development"
    assert env["COMPLIANCE_S3_BUCKET"] == ""


def test_demo_refuses_to_overwrite_existing_data(tmp_path):
    original = tmp_path / "research.json"
    original.write_text('{"important": true}', encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "scripts.demo", "--output-dir", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "must be empty" in result.stderr
    assert json.loads(original.read_text()) == {"important": True}
    assert list(tmp_path.iterdir()) == [original]


@pytest.mark.parametrize("probabilities,expected", [
    ([0.0, 1.0], True), ([0.2, float("nan")], False), ([0.2, 1.01], False),
])
def test_verifier_accepts_probability_endpoints_but_rejects_invalid_values(
    monkeypatch, probabilities, expected,
):
    values = np.asarray(probabilities)
    monkeypatch.setattr(verify_pipeline, "_load_scored_test_sample", lambda **kwargs: (
        pd.DataFrame({"row": [0, 1]}), values, values, values, values, values >= .8,
    ))
    assert verify_pipeline.verify_inference_bounds() is expected


def test_offline_workflow_refuses_a_sample_without_alerts(monkeypatch):
    values = np.array([0.2, 0.3])
    monkeypatch.setattr(verify_pipeline, "_load_scored_test_sample", lambda: (
        pd.DataFrame({"trans_num": ["a", "b"]}), values, values, values, values,
        np.array([False, False]),
    ))
    with pytest.raises(verify_pipeline.NoAlertsInSample):
        verify_pipeline.verify_case_workflow()
