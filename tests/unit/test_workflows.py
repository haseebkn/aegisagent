"""The CI workflow must be valid YAML.

Regression: `run: dbt test --vars '{expected_row_count: 8500}'` written inline made
GitHub Actions fail at parse time with a 0-second run, because YAML read the
"key: value" inside the flow mapping as structure rather than as string content.
A malformed workflow fails silently in the sense that no job ever runs.
"""
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOW_DIR = Path(__file__).resolve().parents[2] / ".github" / "workflows"


def workflow_files():
    return sorted(WORKFLOW_DIR.glob("*.yml")) + sorted(WORKFLOW_DIR.glob("*.yaml"))


def test_at_least_one_workflow_exists():
    assert workflow_files()


@pytest.mark.parametrize("path", workflow_files(), ids=lambda p: p.name)
def test_workflow_is_valid_yaml(path):
    yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", workflow_files(), ids=lambda p: p.name)
def test_workflow_defines_jobs_with_steps(path):
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert doc.get("jobs"), f"{path.name} defines no jobs"
    for name, job in doc["jobs"].items():
        assert job.get("steps"), f"job '{name}' has no steps"
        assert job.get("runs-on"), f"job '{name}' has no runner"
