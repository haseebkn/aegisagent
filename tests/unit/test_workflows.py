"""The CI workflow must be valid YAML.

Regression: `run: dbt test --vars '{expected_row_count: 14500}'` written inline made
GitHub Actions fail at parse time with a 0-second run, because YAML read the
"key: value" inside the flow mapping as structure rather than as string content.
A malformed workflow fails silently in the sense that no job ever runs.
"""
import ast
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOW_DIR = Path(__file__).resolve().parents[2] / ".github" / "workflows"
FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "make_fixture.py"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def test_fixture_reference_is_warm_and_ci_row_contract_matches():
    """The 90-day card counter needs a long enough reference to leave cold start.

    At 6,000 training rows the fixed fixture produced a false PSI 0.354 alert;
    12,000 rows produces PSI 0.055 without changing the production drift threshold.
    """
    tree = ast.parse(FIXTURE_PATH.read_text(encoding="utf-8"))
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"DEFAULT_TRAIN_ROWS", "DEFAULT_TEST_ROWS"}
    }
    assert constants["DEFAULT_TRAIN_ROWS"] >= 4 * constants["DEFAULT_TEST_ROWS"]
    expected = constants["DEFAULT_TRAIN_ROWS"] + constants["DEFAULT_TEST_ROWS"]
    workflow_text = "\n".join(path.read_text(encoding="utf-8") for path in workflow_files())
    assert f"expected_row_count: {expected}" in workflow_text


def test_ci_uses_read_only_token_and_node24_action_generations():
    ci_path = WORKFLOW_DIR / "ci.yml"
    doc = yaml.safe_load(ci_path.read_text(encoding="utf-8"))
    assert doc["permissions"] == {"contents": "read"}
    workflow_text = ci_path.read_text(encoding="utf-8")
    for retired in (
        "actions/checkout@v4",
        "actions/setup-python@v5",
        "hashicorp/setup-terraform@v3",
        "docker/setup-buildx-action@v3",
        "docker/build-push-action@v6",
    ):
        assert retired not in workflow_text


def test_container_defaults_to_non_root_fail_closed_runtime():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    dockerignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "FROM python:3.11-slim@sha256:" in dockerfile
    assert "AEGIS_SECURITY_MODE=production" in dockerfile
    assert "--create-home" in dockerfile
    assert "USER aegis" in dockerfile
    assert "/home/aegis/.aws" in compose
    assert "/root/.aws" not in compose
    for excluded in (".env", ".aws/", "*credentials*", "*.pem"):
        assert excluded in dockerignore
