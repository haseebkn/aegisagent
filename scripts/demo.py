"""Build and verify an isolated synthetic portfolio demo with no cloud credentials.

Run ``python -m scripts.demo --serve`` to open the dashboard after verification.
Fixture scores demonstrate mechanics, not the quality measured in reports/.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def demo_environment(directory: Path) -> dict[str, str]:
    """Keep all generated data and local identities in one isolated demo directory."""
    directory = directory.resolve()
    env = dict(os.environ)
    env.update({
        "AEGIS_RAW_DATA_DIR": str(directory / "raw"),
        "DBT_DB_PATH": str(directory / "demo.duckdb"),
        "DBT_PROFILES_DIR": str(PROJECT_ROOT),
        "DBT_TARGET_PATH": str(directory / "dbt_target"),
        "DBT_LOG_PATH": str(directory / "dbt_logs"),
        "DBT_SEND_ANONYMOUS_USAGE_STATS": "false",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        "MODELS_ARTIFACTS_DIR": str(directory / "models_artifacts"),
        "AEGIS_MODEL_REGISTRY_PATH": str(directory / "models_artifacts" / "registry.json"),
        "COMPLIANCE_LOGS_DIR": str(directory / "cases"),
        "AEGIS_CASE_DB_PATH": str(directory / "cases" / "cases.sqlite3"),
        "AEGIS_EVIDENCE_DIR": str(directory / "cases" / "evidence"),
        "AEGIS_SECURITY_MODE": "development",
        "AEGIS_DEV_SUBJECT": "portfolio-reviewer",
        "AEGIS_DEV_ROLES": "investigator",
        "AEGIS_DEV_ORGANIZATION": "synthetic-portfolio-demo",
        "COMPLIANCE_S3_BUCKET": "",
        "AWS_EC2_METADATA_DISABLED": "true",
        "AEGIS_TRAINING_JOBS": "2",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "PYTHONPATH": str(PROJECT_ROOT),
        "PYTHONUTF8": "1",
    })
    return env


def run(command: list[str], env: dict[str, str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        help="An empty directory for generated demo data; defaults to .demo/run-*")
    parser.add_argument("--serve", action="store_true", help="Launch Streamlit on 127.0.0.1:8501")
    args = parser.parse_args(argv)
    if args.output_dir is not None:
        directory = args.output_dir.resolve()
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            parser.error("--output-dir must be empty; existing demo and research data are preserved")
        directory.mkdir(parents=True, exist_ok=True)
    else:
        demo_root = PROJECT_ROOT / ".demo"
        demo_root.mkdir(exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="run-", dir=demo_root))
    env = demo_environment(directory)
    print(f"Synthetic demonstration directory: {directory}", flush=True)
    try:
        run([sys.executable, str(PROJECT_ROOT / "tests/fixtures/make_fixture.py"),
             "--out-dir", env["AEGIS_RAW_DATA_DIR"]], env)
        run([sys.executable, "-c", "from dbt.cli.main import cli; cli()",
             "run", "--profiles-dir", str(PROJECT_ROOT)], env)
        run([sys.executable, "-m", "scripts.train_models"], env)
        run([sys.executable, "-m", "scripts.verify_pipeline", "--expected-rows", "14500"], env)
        run([sys.executable, "-m", "scripts.drift", "--fail-on-significant"], env)
        run([sys.executable, "tests/smoke_dashboard.py"], env)
    except subprocess.CalledProcessError as exc:
        print(f"Demo stopped: a required stage failed (exit {exc.returncode}).", file=sys.stderr)
        return 1
    summary = {
        "purpose": "synthetic workflow demonstration, not model validation",
        "source": "tests/fixtures/make_fixture.py",
        "rows": 14500,
        "cloud_calls": False,
        "verification": "dbt, inference, case/evidence integrity, drift and dashboard passed",
        "environment": {key: value for key, value in env.items() if key in demo_environment_keys()},
    }
    (directory / "demo-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Demo verified. Summary: {directory / 'demo-summary.json'}", flush=True)
    if args.serve:
        print("Dashboard: http://127.0.0.1:8501 (Ctrl+C to stop)", flush=True)
        run([sys.executable, "-m", "streamlit", "run", "app.py",
             "--server.address", "127.0.0.1", "--server.headless", "true"], env)
    return 0


def demo_environment_keys() -> tuple[str, ...]:
    # Only non-secret paths and local demonstration settings enter the summary.
    return ("AEGIS_RAW_DATA_DIR", "DBT_DB_PATH", "MODELS_ARTIFACTS_DIR",
            "AEGIS_MODEL_REGISTRY_PATH", "COMPLIANCE_LOGS_DIR", "AEGIS_CASE_DB_PATH",
            "AEGIS_EVIDENCE_DIR", "AEGIS_SECURITY_MODE", "AEGIS_DEV_ORGANIZATION")


if __name__ == "__main__":
    raise SystemExit(main())
