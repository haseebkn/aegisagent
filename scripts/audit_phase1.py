"""Read-only data/artifact audit that returns nonzero when a contract fails.

This is an offline check. It does not score the locked tail, contact Bedrock, or
print transaction identifiers or narrative contents. Temporal checks execute the
same reconstruction queries as dbt, not an ORDER BY/LAG tautology.
"""

import argparse
import math
import os
import sys
from pathlib import Path

import duckdb
from jinja2 import Environment, StrictUndefined

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.config import DB_PATH, resolve_model_dir


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_FILES = (
    "model_2_geo_rf.joblib", "model_3_cat_xgb.joblib", "model_4_vel_rf.joblib",
    "model_4_scaler.joblib", "meta_model.joblib", "meta_threshold.txt",
)


def audit_database(connection, expected_rows=1852394):
    results = {}
    environment = Environment(undefined=StrictUndefined)
    variables = {"expected_row_count": expected_rows}
    for path in sorted((ROOT / "tests").glob("assert_*.sql")):
        query = environment.from_string(path.read_text(encoding="utf-8")).render(
            ref=lambda name: name,
            var=lambda name, default=None: variables.get(name, default),
        )
        try:
            violations = connection.execute(f"SELECT COUNT(*) FROM ({query}) audit_result").fetchone()[0]
            results[path.stem] = {"passed": violations == 0, "violations": violations}
        except duckdb.Error as exc:
            results[path.stem] = {"passed": False, "error": type(exc).__name__}
    try:
        duplicate_count = connection.execute("""
            SELECT COUNT(*) FROM (
                SELECT trans_num FROM fct_fraud_features GROUP BY trans_num HAVING COUNT(*) > 1
            ) duplicates
        """).fetchone()[0]
        results["unique_transactions"] = {"passed": duplicate_count == 0, "violations": duplicate_count}
    except duckdb.Error as exc:
        results["unique_transactions"] = {"passed": False, "error": type(exc).__name__}
    return results


def audit_artifacts(directory):
    directory = Path(directory)
    results = {}
    for name in ARTIFACT_FILES:
        path = directory / name
        results[name] = {"passed": path.is_file() and path.stat().st_size > 0}
    try:
        threshold = float((directory / "meta_threshold.txt").read_text(encoding="utf-8"))
        results["threshold_range"] = {"passed": math.isfinite(threshold) and 0 < threshold < 1}
    except (OSError, ValueError):
        results["threshold_range"] = {"passed": False}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--expected-rows", type=int, default=1852394)
    parser.add_argument("--artifacts", type=Path, default=None)
    args = parser.parse_args()
    if args.expected_rows < 1:
        parser.error("--expected-rows must be positive")
    try:
        with duckdb.connect(str(args.db), read_only=True) as connection:
            results = audit_database(connection, args.expected_rows)
    except duckdb.Error as exc:
        results = {"database_access": {"passed": False, "error": type(exc).__name__}}
    try:
        results.update(audit_artifacts(args.artifacts or resolve_model_dir()))
    except (OSError, RuntimeError, ValueError) as exc:
        results["artifact_resolution"] = {"passed": False, "error": type(exc).__name__}
    for name, result in results.items():
        detail = f" ({result['violations']} violations)" if "violations" in result else ""
        detail += f" ({result['error']})" if "error" in result else ""
        print(f"{'PASS' if result['passed'] else 'FAIL'} {name}{detail}")
    failures = sum(not result["passed"] for result in results.values())
    print(f"Audit: {len(results) - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
