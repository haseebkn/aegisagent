"""Compare, promote, inspect, or roll back governed AegisAgent model versions."""

from __future__ import annotations

import argparse
import json

import duckdb

from scripts.config import ARTIFACTS_DIR, DB_PATH, MODEL_REGISTRY_PATH
from scripts.inference_engine import load_models, run_inference
from scripts.model_comparison import build_comparison, write_report
from scripts.model_registry import (
    RegistryError,
    load_registry,
    promote_candidate,
    record_comparison,
    rollback_champion,
)
from scripts.security import (
    Permission,
    local_development_principal,
    require_permission,
)


def _principal(permission: Permission):
    identity = local_development_principal()
    require_permission(identity, permission)
    return identity


def _score(version, frame):
    model_dir = ARTIFACTS_DIR / version
    models = load_models(model_dir)
    probabilities = run_inference(frame, *models)[0]
    return probabilities, models[-1]


def compare(args):
    identity = _principal(Permission.COMPARE_MODELS)
    registry = load_registry(MODEL_REGISTRY_PATH)
    champion = registry["champion"]
    if not champion:
        raise RegistryError("Registry has no champion")
    candidate = registry["versions"].get(args.candidate)
    if not candidate or candidate["status"] != "candidate":
        raise RegistryError(f"{args.candidate} is not a registered candidate")
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        frame = con.execute(
            """SELECT * FROM fct_fraud_features
               WHERE evaluation_role = 'development_holdout'
               ORDER BY trans_date_trans_time, trans_num"""
        ).df()
    if frame.empty or frame["is_fraud"].nunique() < 2:
        raise RegistryError("Development holdout must contain both outcome classes")
    champion_probs, champion_threshold = _score(champion, frame)
    candidate_probs, candidate_threshold = _score(args.candidate, frame)
    report = build_comparison(
        champion_version=champion,
        candidate_version=args.candidate,
        champion_manifest_sha256=registry["versions"][champion]["manifest_sha256"],
        candidate_manifest_sha256=candidate["manifest_sha256"],
        y=frame["is_fraud"].to_numpy(),
        timestamps=frame["trans_date_trans_time"],
        champion_probabilities=champion_probs,
        candidate_probabilities=candidate_probs,
        champion_threshold=champion_threshold,
        candidate_threshold=candidate_threshold,
        bootstrap_samples=args.bootstrap_samples,
    )
    destination = ARTIFACTS_DIR / args.candidate / "promotion_report.json"
    write_report(destination, report)
    record_comparison(destination, principal=identity)
    print(json.dumps(report, indent=2))
    print(f"Promotion report: {destination}")


def promote(args):
    identity = _principal(Permission.PROMOTE_MODEL)
    report = ARTIFACTS_DIR / args.candidate / "promotion_report.json"
    registry = promote_candidate(
        args.candidate,
        principal=identity,
        rationale=args.rationale,
        report_path=report,
    )
    print(f"Promoted {registry['champion']} as champion. Serving pointer changed atomically.")


def rollback(args):
    identity = _principal(Permission.PROMOTE_MODEL)
    registry = rollback_champion(
        args.version,
        principal=identity,
        rationale=args.rationale,
    )
    print(f"Rolled serving back to {registry['champion']}.")


def status(_args):
    _principal(Permission.COMPARE_MODELS)
    registry = load_registry(MODEL_REGISTRY_PATH)
    summary = {
        "champion": registry["champion"],
        "versions": {
            version: record["status"] for version, record in registry["versions"].items()
        },
        "event_chain_length": len(registry["events"]),
        "event_chain_head": registry["events"][-1]["event_hash"] if registry["events"] else "",
    }
    print(json.dumps(summary, indent=2))


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--candidate", required=True)
    comparison.add_argument("--bootstrap-samples", type=int, default=500)
    comparison.set_defaults(function=compare)
    promotion = commands.add_parser("promote")
    promotion.add_argument("--candidate", required=True)
    promotion.add_argument("--rationale", required=True)
    promotion.set_defaults(function=promote)
    rollback_command = commands.add_parser("rollback")
    rollback_command.add_argument("--version", required=True)
    rollback_command.add_argument("--rationale", required=True)
    rollback_command.set_defaults(function=rollback)
    status_command = commands.add_parser("status")
    status_command.set_defaults(function=status)
    return root


def main():
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
