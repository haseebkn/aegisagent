"""Compare, promote, inspect, or roll back governed AegisAgent model versions."""

from __future__ import annotations

import argparse
import json

from scripts.config import ARTIFACTS_DIR, MODEL_REGISTRY_PATH, version_directory
from scripts.model_comparison import write_report
from scripts.model_registry import (
    evaluate_registered_comparison,
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


def compare(args):
    identity = _principal(Permission.COMPARE_MODELS)
    report = evaluate_registered_comparison(
        args.candidate,
        bootstrap_samples=args.bootstrap_samples,
    )
    destination = version_directory(ARTIFACTS_DIR, args.candidate) / "promotion_report.json"
    write_report(destination, report)
    record_comparison(destination, principal=identity)
    print(json.dumps(report, indent=2))
    print(f"Promotion report: {destination}")


def promote(args):
    identity = _principal(Permission.PROMOTE_MODEL)
    report = version_directory(ARTIFACTS_DIR, args.candidate) / "promotion_report.json"
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
