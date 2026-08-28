"""Command-line interface for the Phase 2 human-review case workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.case_management import (
    CaseManagementError,
    CaseStore,
    ReviewerRole,
    case_to_dict,
    event_to_dict,
)


def _role(value: str) -> ReviewerRole:
    try:
        return ReviewerRole(value)
    except ValueError as exc:
        choices = ", ".join(role.value for role in ReviewerRole)
        raise argparse.ArgumentTypeError(f"role must be one of: {choices}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage investigation alerts through an explicit human RGS decision."
    )
    parser.add_argument("--db", help="Override AEGIS_CASE_DB_PATH for this invocation")
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create", help="Create a case from a threshold-breaching alert")
    create.add_argument("--trans-num", required=True)
    create.add_argument("--score", required=True, type=float)
    create.add_argument("--threshold", required=True, type=float)
    create.add_argument("--model-version", required=True)
    create.add_argument("--actor", required=True)
    create.add_argument("--role", type=_role, default=ReviewerRole.INVESTIGATOR)

    start = commands.add_parser("start-review", help="Assign and start human review")
    start.add_argument("case_id")
    start.add_argument("--actor", required=True)
    start.add_argument("--role", type=_role, required=True)
    start.add_argument("--rationale", required=True)
    start.add_argument("--expected-version", required=True, type=int)

    decide = commands.add_parser("decide", help="Record an authorized human RGS disposition")
    decide.add_argument("case_id")
    decide.add_argument("--actor", required=True)
    decide.add_argument("--role", type=_role, required=True)
    decide.add_argument("--rgs", choices=("reached", "not-reached"), required=True)
    decide.add_argument("--rationale", required=True)
    decide.add_argument("--expected-version", required=True, type=int)

    show = commands.add_parser("show", help="Show a case and its complete event history")
    show.add_argument("case_id")

    list_cmd = commands.add_parser("list", help="List cases, newest activity first")
    list_cmd.add_argument(
        "--status",
        choices=("alert_open", "under_review", "rgs_not_reached", "rgs_reached"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    store = CaseStore(args.db) if args.db else CaseStore()
    try:
        if args.command == "create":
            result = case_to_dict(store.create_alert_case(
                trans_num=args.trans_num,
                model_score=args.score,
                threshold=args.threshold,
                model_version=args.model_version,
                actor=args.actor,
                actor_role=args.role,
            ))
        elif args.command == "start-review":
            result = case_to_dict(store.start_review(
                args.case_id,
                actor=args.actor,
                actor_role=args.role,
                rationale=args.rationale,
                expected_version=args.expected_version,
            ))
        elif args.command == "decide":
            result = case_to_dict(store.record_rgs_decision(
                args.case_id,
                reached=args.rgs == "reached",
                actor=args.actor,
                actor_role=args.role,
                rationale=args.rationale,
                expected_version=args.expected_version,
            ))
        elif args.command == "show":
            result = {
                "case": case_to_dict(store.get_case(args.case_id)),
                "history": [event_to_dict(event) for event in store.history(args.case_id)],
            }
        else:
            result = [case_to_dict(case) for case in store.list_cases(args.status)]
    except CaseManagementError as exc:
        raise SystemExit(f"CASE_WORKFLOW_ERROR: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
