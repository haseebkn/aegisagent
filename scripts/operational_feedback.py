"""Aggregate analyst dispositions without treating them as fraud ground truth."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median

from scripts.case_management import CaseStatus, CaseStore
from scripts.security import (
    Permission,
    local_development_principal,
    require_permission,
)


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def build_feedback_report(cases, histories, *, min_cohort_size: int = 5):
    if min_cohort_size < 1:
        raise ValueError("min_cohort_size must be positive")
    cohorts = defaultdict(lambda: {"cases": [], "review_hours": []})
    for case in cases:
        cohort = cohorts[case.model_version]
        cohort["cases"].append(case)
        events = histories.get(case.case_id, [])
        started = next((event for event in events if event.event_type == "review_started"), None)
        decided = next(
            (
                event
                for event in events
                if event.event_type in {"rgs_reached", "rgs_not_reached"}
            ),
            None,
        )
        if started and decided:
            hours = (_parse(decided.occurred_at) - _parse(started.occurred_at)).total_seconds() / 3600
            cohort["review_hours"].append(max(hours, 0.0))

    output = {}
    for version, values in sorted(cohorts.items()):
        items = values["cases"]
        if len(items) < min_cohort_size:
            output[version] = {
                "case_count": len(items),
                "suppressed": True,
                "minimum_cohort_size": min_cohort_size,
            }
            continue
        reached = sum(case.status == CaseStatus.RGS_REACHED.value for case in items)
        not_reached = sum(case.status == CaseStatus.RGS_NOT_REACHED.value for case in items)
        terminal = reached + not_reached
        output[version] = {
            "case_count": len(items),
            "terminal_dispositions": terminal,
            "rgs_reached": reached,
            "rgs_not_reached": not_reached,
            "rgs_reached_rate_among_terminal": reached / terminal if terminal else None,
            "median_review_hours": median(values["review_hours"])
            if values["review_hours"]
            else None,
            "suppressed": False,
        }
    return {
        "schema_version": 1,
        "outcome_semantics": "human_rgs_disposition_not_fraud_ground_truth",
        "training_label_eligible": False,
        "minimum_cohort_size": min_cohort_size,
        "models": output,
        "limitations": (
            "RGS dispositions reflect reviewed case context and selection by the deployed "
            "alert policy. They are not fraud labels, are subject to selection bias, and "
            "must not be joined into model training as ground truth."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--minimum-cohort-size", type=int, default=5)
    args = parser.parse_args()
    identity = local_development_principal()
    require_permission(identity, Permission.EXPORT_OPERATIONAL_FEEDBACK)
    store = CaseStore(principal=identity)
    cases = store.list_cases(principal=identity)
    histories = {case.case_id: store.history(case.case_id, principal=identity) for case in cases}
    report = build_feedback_report(
        cases,
        histories,
        min_cohort_size=args.minimum_cohort_size,
    )
    rendered = json.dumps(report, indent=2)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
        print(f"Operational feedback report written to {destination}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
