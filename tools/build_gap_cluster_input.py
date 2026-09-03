#!/usr/bin/env python3
"""Build a compact, deterministic clustering input from accepted Phase A results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    phase = args.phase_root.resolve(strict=True)
    output = args.output
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing existing output: {output}")

    ledger_path = phase / "ledger.jsonl"
    rows: list[dict[str, Any]] = []
    accepted = 0
    for line_number, line in enumerate(
        ledger_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        event = json.loads(line)
        if event.get("status") != "accepted":
            continue
        accepted += 1
        receipt_path = phase / event["receipt"]
        receipt = load_object(receipt_path)
        if receipt.get("accepted") is not True or receipt.get("deterministic") is not True:
            raise ValueError(f"accepted ledger receipt differs at line {line_number}")
        case_root = receipt_path.parent.parent
        result_path = case_root / receipt["result"]
        result = load_object(result_path)
        if result.get("owner") != "ir_gap":
            continue
        gap = result.get("gap")
        primitive = gap.get("candidate_primitive") if isinstance(gap, dict) else None
        if not isinstance(primitive, dict):
            raise ValueError(f"missing gap primitive: {event['case_id']}")
        required = gap.get("minimum_required_semantics")
        if not isinstance(required, list) or not required:
            raise ValueError(f"missing minimum semantics: {event['case_id']}")
        row = {
            "case_id": event["case_id"],
            "exact_candidate_name": primitive.get("name"),
            "candidate_semantics": primitive.get("semantics"),
            "earliest_gap": gap.get("earliest_gap"),
            "minimum_required_semantics": required,
            "counterexample": gap.get("counterexample"),
            "reason": result.get("reason"),
            "assessment_scope": result.get("assessment_scope"),
            "source_result": result_path.relative_to(phase).as_posix(),
        }
        if any(not isinstance(row[key], str) or not row[key] for key in (
            "case_id",
            "exact_candidate_name",
            "candidate_semantics",
            "earliest_gap",
            "counterexample",
            "reason",
            "assessment_scope",
        )):
            raise ValueError(f"incomplete gap row: {event['case_id']}")
        rows.append(row)

    rows.sort(key=lambda item: (item["exact_candidate_name"], item["case_id"]))
    if accepted != 439 or len(rows) != 326 or len({row["case_id"] for row in rows}) != 326:
        raise ValueError(
            f"source counts differ: accepted={accepted}, gaps={len(rows)}, "
            f"unique={len({row['case_id'] for row in rows})}"
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["exact_candidate_name"]].append(row)
    if len(groups) != 296:
        raise ValueError(f"exact candidate-name count differs: {len(groups)}")

    value = {
        "schema": "open-cake.aka-ir-gap-cluster-input.v1",
        "source": {
            "phase_a_root": str(phase),
            "ledger": "ledger.jsonl",
            "accepted_results": accepted,
            "selection": "deterministic verifier-accepted rows with owner=ir_gap",
        },
        "counts": {
            "gap_cases": len(rows),
            "exact_candidate_names": len(groups),
        },
        "exact_name_groups": [
            {
                "exact_candidate_name": name,
                "case_count": len(groups[name]),
                "members": groups[name],
            }
            for name in sorted(groups)
        ],
        "evidence_boundary": {
            "current_ir_not_expressible": "accepted per-item reviewer conclusion only",
            "cross_item_merge": "not yet established",
            "ir_change_approved": False,
            "implementation_performed": False,
            "gpu_or_performance_claim": False,
        },
    }
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "created", "output": str(output), **value["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
