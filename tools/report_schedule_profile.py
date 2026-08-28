#!/usr/bin/env python3
"""Report an NCU-aligned static profile envelope without running a GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler, Schedule, Target, profile_envelope  # noqa: E402


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _paths(arguments: argparse.Namespace) -> list[Path]:
    if arguments.manifest is not None:
        manifest = json.loads(arguments.manifest.resolve(strict=True).read_text())
        return [
            ROOT / case["schedule"]
            for case in manifest["cases"]
            if arguments.all or case["expected"].get("lowering_eligible") is True
        ]
    if not arguments.schedules:
        raise ValueError("provide at least one Schedule or --manifest")
    return [path.resolve(strict=True) for path in arguments.schedules]


def _metric(document: dict[str, object], name: str) -> dict[str, object]:
    return next(
        metric
        for metric in document["ncu_metrics"]
        if metric["metric"] == name
    )


def _table(rows: list[dict[str, object]]) -> str:
    header = (
        f"{'schedule':<34}{'regs>=':>8}{'CTA/SM<=':>10}{'warps%<=':>10}"
        f"{'barrier':>10}{'scoreboard':>12}{'source B':>12}  coverage"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        profile = row["profile"]
        registers = _metric(profile, "launch__registers_per_thread")
        occupancy = profile["residency"] or {}
        active = _metric(
            profile, "sm__warps_active.avg.pct_of_peak_sustained_elapsed"
        )
        barrier = _metric(
            profile,
            "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
        )
        scoreboard = _metric(
            profile,
            "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        )
        source_bytes = profile["lowering"]["generated_source_bytes"]
        active_value = active["value"]
        active_text = "-" if active_value is None else f"{float(active_value):.1f}"
        lines.append(
            f"{str(row['schedule_id'])[:33]:<34}"
            f"{str(registers['value']):>8}"
            f"{str(occupancy.get('ctas_per_sm_upper_bound')):>10}"
            f"{active_text:>10}"
            f"{str(barrier['value']):>10}"
            f"{str(scoreboard['value']):>12}"
            f"{str(source_bytes):>12}  "
            f"{len(profile['abstentions'])} abstention(s)"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision",
        type=Path,
        default=ROOT / "compiler/revision.lock.json",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("schedules", nargs="*", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.manifest is not None and arguments.schedules:
        raise ValueError("--manifest and positional Schedules are exclusive")
    compiler = Compiler.load(ROOT, arguments.revision.resolve(strict=True))
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    targets: dict[str, Target] = {}
    for path in _paths(arguments):
        assessment = compiler.assess_file(path)
        findings = [finding.code for finding in assessment.findings]
        if not assessment.lowering_eligible:
            skipped.append(
                {
                    "schedule": _display_path(path),
                    "findings": findings,
                }
            )
            continue
        schedule = Schedule.load(path)
        target = targets.setdefault(
            schedule.target,
            Target.load(ROOT / f"compiler/targets/{schedule.target}.json"),
        )
        lowering = compiler.lower(assessment)
        rows.append(
            {
                "schedule": _display_path(path),
                "schedule_id": schedule.schedule_id,
                "findings": findings,
                "profile": profile_envelope(
                    schedule, target, lowered_source=lowering.source
                ).as_dict(),
            }
        )
    document = {"schema_version": 1, "rows": rows, "skipped": skipped}
    if arguments.json:
        print(json.dumps(document, indent=2, sort_keys=True))
    else:
        print(_table(rows))
        if skipped:
            print("\nskipped:")
            for row in skipped:
                print(f"  {row['schedule']}: {','.join(row['findings'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
