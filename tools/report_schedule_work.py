#!/usr/bin/env python3
"""Report the work every Corpus Schedule declares, before any of it reaches a GPU.

`docs/ANALYSIS_CALIBRATION.md` compares a residency bound against Nsight. There was no
comparable table for work, so a measured time had no denominator: a lowering that ran was
correct or incorrect and never efficient or inefficient against anything.

This prints the declared half of that table. Every column is derived from the Schedule
alone -- no Target, no clock, no bandwidth, no measurement -- so it runs anywhere and the
numbers do not depend on which machine ran it.

Two markers carry the direction of error, and they are printed rather than footnoted:

  ~flops   at least one operation's arithmetic could not be priced, so the count is a
           lower bound and the operations that abstained are named.
  ~bytes   at least one Buffer is addressed by a runtime coordinate, a valid prefix or a
           subrange, so charging it in full is an upper bound.

Arithmetic intensity is therefore a lower bound whenever either marker appears, which is
the direction that keeps a memory-bound Schedule from reading as compute-bound.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler.ir import Schedule, ScheduleParseError  # noqa: E402
from open_cake_ir.compiler.target import Target  # noqa: E402
from open_cake_ir.compiler.performance.utilization import MemoryScope, roofline_seconds  # noqa: E402
from open_cake_ir.compiler.performance.work import WorkBound, work_bound  # noqa: E402


def _row(
    case_id: str, schedule: Schedule, bound: WorkBound, target: Target | None
) -> dict[str, object]:
    intensity = bound.arithmetic_intensity
    return {
        "roofline_seconds": None if target is None else roofline_seconds(bound, target),
        "memory_scope": MemoryScope.LOGICAL.value,
        "case_id": case_id,
        "schedule_id": schedule.schedule_id,
        "program_tiles": bound.program_tiles,
        "flops": bound.flops,
        "flops_exact": bound.flops_exact,
        "mma_flops": bound.mma_flops,
        "mma_fraction": bound.mma_fraction,
        "compulsory_read_bytes": bound.compulsory_read_bytes,
        "compulsory_written_bytes": bound.compulsory_written_bytes,
        "compulsory_bytes_exact": bound.compulsory_bytes_exact,
        "arithmetic_intensity": intensity,
        "uncounted_arithmetic": list(bound.uncounted_arithmetic),
        "partially_addressed": list(bound.partially_addressed),
    }


def _table(rows: list[dict[str, object]], floors: bool) -> str:
    header = (
        f"{'case':<34}{'tiles':>7}{'FLOPs':>16}{'mma':>7}"
        f"{'bytes':>13}{'FLOP/byte':>12}"
        + (f"{'floor ns':>11}" if floors else "")
        + "  notes"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        intensity = row["arithmetic_intensity"]
        fraction = row["mma_fraction"]
        notes = []
        if row["uncounted_arithmetic"]:
            notes.append("~flops: " + ",".join(row["uncounted_arithmetic"]))
        if row["partially_addressed"]:
            notes.append("~bytes: " + ",".join(row["partially_addressed"]))
        lines.append(
            f"{str(row['case_id'])[:33]:<34}"
            f"{row['program_tiles']:>7}"
            f"{row['flops']:>15}{'~' if not row['flops_exact'] else ' '}"
            f"{('-' if fraction is None else f'{fraction:.0%}'):>7}"
            f"{row['compulsory_read_bytes'] + row['compulsory_written_bytes']:>12}"
            f"{'~' if not row['compulsory_bytes_exact'] else ' '}"
            f"{('-' if intensity is None else f'{intensity:.2f}'):>12}"
            + (_floor(row) if floors else "")
            + f"  {'; '.join(notes)}"
        )
    return "\n".join(lines)


def _floor(row: dict[str, object]) -> str:
    """Nanoseconds the declared work must take at the Target's declared rates.

    A lower bound on any measurement of this Schedule, not a predicted time: what
    separates the two is everything the declarations cannot see.
    """

    seconds = row["roofline_seconds"]
    return f"{('-' if seconds is None else f'{seconds * 1e9:.1f}'):>11}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="corpus/manifest.json")
    parser.add_argument(
        "--target",
        help="Target whose declared peak rates turn the counts into a roofline floor",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="include cases the gates refuse to lower, whose work is declared anyway",
    )
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()

    target = None
    if arguments.target:
        target = Target.load(ROOT / arguments.target)
        if target.peak is None:
            # Said once, up front. A column of dashes would read as "these Schedules
            # have no floor" rather than "this device has no measured rates".
            print(
                f"{target.target_id} declares no peak rates, so no floor is derivable. "
                "Measure them with tools/observe_target_peak.py.\n"
            )

    manifest = json.loads((ROOT / arguments.manifest).read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    skipped: list[str] = []
    for case in manifest["cases"]:
        if not arguments.all and not case["expected"].get("lowering_eligible"):
            continue
        try:
            schedule = Schedule.load(ROOT / case["schedule"])
        except ScheduleParseError as error:
            # A Corpus negative may be refused by the parser itself, which is a fact
            # about the case rather than a failure here. Naming it beats a short table.
            skipped.append(f"{case['case_id']}: {error}")
            continue
        bound = work_bound(schedule)
        if bound is None:
            skipped.append(f"{case['case_id']}: no derivable work domain")
            continue
        rows.append(_row(case["case_id"], schedule, bound, target))

    if arguments.json:
        print(json.dumps({"rows": rows, "skipped": skipped}, indent=1, sort_keys=True))
        return 0

    print(_table(rows, floors=target is not None and target.peak is not None))
    if skipped:
        print("\nnot reported:")
        for entry in skipped:
            print(f"  {entry}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
