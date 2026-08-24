#!/usr/bin/env python3
"""Print everything a Schedule can say, and answer whether it can say a given thing.

Checking a surveyed kernel axis against the IR by memory is how a field gets called
missing when it exists, and called present when the match was a different concept wearing
the same word -- `register` the memory space read as a register budget, `policy` the NaN
policy read as a cache policy. Both happened.

The vocabulary is derived from the typed IR, so this cannot drift from it. So is the
last section, which is the other half of the same question: an operation kind the IR can
express but no backend can lower is a word the Schedule may say and the Compiler cannot
answer, and that gap used to be discovered by hitting it.

    ir_vocabulary.py                       every enum, structure, and backend body
    ir_vocabulary.py --probe raster grid   whether those words appear, and where
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from enum import Enum
from pathlib import Path

from open_cake_ir.compiler import emit_cutedsl, emit_triton, ir

ROOT = Path(__file__).resolve().parents[1]

_BACKENDS = {"triton": emit_triton, "cute-dsl": emit_cutedsl}


def _used_members() -> set[Enum]:
    """Every enum member some corpus Schedule actually selects.

    Walked over the parsed Schedules rather than their JSON. A string search would count
    `sum` in a buffer name as a use of `ReduceOp.SUM` and report an unused member as used,
    which is the direction that hides the thing this is looking for.
    """

    used: set[Enum] = set()

    def visit(value, seen: set[int]) -> None:
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, Enum):
            used.add(value)
        elif dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                visit(getattr(value, field.name), seen)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item, seen)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item, seen)

    manifest = json.loads((ROOT / "corpus/manifest.json").read_text(encoding="utf-8"))
    for case in manifest["cases"]:
        document = json.loads((ROOT / case["schedule"]).read_text(encoding="utf-8"))
        try:
            schedule = ir.Schedule.from_dict(document)
        except ir.ScheduleParseError:
            continue  # a case whose whole point is that it does not parse
        visit(schedule, set())
    return used


def _enums() -> dict[str, list[str]]:
    return {
        name: [member.value for member in obj]
        for name, obj in sorted(vars(ir).items())
        if isinstance(obj, type) and issubclass(obj, Enum) and obj is not Enum
    }


def _structures() -> dict[str, list[str]]:
    return {
        name: [field.name for field in dataclasses.fields(obj)]
        for name, obj in sorted(vars(ir).items())
        if dataclasses.is_dataclass(obj)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe",
        nargs="+",
        metavar="WORD",
        help="report where each word appears in the vocabulary, or that it does not",
    )
    arguments = parser.parse_args()
    enums, structures = _enums(), _structures()

    if not arguments.probe:
        print(f"# closed vocabularies ({len(enums)})")
        for name, members in enums.items():
            print(f"{name:22s} {len(members):2d}  {', '.join(members)}")
        print(f"\n# structures ({len(structures)})")
        for name, fields in structures.items():
            print(f"{name:26s} {len(fields):2d}  {', '.join(fields)}")

        used = _used_members()
        print(f"\n# enum members no corpus Schedule selects")
        print("# (a member the IR derives rather than a Schedule declares is not")
        print("#  selected and appears here: PipelineKind is derived from a producer's")
        print("#  own kind and movement, and both of its members reach the backend.)")
        idle = []
        for name, obj in sorted(vars(ir).items()):
            if not (isinstance(obj, type) and issubclass(obj, Enum) and obj is not Enum):
                continue
            absent = [m.value for m in obj if m not in used]
            if absent:
                idle.append(f"{name:22s} {', '.join(absent)}")
        # Not a defect list, and not a dead-code list either. A member here may be
        # supported and merely unused, or derived rather than declared. What it is good
        # for is the question: is this one a promise on the strength of an argument
        # rather than a kernel? That question found `int64`.
        print("\n".join(idle) if idle else "(every member is selected by some Schedule)")

        print(f"\n# what a backend can lower")
        names = sorted(_BACKENDS)
        print(f"{'':18s} " + "  ".join(f"{name:9s}" for name in names))
        unreachable = []
        for members, attribute in (
            (ir.OperationKind, "SUPPORTED_OPERATION_KINDS"),
            (ir.DType, "SUPPORTED_DTYPES"),
        ):
            for member in members:
                marks = [
                    member in getattr(_BACKENDS[name], attribute) for name in names
                ]
                cells = "  ".join(f"{'yes' if mark else '--':9s}" for mark in marks)
                print(f"{member.value:18s} {cells}")
                if not any(marks):
                    unreachable.append(member.value)
        if unreachable:
            # A kind the IR can express and nothing can lower. Not a defect on its own --
            # the vocabulary may lead the backends deliberately -- but it should be a
            # thing someone chose rather than a thing an author discovers.
            print(f"\nno backend lowers: {', '.join(unreachable)}")
        return 0

    # A word can match a field name, an enum member, or an enum type. Reporting which
    # keeps a match from being read as the concept the word meant in the survey.
    absent = 0
    for word in arguments.probe:
        needle = word.lower()
        hits = [
            f"{owner}.{field}"
            for owner, fields in structures.items()
            for field in fields
            if needle in field.lower()
        ]
        hits += [
            f"{owner}={member}"
            for owner, members in enums.items()
            for member in members
            if needle in member.lower()
        ]
        hits += [f"enum {owner}" for owner in enums if needle in owner.lower()]
        if hits:
            print(f"{word:22s} {', '.join(sorted(set(hits)))}")
        else:
            absent += 1
            print(f"{word:22s} -- absent")
    return 1 if absent else 0


if __name__ == "__main__":
    raise SystemExit(main())
