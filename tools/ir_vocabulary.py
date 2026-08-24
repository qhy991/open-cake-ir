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
from enum import Enum

from open_cake_ir.compiler import emit_cutedsl, emit_triton, ir

_BACKENDS = {"triton": emit_triton, "cute-dsl": emit_cutedsl}


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

        print(f"\n# operation kinds a backend can lower")
        names = sorted(_BACKENDS)
        print(f"{'kind':18s} " + "  ".join(f"{name:9s}" for name in names))
        unreachable = []
        for kind in ir.OperationKind:
            marks = [kind in _BACKENDS[name].SUPPORTED_OPERATION_KINDS for name in names]
            cells = "  ".join(f"{'yes' if mark else '--':9s}" for mark in marks)
            print(f"{kind.value:18s} {cells}")
            if not any(marks):
                unreachable.append(kind.value)
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
