#!/usr/bin/env python3
"""Check every finding parses, carries the record contract's fields, and is indexed."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINDINGS = ROOT / "findings"
KINDS = {"bug", "capacity", "behavior", "protocol"}


def main() -> int:
    readme = (FINDINGS / "README.md").read_text(encoding="utf-8")
    contract = readme.partition("## Record contract")[2].partition("## Index")[0]
    named = set(re.findall(r"`(\w+)`", contract)) - KINDS - {"NNN", "kind"}
    records = {path: json.loads(path.read_text(encoding="utf-8"))
               for path in sorted(FINDINGS.glob("*.json"))}
    required = named | set.intersection(*(set(record) for record in records.values()))
    print(f"required fields ({len(required)}): {', '.join(sorted(required))}")
    problems = []
    for path, record in records.items():
        missing = required - record.keys()
        if missing:
            problems.append(f"{path.name}: missing {', '.join(sorted(missing))}")
        if record.get("kind") not in KINDS:
            problems.append(f"{path.name}: kind {record.get('kind')!r} is not one of {sorted(KINDS)}")
        finding_id = f"F-{path.stem[:14]}"
        if record.get("id") != finding_id:
            problems.append(f"{path.name}: id {record.get('id')!r} does not match its filename")
        if finding_id not in readme:
            problems.append(f"{path.name}: {finding_id} is not in findings/README.md")
    print("\n".join(problems) if problems else f"{len(records)} findings conform")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
