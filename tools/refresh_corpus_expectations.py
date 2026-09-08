#!/usr/bin/env python3
"""Show, and on request adopt, the Corpus Gate expectations of the draft Compiler.

The Corpus Gate pins the exact finding codes, Schedule digests, and lowering source
digests each case must produce. That pin is only worth holding if adopting a new one is a
deliberate act: a release path that recomputed expectations before gating would report a
match it had just manufactured, and unintended semantic drift would pass silently.

So this runs separately from the release. It prints what would change and exits non-zero
while a difference stands; `--write` adopts the difference, and the approval basis
recorded at release is where the reason for adopting it belongs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from open_cake_ir.compiler import Compiler


def _expected(compiler: Compiler, root: Path, schedule: str) -> dict[str, object]:
    assessment = compiler.assess_file(root / schedule)
    return {
        "accepted": assessment.accepted,
        "lowering_eligible": assessment.lowering_eligible,
        "finding_codes": [finding.code for finding in assessment.findings],
        "schedule_sha256": assessment.schedule_sha256,
        "lowering_source_sha256": (
            compiler.lower(assessment).source_sha256
            if assessment.lowering_eligible
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--write", action="store_true", help="adopt the differences")
    arguments = parser.parse_args()
    root = arguments.project_root.resolve(strict=True)
    compiler = Compiler.load(root, root / "compiler/revision.json")
    manifest_path = root / "corpus/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    differences = 0
    for case in manifest["cases"]:
        observed = _expected(compiler, root, case["schedule"])
        recorded = case.get("expected", {})
        changed = {
            field: (recorded.get(field), value)
            for field, value in observed.items()
            if recorded.get(field) != value
        }
        if not changed:
            continue
        differences += 1
        print(f"{case['case_id']}")
        for field, (before, after) in sorted(changed.items()):
            print(f"    {field}\n      pinned   {before}\n      observed {after}")
        case["expected"] = observed

    if not differences:
        print("corpus expectations already match the draft Compiler")
        return 0
    if not arguments.write:
        print(f"\n{differences} case(s) differ; re-run with --write to adopt")
        return 1
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nadopted {differences} case(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
