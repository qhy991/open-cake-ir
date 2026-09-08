"""Executable protocol double; it establishes no kernel qualification or custody."""
from pathlib import Path


def write_parent_validator_fixture(path: Path) -> Path:
    path.write_text('''"""CPU test double for the external parent-validator protocol only."""
import json
from pathlib import Path
import sys

source = Path(sys.argv[1])
completion = json.loads(source.read_text())
case_id = completion["case_id"]
outcome = completion["outcome"]
if outcome not in {"qualified", "runnable_unqualified", "qualification_unknown", "missing"}:
    raise SystemExit(2)
if "--finalize" in sys.argv:
    marker = {
        "schema": "aka.kernel-parent-completion-done.v1",
        "case_id": case_id, "outcome": outcome,
        "recovery_mode": completion["recovery_mode"],
        "verified_by": "complete-kernel-parent-validator",
        "completion": source.name,
        "counts_as_executable_parent": outcome == "qualified",
        "counts_as_runnable_bundle": outcome in {"qualified", "runnable_unqualified", "qualification_unknown"},
        "training_eligibility": False,
        "verified_at": "synthetic CPU protocol fixture; no GPU verification",
    }
    source.with_name("PARENT_DONE.json").write_text(json.dumps(marker))
print(json.dumps({"valid": True, "case_id": case_id, "outcome": outcome, "fixture_only": True}))
''', encoding="utf-8")
    return path
