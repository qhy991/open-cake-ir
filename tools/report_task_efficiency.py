#!/usr/bin/env python3
"""Audit an opted-in task workspace using its exact source checkout; never backfill a Study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _external_path(path: Path, *, directory: bool = False) -> Path:
    resolved = path.resolve(strict=directory)
    if (not path.is_absolute() or path != resolved or
            any((parent / ".git").exists() for parent in (resolved, *resolved.parents))):
        raise ValueError("report and workspace paths must be canonical and outside every Git checkout")
    if directory and not resolved.is_dir():
        raise ValueError("workspace must be an existing directory")
    return resolved


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True,
                        help="exact source checkout used by the Campaign; no source fallback")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="optional new external JSON report; existing files are refused")
    args = parser.parse_args(argv)
    root = args.project_root.resolve(strict=True)
    if not (root / "src/open_cake_ir/tasks/runtime.py").is_file():
        raise ValueError("--project-root has no TaskLab source")
    workspace = _external_path(args.workspace, directory=True)
    output = _external_path(args.output) if args.output else None
    if output is not None and output.exists():
        raise FileExistsError("report output already exists; choose a new path or use stdout")
    sys.path.insert(0, str(root / "src"))
    import open_cake_ir
    if Path(open_cake_ir.__file__).resolve() != root / "src/open_cake_ir/__init__.py":
        raise ValueError("loaded source differs from --project-root; run in a fresh Python process")
    from open_cake_ir.cli import _json_projection
    from open_cake_ir.lab.contracts import CampaignLock
    from open_cake_ir.lab.efficiency_policy import performance_reporting_policy
    from open_cake_ir.tasks.runtime import TaskLab

    lock = CampaignLock.load(workspace / "campaign-lock.json")
    if performance_reporting_policy(lock.analysis_plan, lock.claim_scope) is None:
        raise ValueError("Campaign has no performance_reporting policy; historical backfill is forbidden")
    lab = TaskLab(root)
    campaign = lab.reference_campaign(lock, workspace / "campaign-evidence")
    report = lab.audit(campaign)
    document = _json_projection(report)
    payload = json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if output is not None:
        with output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
