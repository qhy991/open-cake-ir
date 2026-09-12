#!/usr/bin/env python3
"""Promote one audited material winner into an external task-incumbent registry."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.lab.incumbents import promote_task_incumbent  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--registry-root", type=Path, required=True)
    parser.add_argument("--campaign-lock", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--run-id")
    arguments = parser.parse_args(argv)
    project = arguments.project_root.resolve(strict=True)
    record = promote_task_incumbent(
        project_root=project,
        registry_root=arguments.registry_root,
        campaign_lock_path=arguments.campaign_lock,
        evidence_root=arguments.evidence_root,
        run_id=arguments.run_id,
    )
    print(
        json.dumps(
            {
                "registry_root": str(arguments.registry_root.absolute()),
                "run_id": record["run_id"],
                "generation": record["generation"],
                "candidate_sha256": record["candidate"]["candidate_sha256"],
                "speedup": record["comparison"]["speedup"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
