#!/usr/bin/env python3
"""Report every pack task and exercise existing authoring routes without GPU allocation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.frontend import parse
from open_cake_ir.tasks.solx_fib.catalog import TASK_IDS, task_owner
from open_cake_ir.tasks.workloads import create_task
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.source_identity import checkout_commit


def check(backend: str) -> dict:
    commit = checkout_commit(ROOT)
    compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
    results = []
    for task_id in TASK_IDS:
        task, owner = task_owner(task_id)
        row = {"task_id": task_id, "task": task, "device_test": "not_run"}
        if owner is None:
            row.update(status="not_integrated", reason="no_task_contract_oracle_or_starter")
        elif hasattr(owner, 'author_plan'):
            try:
                if backend != 'triton-b300':
                    raise ValueError('these launch-plan contracts declare exact target sm_103a on triton-b300')
                variants = []
                for variant in owner.VARIANTS:
                    workload = WorkloadContract(owner.workload_document(task, variant=variant))
                    plan = owner.launch_plan(workload)
                    compiled = compiler.lower_program(plan)
                    variants.append({'variant': variant, 'workload_id': workload.workload_id,
                                     'stages': len(compiled.lowerings)})
                row.update(status='offline_lowering_passed', route='cake_launch_plan', variants=variants,
                           target=plan.target)
            except ValueError as error:
                row.update(status='refused', reason=str(error))
        else:
            spec = owner.SPECS[task]
            rows = owner.default_rows(task) if hasattr(owner, "default_rows") else min(spec["batches"])
            columns = spec["hidden"] if "hidden" in spec else spec["N"]
            try:
                document, source = create_task(task, backend=backend, rows=rows, columns=columns)
                assessment = compiler.assess(parse(source, filename="starter.py").document)
                if not assessment.lowering_eligible:
                    row.update(status="refused", findings=[f.code for f in assessment.findings])
                else:
                    lowering = compiler.lower(assessment)
                    row.update(status="offline_lowering_passed", target=lowering.target,
                               workload_id=document["workload_id"], shape=document["cases"][0]["shape"])
            except ValueError as error:
                row.update(status="refused", reason=str(error))
        results.append(row)
    return {"source_commit": commit, "backend": backend, "task_count": len(results),
            "offline_lowering_passed": sum(r["status"] == "offline_lowering_passed" for r in results),
            "device_test": "not_run", "tasks": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="triton-b300")
    args = parser.parse_args()
    report = check(args.backend)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["offline_lowering_passed"] == report["task_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
