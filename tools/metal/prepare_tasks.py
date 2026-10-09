#!/usr/bin/env python3
"""Materialize main's registered task starters for Metal; CPU inspection only.

This creates development inputs, not admitted Runs or device qualification.
Task semantics and member names stay with the existing catalog/workload owners.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.tasks.catalog import task_names, default_shape, matrix_depth, task_entry
from open_cake_ir.tasks.workloads import create_task
from open_cake_ir.tasks.devices import admit_cohort_payload
from open_cake_ir.evaluation.workload import WorkloadContract


def inspect_task(compiler, name, *, rows=None, columns=None, depth=None):
    r, c = default_shape(name, rows, columns)
    k = matrix_depth(name, depth)
    result = {'task': name, 'family': task_entry(name).family,
              'shape': {'rows': r, 'columns': c, 'depth': k},
              'status': 'not_examined', 'device_qualified': False}
    artifacts = {}
    stage = 'workload'
    try:
        document, source = create_task(name, backend='metal-m4', rows=r, columns=c, depth=k)
        artifacts.update({'workload.json': document, 'starter.py': source})
        stage = 'construction'
        schedule = frontend.parse(source, filename='starter.py').document
        stage = 'assessment'
        assessed = compiler.assess(schedule)
        result['findings'] = [{'code': f.code, 'path': f.path, 'message': f.message,
                               'blocks_lowering': f.blocks_lowering} for f in assessed.findings]
        if not assessed.lowering_eligible:
            result['status'] = 'assessment_refused'
            return result, artifacts
        stage = 'lowering'
        artifacts['starter.metal'] = compiler.lower(assessed).source
        result['status'] = 'msl_generated'
        stage = 'cohort_payload'
        admit_cohort_payload(WorkloadContract(document), 'primary', 18)
        result['mean30_payload_admitted'] = True
    except ValueError as error:
        result.update(status=f'{stage}_refused', reason=str(error))
    return result, artifacts


def prepare(root, output, *, tasks=None, rows=None, columns=None, depth=None):
    output = Path(output)
    if not output.is_absolute() or output.resolve() != output:
        raise ValueError('output must be an absolute canonical external directory')
    if any((p / '.git').exists() for p in (output, *output.parents)):
        raise ValueError('generated task materials must stay outside Git checkouts')
    selected = tuple(task_names() if tasks is None else tasks)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(task_names()):
        raise ValueError('select distinct registered tasks')
    compiler = Compiler.load(root, root / 'compiler/revision.json')
    if compiler.commit is None:
        raise ValueError('prepare from a clean committed checkout')
    output.mkdir(parents=True, exist_ok=False)
    rows_out = []
    for name in selected:
        row, artifacts = inspect_task(compiler, name, rows=rows, columns=columns, depth=depth)
        folder = output / name
        folder.mkdir()
        for filename, value in artifacts.items():
            (folder / filename).write_text(value if isinstance(value, str) else json.dumps(value, indent=2)+'\n')
        (folder / 'assessment.json').write_text(json.dumps(row, indent=2)+'\n')
        (folder / 'TASK.md').write_text(
            f'# Metal development task: {name}\n\n'
            'Development input only; not a Bench result or an admitted Run.\n'
            'Use the canonical tools/launch_task.py with --backend metal-m4.\n'
            'Keep this source commit fixed; create a fresh external Run workspace.\n'
            'Omit --token-budget (accounting only); declare --wall-seconds 10800.\n'
            'Use own MSL to test a concrete Cake hypothesis. Preserve diagnostic codes,\n'
            'negative results and proposed ownership. Do not modify the oracle or Compiler.\n'
            'MSL generation does not prove native compilation or device correctness.\n'
            'Do not read another checkout or treat this folder as read isolation.\n')
        rows_out.append(row)
    report = {'source_commit': compiler.commit, 'target': 'apple_gpu_family9',
              'scope': 'registered starter construction, assessment, MSL generation and mean30 payload bound only',
              'provider_calls': 0, 'gpu_calls': 0, 'tasks': rows_out,
              'counts': dict(Counter(row['status'] for row in rows_out))}
    (output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task', choices=task_names(), action='append')
    p.add_argument('--rows', type=int)
    p.add_argument('--columns', type=int)
    p.add_argument('--depth', type=int)
    a = p.parse_args()
    report = prepare(ROOT, a.output, tasks=a.task, rows=a.rows, columns=a.columns, depth=a.depth)
    print(json.dumps({'output': str(a.output), 'counts': report['counts'], 'gpu_calls': 0}))


if __name__ == '__main__':
    main()
