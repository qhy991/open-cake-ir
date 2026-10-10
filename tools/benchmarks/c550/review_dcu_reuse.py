#!/usr/bin/env python3
"""Read-only exact-target source review, not a GPU experiment or performance test."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--compare', type=Path, nargs=2, metavar=('BEFORE', 'AFTER'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.compare:
        before, after = [json.loads(path.read_text()) for path in args.compare]
        old = {row['task']: row for row in before['tasks']}
        if set(old) != {row['task'] for row in after['tasks']}:
            parser.error('Task sets differ; this is not the same-panel comparison')
        rows = []
        for row in after['tasks']:
            prior = old[row['task']]
            for key in ('status', 'accepted', 'lowering_eligible', 'blockers'):
                if row.get(key) != prior.get(key):
                    parser.error(f"{row['task']}: {key} changed")
            if not row.get('lowering_eligible'):
                continue
            for key in ('workload', 'schedule', 'kernel_body', 'signature', 'grid', 'compile_options'):
                if row[key] != prior[key]:
                    parser.error(f"{row['task']}: {key} changed")
            if any(prior['constants'].get(key) != value for key, value in row['constants'].items()):
                parser.error(f"{row['task']}: live constant added or changed")
            rows.append(dict(task=row['task'], old_constants=len(prior['constants']),
                new_constants=len(row['constants']),
                removed=sorted(set(prior['constants']) - set(row['constants']))))
        result = dict(before=before['commit'], after=after['commit'],
            scope='CPU source comparison; no native compilation or device speed claim',
            lowerable=len(rows), pruned_tasks=sum(bool(r['removed']) for r in rows),
            old_constants=sum(r['old_constants'] for r in rows),
            new_constants=sum(r['new_constants'] for r in rows),
            workload_schedule_body_abi_grid_options_unchanged=True, tasks=rows)
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps({k: v for k, v in result.items() if k != 'tasks'}))
        return
    if args.source_root is None:
        parser.error('Pass --source-root or --compare BEFORE AFTER')
    root = args.source_root.resolve()
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain']).strip():
        parser.error('source-root must be a clean fixed checkout')
    sys.path.insert(0, str(root / 'src'))
    from open_cake_ir.compiler import Compiler, frontend
    from open_cake_ir.tasks.catalog import task_names, default_shape, matrix_depth
    from open_cake_ir.tasks.workloads import create_task
    compiler = Compiler.load(root)
    rows = []
    for name in task_names():
        row = {'task': name}
        rows.append(row)
        m, n = default_shape(name)
        try:
            workload, author = create_task(name, backend='triton-metax', rows=m,
                columns=n, depth=matrix_depth(name, None))
        except ValueError as error:
            row.update(status='factory_refused', error=str(error))
            continue
        document = frontend.parse(author).document
        assessment = compiler.assess(document)
        row.update(status='assessed', accepted=assessment.accepted,
                   lowering_eligible=assessment.lowering_eligible,
                   blockers=[f.code for f in assessment.findings
                             if f.blocks_acceptance or f.blocks_lowering])
        row['workload'] = workload
        row['schedule'] = document
        if assessment.lowering_eligible:
            lowered = compiler.lower(assessment)
            requirements = lowered.toolchain_requirements
            tree = ast.parse(lowered.source)
            kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                          and n.name == requirements['kernel_entry_point'])
            row.update(kernel_body=ast.dump(ast.Module(body=kernel.body, type_ignores=[])),
                       constants=dict(requirements['compile_constants']),
                       signature=dict(requirements['signature']),
                       grid=list(requirements['grid']),
                       compile_options=dict(requirements['compile_options']))
        if name in {'rmsnorm', 'gemm', 'gemm_silu', 'aka_gemm_nt_bias'}:
            width = compiler.specialize_triton_warps(document, num_warps=4,
                schedule_id='review_width', entry_point='review_width')
            row['width_action'] = dict(applied=width.applied, reason=width.reason, message=width.message)
            if hasattr(compiler, 'specialize_fp32_contraction'):
                result = compiler.specialize_fp32_contraction(document, row_tile=16,
                    column_tile=16, k_tile=32, num_warps=4, num_stages=1,
                    schedule_id='review_contraction', entry_point='review_contraction')
                row['contraction_action'] = dict(applied=result.applied,
                    reason=result.reason, message=result.message)
    payload = dict(commit=commit, target='xcore1002', scope='CPU construction, assessment and lowering only', tasks=rows)
    # The report is external to the source checkout and created once.
    with args.output.open('x') as stream:
        json.dump(payload, stream, indent=2)
    print(json.dumps({'commit': commit, 'tasks': len(rows),
        'lowerable': sum(r.get('lowering_eligible', False) for r in rows),
        'factory_refused': [r['task'] for r in rows if r['status']=='factory_refused']}))


if __name__ == '__main__':
    main()
