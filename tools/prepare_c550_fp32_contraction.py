#!/usr/bin/env python3
"""Prepare original-task Programs for C550 NT contraction qualification.

Use qualify_tensor_program.py for isolated build and original-case evaluation.
This command does not allocate a device, run a provider or decide promotion.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from open_cake_ir.compiler import Compiler, Program, frontend
from open_cake_ir.source_identity import checkout_commit
from open_cake_ir.tasks.workloads import create_task

# The original task factory requires power-of-two N/K. M and the selected tiles
# exercise masked rows and N smaller than a tile without changing its contract.
CASES = (('original', 24, 32, 64), ('mma', 24, 32, 64),
         ('mma', 5, 8, 128), ('mma', 1, 16, 32))
PARAMETERS = dict(row_tile=16, column_tile=16, k_tile=16, num_warps=4, num_stages=1)


def candidates(compiler):
    for kind, rows, columns, depth in CASES:
        workload, source = create_task('aka_gemm_nt_bias', backend='triton-metax',
                                      rows=rows, columns=columns, depth=depth)
        original = frontend.parse(source).document
        name = f'{kind}-m{rows}-n{columns}-k{depth}'
        if kind == 'mma':
            saved = deepcopy(original)
            result = compiler.specialize_fp32_contraction(original, **PARAMETERS,
                schedule_id=name, entry_point=original['lowering']['entry_point'])
            if not result.applied:
                raise ValueError(f'{name}: {result.reason}: {result.message}')
            document = result.schedule
            assert original == saved
            assert document.get('metadata') == original.get('metadata')
            assert [b for b in document['buffers'] if b['space'] == 'global'] == [
                b for b in original['buffers'] if b['space'] == 'global']
        else:
            document = original
        program = Program.from_schedule(document)
        compiler.lower_program(program)
        yield name, workload, source, program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    commit = checkout_commit(ROOT)
    if commit is None:
        parser.error('Require a clean source commit')
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error('Qualification outputs must stay outside source')
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        for name, workload, source, program in candidates(Compiler.load(ROOT)):
            folder = output / name
            folder.mkdir()
            (folder / 'workload.json').write_text(json.dumps(workload, indent=2))
            (folder / 'program.json').write_bytes(program.document_bytes)
            (folder / 'original.py').write_text(source)
            rows.append(dict(name=name, original_cases=[c['case_id'] for c in workload['cases']]))
    except Exception as error:
        (output / 'failure.json').write_text(json.dumps({'error':str(error)}))
        raise
    (output / 'plan.json').write_text(json.dumps(dict(source_commit=commit, candidates=rows,
        parameters=PARAMETERS, timing='none', provider_calls=0,
        scope='original-task NT rewrite correctness, not performance'), indent=2))
    print(json.dumps({'prepared':len(rows), 'original_checks':sum(len(r['original_cases']) for r in rows)}))


if __name__ == '__main__':
    main()
