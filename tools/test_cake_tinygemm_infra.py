#!/usr/bin/env python3
"""Bounded TinyGEMM development qualification behind the existing GPU Infra broker.

Checks two complete generated candidates on the three public fixtures. This is not
a provider Campaign, a timing result, or a claim that Triton preserved the CAKE schedule.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.evaluation.admission import observe_exclusive_cuda
from open_cake_ir.evaluation.core import compare_tile_outputs
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.source_identity import checkout_commit
from open_cake_ir.tasks.tinygemm import reproduction as task
from tools.compare_flashinfer_reference import admit_judge_source


def specification(path):
    value = json.loads(path.read_text())
    if (not isinstance(value, dict) or set(value) != {'task', 'target'}
            or value != {'task': task.TASK, 'target': 'sm_103a'}):
        raise ValueError('qualification input must name exact TinyGEMM task on sm_103a')
    return value


def qualify(output, compiler, torch):
    rows = []
    catalog = json.loads((ROOT / 'experiments/flashinfer_rewrites/catalog.json').read_text())
    spec = next(r['assessment'] for r in catalog['tasks'] if r['id'] == '029_cake_tinygemm2')
    for shape in spec['benchmark_rows']:
        document = task.workload_document(rows=shape['batch'], columns=shape['output_features'],
                                          depth=shape['input_features'], backend='triton-b300')
        workload = WorkloadContract(document)
        folder = output / shape['id']
        folder.mkdir(parents=True)
        (folder / 'workload.json').write_text(json.dumps(document, indent=2) + '\n')
        entries = {}
        for stages in (4, 8):
            source = task.starter_source(workload, stages=stages)
            assessment = compiler.assess(frontend.parse(source).document)
            (folder / f'assessment-s{stages}.json').write_text(json.dumps(
                [f.to_dict() for f in assessment.findings], indent=2) + '\n')
            lowering = compiler.lower(assessment)
            (folder / f'candidate-s{stages}.py').write_text(source)
            path = folder / f'lowered-s{stages}.py'
            path.write_text(lowering.source)
            name = f'cake_tiny_{shape["id"]}_s{stages}'
            module_spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[name] = module
            module_spec.loader.exec_module(module)
            entries[stages] = getattr(module, lowering.route.entry_point)
        for case_id in workload.case_ids:
            cpu = task.materialize_case(workload, case_id)
            before = {name: list(values) for name, values in cpu.items()}
            expected = task.reference_outputs(workload, case_id, cpu)
            arguments = workload.tensor_abi(case_id)
            inputs = {a.name: torch.tensor(cpu[a.name], dtype=torch.bfloat16, device='cuda').reshape(a.shape)
                      for a in arguments if a.mode == 'input'}
            out_arg = next(a for a in arguments if a.mode == 'output')
            for stages, entry in entries.items():
                result = torch.full(out_arg.shape, float('nan'), dtype=torch.bfloat16, device='cuda')
                returned = entry(**inputs, out=result)
                torch.cuda.synchronize()
                if returned.data_ptr() != result.data_ptr():
                    raise ValueError('generated candidate replaced the caller output')
                observed = {'out': result.cpu().reshape(-1).tolist()}
                after = {name: value.cpu().reshape(-1).tolist() for name, value in inputs.items()}
                passed, metrics = compare_tile_outputs(workload, before, expected, observed, after)
                row = {'shape': shape['id'], 'case_id': case_id, 'stages': stages,
                       'passed': passed, 'metrics': metrics, 'elements': result.numel()}
                rows.append(row)
                # Whole outputs are retained for independent replay, including failures.
                (folder / f'{case_id}-s{stages}.json').write_text(json.dumps(
                    {'expected': expected, 'observed': observed, 'metrics': metrics}, allow_nan=False) + '\n')
                (output / 'progress.json').write_text(json.dumps(rows, indent=2) + '\n')
                print(shape['id'], case_id, stages, metrics, flush=True)
    return rows


def main():
    stage = Path(os.environ['KERNELINFRA_STAGE_DIR'])
    destination = Path(os.environ['KERNELINFRA_RESULT'])
    result = {'schema': 'kernelinfra.stage-result.v1', 'status': 'failed', 'validity': 'unknown',
              'summary': 'not evaluated', 'artifacts': {}}
    try:
        commit = checkout_commit(ROOT)
        admit_judge_source(commit, json.loads(Path(os.environ['KERNELINFRA_TASK']).read_text()),
                           os.environ['KERNELINFRA_STAGE_ID'])
        specification(Path(os.environ['KERNELINFRA_CANDIDATE_DIR']) / 'qualification.json')
        compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')
        if not compiler.check_corpus().passed:
            raise ValueError('Corpus Gate failed before GPU execution')
        admission = observe_exclusive_cuda('sm_103a')
        import torch
        import triton
        import flashinfer
        if torch.cuda.device_count() != 1:
            raise ValueError('one broker-owned GPU is required')
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        output = stage / 'checks'
        output.mkdir(exist_ok=False)
        rows = qualify(output, compiler, torch)
        passed = len(rows) == 30 and all(row['passed'] for row in rows)
        report = {'source_commit': commit, 'target': 'sm_103a', 'device': torch.cuda.get_device_name(),
                  'allocation': admission.mode, 'torch': torch.__version__, 'triton': triton.__version__,
                  'flashinfer_jit_sdk': flashinfer.__version__, 'peer_source_commit': task.PEER_COMMIT,
                  'scope': 'development_correctness_only; no_Campaign_or_promotion',
                  'passed': passed, 'timing': 'not_measured', 'mechanism_equivalence': 'not_established',
                  'rows': rows}
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        result.update(status='passed' if passed else 'failed', validity='valid' if passed else 'invalid',
                      summary=f'{sum(r["passed"] for r in rows)}/{len(rows)} strict peer checks passed; no performance claim',
                      artifacts={'report': 'checks/report.json'})
    except Exception as error:
        result['summary'] = f'{type(error).__name__}: {error}'
        (stage / 'error.txt').write_text(traceback.format_exc())
        result['artifacts']['error'] = 'error.txt'
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
