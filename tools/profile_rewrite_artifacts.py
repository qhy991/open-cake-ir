#!/usr/bin/env python3
"""Separate NCU attribution of sealed comparison participants; never a timing score."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from tools.compare_rewrite_artifacts import (
    RetainedExternal, correctness, load_reference, reference_arguments, regular,
)
from tools.compare_flashinfer_reference import admit_judge_source, write
from open_cake_ir.evaluation.admission import observe_exclusive_cuda
from open_cake_ir.evaluation.core import LoadedTorchTensorCandidate, compare_tile_outputs
from open_cake_ir.evaluation.gpuq import observe_allocation
from open_cake_ir.evaluation.paired import validate_pair_candidates
from open_cake_ir.evaluation.profiler import NCU_ATTRIBUTION_METRICS
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.lab.bindings import load_baseline_bundle
from open_cake_ir.lab.ncu_process import profile_output_owner, run_ncu, write_new
from open_cake_ir.source_identity import checkout_commit
from open_cake_ir.tasks.workloads import materialize_case, reference_outputs

METRICS = (*NCU_ATTRIBUTION_METRICS, 'launch__block_size', 'launch__grid_size',
           'launch__shared_mem_per_block', 'dram__bytes_read.sum', 'dram__bytes_write.sum',
           'lts__t_bytes.sum', 'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum',
           'l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum')


def parse_metrics(raw):
    lines = raw.splitlines()
    for index, line in enumerate(lines):
        if {'ID', 'Kernel Name', 'Metric Name', 'Metric Unit', 'Metric Value'} <= set(next(csv.reader([line]))):
            break
    else:
        raise ValueError('NCU metric CSV is missing')
    kernels = {}
    for row in csv.DictReader(io.StringIO('\n'.join(lines[index:]))):
        name = row.get('Metric Name')
        if name not in METRICS:
            continue
        key = row['ID']
        kernel = kernels.setdefault(key, {'kernel': row['Kernel Name'], 'metrics': {}})
        value = float(row['Metric Value'].replace(',', ''))
        # NCU prints these two dimensionless launch counts with an empty unit.
        unitless_count = name in {'launch__block_size', 'launch__grid_size'}
        if (kernel['kernel'] != row['Kernel Name'] or name in kernel['metrics']
                or not math.isfinite(value) or value < 0
                or (not row['Metric Unit'] and not unitless_count)):
            raise ValueError('NCU metric identity, value or unit differs')
        kernel['metrics'][name] = {'value': value, 'unit': row['Metric Unit']}
    if not kernels or any(set(k['metrics']) != set(METRICS) for k in kernels.values()):
        raise ValueError('NCU metrics are incomplete')
    return list(kernels.values())


def child(role, output, root, owner):
    admission = observe_exclusive_cuda('sm_103a')
    import torch
    torch.set_num_threads(4)
    sys.dont_write_bytecode = True
    workload = WorkloadContract(json.loads(regular(root, 'workload.json').read_bytes()))
    if workload.target != 'sm_103a':
        raise ValueError('profile target differs')
    names = reference_arguments(workload)
    candidates = {r: load_baseline_bundle(ROOT, regular(root, r + '/candidate.json'))
                  for r in ('optimized', 'starter')}
    manifests = validate_pair_candidates(candidates['optimized'], candidates['starter'], workload, 'primary')
    report = {'role': role, 'roles': {'external': 'unchanged supplied external implementation'}, 'cases': [],
              'broker_job_id': admission.broker_job_id,
              'comparison': json.loads((root / 'comparison.json').read_text())}
    loaded = None
    try:
        if role == 'external':
            refs = load_reference(root, output, report, 1)
            launches = [lambda values, out, fn=fn: fn(*(values[n] for n in names)) for fn in refs]
        values_by_case = {c: materialize_case(workload, c) for c in workload.case_ids}
        expected_by_case = {c: reference_outputs(workload, c, v) for c, v in values_by_case.items()}
        for case, values in values_by_case.items():
            if role == 'external':
                loaded = RetainedExternal(workload, values, launches, torch,
                                          cached_output=report['reference'].get('cached_output', False))
            else:
                manifest = manifests['candidate' if role == 'optimized' else 'baseline']
                manifest.check_validation_case(workload, case)
                loaded = LoadedTorchTensorCandidate(candidates[role], manifest, values, admission)
            check = correctness(loaded, workload, values, expected_by_case[case])
            report['cases'].append({'case': case, **check})
            if not check['passed']:
                raise ArithmeticError('profile participant failed ' + case)
            if case != 'primary' and role != 'external':
                loaded.close()
            if case == 'primary':
                primary = loaded
        loaded = primary
        argument = loaded.fresh_argument_sets(1)[0]
        torch.cuda.synchronize()
        # NCU owns cold-cache kernel replay. Only this complete callable is inside
        # the capture range: compilation, input creation and poison stay outside.
        torch.cuda.profiler.start()
        loaded.launch(argument)
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        observed, after = loaded.snapshot(argument)
        passed, metrics = compare_tile_outputs(workload, values_by_case['primary'],
            expected_by_case['primary'], observed, after)
        report['profiled_output'] = {'passed': passed, 'metrics': metrics}
        if not passed:
            raise ArithmeticError('profiled output failed')
        if role != 'external':
            report['compiled_resources'] = loaded.loaded.resources
        report['passed'] = True
    except Exception as error:
        report.update(passed=False, error=f'{type(error).__name__}: {error}',
                      failure_kind='correctness' if isinstance(error, ArithmeticError) else 'infrastructure')
        raise
    finally:
        if role != 'external' and loaded is not None:
            loaded.close()
        write_new(output / 'child.json', (json.dumps(report, indent=2, allow_nan=False) + '\n').encode(), owner)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', choices=['optimized', 'starter', 'external'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--task-document', type=Path)
    parser.add_argument('--stage-id')
    parser.add_argument('--owner', type=int, nargs=2)
    args = parser.parse_args()
    commit = checkout_commit(ROOT)
    task_path = args.task_document if args.child else Path(os.environ['KERNELINFRA_TASK'])
    stage_id = args.stage_id if args.child else os.environ['KERNELINFRA_STAGE_ID']
    admit_judge_source(commit, json.loads(task_path.read_bytes()), stage_id)
    if args.child:
        owner = profile_output_owner({'uid': args.owner[0], 'gid': args.owner[1]})
        child(args.child, args.output, args.input, owner)
        return 0
    output = Path(os.environ['KERNELINFRA_STAGE_DIR'])
    report = {'scope': 'separate NCU attribution; not performance timing or promotion',
              'source_commit': commit, 'promotion_disposition': 'No promotion', 'roles': {}}
    validity = 'unknown'
    try:
        report['allocation'] = observe_allocation('sm_103a')
        # Device admission belongs to each child before it initializes CUDA; the
        # parent must not create a context that would make that check see a peer.
        host = json.loads((ROOT / 'runtime/hosts/sm_103a.json').read_text())
        ncu = host['host_environment']['nsight_compute']['path']
        probe = subprocess.run([ncu, '--version'], capture_output=True, text=True, timeout=20, check=True)
        report['ncu_version'] = probe.stdout
        for role in ('optimized', 'starter', 'external'):
            directory = output / role
            directory.mkdir()
            command = [ncu, '--csv', '--print-units', 'base', '--profile-from-start', 'off',
                       '--target-processes', 'application-only', '--replay-mode', 'kernel',
                       '--cache-control', 'all', '--metrics', ','.join(METRICS),
                       sys.executable, str(Path(__file__).resolve()), '--child', role,
                       '--output', str(directory), '--input', os.environ['KERNELINFRA_CANDIDATE_DIR'],
                       '--task-document', str(task_path), '--stage-id', stage_id,
                       '--owner', str(os.geteuid()), str(os.getegid())]
            # Do not cap launch count: a reference's multi-kernel callable must remain visible.
            try:
                result = run_ncu(command, cwd=ROOT, environment=dict(os.environ), timeout_seconds=600)
            except Exception as error:
                write_new(directory / 'stdout.log', getattr(error, 'stdout', b''))
                write_new(directory / 'stderr.log', getattr(error, 'stderr', b''))
                raise
            write_new(directory / 'stdout.log', result.stdout)
            write_new(directory / 'stderr.log', result.stderr)
            row = {'command': command, 'returncode': result.returncode}
            report['roles'][role] = row
            if (directory / 'child.json').is_file():
                row['validation'] = json.loads((directory / 'child.json').read_text())
                if row['validation'].get('failure_kind') == 'correctness':
                    raise ArithmeticError('profile child correctness failed for ' + role)
            if result.returncode:
                raise ValueError('NCU child failed for ' + role)
            if not row.get('validation', {}).get('passed'):
                raise ValueError('profile child did not validate output')
            row['kernels'] = parse_metrics((directory / 'stdout.log').read_text())
        validity = 'valid'
    except Exception as error:
        if isinstance(error, ArithmeticError):
            validity = 'invalid'
        report.update(error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
    write(output / 'profile-report.json', report)
    write(Path(os.environ['KERNELINFRA_RESULT']), {'schema': 'kernelinfra.stage-result.v1',
          'status': 'passed' if validity == 'valid' else 'failed', 'validity': validity,
          'summary': 'attribution complete' if validity == 'valid' else report['error'],
          'artifacts': {'profile': 'profile-report.json'}})
    return 1 if validity == 'unknown' else 0


if __name__ == '__main__':
    raise SystemExit(main())
