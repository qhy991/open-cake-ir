#!/usr/bin/env python3
"""CPU preparation, exclusive NCU capture, then CPU validation; never a timing score."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import subprocess
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from tools.compare_rewrite_artifacts import (
    RetainedExternal, load_reference, reference_arguments, regular, comparison_roles,
)
from tools.compare_flashinfer_reference import write
from open_cake_ir.evaluation.admission import observe_exclusive_cuda
from open_cake_ir.evaluation.core import LoadedTorchTensorCandidate, compare_tile_outputs
from open_cake_ir.evaluation.gpuq import observe_allocation
from open_cake_ir.evaluation.paired import candidate_identity
from open_cake_ir.evaluation.profiler import NCU_ATTRIBUTION_METRICS
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.lab.ncu_process import profile_output_owner, run_ncu, write_new
from open_cake_ir.source_identity import checkout_commit
from tools.check_alignment_candidate import admit_cases, case_data, phase_contract
from tools.staged_rewrite_comparison import (
    prepare, prepared_library, load_participants, SnapshotWriter, SnapshotReader,
    INPUT_REFERENCES, WIDTHS,
)

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



ROLES = ('optimized', 'starter', 'external')


def snapshot_plan(workload):
    return [{'case':case, 'phase':'validation'} for case in workload.case_ids] + [
        {'case':'primary', 'phase':'profiled'}]


def snapshot_bytes(workload, case):
    # Each file is one complete observation, so every input is a first literal.
    return sum(math.prod(arg.shape)*WIDTHS[arg.dtype] + (arg.mode == 'input')
               for arg in workload.tensor_abi(case))


def preparation_metadata(prepared, workload):
    admit_cases(prepared, workload)
    return json.loads(regular(prepared, 'preparation.json').read_text())


def retain_snapshot(output, index, arguments, case, workload, owner):
    stream = io.BytesIO()
    writer = SnapshotWriter(stream, workload, snapshot_bytes(workload, case))
    writer.append(arguments, case)
    write_new(output / f'{index}.bin', stream.getvalue(), owner)


def child(role, output, root, prepared, owner):
    admission = observe_exclusive_cuda('sm_103a')
    import torch
    torch.set_num_threads(4)
    sys.dont_write_bytecode = True
    workload = WorkloadContract(json.loads(regular(root, 'workload.json').read_bytes()))
    if workload.target != 'sm_103a':
        raise ValueError('profile target differs')
    metadata = preparation_metadata(prepared, workload)
    if metadata['torch'] != torch.__version__:
        raise ValueError('prepared native runtime differs')
    names = reference_arguments(workload)
    candidates, manifests = load_participants(root, workload)
    report = {'role':role, 'broker_job_id':admission.broker_job_id,
              'roles':comparison_roles(json.loads(regular(root, 'comparison.json').read_text())),
              'workload_sha256':workload.canonical_sha256, 'byteorder':sys.byteorder,
              'snapshot_encoding':INPUT_REFERENCES, 'observations':[], 'capture_complete':False}
    primary = loaded = None
    try:
        if role == 'external':
            refs = load_reference(root, output, report, 1,
                                  prebuilt_library=prepared_library(prepared, metadata))
            if report['reference'] != metadata['reference']:
                raise ValueError('prepared reference specification differs')
            launches = [lambda values, out, fn=fn: fn(*(values[n] for n in names)) for fn in refs]
        for index, case in enumerate(workload.case_ids):
            values = case_data(prepared, workload, index, 'input')
            loaded = (RetainedExternal(workload, values, launches, torch,
                        cached_output=metadata['reference'].get('cached_output', False))
                      if role == 'external' else
                      LoadedTorchTensorCandidate(candidates[role], manifests[role], values, admission))
            argument = loaded.fresh_argument_sets(1)[0]
            try:
                loaded.launch(argument)
                torch.cuda.synchronize()
                retain_snapshot(output, index, argument, case, workload, owner)
            finally:
                release = getattr(loaded, 'release_argument_sets', None)
                if release is not None:release([argument])
            report['observations'].append({'case':case, 'phase':'validation'})
            if case == 'primary':
                primary = loaded
            elif role != 'external':
                loaded.close()
            loaded = None
        loaded = primary
        argument = loaded.fresh_argument_sets(1)[0]
        try:
            torch.cuda.synchronize()
            # Keep the entire callable inside one range; NCU owns cold-cache
            # kernel replay. Host materialization, compilation and validation
            # are separate phases; each captured public tensor is retained.
            torch.cuda.profiler.start()
            try:
                loaded.launch(argument)
                torch.cuda.synchronize()
            finally:
                torch.cuda.profiler.stop()
            retain_snapshot(output, len(workload.case_ids), argument, 'primary', workload, owner)
        finally:
            release = getattr(loaded, 'release_argument_sets', None)
            if release is not None:release([argument])
        report['observations'].append({'case':'primary', 'phase':'profiled'})
        if role != 'external' and not candidates[role].is_program:
            report['compiled_resources'] = loaded.loaded.resources
        report['capture_complete'] = True
    finally:
        if role != 'external':
            for item in (loaded, primary if primary is not loaded else None):
                if item is not None:item.close()
        write_new(output / 'child.json', (json.dumps(report, indent=2, allow_nan=False) + '\n').encode(), owner)


def capture(root, output, prepared, workload, task_path, stage_id, commit):
    metadata = preparation_metadata(prepared, workload)
    candidates, _ = load_participants(root, workload)
    required_bytes = len(ROLES)*sum(snapshot_bytes(workload, r['case']) for r in snapshot_plan(workload))
    if shutil.disk_usage(output).free < required_bytes:
        raise OSError('insufficient storage for all profile observations')
    report = {'source_commit':commit, 'capture_complete':False, 'roles':{},
              'workload_sha256':workload.canonical_sha256, 'reference':metadata['reference'],
              'participants':{r:candidate_identity(c) for r,c in candidates.items()},
              'snapshot_bytes_budget':required_bytes, 'allocation':observe_allocation('sm_103a')}
    # The parent has no CUDA context. Each child independently admits the live
    # allocation before initializing the framework, as required by run_ncu.
    host = json.loads((ROOT / 'runtime/hosts/sm_103a.json').read_text())
    ncu = host['host_environment']['nsight_compute']['path']
    probe = subprocess.run([ncu, '--version'], capture_output=True, text=True, timeout=20, check=True)
    report['ncu_version'] = probe.stdout
    for role in ROLES:
        directory = output / role
        directory.mkdir()
        command = [ncu, '--csv', '--print-units', 'base', '--profile-from-start', 'off',
                   '--target-processes', 'application-only', '--replay-mode', 'kernel',
                   '--cache-control', 'all', '--metrics', ','.join(METRICS),
                   sys.executable, str(Path(__file__).resolve()), '--child', role,
                   '--output', str(directory), '--input', str(root), '--prepared', str(prepared),
                   '--task-document', str(task_path), '--stage-id', stage_id,
                   '--owner', str(os.geteuid()), str(os.getegid())]
        # No launch cap: all kernels in an external multi-launch callable count.
        try:
            result = run_ncu(command, cwd=ROOT, environment=dict(os.environ), timeout_seconds=600)
        except Exception as error:
            write_new(directory / 'stdout.log', getattr(error, 'stdout', b''))
            write_new(directory / 'stderr.log', getattr(error, 'stderr', b''))
            raise
        write_new(directory / 'stdout.log', result.stdout)
        write_new(directory / 'stderr.log', result.stderr)
        report['roles'][role] = {'command':command, 'returncode':result.returncode}
        if result.returncode:
            raise ValueError('NCU child failed for ' + role)
    report['capture_complete'] = True
    write(output / 'capture.json', report)


def verify(root, output, prepared, captured, workload, commit):
    metadata = preparation_metadata(prepared, workload)
    raw = json.loads(regular(captured, 'capture.json').read_text())
    candidates, _ = load_participants(root, workload)
    if (raw.get('capture_complete') is not True or raw.get('source_commit') != commit
        or raw.get('workload_sha256') != workload.canonical_sha256
        or raw.get('reference') != metadata['reference']
        or raw.get('participants') != {r:candidate_identity(c) for r,c in candidates.items()}
        or set(raw.get('roles', {})) != set(ROLES)):
        raise ValueError('profile capture authority differs')
    plan = snapshot_plan(workload)
    report = {'scope':'separate NCU attribution; not performance timing or promotion',
              'source_commit':commit, 'promotion_disposition':'No promotion', 'roles':{},
              'ncu_version':raw['ncu_version'], 'correctness_passed':True,
              'verification_phase':'CPU after profile workers exit'}
    for role in ROLES:
        directory = captured / role
        child_report = json.loads(regular(directory, 'child.json').read_text())
        if (raw['roles'][role].get('returncode') != 0
            or child_report.get('capture_complete') is not True
            or child_report.get('role') != role
            or child_report.get('workload_sha256') != workload.canonical_sha256
            or child_report.get('byteorder') != sys.byteorder
            or child_report.get('snapshot_encoding') != INPUT_REFERENCES
            or child_report.get('observations') != plan):
            raise ValueError('profile child does not cover all required observations')
        checks = []
        for index, row in enumerate(plan):
            case = row['case'];case_index = workload.case_ids.index(case)
            with regular(directory, f'{index}.bin').open('rb') as stream:
                observed, after = SnapshotReader(stream, workload, INPUT_REFERENCES).read(case)
                if stream.read(1):raise ValueError('profile snapshot contains trailing data')
            before = case_data(prepared, workload, case_index, 'input')
            expected = case_data(prepared, workload, case_index, 'output')
            passed, metrics = compare_tile_outputs(workload, before, expected, observed, after)
            checks.append({**row, 'passed':passed, 'metrics':metrics})
            report['correctness_passed'] &= passed
        report['roles'][role] = {'checks':checks, 'participant_roles':child_report['roles']}
        if 'compiled_resources' in child_report:
            report['roles'][role]['compiled_resources'] = child_report['compiled_resources']
    # Invalid outputs never produce an accepted attribution report. Raw profiler
    # CSV remains retained even when it is incomplete or numerical checks fail.
    if report['correctness_passed']:
        for role in ROLES:
            report['roles'][role]['kernels'] = parse_metrics(regular(captured / role, 'stdout.log').read_text())
    write(output / 'profile-report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', choices=ROLES)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--task-document', type=Path)
    parser.add_argument('--stage-id')
    parser.add_argument('--owner', type=int, nargs=2)
    args = parser.parse_args()
    phase = args.stage_id if args.child else os.environ['KERNELINFRA_STAGE_ID']
    if phase not in {'prepare', 'profile', 'verify'}:
        raise ValueError('profile requires separate prepare, profile and verify phases')
    if args.child and phase != 'profile':raise ValueError('NCU child belongs to the profile phase')
    if phase != 'profile':os.environ['CUDA_VISIBLE_DEVICES'] = ''
    commit = checkout_commit(ROOT)
    task_path = args.task_document if args.child else Path(os.environ['KERNELINFRA_TASK'])
    phase_contract(commit, json.loads(task_path.read_bytes()), gpu_stage='profile')
    if args.child:
        owner = profile_output_owner({'uid':args.owner[0], 'gid':args.owner[1]})
        child(args.child, args.output, args.input, args.prepared, owner)
        return 0
    output = Path(os.environ['KERNELINFRA_STAGE_DIR'])
    root = Path(os.environ['KERNELINFRA_CANDIDATE_DIR'])
    run = Path(os.environ['KERNELINFRA_RUN_DIR'])
    result = {'schema':'kernelinfra.stage-result.v1', 'status':'failed', 'validity':'unknown', 'artifacts':{}}
    try:
        workload = WorkloadContract(json.loads(regular(root, 'workload.json').read_text()))
        if workload.target != 'sm_103a':raise ValueError('profile target differs')
        reference_arguments(workload)
        if phase == 'prepare':
            prepare(root, output, workload)
            summary = 'CPU cases, oracle and native reference prepared without a GPU lease'
            artifacts = {'preparation':'preparation.json'}
        elif phase == 'profile':
            capture(root, output, run / 'stages/prepare', workload, task_path, phase, commit)
            summary = 'NCU capture complete; all numerical verification remains pending'
            artifacts = {'capture':'capture.json'}
        else:
            report = verify(root, output, run / 'stages/prepare', run / 'stages/profile', workload, commit)
            artifacts = {'profile':'profile-report.json'}
            summary = 'attribution complete' if report['correctness_passed'] else 'profile participant numerical checks failed'
            if not report['correctness_passed']:
                result.update(status='failed', validity='invalid', summary=summary, artifacts=artifacts)
                write(Path(os.environ['KERNELINFRA_RESULT']), result)
                return 0
        result.update(status='passed', validity='valid', summary=summary, artifacts=artifacts)
    except Exception as error:
        result['summary'] = f'{type(error).__name__}: {error}'
        write(output / 'failure.json', {'error':str(error), 'traceback':traceback.format_exc()})
        result['artifacts']['failure'] = 'failure.json'
    write(Path(os.environ['KERNELINFRA_RESULT']), result)
    return 1 if result['validity'] == 'unknown' else 0


if __name__ == '__main__':
    raise SystemExit(main())
