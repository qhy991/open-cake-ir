#!/usr/bin/env python3
"""Bounded pre-admission probe of existing register transpose on exact C550.

The committed Target stays closed. Build uses a synthetic in-memory declaration;
device checks use sealed native artifacts and the existing MACA allocator/loader.
This command cannot issue an optimization Run or a performance result.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]

DTYPES = ('fp32', 'fp16', 'bf16', 'int32', 'int64', 'bool')
REFUSED_DTYPES = ('fp8_e4m3',)
SHAPES = ((16, 32), (35, 67), (1, 3))


def source(rows, columns, dtype):
    if dtype not in DTYPES + REFUSED_DTYPES or (rows, columns) not in SHAPES:
        raise ValueError('transpose probe is limited to its declared dtype and shape set')
    return f'''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="transpose_probe", target="xcore1002", backend="triton", entry_point="transpose_probe")
def candidate(lm, x: cake.Tensor(({rows}, {columns}), "{dtype}"), out: cake.Tensor(({columns}, {rows}), "{dtype}", mode="output")):
    compute = lm.role(execution_groups=[0, 1, 2, 3])
    row = lm.program(x, axis=0, dimension=0, tile=16)
    column = lm.program(x, axis=1, dimension=1, tile=32)
    with compute:
        values = lm.load(x[row, column], id="load")
        transposed = lm.transpose(values, id="transpose")
        lm.store(out[column, row], transposed, id="store")
'''


def workload_document(rows, columns, dtype):
    return {'schema_version': 1, 'workload_id': f'transpose-{dtype}-{rows}-{columns}',
        'revision': '1', 'state': 'frozen', 'operator': 'register_transpose_probe',
        'provenance': [{'kind': 'qualification_contract', 'path': 'tools/qualify_metax_transpose.py'}],
        'cases': [{'case_id': 'primary', 'shape': {'R': rows, 'C': columns}, 'seed': 418, 'mode': 'storage_patterns'}],
        'tensors': {'x': {'shape': ['R', 'C'], 'dtype': dtype, 'layout': 'contiguous_row_major'},
                    'out': {'shape': ['C', 'R'], 'dtype': dtype, 'layout': 'contiguous_row_major'}},
        'semantics': {'target': 'xcore1002', 'definition': 'out[c,r] = x[r,c]',
                      'candidate_abi': {'inputs': ['x'], 'outputs': ['out']},
                      'input_effects': 'unchanged', 'output_storage': 'fresh_contiguous_nonaliasing'},
        'oracle': {'kind': 'independent_cpu_byte_permutation', 'callable': 'torch.Tensor.permute',
                   'scope': 'CPU storage movement; no candidate arithmetic'},
        'validation': {'primary_case': 'primary', 'all_cases_required': True,
                       'comparison': 'exact_storage_bytes', 'atol': 0, 'rtol': 0}}


def write(path, document):
    from open_cake_ir.serialization import canonical_json_bytes
    from open_cake_ir.cli import _json_projection
    with path.open('xb') as stream:
        stream.write(canonical_json_bytes(_json_projection(document)))


def storage_patterns(rows, columns, dtype):
    """Give every logical element a unique signature across the test inputs."""
    from open_cake_ir.compiler.ir import DType
    count = rows * columns
    size = DType(dtype).itemsize
    if dtype == 'bool':
        return tuple(bytes((index >> bit) & 1 for index in range(count))
                     for bit in range(max(1, (count - 1).bit_length())))
    prefix = {'fp32': 0x3F000000, 'fp16': 0x3000, 'bf16': 0x3000,
              'int32': 2**24, 'int64': 2**54}[dtype]
    unique = b''.join((prefix + index).to_bytes(size, 'little') for index in range(count))
    varied = tuple(bytes((i * (37 + pattern * 2) + pattern * 71) % 256
                         for i in range(count * size)) for pattern in range(3))
    return (unique, *varied)


def bound_candidates(built_root, source_commit):
    """Bind each retained artifact before acquiring a device allocation."""
    built_root = built_root.resolve(strict=True)
    from open_cake_ir.compiler import Target, frontend
    from open_cake_ir.compiler.backends.triton import emit
    from open_cake_ir.compiler.ir import OperationKind, Schedule
    from open_cake_ir.evaluation.workload import WorkloadContract
    from open_cake_ir.evaluation.paired import candidate_from_identity
    from open_cake_ir.evaluation.core import TensorLaunchManifest
    from open_cake_ir.lab import CandidateSubmission, OpenCakeEnvironment
    from open_cake_ir.serialization import canonical_json_bytes
    built = json.loads((built_root / 'result.json').read_text())
    expected = [(dtype, list(shape)) for dtype in DTYPES for shape in SHAPES]
    refused = [{'name': f'fp8_e4m3-{rows}-{columns}',
                'codes': ['MACA_FP8_OPERATION_UNQUALIFIED', 'VALUE_OPERATION_TYPE']}
               for rows, columns in SHAPES]
    if (not source_commit or built.get('source_commit') != source_commit or built.get('target') != 'xcore1002'
            or not built.get('passed') or [(c['dtype'], c['shape']) for c in built['cases']] != expected
            or not all(c['native_compiled'] for c in built['cases'])):
        raise ValueError('all declared builds must belong to this exact source commit and target')
    if [{key: row[key] for key in ('name', 'codes')} for row in built.get('refused_cases', [])] != refused:
        raise ValueError('all declared FP8 refusal controls must be retained')
    target = Target.load(ROOT / 'compiler/targets/xcore1002.json')
    if OperationKind.TRANSPOSE in target.operation_kinds:
        raise ValueError('pre-admission probe requires the production Target to remain closed')
    probe = replace(target, operation_kinds=target.operation_kinds | {OperationKind.TRANSPOSE})
    bound = []
    for item in built['cases']:
        rows, columns = item['shape']
        expected_name = f'{item["dtype"]}-{rows}-{columns}'
        if item['name'] != expected_name:
            raise ValueError('transpose build path differs from its fixed case')
        directory = built_root / expected_name
        identity = json.loads((directory / 'candidate.json').read_text())
        from open_cake_ir.tasks.evaluate import _input_path
        candidate = candidate_from_identity(identity, {role: _input_path(directory, role + '.bin', role).read_bytes()
                                                       for role in identity['artifact_roles']})
        manifest = TensorLaunchManifest.from_dict(json.loads(candidate.artifact_payloads['launch_manifest']))
        workload = WorkloadContract(workload_document(rows, columns, item['dtype']))
        manifest.check_workload(workload, 'primary')
        raw = frontend.parse(source(rows, columns, item['dtype'])).document
        submission = CandidateSubmission.seal(OpenCakeEnvironment.media_type, canonical_json_bytes(raw))
        emission = emit(Schedule.from_dict(raw), probe)
        if (candidate.candidate_sha256 != submission.sha256
                or candidate.artifact_payloads.get('lowered_source') != emission.source.encode()
                or manifest.kernel_name != emission.toolchain['kernel_entry_point']
                or list(manifest.grid) != emission.toolchain['grid']
                or tuple(manifest.block) != (emission.toolchain['compile_options']['num_warps'] * target.warp_size, 1, 1)):
            raise ValueError('retained transpose candidate differs from the current fixed probe emission')
        bound.append((item, candidate, manifest))
    return tuple(bound)


def build(args, result):
    from open_cake_ir.compiler import Compiler, Target, frontend
    from open_cake_ir.compiler.backends.triton import emit, preflight
    from open_cake_ir.compiler.ir import OperationKind, Schedule
    from open_cake_ir.compiler.verifier import verify
    from open_cake_ir.compiler.toolchain import project_triton_kernel
    from open_cake_ir.evaluation.workload import WorkloadContract
    from open_cake_ir.evaluation.paired import candidate_identity
    from open_cake_ir.lab import CandidateSubmission, OpenCakeEnvironment
    from open_cake_ir.lab.build import BuildRequest, TritonToolchainBuilder
    from open_cake_ir.lab.executor import ExecutorRevision
    from open_cake_ir.lab.triton_build import IsolatedTritonCompiler
    from open_cake_ir.serialization import canonical_json_bytes
    from launch_task import _triton_toolchain_config
    compiler = Compiler.load(ROOT)
    target = Target.load(ROOT / 'compiler/targets/xcore1002.json')
    if OperationKind.TRANSPOSE in target.operation_kinds:
        raise ValueError('pre-admission probe requires the production Target to remain closed')
    probe = replace(target, operation_kinds=target.operation_kinds | {OperationKind.TRANSPOSE})
    gate = compiler.check_corpus()
    write(args.output / 'compiler-gate.json', gate)
    if not gate.passed:
        raise ValueError('Corpus Gate failed')
    isolated = None
    if args.native:
        executor = ExecutorRevision.for_target(ROOT, 'xcore1002')
        executor.admit_host()
        isolated = IsolatedTritonCompiler(**_triton_toolchain_config(executor))
        isolated.check_executor(executor, author_workspace=args.output)
    result.update(cases=[], refused_cases=[])
    for dtype in DTYPES + REFUSED_DTYPES:
        for rows, columns in SHAPES:
            name = f'{dtype}-{rows}-{columns}'
            directory = args.output / name
            directory.mkdir()
            text = source(rows, columns, dtype)
            (directory / 'candidate.cake.py').write_text(text)
            raw = frontend.parse(text).document
            schedule = Schedule.from_dict(raw)
            production = compiler.assess(raw)
            if production.lowering_eligible:
                raise ValueError('production Target unexpectedly admitted transpose')
            write(directory / 'production-findings.json', production.findings)
            findings = (*verify(schedule, probe), *preflight(schedule, probe))
            write(directory / 'probe-findings.json', findings)
            if dtype in REFUSED_DTYPES:
                codes = {f.code for f in findings if f.blocks_lowering}
                if codes != {'VALUE_OPERATION_TYPE', 'MACA_FP8_OPERATION_UNQUALIFIED'}:
                    raise ValueError(f'{name}: FP8 refusal ownership differs')
                result['refused_cases'].append({'name': name, 'codes': sorted(codes),
                                                'scope': 'existing type and target restriction; no device call'})
                continue
            if any(f.blocks_lowering for f in findings):
                raise ValueError(f'{name}: synthetic transpose has a blocking finding')
            emission = emit(schedule, probe)
            requirements = {'compiler': 'triton', 'source_language': 'python',
                            'target': target.target_id, **emission.toolchain}
            (directory / 'kernel.triton.py').write_bytes(project_triton_kernel(emission.source.encode(), requirements))
            write(directory / 'requirements.json', requirements)
            document = workload_document(rows, columns, dtype)
            write(directory / 'workload.json', document)
            row = {'name': name, 'dtype': dtype, 'shape': [rows, columns], 'native_compiled': False}
            result['cases'].append(row)
            if isolated is not None:
                submission = CandidateSubmission.seal(OpenCakeEnvironment.media_type, canonical_json_bytes(raw))
                from hashlib import sha256
                payload = emission.source.encode()
                request = BuildRequest(submission.sha256, payload, 'lowered_source', sha256(payload).hexdigest(),
                                       target.target_id, requirements['kernel_entry_point'], requirements)
                builder = TritonToolchainBuilder(workload=WorkloadContract(document), case_id='primary', isolated_compiler=isolated)
                candidate = builder.build(request)
                for role, data in candidate.artifact_payloads.items():
                    (directory / (role + '.bin')).write_bytes(data)
                write(directory / 'candidate.json', candidate_identity(candidate))
                row['native_compiled'] = True
            write(directory / 'build-result.json', row)
    result['passed'] = True


def evaluate(args, result):
    from open_cake_ir.evaluation.local_broker import admit_local_job
    from open_cake_ir.evaluation.triton_metax import observe_local_metax
    from open_cake_ir.evaluation.metax_driver import LoadedMetaxCandidate
    from open_cake_ir.lab.executor import ExecutorRevision
    import torch
    candidates = bound_candidates(args.built, result['source_commit'])
    host = ExecutorRevision.for_target(ROOT, 'xcore1002').admit_host()
    job = admit_local_job('maca', device=args.physical_device, runtime_device=args.runtime_device,
                         expected_pci=args.expected_pci, lock_scope='device', queue_seconds=300)
    admission = observe_local_metax('xcore1002', runtime_library=host['runtime_library'])
    result.update(job_id=job, admission=asdict(admission), cases=[])
    dtype_names = dict(fp32='float32', fp16='float16', bf16='bfloat16', fp8_e4m3='float8_e4m3fn',
                       int32='int32', int64='int64', bool='bool')
    for item, candidate, manifest in candidates:
        rows, columns = item['shape']
        dtype = getattr(torch, dtype_names[item['dtype']])
        size = torch.empty((), dtype=dtype).element_size()
        loaded = LoadedMetaxCandidate.load(candidate, manifest, admission)
        checks = []
        primary = None
        try:
            for pattern, raw in enumerate(storage_patterns(rows, columns, item['dtype'])):
                count = rows * columns * size
                cpu_bytes = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
                expected = cpu_bytes.reshape(rows, columns, size).permute(1, 0, 2).contiguous().reshape(-1)
                x = cpu_bytes.view(dtype).reshape(rows, columns).to('cuda:0')
                out = torch.empty((columns, rows), dtype=dtype, device='cuda:0')
                out.view(torch.uint8).fill_(0xA5)
                loaded.launch((x, out), tensor_contract=manifest, stream=int(torch.cuda.current_stream().cuda_stream))
                torch.cuda.synchronize()
                observed = out.cpu().view(torch.uint8).reshape(-1)
                unchanged = torch.equal(x.cpu().view(torch.uint8).reshape(-1), cpu_bytes)
                mismatches = int((observed != expected).sum().item())
                checks.append({'pattern': pattern, 'bytes_checked': count, 'byte_mismatches': mismatches,
                               'input_unchanged': unchanged})
                if mismatches or not unchanged:
                    raise ValueError(f'{item["name"]} failed original-byte transpose')
        except BaseException as error:
            primary = error
        finally:
            try:
                loaded.close(synchronize=torch.cuda.synchronize, primary=primary)
            except BaseException as error:
                primary = error
            result['cases'].append({'name': item['name'], 'checks': checks, 'kernel_calls': loaded.launch_calls,
                                    'module_closed': loaded.closed, 'resources': loaded.resources})
            write(args.output / (item['name'] + '.json'), result['cases'][-1])
        if primary is not None:
            raise primary
    result['passed'] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    builder = sub.add_parser('build')
    builder.add_argument('--native', action='store_true')
    runner = sub.add_parser('evaluate')
    runner.add_argument('--built', type=Path, required=True)
    runner.add_argument('--physical-device', type=int, required=True)
    runner.add_argument('--runtime-device', type=int, required=True)
    runner.add_argument('--expected-pci', required=True)
    for command in (builder, runner):
        command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from open_cake_ir.source_identity import checkout_commit
    commit = checkout_commit(ROOT)
    if commit is None or os.environ.get('METAL_BROKER_LOCK_FD') or os.environ.get('GPUQ_JOB_ID'):
        parser.error('probe requires clean committed source and starts outside an allocation')
    args.output = args.output.resolve()
    if args.output == ROOT or ROOT in args.output.parents:
        parser.error('qualification evidence must stay outside source')
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'source_commit': commit, 'passed': False, 'target': 'xcore1002',
              'scope': 'pre-admission transpose qualification; no optimization or performance',
              'target_admission_changed': False}
    try:
        (build if args.command == 'build' else evaluate)(args, result)
    except Exception as error:
        result.update(error=str(error), failure_class=type(error).__name__)
        (args.output / 'failure.txt').write_text(traceback.format_exc())
    write(args.output / 'result.json', result)
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
