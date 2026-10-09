#!/usr/bin/env python3
"""Bounded MACA sin/cos correctness qualification with production admission closed.

Build uses the existing isolated compiler. Evaluate loads only its sealed native
artifacts under the existing physical-device lock. No timing or provider is used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import struct
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from qualify_metax_trig import probe_source

OPERATIONS = ('sin', 'cos')
SHAPE = (4, 256)
COUNT = math.prod(SHAPE)
ATOL = 2e-6
RTOL = 2e-6


def fp32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def words(raw):
    if len(raw) != COUNT * 4:
        raise ValueError('trig storage must contain exactly 4x256 FP32 values')
    return struct.unpack('<' + 'I' * COUNT, raw)


def input_patterns():
    """Encode inputs before computing references, including signed and special bits."""
    def sweep(bound, seed):
        result = []
        for _ in range(COUNT):
            seed = (1664525 * seed + 1013904223) & 0xffffffff
            result.append(fp32(bound * seed / 2**32))
        return result

    special = [0x00000000, 0x80000000, 0x7f800000, 0xff800000, 0x7fc12345,
               0xffc54321, 0x00000001, 0x80000001, 0x007fffff, 0x807fffff,
               0x00800000, 0x80800000]
    for multiple in range(-128, 129):
        if multiple:
            center = struct.unpack('<I', struct.pack('<f', multiple * math.pi / 2))[0]
            special.extend((center - 1, center, center + 1))
    fill = sweep(2047, 401)
    special.extend(struct.unpack('<I', struct.pack('<f', value))[0]
                   for value in fill[:COUNT - len(special)])
    rope = sweep(2047, 402)
    rope[:8] = [0., 2047., 128., 131., 511., 1024., fp32(1/3), fp32(17/13)]
    positive = sweep(65536, 403)
    positive[:3] = [2047., 65536., fp32(65536 - .00390625)]
    negative = [-value for value in sweep(65536, 404)]
    negative[:3] = [-2047., -65536., fp32(-65536 + .00390625)]
    # Bit-distributed magnitudes exercise normal/subnormal values near zero.
    small = [((i * 2654435761) % 0x3f800001) | ((i % 2) << 31) for i in range(COUNT)]
    pack_values = lambda values: struct.pack('<' + 'f' * COUNT, *values)
    pack_words = lambda values: struct.pack('<' + 'I' * COUNT, *values)
    return (('special_and_quadrants', pack_words(special)), ('rope_envelope', pack_values(rope)),
            ('wide_positive', pack_values(positive)), ('wide_negative', pack_values(negative)),
            ('small_magnitudes', pack_words(small)))


def numeric_contract():
    return {'shape': list(SHAPE), 'input_encoding': 'little_endian_fp32_bits',
            'patterns': [name for name, _ in input_patterns()], 'finite_sample_domain': [-65536., 65536.],
            'reference': 'Python math.sin/cos of the exact decoded FP32 input',
            'atol': ATOL, 'rtol': RTOL, 'comparison': 'abs_error <= atol + rtol*abs(reference)',
            'nonfinite_inputs': 'NaN and either infinity require NaN; payload unspecified',
            'signed_zero': 'sin preserves sign; cos returns exactly +1',
            'output_poisons': [2., -2.], 'scope': 'sampled component correctness; not exhaustive FP32 or RoPE qualification'}


def check_outputs(op, inputs, observed):
    if op not in OPERATIONS:
        raise ValueError('trig operation differs')
    source, output = words(inputs), words(observed)
    failures, mismatches, maximum = [], 0, 0.
    for index, (x_word, y_word) in enumerate(zip(source, output, strict=True)):
        x, y = (struct.unpack('<f', struct.pack('<I', word))[0] for word in (x_word, y_word))
        if not math.isfinite(x):
            passed, expected = math.isnan(y), 'nan'
        elif x == 0:
            expected_word = x_word if op == 'sin' else 0x3f800000
            passed, expected = y_word == expected_word, f'bits:{expected_word:08x}'
        else:
            reference = getattr(math, op)(x)
            error = abs(y - reference)
            passed = math.isfinite(y) and error <= ATOL + RTOL * abs(reference)
            expected = reference
            if math.isfinite(error):
                maximum = max(maximum, error)
        if not passed:
            mismatches += 1
            if len(failures) < 16:
                failures.append({'index': index, 'input_bits': f'{x_word:08x}',
                                 'observed_bits': f'{y_word:08x}', 'expected': expected})
    return {'passed': mismatches == 0, 'elements_checked': COUNT, 'mismatches': mismatches,
            'maximum_finite_abs_error': maximum, 'first_failures': failures}


def workload_document(op):
    if op not in OPERATIONS:
        raise ValueError('trig operation differs')
    return {'schema_version': 1, 'workload_id': f'maca-{op}-component-probe', 'revision': '1',
            'state': 'frozen', 'operator': 'trig_component_probe',
            'provenance': [{'kind': 'qualification_contract', 'path': 'tools/qualify_metax_trig_device.py'}],
            'cases': [{'case_id': 'primary', 'shape': {'R': SHAPE[0], 'C': SHAPE[1]},
                       'seed': 401, 'mode': 'fixed_fp32_bit_patterns'}],
            'tensors': {name: {'shape': ['R', 'C'], 'dtype': 'fp32', 'layout': 'contiguous_row_major'}
                        for name in ('x', 'out')},
            'semantics': {'target': 'xcore1002', 'definition': f'out = {op}(x)',
                          'candidate_abi': {'inputs': ['x'], 'outputs': ['out']},
                          'input_effects': 'unchanged', 'qualification': numeric_contract()},
            'oracle': {'kind': 'independent_python_math', 'callable': f'math.{op}'},
            'validation': {'primary_case': 'primary', 'all_cases_required': True,
                           'comparison': 'declared_finite_tolerance_and_special_classification',
                           'atol': ATOL, 'rtol': RTOL}}


def write(path, value):
    from open_cake_ir.cli import _json_projection
    from open_cake_ir.serialization import canonical_json_bytes
    with path.open('xb') as stream:
        stream.write(canonical_json_bytes(_json_projection(value)))


def probe_emission(op):
    from open_cake_ir.compiler import Target, frontend
    from open_cake_ir.compiler.ir import Schedule
    from open_cake_ir.compiler.backends.triton import emit, preflight
    from open_cake_ir.compiler.verifier import verify
    target = Target.load(ROOT / 'compiler/targets/xcore1002.json')
    if {f'maca.{name}.f32' for name in OPERATIONS} & target.instruction_contracts:
        raise ValueError('pre-admission trig probe requires the production Target to remain closed')
    document = frontend.parse(probe_source(op)).document
    schedule = Schedule.from_dict(document)
    probe = replace(target, instruction_contracts=target.instruction_contracts | {f'maca.{op}.f32'})
    findings = (*verify(schedule, probe), *preflight(schedule, probe))
    if any(item.blocks_lowering for item in findings):
        raise ValueError('trig qualification has a blocking software finding')
    return document, emit(schedule, probe), target


def build(args, result):
    from hashlib import sha256
    from open_cake_ir.compiler import Compiler
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
    gate = compiler.check_corpus()
    write(args.output / 'compiler-gate.json', gate)
    if not gate.passed:
        raise ValueError('Corpus Gate failed')
    isolated = None
    if args.native:
        executor = ExecutorRevision.for_target(ROOT, 'xcore1002')
        write(args.output / 'host-admission.json', executor.admit_host())
        isolated = IsolatedTritonCompiler(**_triton_toolchain_config(executor))
        isolated.check_executor(executor, author_workspace=args.output)
    result['operations'] = []
    write(args.output / 'numeric-contract.json', numeric_contract())
    for name, payload in input_patterns():
        (args.output / (name + '.fp32')).write_bytes(payload)
    for op in OPERATIONS:
        folder = args.output / op
        folder.mkdir()
        document, emission, _ = probe_emission(op)
        production = compiler.assess(document)
        if production.lowering_eligible or 'TARGET_INSTRUCTION_UNSUPPORTED' not in {f.code for f in production.findings}:
            raise ValueError('production trig instruction refusal differs')
        write(folder / 'production-findings.json', production.findings)
        (folder / 'candidate.cake.py').write_text(probe_source(op))
        requirements = {'compiler': 'triton', 'source_language': 'python', 'target': 'xcore1002', **emission.toolchain}
        payload = emission.source.encode()
        (folder / 'kernel.triton.py').write_bytes(project_triton_kernel(payload, requirements))
        write(folder / 'requirements.json', requirements)
        workload = WorkloadContract(workload_document(op))
        write(folder / 'workload.json', workload.document)
        row = {'operation': op, 'native_compiled': False}
        result['operations'].append(row)
        if isolated is not None:
            submission = CandidateSubmission.seal(OpenCakeEnvironment.media_type, canonical_json_bytes(document))
            request = BuildRequest(submission.sha256, payload, 'lowered_source', sha256(payload).hexdigest(),
                                   'xcore1002', requirements['kernel_entry_point'], requirements)
            builder = TritonToolchainBuilder(workload=workload, case_id='primary', isolated_compiler=isolated)
            abi = tuple((a.name, a.shape, a.dtype, a.mode) for a in workload.tensor_abi('primary'))
            candidate = builder.build_stage(request, abi)
            for role, data in candidate.artifact_payloads.items():
                (folder / (role + '.bin')).write_bytes(data)
            write(folder / 'candidate.json', candidate_identity(candidate))
            row['native_compiled'] = True
    result['passed'] = True


def bound_candidates(built_root, commit):
    """Bind current source, contract, inputs and every sealed artifact before allocation."""
    from open_cake_ir.evaluation.core import TensorLaunchManifest
    from open_cake_ir.evaluation.paired import candidate_from_identity
    from open_cake_ir.evaluation.program import check_triton_launch_record
    from open_cake_ir.evaluation.workload import WorkloadContract
    from open_cake_ir.lab import CandidateSubmission, OpenCakeEnvironment
    from open_cake_ir.serialization import canonical_json_bytes
    from open_cake_ir.tasks.evaluate import _input_path
    built_root = built_root.resolve(strict=True)
    result = json.loads(_input_path(built_root, 'result.json', 'result').read_text())
    if (not commit or result.get('source_commit') != commit or result.get('target') != 'xcore1002'
            or result.get('passed') is not True or result.get('operations') !=
            [{'operation': op, 'native_compiled': True} for op in OPERATIONS]):
        raise ValueError('both native trig builds must belong to this exact source commit and target')
    if json.loads(_input_path(built_root, 'numeric-contract.json', 'contract').read_text()) != numeric_contract():
        raise ValueError('retained trig numeric contract differs')
    for name, payload in input_patterns():
        if _input_path(built_root, name + '.fp32', 'inputs').read_bytes() != payload:
            raise ValueError('retained trig input bits differ')
    bound = []
    for op in OPERATIONS:
        folder = built_root / op
        if folder.is_symlink():
            raise ValueError('trig artifact directory must not be a symlink')
        identity = json.loads(_input_path(folder, 'candidate.json', 'candidate').read_text())
        payloads = {role: _input_path(folder, role + '.bin', role).read_bytes() for role in identity['artifact_roles']}
        candidate = candidate_from_identity(identity, payloads)
        manifest = TensorLaunchManifest.from_dict(json.loads(payloads['launch_manifest']))
        document, emission, target = probe_emission(op)
        workload = WorkloadContract(workload_document(op))
        manifest.check_workload(workload, 'primary')
        submission = CandidateSubmission.seal(OpenCakeEnvironment.media_type, canonical_json_bytes(document))
        expected_requirements = {'compiler': 'triton', 'source_language': 'python', 'target': 'xcore1002', **emission.toolchain}
        if (json.loads(_input_path(folder, 'requirements.json', 'requirements').read_text()) != expected_requirements
                or candidate.candidate_sha256 != submission.sha256 or candidate.target != 'xcore1002'
                or payloads.get('lowered_source') != emission.source.encode()
                or candidate.entry_point != emission.toolchain['kernel_entry_point']
                or manifest.kernel_name != candidate.entry_point or candidate.launch_spec_sha256 != manifest.canonical_sha256
                or list(manifest.grid) != emission.toolchain['grid']
                or manifest.block != (emission.toolchain['compile_options']['num_warps'] * target.warp_size, 1, 1)
                or manifest.aligned_variant or manifest.pointer_alignments):
            raise ValueError('retained trig candidate differs from the current fixed probe emission')
        check_triton_launch_record(candidate, manifest, candidate.artifact_roles['lowered_source'])
        bound.append((op, candidate, manifest))
    return tuple(bound)


def evaluate(args, result):
    from open_cake_ir.evaluation.local_broker import admit_local_job
    from open_cake_ir.evaluation.triton_metax import observe_local_metax
    from open_cake_ir.evaluation.metax_driver import LoadedMetaxCandidate
    from open_cake_ir.lab.executor import ExecutorRevision
    candidates = bound_candidates(args.built, result['source_commit'])
    host = ExecutorRevision.for_target(ROOT, 'xcore1002').admit_host()
    job = admit_local_job('maca', device=args.physical_device, runtime_device=args.runtime_device,
                         expected_pci=args.expected_pci, lock_scope='device', queue_seconds=300)
    admission = observe_local_metax('xcore1002', runtime_library=host['runtime_library'])
    import torch
    result.update(job_id=job, admission=asdict(admission), operations=[], device_execution=True)
    for op, candidate, manifest in candidates:
        loaded = LoadedMetaxCandidate.load(candidate, manifest, admission)
        checks, primary = [], None
        try:
            for name, raw in input_patterns():
                cpu = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
                x = cpu.view(torch.float32).reshape(SHAPE).to('cuda:0')
                for poison in numeric_contract()['output_poisons']:
                    out = torch.full(SHAPE, poison, dtype=torch.float32, device='cuda:0')
                    loaded.launch((x, out), tensor_contract=manifest, stream=int(torch.cuda.current_stream().cuda_stream))
                    torch.cuda.synchronize()
                    observed = bytes(out.cpu().view(torch.uint8).reshape(-1).tolist())
                    unchanged = torch.equal(x.cpu().view(torch.uint8).reshape(-1), cpu)
                    check = {**check_outputs(op, raw, observed), 'pattern': name,
                             'output_poison': poison, 'input_unchanged': unchanged}
                    checks.append(check)
                    (args.output / f'{op}-{name}-{len(checks)}.fp32').write_bytes(observed)
                    if not check['passed'] or not unchanged:
                        raise ValueError(f'{op}/{name} failed the declared trig contract')
        except BaseException as error:
            primary = error
        finally:
            try:
                loaded.close(synchronize=torch.cuda.synchronize, primary=primary)
            except BaseException as error:
                primary = error
            row = {'operation': op, 'checks': checks, 'kernel_calls': loaded.launch_calls,
                   'module_closed': loaded.closed, 'resources': loaded.resources}
            result['operations'].append(row)
            write(args.output / (op + '.json'), row)
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
    result = {'source_commit': commit, 'target': 'xcore1002', 'passed': False,
              'target_admission_changed': False, 'device_execution': False,
              'scope': 'bounded sin/cos component correctness; no timing or optimization'}
    try:
        (build if args.command == 'build' else evaluate)(args, result)
    except Exception as error:
        result.update(error=str(error), failure_class=type(error).__name__)
        for role, data in getattr(error, 'artifact_payloads', {}).items():
            if isinstance(role, str) and role.isidentifier() and isinstance(data, bytes):
                (args.output / ('failure-' + role)).write_bytes(data)
        (args.output / 'failure.txt').write_text(traceback.format_exc())
    write(args.output / 'result.json', result)
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
