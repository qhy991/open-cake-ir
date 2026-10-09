#!/usr/bin/env python3
"""Original-domain RoPE angle comparison; production TF32 admission stays closed."""
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
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]

TASK = 'L1/011_rotary_position_embedding'
TARGET = 'xcore1002'
SHAPES = ((1, 2048), (16, 256), (2, 131))
PRECISIONS = ('tf32', 'ieee')
TF32 = 'triton.dot.fp32_tf32'


def angle_source(batch, sequence, precision):
    if ((batch, sequence) not in SHAPES or type(batch) is not int
            or type(sequence) is not int or precision not in PRECISIONS):
        raise ValueError('angle probe requires one of its three original shapes and explicit precision')
    return f'''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="rope_angle_{precision}", target="xcore1002", backend="triton", entry_point="rope_angle_{precision}")
def candidate(lm, position_ids: cake.Tensor(({batch},{sequence},1), "int64"),
              inv_freq: cake.Tensor((64,1), "fp32"), angles: cake.Tensor(({batch},{sequence},64), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0,1,2,3])
    frequency = lm.program(inv_freq,axis=0,dimension=0,tile=16)
    row = lm.program(position_ids,axis=1,dimension=1,tile=32)
    batch = lm.program(position_ids,axis=2,dimension=0,tile=1)
    with compute:
        k = lm.coordinate(source="range",start=0,extent=32)
        frequencies = lm.load(inv_freq[frequency,k],id="load_frequency_kpad")
        positions_i64 = lm.load(position_ids[batch,row,k],id="load_positions_kpad")
        positions = lm.cast(positions_i64,to="fp32",id="positions_to_fp32")
        product = lm.mma(frequencies,positions,instruction={{"contract":"triton.dot.fp32_{precision}"}},tile_shape=(16,32,32),id="angle_product")
        transposed = lm.transpose(product,id="angle_transpose")
        lm.store(angles[batch,row,frequency],transposed,id="store_angles")
'''


def probe_emission(batch, sequence, precision):
    from open_cake_ir.compiler import Target, frontend
    from open_cake_ir.compiler.ir import Schedule
    from open_cake_ir.compiler.backends.triton import emit, preflight
    from open_cake_ir.compiler.verifier import verify
    target = Target.load(ROOT / 'compiler/targets/xcore1002.json')
    if TF32 in target.instruction_contracts:
        raise ValueError('pre-admission angle probe requires production TF32 to remain closed')
    document = frontend.parse(angle_source(batch, sequence, precision)).document
    schedule = Schedule.from_dict(document)
    probe = replace(target, instruction_contracts=target.instruction_contracts | {TF32})
    if any(item.blocks_lowering for item in (*verify(schedule, probe), *preflight(schedule, probe))):
        raise ValueError('angle probe has a blocking software finding')
    return document, emit(schedule, probe), target


def compare_angles(expected, observed, batch, sequence):
    """Retain bit equality and finite error without inventing a Bench tolerance."""
    count = batch * sequence * 64
    if len(expected) != count * 4 or len(observed) != count * 4:
        raise ValueError('angle comparison requires the complete declared FP32 output')
    reference_words = struct.unpack('<' + 'I' * count, expected)
    output_words = struct.unpack('<' + 'I' * count, observed)
    reference = struct.unpack('<' + 'f' * count, expected)
    actual = struct.unpack('<' + 'f' * count, observed)
    if any(not math.isfinite(value) for value in reference):
        raise ValueError('original sampled angle reference must be finite')
    mismatch = sum(left != right for left, right in zip(reference_words, output_words, strict=True))
    finite_errors = [abs(left - right) for left, right in zip(reference, actual, strict=True)
                     if math.isfinite(right)]
    return {'elements': count, 'bitwise_equal': mismatch == 0, 'bit_mismatches': mismatch,
            'nonfinite_outputs': sum(not math.isfinite(value) for value in actual),
            'maximum_finite_absolute_error': max(finite_errors) if finite_errors else None,
            'comparison': 'exact_fp32_bits; finite errors are diagnostic only'}


def select_cases(problem):
    """Resolve three original identities, without inventing or resizing a workload."""
    if problem.task_id != TASK:
        raise ValueError('angle probe requires the original RoPE task')
    cases = []
    for batch, sequence in SHAPES:
        matches = [row for row in problem.workloads
                   if row.axes == {'batch_size': batch, 'seq_len': sequence}]
        if len(matches) != 1:
            raise ValueError('each angle probe shape must identify one original workload')
        selected = matches[0]
        if str(UUID(selected.uuid)) != selected.uuid:
            raise ValueError('original workload identity must be a canonical UUID')
        shapes = {name: None if shape is None else tuple(shape)
                  for name, shape in problem.definition.get_input_shapes(selected.axes).items()}
        outputs = {name: tuple(shape) for name, shape in problem.definition.get_output_shapes(selected.axes).items()}
        if (shapes != {'position_ids': (batch, sequence), 'inv_freq': (64,), 'attention_scaling': None}
                or outputs != {'cos_sin': (batch, sequence, 128, 2)}):
            raise ValueError('original RoPE input or output geometry differs')
        cases.append({'workload_uuid': selected.uuid, 'batch': batch, 'sequence': sequence})
    return cases


def hypothesis_policy(value):
    from open_cake_ir.tasks.c550_bench.binding import validate_oracle_numerics
    policy = validate_oracle_numerics(value)
    if (policy['float32_matmul_precision'] != 'high' or policy['allow_tf32'] is not True
            or policy['initialization']['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] != '1'):
        raise ValueError('this hypothesis requires the explicitly observed original HIGH override policy')
    return policy


def numeric_policy(path):
    from open_cake_ir.lab.bindings import external_file
    return hypothesis_policy(json.loads(
        external_file(ROOT, str(path), 'oracle numerics observation').read_text()))


def stage_abi(case):
    b, s = case['batch'], case['sequence']
    return (('position_ids', (b, s, 1), 'int64', 'input'),
            ('inv_freq', (64, 1), 'fp32', 'input'),
            ('angles', (b, s, 64), 'fp32', 'output'))


def original_workload(problem, case, policy):
    from open_cake_ir.tasks.c550_bench.workload import BenchWorkload
    workload = BenchWorkload(problem.workload_document(case['workload_uuid'], oracle_numerics=policy))
    b, s = case['batch'], case['sequence']
    expected = (('position_ids', (b, s), 'int64', 'input'), ('inv_freq', (64,), 'fp32', 'input'),
                ('cos_sin', (b, s, 128, 2), 'bf16', 'output'))
    if (tuple((arg.name, arg.shape, arg.dtype, arg.mode) for arg in workload.tensor_abi('primary')) != expected
            or workload.document['semantics']['fixed_scalar_inputs']['attention_scaling']['value'] != 1.0):
        raise ValueError('original RoPE tensor or scalar contract differs')
    return workload


def write(path, value):
    from open_cake_ir.cli import _json_projection
    from open_cake_ir.serialization import canonical_json_bytes
    with path.open('xb') as stream:
        stream.write(canonical_json_bytes(_json_projection(value)))


def build(args, result):
    from hashlib import sha256
    from open_cake_ir.compiler import Compiler
    from open_cake_ir.evaluation.paired import candidate_identity
    from open_cake_ir.lab import CandidateSubmission, OpenCakeEnvironment
    from open_cake_ir.lab.build import BuildRequest, TritonToolchainBuilder
    from open_cake_ir.lab.executor import ExecutorRevision
    from open_cake_ir.lab.triton_build import IsolatedTritonCompiler
    from open_cake_ir.serialization import canonical_json_bytes
    from open_cake_ir.tasks.c550_bench.binding import BenchProblem, BENCH_COMMIT
    from launch_task import _triton_toolchain_config
    policy = numeric_policy(args.oracle_numerics)
    problem = BenchProblem.open(args.bench_root, TASK)
    cases = select_cases(problem)
    compiler = Compiler.load(ROOT)
    gate = compiler.check_corpus()
    write(args.output / 'compiler-gate.json', gate)
    if not gate.passed:
        raise ValueError('Corpus Gate failed')
    isolated = None
    if args.native:
        executor = ExecutorRevision.for_target(ROOT, TARGET)
        write(args.output / 'host-admission.json', executor.admit_host())
        isolated = IsolatedTritonCompiler(**_triton_toolchain_config(executor))
        isolated.check_executor(executor, author_workspace=args.output)
    write(args.output / 'binding.json', {'bench_commit': BENCH_COMMIT, 'task': TASK,
        'cases': cases, 'oracle_numerics': policy, 'precisions': list(PRECISIONS),
        'scope': 'angle stage only; original RoPE Workload remains the parent authority'})
    result['cases'] = []
    for case in cases:
        directory = args.output / case['workload_uuid']
        directory.mkdir()
        workload = original_workload(problem, case, policy)
        write(directory / 'original-workload.json', workload.document)
        row = {**case, 'precisions': []}
        result['cases'].append(row)
        for precision in PRECISIONS:
            folder = directory / precision
            folder.mkdir()
            document, emission, _ = probe_emission(case['batch'], case['sequence'], precision)
            assessment = compiler.assess(document)
            codes = {f.code for f in assessment.findings if f.blocks_lowering}
            if ((precision == 'tf32' and codes != {'TARGET_INSTRUCTION_UNSUPPORTED'})
                    or (precision == 'ieee' and not assessment.lowering_eligible)):
                raise ValueError('original production precision admission differs')
            write(folder / 'production-findings.json', assessment.findings)
            (folder / 'candidate.cake.py').write_text(angle_source(case['batch'], case['sequence'], precision))
            payload = emission.source.encode()
            (folder / 'kernel.triton.py').write_bytes(payload)
            requirements = {'compiler': 'triton', 'source_language': 'python', 'target': TARGET,
                            **emission.toolchain}
            write(folder / 'requirements.json', requirements)
            entry = {'precision': precision, 'native_compiled': False}
            row['precisions'].append(entry)
            if isolated is not None:
                submission = CandidateSubmission.seal(OpenCakeEnvironment.media_type, canonical_json_bytes(document))
                request = BuildRequest(submission.sha256, payload, 'lowered_source', sha256(payload).hexdigest(),
                                       TARGET, requirements['kernel_entry_point'], requirements)
                builder = TritonToolchainBuilder(workload=workload, case_id='primary', isolated_compiler=isolated)
                candidate = builder.build_stage(request, stage_abi(case))
                for role, data in candidate.artifact_payloads.items():
                    (folder / (role + '.bin')).write_bytes(data)
                write(folder / 'candidate.json', candidate_identity(candidate))
                entry['native_compiled'] = True
    result['passed'] = True


def bound_candidates(built, commit, problem, policy):
    """Bind original case and exact stage bytes before any device allocation."""
    from open_cake_ir.evaluation.core import TensorLaunchManifest
    from open_cake_ir.evaluation.paired import candidate_from_identity
    from open_cake_ir.evaluation.program import check_triton_launch_record
    from open_cake_ir.evaluation.loaders import check_candidate_authority
    from open_cake_ir.compiler.metax_toolchain import native_pointer_parameters
    from open_cake_ir.lab import CandidateSubmission, OpenCakeEnvironment
    from open_cake_ir.serialization import canonical_json_bytes
    from open_cake_ir.tasks.c550_bench.binding import BENCH_COMMIT
    from open_cake_ir.tasks.evaluate import _input_path
    built = built.resolve(strict=True)
    policy = hypothesis_policy(policy)
    cases = select_cases(problem)
    result = json.loads(_input_path(built, 'result.json', 'build result').read_text())
    expected_rows = [{**case, 'precisions': [{'precision': p, 'native_compiled': True}
                                          for p in PRECISIONS]} for case in cases]
    if (not commit or result.get('source_commit') != commit or result.get('target') != TARGET
            or result.get('passed') is not True or result.get('cases') != expected_rows):
        raise ValueError('all six native angle builds must belong to this exact source and original cases')
    expected_binding = {'bench_commit': BENCH_COMMIT, 'task': TASK, 'cases': cases,
        'oracle_numerics': policy, 'precisions': list(PRECISIONS),
        'scope': 'angle stage only; original RoPE Workload remains the parent authority'}
    if json.loads(_input_path(built, 'binding.json', 'probe binding').read_text()) != expected_binding:
        raise ValueError('retained original case or oracle policy binding differs')
    bound = []
    for case in cases:
        directory = built / case['workload_uuid']
        if directory.is_symlink() or directory.resolve(strict=True) != directory:
            raise ValueError('angle case directory must not be a symlink or escape its build root')
        workload = original_workload(problem, case, policy)
        if json.loads(_input_path(directory, 'original-workload.json', 'original Workload').read_text()) != workload.document:
            raise ValueError('retained original Workload differs')
        routes = []
        for precision in PRECISIONS:
            folder = directory / precision
            if folder.is_symlink():
                raise ValueError('angle artifact directory must not be a symlink')
            identity = json.loads(_input_path(folder, 'candidate.json', 'candidate').read_text())
            payloads = {role: _input_path(folder, role + '.bin', role).read_bytes()
                        for role in identity['artifact_roles']}
            candidate = candidate_from_identity(identity, payloads)
            manifest = TensorLaunchManifest.from_dict(json.loads(payloads['launch_manifest']))
            check_candidate_authority(candidate, payloads['mcfatbin'], 'mcfatbin', manifest)
            document, emission, target = probe_emission(case['batch'], case['sequence'], precision)
            submission = CandidateSubmission.seal(OpenCakeEnvironment.media_type, canonical_json_bytes(document))
            requirements = {'compiler': 'triton', 'source_language': 'python', 'target': TARGET, **emission.toolchain}
            if (json.loads(_input_path(folder, 'requirements.json', 'requirements').read_text()) != requirements
                    or candidate.candidate_sha256 != submission.sha256
                    or payloads.get('lowered_source') != emission.source.encode()
                    or candidate.entry_point != emission.toolchain['kernel_entry_point']
                    or manifest.workload_sha256 != workload.canonical_sha256 or manifest.case_id != 'primary'
                    or manifest.tensor_abi != stage_abi(case) or manifest.target != TARGET
                    or list(manifest.grid) != emission.toolchain['grid']
                    or manifest.block != (emission.toolchain['compile_options']['num_warps'] * target.warp_size, 1, 1)
                    or manifest.aligned_variant or manifest.pointer_alignments):
                raise ValueError('retained angle stage differs from its current original-case emission')
            manifest.check_complete_domain()
            check_triton_launch_record(candidate, manifest, candidate.artifact_roles['lowered_source'])
            hidden = native_pointer_parameters(payloads['mcfatbin'], target.architecture,
                                               manifest.kernel_name) - len(manifest.tensor_abi)
            if hidden not in (0, 2) or hidden != manifest.hidden_null_pointer_parameters:
                raise ValueError('angle stage pointer ABI differs from the native kernel')
            routes.append((precision, candidate, manifest))
        bound.append((case, tuple(routes)))
    return tuple(bound)


def evaluate(args, result):
    from open_cake_ir.evaluation.local_broker import admit_local_job
    from open_cake_ir.evaluation.triton_metax import observe_local_metax
    from open_cake_ir.evaluation.metax_driver import LoadedMetaxCandidate
    from open_cake_ir.lab.executor import ExecutorRevision
    from open_cake_ir.tasks.c550_bench.binding import BenchProblem, require_oracle_numerics
    policy = numeric_policy(args.oracle_numerics)
    problem = BenchProblem.open(args.bench_root, TASK)
    candidates = bound_candidates(args.built, result['source_commit'], problem, policy)
    host = ExecutorRevision.for_target(ROOT, TARGET).admit_host()
    require_oracle_numerics(policy, phase='before_angle_probe_allocation')
    job = admit_local_job('maca', device=args.physical_device, runtime_device=args.runtime_device,
                         expected_pci=args.expected_pci, lock_scope='device', queue_seconds=300)
    admission = observe_local_metax(TARGET, runtime_library=host['runtime_library'])
    import torch
    result.update(job_id=job, admission=asdict(admission), device_execution=True,
                  oracle_numerics=policy, cases=[])
    lock = os.fstat(int(os.environ['METAL_BROKER_LOCK_FD']))
    legacy = os.fstat(int(os.environ['OPEN_CAKE_LOCAL_LEGACY_FD']))
    result['lease_binding'] = {'physical_device': args.physical_device, 'runtime_device': args.runtime_device,
        'expected_pci': args.expected_pci, 'lock_scope': 'device',
        'device_lock': {'device': lock.st_dev, 'inode': lock.st_ino, 'uid': lock.st_uid},
        'legacy_lock': {'device': legacy.st_dev, 'inode': legacy.st_ino, 'uid': legacy.st_uid}}
    def raw(value):
        return value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    for case, routes in candidates:
        directory = args.output / case['workload_uuid']
        directory.mkdir()
        arguments, _, actual_admission = problem.original_inputs_on_target(case['workload_uuid'],
            runtime_library=host['runtime_library'], oracle_numerics=policy)
        if actual_admission != admission:
            raise ValueError('original factory changed the device lease')
        values = dict(zip(problem.definition.inputs, arguments, strict=True))
        positions, frequencies = values['position_ids'], values['inv_freq']
        if (tuple(positions.shape) != (case['batch'], case['sequence']) or positions.dtype != torch.int64
                or tuple(frequencies.shape) != (64,) or frequencies.dtype != torch.float32
                or not positions.is_contiguous() or not frequencies.is_contiguous()
                or type(values['attention_scaling']) is not float or values['attention_scaling'] != 1.0):
            raise ValueError('original factory tensor or scalar ABI differs')
        inputs = {'position_ids': raw(positions), 'inv_freq': raw(frequencies)}
        for name, data in inputs.items():
            (directory / (name + '.input.bin')).write_bytes(data)
        require_oracle_numerics(policy, phase='before_original_angle_matmul')
        try:
            reference = (frequencies[None, :, None].float().expand(case['batch'], -1, 1)
                         @ positions[:, None, :].float()).transpose(1, 2)
            torch.cuda.synchronize()
        finally:
            require_oracle_numerics(policy, phase='after_original_angle_matmul')
        expected = raw(reference)
        (directory / 'original-high.angles.fp32').write_bytes(expected)
        if any(raw(values[name]) != data for name, data in inputs.items()):
            raise ValueError('original angle reference changed input bits')
        views = (positions.unsqueeze(-1), frequencies.unsqueeze(-1))
        for original, view in zip((positions, frequencies), views, strict=True):
            if (view.data_ptr() != original.data_ptr() or view.storage_offset() != original.storage_offset()
                    or view.numel() != original.numel() or not view.is_contiguous()):
                raise ValueError('candidate singleton view changed original input storage')
        row = {**case, 'reference_inputs_unchanged': True, 'routes': []}
        result['cases'].append(row)
        for precision, candidate, manifest in routes:
            loaded = LoadedMetaxCandidate.load(candidate, manifest, admission)
            primary = None
            observation = {'precision': precision, 'comparison': None}
            try:
                output = torch.full((case['batch'], case['sequence'], 64), float('nan'),
                                    dtype=torch.float32, device='cuda:0')
                require_oracle_numerics(policy, phase='before_' + precision + '_candidate')
                loaded.launch((*views, output), tensor_contract=manifest,
                              stream=int(torch.cuda.current_stream().cuda_stream))
                torch.cuda.synchronize()
                observed = raw(output)
                (directory / (precision + '.angles.fp32')).write_bytes(observed)
                observation['comparison'] = compare_angles(expected, observed, case['batch'], case['sequence'])
                observation['inputs_unchanged'] = all(raw(values[name]) == data for name, data in inputs.items())
                require_oracle_numerics(policy, phase='after_' + precision + '_candidate')
                if not observation['inputs_unchanged']:
                    raise ValueError('candidate changed original input bits')
            except BaseException as error:
                primary = error
            finally:
                try:
                    loaded.close(synchronize=torch.cuda.synchronize, primary=primary)
                except BaseException as error:
                    primary = error
                observation.update(kernel_calls=loaded.launch_calls, module_closed=loaded.closed,
                                   resources=loaded.resources)
                row['routes'].append(observation)
                write(directory / (precision + '.json'), observation)
            if primary is not None:
                raise primary
            if loaded.launch_calls != 1 or not loaded.closed:
                raise ValueError('angle native call count or module teardown differs')
        write(directory / 'result.json', row)
    result['comparison_completed'] = True
    require_oracle_numerics(policy, phase='after_angle_probe')
    result['tf32_matches_original_high'] = all(row['routes'][0]['comparison']['bitwise_equal']
                                               for row in result['cases'])
    result['passed'] = result['tf32_matches_original_high']


def main():
    from open_cake_ir.lab.custody import admit_new_campaign_path
    from open_cake_ir.source_identity import checkout_commit
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
        command.add_argument('--bench-root', type=Path, required=True)
        command.add_argument('--oracle-numerics', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    commit = checkout_commit(ROOT)
    if commit is None or os.environ.get('METAL_BROKER_LOCK_FD') or os.environ.get('GPUQ_JOB_ID'):
        parser.error('angle probe needs clean committed source and starts outside an allocation')
    args.output = admit_new_campaign_path(ROOT, args.output, role='RoPE angle qualification')
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'source_commit': commit, 'target': TARGET, 'passed': False,
              'device_execution': False, 'target_admission_changed': False,
              'scope': 'original-domain angle comparison only; no full RoPE, timing or optimization qualification'}
    try:
        (build if args.command == 'build' else evaluate)(args, result)
    except Exception as error:
        result.update(error=str(error), failure_class=type(error).__name__)
        for role, data in getattr(error, 'artifact_payloads', {}).items():
            if isinstance(role, str) and role.isidentifier() and isinstance(data, bytes):
                (args.output / ('failure-' + role + '.bin')).write_bytes(data)
        (args.output / 'failure.txt').write_text(traceback.format_exc())
    write(args.output / 'result.json', result)
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
