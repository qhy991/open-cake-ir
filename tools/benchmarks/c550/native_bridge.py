"""Sealed baseline binding and original-Bench native calls, without an oracle."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from open_cake_ir.compiler import Compiler, Target, Program
from open_cake_ir.evaluation.core import TensorLaunchManifest, _TORCH_DTYPE_NAMES
from open_cake_ir.evaluation.loaders import check_candidate_authority, LifecycleError
from open_cake_ir.evaluation.program import (
    ProgramLaunchManifest, program_components, check_triton_launch_record,
    single_kernel_lowering)
from open_cake_ir.lab import CandidateSubmission, OpenCakeEnvironment
from open_cake_ir.lab.bindings import external_file, load_prepared_baseline
from open_cake_ir.lab.python_reference import parse_skeleton
from open_cake_ir.lab.workload_binding import bind_program_workload, bind_schedule_workload
from open_cake_ir.serialization import canonical_json_bytes
from open_cake_ir.tasks.c550_bench.binding import (
    BENCH_COMMIT, TARGET, physical_input_view, validate_input_view_observation,
    validate_oracle_numerics, require_oracle_numerics)
from open_cake_ir.tasks.c550_bench.workload import BenchWorkload

ROOT = Path(__file__).resolve().parents[3]


def _file(root, value, context):
    """Retain the existing external-file rules and constrain the owning folder."""
    root = root.resolve(strict=True)
    path = Path(value)
    path = path if path.is_absolute() else root / path
    result = external_file(ROOT, str(path), context)
    if root not in result.parents:
        raise ValueError(f'{context} escapes its owning folder')
    return result


@dataclass(frozen=True)
class BoundCase:
    uuid: str
    workload: BenchWorkload
    candidate: object
    manifest: object


def _leaf(candidate, manifest, lowering, target):
    check_candidate_authority(candidate, candidate.artifact_payloads['mcfatbin'], 'mcfatbin', manifest)
    requirements = lowering.toolchain_requirements
    if (candidate.target != TARGET or candidate.entry_point != requirements['kernel_entry_point']
            or candidate.artifact_payloads.get('lowered_source') != lowering.source.encode()
            or list(manifest.grid) != list(requirements['grid'])
            or manifest.block != (requirements['compile_options']['num_warps'] * target.warp_size, 1, 1)
            or manifest.aligned_variant or manifest.pointer_alignments):
        raise ValueError('prepared native leaf differs from its source or launch')
    if 'stage_compilation' in candidate.artifact_payloads:
        check_triton_launch_record(candidate, manifest, lowering.source_sha256)
    from open_cake_ir.compiler.metax_toolchain import native_pointer_parameters
    hidden = native_pointer_parameters(candidate.artifact_payloads['mcfatbin'], target.architecture,
                                       manifest.kernel_name) - len(manifest.tensor_abi)
    if hidden not in (0, 2) or manifest.hidden_null_pointer_parameters != hidden:
        raise ValueError('prepared native pointer ABI differs')
    from open_cake_ir.lab.build import compiled_allocation_feedback
    compiled_allocation_feedback(candidate)


def bind_case(problem, uuid, prepared_path, compiler, views):
    """Read the existing baseline-only handoff; never rebuild or reselect it."""
    prepared_path = external_file(ROOT, str(prepared_path), 'prepared baseline')
    workspace = prepared_path.parent
    gate = json.loads(_file(workspace, 'compiler-gate.json', 'compiler gate').read_text())
    saved = json.loads(_file(workspace, 'workload.json', 'baseline Workload').read_text())
    numerics = validate_oracle_numerics(saved.get('semantics', {}).get('oracle_numerics'))
    expected = problem.workload_document(uuid, input_views=views, oracle_numerics=numerics)
    if saved != expected:
        raise ValueError('prepared baseline differs from the original Workload or qualified views')
    workload = BenchWorkload(saved)
    source = _file(workspace, 'starter.py', 'baseline source').read_text()
    skeleton = parse_skeleton(source, filename='baseline-starter.py')
    envelope = ({'python_program_source': source, 'program_id': skeleton['program_id']}
                if 'program_id' in skeleton else {'python_source': source})
    submission = CandidateSubmission.seal(OpenCakeEnvironment.media_type, canonical_json_bytes(envelope))
    handoff = json.loads(prepared_path.read_text())
    _file(workspace, handoff['fixed_baseline_bundle_path'], 'baseline bundle')
    _, candidate, _ = load_prepared_baseline(ROOT, prepared_path)
    if candidate.candidate_sha256 != submission.sha256 or candidate.target != TARGET:
        raise ValueError('prepared baseline source or target differs from its sealed selection')
    target = Target.load(ROOT / 'compiler/targets/xcore1002.json')
    if 'program_id' in skeleton:
        program = bind_program_workload(Program.from_dict(skeleton), workload.canonical_sha256)
        lowered = compiler.lower_program(program)
        single = single_kernel_lowering(lowered)
        if single is None:
            manifest, children, child_manifests = program_components(candidate)
            if manifest.program.document != program.document or manifest.aligned_stages:
                raise ValueError('prepared Program graph differs from its source')
            for stage, lowering in zip(program.stages, lowered.lowerings, strict=True):
                _leaf(children[stage.name], child_manifests[stage.name], lowering, target)
        else:
            if candidate.is_program:
                raise ValueError('prepared single-stage projection differs')
            manifest = TensorLaunchManifest.from_dict(json.loads(candidate.artifact_payloads['launch_manifest']))
            _leaf(candidate, manifest, single, target)
        revision = lowered.compiler_revision_id
    else:
        if candidate.is_program:
            raise ValueError('prepared Schedule became a different Program')
        assessment = compiler.assess(bind_schedule_workload(skeleton, workload.canonical_sha256))
        if not assessment.lowering_eligible:
            raise ValueError('prepared Schedule is not admitted by the current Compiler')
        public = tuple((buffer.name, buffer.shape, buffer.dtype.value, buffer.mode.value)
                       for buffer in assessment.typed_schedule.buffers if buffer.space.value == 'global')
        expected_abi = tuple((arg.name, tuple(arg.shape), arg.dtype, arg.mode)
                             for arg in workload.tensor_abi('primary'))
        if public != expected_abi:
            raise ValueError('prepared Schedule public ABI differs from the original Workload')
        lowering = compiler.lower(assessment)
        manifest = TensorLaunchManifest.from_dict(json.loads(candidate.artifact_payloads['launch_manifest']))
        _leaf(candidate, manifest, lowering, target)
        revision = lowering.compiler_revision_id
    manifest.check_workload(workload, 'primary')
    if gate.get('passed') is not True or gate.get('compiler_revision_id') != revision:
        raise ValueError('prepared baseline requires the same fixed Compiler source')
    return BoundCase(uuid, workload, candidate, manifest)


def bind_all(problem, index_path, *, input_views=None, compiler=None):
    """Bind all original UUIDs and immutable payloads before any allocation."""
    compiler = Compiler.load(ROOT) if compiler is None else compiler
    from open_cake_ir.source_identity import checkout_commit
    if compiler.commit is None or compiler.commit != checkout_commit(ROOT):
        raise ValueError('Bench bridge requires clean committed source')
    index_path = external_file(ROOT, str(Path(index_path)), 'baseline locator index')
    index = json.loads(index_path.read_text())
    expected = [row.uuid for row in problem.workloads]
    if (len(expected) != 16 or len(set(expected)) != 16 or not isinstance(index, dict)
            or set(index) != {'bench_commit', 'task', 'cases'} or index['bench_commit'] != BENCH_COMMIT
            or index['task'] != problem.task_id or not isinstance(index['cases'], list)
            or any(not isinstance(row, dict) or set(row) != {'uuid', 'prepared_baseline'} for row in index['cases'])
            or [row['uuid'] for row in index['cases']] != expected):
        raise ValueError('baseline locators must cover all sixteen ordered original UUIDs')
    qualified = (validate_input_view_observation(problem, input_views) if input_views is not None else {})
    cases = []
    for row in index['cases']:
        path = external_file(ROOT, row['prepared_baseline'], 'prepared baseline locator')
        saved = json.loads(_file(path.parent, 'workload.json', 'baseline Workload').read_text())
        views = saved['semantics']['input_views']
        if any(order != list(range(len(order))) for order in views.values()) and row['uuid'] not in qualified:
            raise ValueError('nonidentity physical views require the original complete qualification observation')
        if qualified and views != qualified[row['uuid']]:
            raise ValueError('prepared input views differ from the qualified original observation')
        cases.append(bind_case(problem, row['uuid'], path, compiler, qualified.get(row['uuid'])))
    policies = [case.workload.document['semantics']['oracle_numerics'] for case in cases]
    if any(policy != policies[0] for policy in policies):
        raise ValueError('one original Bench loop cannot switch oracle numeric policies')
    return tuple(cases)


class NativeBench:
    """Only the pinned Bench all/ten-round callback order selects a case here."""
    def __init__(self, cases, rounds, admission, trace, *, torch_module=None, loader=None):
        if len(cases) != 16 or rounds != 10 or len({case.uuid for case in cases}) != 16:
            raise ValueError('native bridge requires original all/rounds=10 scope')
        if torch_module is None:
            import torch as torch_module
        if loader is None:
            from open_cake_ir.evaluation.metax_driver import LoadedMetaxCandidate
            loader = LoadedMetaxCandidate.load
        self.cases, self.rounds, self.admission = cases, rounds, admission
        self.torch, self.loader, self.trace = torch_module, loader, trace
        self.calls = 0

    def _check(self, value, name, shape, dtype):
        expected = getattr(self.torch, _TORCH_DTYPE_NAMES[dtype])
        if (tuple(value.shape) != tuple(shape) or value.dtype != expected
                or str(value.device) != 'cuda:0' or not value.is_contiguous()
                or self.torch.cuda.current_device() != 0):
            raise ValueError(f'native Bench tensor {name} differs from its sealed ABI')

    def run(self, *original):
        from open_cake_ir.evaluation.core import load_torch_program, _output_poison
        if self.calls >= len(self.cases) * self.rounds:
            raise ValueError('original Bench made an unexpected extra callback')
        case = self.cases[self.calls // self.rounds]
        ordinal = self.calls
        self.calls += 1
        semantics = case.workload.document['semantics']
        names = semantics['ordered_original_inputs']
        abi = case.workload.tensor_abi('primary')
        input_abi = [row for row in abi if row.mode == 'input']
        output_abi = [row for row in abi if row.mode == 'output']
        physical = []
        loaded = None
        failure = None
        unchanged = None
        calls = 0
        outputs = []
        try:
            require_oracle_numerics(semantics.get('oracle_numerics'), phase='native Bench callback after original reference')
            if len(original) != len(names):
                raise ValueError('original Bench input count differs')
            values = dict(zip(names, original, strict=True))
            for name, scalar in semantics['fixed_scalar_inputs'].items():
                actual = values[name]
                allowed = (bool,) if scalar['dtype'] == 'bool' else (int,) if scalar['dtype'].startswith('int') else (int, float)
                if type(actual) not in allowed or actual != scalar['value']:
                    raise ValueError(f'original Bench scalar {name} differs')
            for row in input_abi:
                value = values[row.name]
                if tuple(value.shape) != tuple(semantics['original_tensor_shapes'][row.name]):
                    raise ValueError(f'original Bench logical shape {row.name} differs')
                view = physical_input_view(value, semantics['input_views'][row.name])
                self._check(view, row.name, row.shape, row.dtype)
                physical.append(view)
            before = [value.view(self.torch.uint8).cpu().clone() for value in physical]
            for row in output_abi:
                dtype = getattr(self.torch, _TORCH_DTYPE_NAMES[row.dtype])
                outputs.append(self.torch.full(row.shape, _output_poison(row.dtype), dtype=dtype, device='cuda:0'))
            arguments = tuple(physical + outputs)
            spans = [(value.data_ptr(), value.data_ptr()+value.numel()*value.element_size()) for value in arguments]
            for index in range(len(physical), len(arguments)):
                start,end = spans[index]
                if any(start<other_end and other_start<end for other_start,other_end in spans[:index]):
                    raise ValueError('native Bench output aliases an input or another output')
            if isinstance(case.manifest, ProgramLaunchManifest):
                loaded, _ = load_torch_program(case.candidate, case.manifest, arguments, self.admission, self.loader)
            else:
                loaded = self.loader(case.candidate, case.manifest, self.admission)
            start = loaded.launch_calls
            loaded.launch(arguments, tensor_contract=case.manifest,
                          stream=int(self.torch.cuda.current_stream().cuda_stream))
            self.torch.cuda.synchronize()
            calls = loaded.launch_calls-start
            if calls != case.manifest.kernels_per_call:
                raise ValueError('native Bench stage call count differs')
            unchanged = all(self.torch.equal(value.view(self.torch.uint8).cpu(), old)
                            for value,old in zip(physical,before,strict=True))
            if not unchanged:
                raise ValueError('native Bench candidate changed an original input')
            for index,(row,value) in enumerate(zip(output_abi,outputs,strict=True),len(physical)):
                self._check(value,row.name,row.shape,row.dtype)
                if value.data_ptr()!=spans[index][0]:
                    raise ValueError('native Bench output storage changed')
        except BaseException as error:
            failure = error
        finally:
            if loaded is not None:
                calls = loaded.launch_calls
                try:
                    loaded.close(synchronize=self.torch.cuda.synchronize)
                except BaseException as cleanup:
                    failure = LifecycleError(failure,cleanup) if failure is not None else cleanup
            self.trace({'uuid':case.uuid,'round':ordinal % self.rounds,
                'expected_stage_calls':case.manifest.kernels_per_call,'kernel_calls':calls,
                'input_unchanged':unchanged,'module_closed':loaded.closed if loaded is not None else None,
                'error':str(failure) if failure is not None else None})
        if failure is not None:
            raise failure
        return outputs[0] if len(outputs)==1 else tuple(outputs)


_ACTIVE = None


def run(*inputs):
    if _ACTIVE is None:
        raise ValueError('native Bench adapter must be bound by its qualification command')
    return _ACTIVE.run(*inputs)
