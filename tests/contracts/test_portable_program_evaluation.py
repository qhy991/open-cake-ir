"""Complete non-CUDA Programs through real software boundaries, CPU substitutes only."""
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from open_cake_ir.compiler import Compiler, Program, frontend
from open_cake_ir.compiler.toolchain import TritonCompilation, triton_route
from open_cake_ir.evaluation.core import EvaluationProtocol, load_torch_program, _MODULE_LOADERS
from open_cake_ir.evaluation.paired import candidate_identity, validate_receipt_policy
from open_cake_ir.evaluation.platforms import platform_for
from open_cake_ir.evaluation.program import (
    admit_program_execution, program_components, seal_program_candidate, stage_abi)
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.lab import CandidateSubmission, OpenCakeEnvironment, TritonToolchainBuilder
from open_cake_ir.lab.build import BuildRequest
from open_cake_ir.tasks import evaluate as worker
from open_cake_ir.tasks.normalization.study import evaluation_policy
from open_cake_ir.tasks.tiles.evaluation import evaluate_tile_workload
from open_cake_ir.tasks.workloads import create_task
from tests.contracts.test_epilogue_fusion import execute
from tests.contracts.test_program_evaluation import replay_program_candidate
from tests.contracts.test_qsa_common_program import Torch

ROOT = Path(__file__).resolve().parents[2]
BACKENDS = ('triton-gfx1151', 'triton-dcu', 'triton-metax')


class NativeCompiler:
    """Return explicit ABI metadata; these bytes are never sent to a GPU."""
    def __init__(self): self.requests = []

    def compile(self, source, requirements):
        self.requests.append((source, requirements))
        route = triton_route(requirements)
        artifacts = {role: b'CPU fixture' for role in route.artifact_roles}
        artifacts['source'] = b'# CPU compiler expansion\n' + source
        artifacts[route.binary_role] = b'\x7fELF CPU fixture, not device code'
        count = len(requirements['signature'])
        if route.gpu_backend == 'hip':
            # The two hidden slots are part of this compiled fixture's metadata,
            # not a lane-width or vendor-derived guess in the Program builder.
            entries = '\n'.join('      - .address_space: global\n'
                f'        .offset: {index * 8}\n        .size: 8\n'
                '        .value_kind: global_buffer' for index in range(count + 2))
            artifacts['amdgcn'] = ('\t.amdgpu_metadata\n---\namdhsa.kernels:\n  - .args:\n'
                f'{entries}\n    .kernarg_segment_size: {(count + 2) * 8}\n    .name: k\n...\n'
                '\t.end_amdgpu_metadata\n').encode()
        else:
            parameters = ', '.join(f'%arg{index}: !tt.ptr<f32> ' for index in range(count))
            artifacts['ttgir'] = f'tt.func public @k({parameters}) attributes {{}}'.encode()
        return TritonCompilation(source, requirements['target'], requirements['kernel_entry_point'],
            artifacts, requirements['compile_options']['num_warps'] * requirements['warp_size'],
            0, 'CPU fixture', route.code_object.value)


def task_program(backend):
    document, source = create_task('softsign', backend=backend, rows=2, columns=8)
    workload = WorkloadContract(document)
    first = frontend.parse(source).document
    copy = frontend.parse(f'''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="copy-output", target="{workload.target}", backend="triton", entry_point="copy_output")
def candidate(lm, x: cake.Tensor((2,8), "fp32"), out: cake.Tensor((2,8), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0])
    row = lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
        values = lm.load(x[row, :], id="load")
        lm.store(out[row, :], values, coalesced=False, id="store")
''').document
    program = Program.from_dict({'schema_version': 1, 'program_id': 'portable-softsign',
        'target': workload.target, 'inputs': ['x'], 'outputs': ['out'],
        'tensors': {'x': {'shape': [2,8], 'dtype': 'fp32'},
                    'out': {'shape': [2,8], 'dtype': 'fp32'},
                    'middle': {'shape': [1,2,8], 'dtype': 'fp32'}},
        'stages': [{'name': 'softsign', 'schedule': first,
                    'bindings': {'x': 'x', 'out': {'tensor': 'middle', 'view': 'singleton_axes'}}},
                   {'name': 'copy', 'schedule': copy,
                    'bindings': {'x': {'tensor': 'middle', 'view': 'singleton_axes'}, 'out': 'out'}}]})
    return workload, program


class PortableProgramEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT/'compiler/revision.json')

    def build(self, backend, *, single=False):
        workload, program = task_program(backend)
        if single:
            program = Program.from_schedule(program.document['stages'][0]['schedule'])
        lowered = self.compiler.lower_program(program)
        compilation = NativeCompiler()
        builder = TritonToolchainBuilder(workload=workload, case_id='primary', isolated_compiler=compilation)
        identity = sha256(program.document_bytes).hexdigest()
        children = {}
        for stage, lowering in zip(program.stages, lowered.lowerings, strict=True):
            request = BuildRequest(identity, lowering.source.encode(), 'lowered_source', lowering.source_sha256,
                lowering.target, lowering.route.entry_point, lowering.toolchain_requirements)
            children[stage.name] = builder.build_stage(request, stage_abi(stage))
        candidate = seal_program_candidate(lowered, children, candidate_sha256=identity,
                                           workload=workload, case_id='primary')
        return workload, program, candidate, compilation

    def load(self, program, candidate, *, broken_last=False, torch=None):
        torch = torch or Torch()
        manifest, children, _ = program_components(candidate)
        kernels, calls = [], []
        by_entry = {children[stage.name].entry_point: stage for stage in program.stages}
        class Kernel:
            closed = False
            launch_calls = 0
            resources = {'CPU_fixture': True}
            def __init__(self, stage): self.stage = stage
            def launch(self, args, *, tensor_contract, stream):
                self.launch_calls += 1; calls.append(self.stage.name)
                if broken_last and self.stage.name == program.stages[-1].name: return
                inputs = {name: arg.data for (name, _, _, mode), arg in zip(tensor_contract.tensor_abi, args, strict=True)
                          if mode == 'input'}
                outputs, _ = execute(json.loads(self.stage.schedule_bytes), inputs)
                for (name, _, _, mode), arg in zip(tensor_contract.tensor_abi, args, strict=True):
                    if mode == 'output': arg.data[:] = outputs[name]
            def close(self, *, synchronize): synchronize(); self.closed = True
        def loader(bound, *args):
            kernel = Kernel(by_entry[bound.entry_point]); kernels.append(kernel); return kernel
        native = ('open_cake_ir.evaluation.hip_driver.LoadedHipModuleCandidate.load'
                  if candidate.target.startswith('gfx') else
                  'open_cake_ir.evaluation.metax_driver.LoadedMetaxCandidate.load')
        admission = SimpleNamespace(device_arch=candidate.target)
        arguments = [torch.full(shape, 0., dtype=dtype, device='cuda:0')
                     for _, shape, dtype, _ in manifest.tensor_abi]
        with patch.dict('sys.modules', {'torch': torch}), patch(native, side_effect=loader) as native_load:
            loaded, tensors = load_torch_program(candidate, manifest, arguments, admission,
                _MODULE_LOADERS[platform_for(candidate.target).code_object])
        self.assertEqual(native_load.call_count, len(program.stages))
        return torch, loaded, manifest, arguments, tensors, kernels, calls

    def assay(self, workload, program, candidate, *, broken_last=False):
        torch, loaded, manifest, arguments, tensors, kernels, calls = self.load(program, candidate, broken_last=broken_last)
        class Launcher:
            def launch_tensors(self, bound, spec, values):
                arguments[0].data[:] = values['x']
                arguments[1].data[:] = [float('nan')] * len(arguments[1].data)
                loaded.launch(arguments, tensor_contract=spec, stream=7)
                return {'out': arguments[1].data[:]}, {'x': arguments[0].data[:]}, {
                    'candidate_sha256': bound.candidate_sha256, 'kernel_calls': loaded.launch_calls,
                    'fallback_calls': 0}
        try:
            with patch.dict('sys.modules', {'torch': torch}):
                receipt = evaluate_tile_workload(candidate, workload,
                    EvaluationProtocol('portable-cpu-fixture', 'confirmatory', workload.canonical_sha256,
                                       'primary', 'none'), Launcher())
                validate_receipt_policy(receipt, {}, None, candidate)
            if len(program.stages) > 1: self.assertEqual(tensors['middle'].shape, (1,2,8))
            return receipt, calls
        finally:
            loaded.close(synchronize=lambda: None)
            self.assertTrue(all(kernel.closed for kernel in kernels))

    def test_complete_program_and_single_stage_share_external_oracle_and_replay(self):
        for backend in BACKENDS:
            for single in (False, True):
                with self.subTest(backend=backend, single=single):
                    workload, program, candidate, compilation = self.build(backend, single=single)
                    replayed = replay_program_candidate(self.compiler, program, candidate, candidate.artifact_payloads)
                    self.assertEqual(replayed.canonical_sha256, candidate.canonical_sha256)
                    _, _, children = program_components(candidate)
                    hidden = 0 if backend == 'triton-metax' else 2
                    self.assertTrue(all(child.hidden_null_pointer_parameters == hidden for child in children.values()))
                    receipt, calls = self.assay(workload, program, candidate)
                    self.assertTrue(receipt.correctness_passed)
                    self.assertIsNone(receipt.timing)
                    self.assertEqual(calls, [stage.name for stage in program.stages])
                    self.assertEqual(receipt.kernel_calls, len(program.stages))
                    self.assertEqual(len(compilation.requests), len(program.stages))

    def test_a_stage_that_leaves_output_unwritten_fails_the_original_task_oracle(self):
        for backend in BACKENDS:
            workload, program, candidate, _ = self.build(backend)
            with self.subTest(backend=backend):
                receipt, calls = self.assay(workload, program, candidate, broken_last=True)
                self.assertFalse(receipt.correctness_passed)
                self.assertGreater(receipt.correctness['output_mismatches'], 0)
                self.assertEqual(calls, ['softsign', 'copy'])

    def test_wrong_dtype_device_shape_storage_or_view_refuses_before_first_dispatch(self):
        _, program, candidate, _ = self.build('triton-gfx1151')
        mutations = ('dtype', 'device', 'shape', 'storage', 'view', 'current_device')
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                torch, loaded, manifest, args, tensors, kernels, calls = self.load(program, candidate)
                try:
                    if mutation == 'dtype': args[0].dtype = 'int32'  # Same size, different ABI.
                    elif mutation == 'device': args[0].device = 'cpu'
                    elif mutation == 'shape': args[0].shape = (16,)
                    elif mutation == 'storage': args[0].address += 256
                    elif mutation == 'view': loaded._prepared[id(args)][3]['copy'][0].address += 256
                    else: torch.cuda = SimpleNamespace(current_device=lambda: 1)
                    with patch.dict('sys.modules', {'torch': torch}), self.assertRaisesRegex(ValueError, 'Program'):
                        loaded.launch(args, tensor_contract=manifest, stream=7)
                    self.assertEqual(calls, [])
                finally: loaded.close(synchronize=lambda: None)
                self.assertTrue(all(kernel.closed for kernel in kernels))

    def test_invalid_public_tensors_refuse_before_module_loading(self):
        _, _, candidate, _ = self.build('triton-dcu')
        manifest, _, _ = program_components(candidate)
        for field, value in (('dtype', 'int32'), ('device', 'cpu'), ('shape', (16,))):
            torch = Torch()
            arguments = [torch.full(shape, 0., dtype=dtype, device='cuda:0')
                         for _, shape, dtype, _ in manifest.tensor_abi]
            setattr(arguments[0], field, value)
            loader = Mock()
            with patch.dict('sys.modules', {'torch': torch}), self.assertRaisesRegex(ValueError, 'tensor shape'):
                load_torch_program(candidate, manifest, arguments, None, loader)
            loader.assert_not_called()

    def test_unsupported_measurement_refuses_candidate_or_baseline_before_device_admission(self):
        for backend in BACKENDS:
            workload, program, candidate, _ = self.build(backend)
            _, children, _ = program_components(candidate)
            single = next(iter(children.values()))
            evaluate = worker._evaluate_metax_candidate if backend == 'triton-metax' else worker._evaluate_hip_candidate
            for as_baseline in (False, True):
                for purpose, timed in (('search', True), ('attribution', False)):
                    authority = SimpleNamespace(candidate=single if as_baseline else candidate,
                        baseline=candidate if as_baseline else single, request={'purpose': purpose}, executor=Mock())
                    with self.subTest(backend=backend, baseline=as_baseline, purpose=purpose), \
                         patch('open_cake_ir.evaluation.triton_hip.observe_local_hip') as hip, \
                         patch('open_cake_ir.evaluation.triton_metax.observe_local_metax') as maca, \
                         self.assertRaisesRegex(ValueError, 'ordered Program (timing|attribution)'):
                        evaluate(authority, {}, collect_timing=timed)
                    hip.assert_not_called(); maca.assert_not_called(); authority.executor.admit_host.assert_not_called()
            receipt = SimpleNamespace(kernel_calls=candidate.kernels_per_call, purpose='search', timing=None)
            with self.assertRaisesRegex(ValueError, 'ordered Program timing'):
                validate_receipt_policy(receipt, evaluation_policy(workload), candidate_identity(single), candidate)
            receipt.kernel_calls = 1
            with self.assertRaisesRegex(ValueError, 'ordered Program timing'):
                validate_receipt_policy(receipt, evaluation_policy(workload), candidate_identity(candidate), single)
            toolchain = Mock()
            environment = OpenCakeEnvironment(self.compiler, toolchain, workload=workload, case_id='primary',
                authority_document={'lowering_route': program.document['stages'][0]['schedule']['lowering']})
            result = environment.build(CandidateSubmission.seal(environment.media_type, program.document_bytes))
            self.assertNotEqual(result.disposition, 'launchable')
            self.assertIn('one dispatch', str(result.feedback))
            toolchain.build_stage.assert_not_called()

    def test_metal_composition_still_refuses_without_a_native_program_adapter(self):
        with self.assertRaisesRegex(ValueError, 'metal_binary_archive'):
            admit_program_execution('apple_gpu_family8')
