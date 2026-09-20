"""Whole-Program build/oracle/profile contracts with CPU-only device doubles."""
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler, Program, frontend
from open_cake_ir.evaluation.core import EvaluationProtocol, LaunchableCandidate, LoadedTorchTensorCandidate
from open_cake_ir.evaluation.loaders import LifecycleError
from open_cake_ir.compiler.target import CodeObject
from open_cake_ir.evaluation.program import ProgramLaunchManifest, LoadedProgram, program_components
from open_cake_ir.evaluation.kernel_bundle import pack_candidates
from open_cake_ir.evaluation.paired import validate_pair_candidates, candidate_identity, participant_work
from open_cake_ir.evaluation.profiler import NCU_ATTRIBUTION_METRICS, build_ncu_program_profile, load_ncu_program_profile
from open_cake_ir.lab import CandidateSubmission, OpenCakeEnvironment, TritonToolchainBuilder
from open_cake_ir.tasks.tiles.evaluation import evaluate_tile_workload
from open_cake_ir.serialization import canonical_json_bytes
from tests.contracts.test_program_rewrites import epilogue_program
from tests.contracts.test_epilogue_fusion import execute, rounded
from tests.contracts.test_native_triton_pairing import CompilationFixture

ROOT = Path(__file__).resolve().parents[2]


def workload_for(program):
    abi = [SimpleNamespace(name=name, shape=program.tensors[name].shape,
                           dtype=program.tensors[name].dtype.value, mode=mode)
           for mode, names in (('input', program.inputs), ('output', program.outputs)) for name in names]
    return SimpleNamespace(canonical_sha256='1'*64, target=program.target,
        document={'semantics': {'target': program.target, 'candidate_abi': {}},
                  'validation': {'comparison': 'elementwise_atol_rtol', 'atol': 0., 'rtol': 0.}},
        tensor_abi=lambda case: abi)


@dataclass
class Tensor:
    data: list
    shape: tuple
    dtype: str
    address: int


class ProgramEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT/'compiler/revision.json')

    def build(self, document=None):
        document = epilogue_program() if document is None else document
        program = Program.from_dict(document)
        workload = workload_for(program)
        fixture = CompilationFixture()
        builder = TritonToolchainBuilder(workload=workload, case_id='primary', isolated_compiler=fixture)
        environment = OpenCakeEnvironment(self.compiler, builder, workload=workload, case_id='primary',
            authority_document={'lowering_route': {'backend': 'triton', 'entry_point': 'starter'},
                                'input_format': 'schedule_or_python_v1'})
        result = environment.build(CandidateSubmission.seal(environment.media_type, canonical_json_bytes(document)))
        self.assertEqual(result.disposition, 'launchable', result.feedback)
        return result.launchable, workload, fixture

    def loaded(self, candidate, *, failing_stage=None):
        manifest, children, _ = program_components(candidate)
        position = 1000
        def tensor(values, shape, dtype):
            nonlocal position
            position += 10000
            return Tensor(list(values), shape, dtype, position)
        def allocate(spec):
            return tensor([float('nan')]*math.prod(spec.shape), spec.shape, spec.dtype.value)
        calls = []
        kernels = {}
        class Kernel:
            launch_calls = 0
            closed = False
            resources = {'CPU_fixture': True}
            def __init__(self, stage): self.stage = stage
            def launch(self, arguments, *, tensor_contract, stream):
                self.launch_calls += 1; calls.append(self.stage.name)
                if self.stage.name == failing_stage: return
                inputs = {name: arg.data for (name, shape, dtype, mode), arg in zip(tensor_contract.tensor_abi, arguments, strict=True) if mode == 'input'}
                outputs, _ = execute(json.loads(self.stage.schedule_bytes), inputs)
                for (name, shape, dtype, mode), arg in zip(tensor_contract.tensor_abi, arguments, strict=True):
                    if mode == 'output': arg.data[:] = outputs[name]
            def close(self, *, synchronize):
                synchronize(); self.closed = True
        by_entry = {children[s.name].entry_point: s for s in manifest.program.stages}
        def loader(child, spec, admission):
            kernel = Kernel(by_entry[child.entry_point]); kernels[kernel.stage.name] = kernel
            return kernel
        loaded = LoadedProgram(candidate, manifest, None, loader, allocate=allocate,
            view=lambda t, shape: Tensor(t.data, shape, t.dtype, t.address),
            storage_span=lambda t: ('fixture', t.address, t.address + math.prod(t.shape)*(2 if t.dtype=='bf16' else 4)),
            stream='fixed-stream')
        return loaded, manifest, tensor, calls, kernels

    def assay(self, candidate, workload, *, failing_stage=None):
        loaded, manifest, tensor, calls, _ = self.loaded(candidate, failing_stage=failing_stage)
        inputs = {'a': [0.]*16, 'b': [0.]*64, 'bias': [1.00390625, -1., .1, 2., -.1, .5, -.5, 0.]}
        # Independent scalar oracle: rounded materialization, then SiLU, then rounded output.
        expected = {'out': [rounded(rounded(value, 'bf16')/(1+math.exp(-rounded(value, 'bf16'))), 'bf16')
                            for _ in range(2) for value in inputs['bias']]}
        arguments = [tensor(inputs[name] if mode=='input' else [float('nan')]*math.prod(shape), shape, dtype)
                     for name, shape, dtype, mode in manifest.tensor_abi]
        loaded.prepare_arguments(arguments)
        class Launcher:
            def launch_tensors(self, bound, spec, values):
                loaded.launch(arguments, tensor_contract=spec, stream='fixed-stream')
                observed = {name: list(arg.data) for (name, _, _, mode), arg in zip(spec.tensor_abi, arguments, strict=True) if mode=='output'}
                after = {name: list(arg.data) for (name, _, _, mode), arg in zip(spec.tensor_abi, arguments, strict=True) if mode=='input'}
                return observed, after, {'candidate_sha256': bound.candidate_sha256,
                    'kernel_calls': loaded.launch_calls, 'fallback_calls': 0}
        try:
            with patch('open_cake_ir.tasks.workloads.materialize_case', return_value=deepcopy(inputs)), \
                 patch('open_cake_ir.tasks.workloads.reference_outputs', return_value=expected):
                receipt = evaluate_tile_workload(candidate, workload,
                    EvaluationProtocol('cpu-fixture', 'confirmatory', workload.canonical_sha256, 'primary', 'none'), Launcher())
            return receipt, calls
        finally:
            loaded.close(synchronize=lambda: None)

    def test_unfused_and_fused_programs_share_the_complete_oracle_boundary(self):
        candidate, workload, fixture = self.build()
        receipt, calls = self.assay(candidate, workload)
        self.assertTrue(receipt.correctness_passed)
        self.assertEqual(receipt.kernel_calls, 2)
        self.assertEqual(calls, ['producer', 'epilogue'])
        self.assertEqual(len(fixture.requests), 2)
        rewrite = self.compiler.rewrite_program(Program.from_dict(epilogue_program()), 'fuse_pointwise_epilogue',
            {'producer':'producer', 'epilogue':'epilogue', 'schedule_id':'fused', 'entry_point':'fused'})
        fused, _, _ = self.build(rewrite.program.document)
        fused_receipt, fused_calls = self.assay(fused, workload)
        self.assertTrue(fused_receipt.correctness_passed)
        self.assertEqual(fused_receipt.kernel_calls, 1)
        self.assertEqual(fused_calls, ['fused'])
        manifests = validate_pair_candidates(fused, candidate, workload, 'primary')
        work = participant_work({'participants': {'candidate': candidate_identity(fused), 'baseline': candidate_identity(candidate)},
                                 'launch_manifests': {role: value.as_dict() for role,value in manifests.items()}})
        self.assertEqual(work, {'candidate': {'modules':1,'kernels':1}, 'baseline': {'modules':2,'kernels':2}})

    def test_single_stage_program_preserves_each_target_kernel_build_and_public_abi(self):
        from open_cake_ir.evaluation.artifacts import required_build_roles
        from open_cake_ir.evaluation.core import TensorLaunchManifest
        from open_cake_ir.evaluation.metal_manifest import MetalTensorLaunchManifest
        from open_cake_ir.evaluation.workload import WorkloadContract
        from open_cake_ir.tasks.devices import BACKENDS
        from open_cake_ir.tasks.workloads import create_task
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                document,source = create_task('softsign',backend=backend,rows=2,columns=32)
                workload = WorkloadContract(document)
                schedule = frontend.parse(source).document
                program = Program.from_schedule(schedule)
                requests = []
                class KernelBuilder:
                    # Deliberately only the existing single-kernel interface.
                    def build(self,request):
                        requests.append(request)
                        requirements = request.toolchain_requirements
                        if requirements['source_language']=='metal':
                            manifest = MetalTensorLaunchManifest.for_workload(workload,'primary',target=request.target,
                                kernel_name=request.entry_point,grid=requirements['threadgroups_per_grid'],
                                block=requirements['threads_per_threadgroup'],
                                threadgroup_memory_bytes=requirements['threadgroup_memory_bytes'])
                        else:
                            manifest = TensorLaunchManifest.for_workload(workload,'primary',target=request.target,
                                kernel_name=requirements['kernel_entry_point'],grid=requirements['grid'],
                                block=[requirements['compile_options']['num_warps']*requirements['warp_size'],1,1],
                                dynamic_shared_memory_bytes=0,hidden_null_pointer_parameters=0)
                        payloads = {role:b'CPU ABI fixture; not executable' for role in required_build_roles(request.target)}
                        payloads.update(lowered_source=request.source,launch_manifest=canonical_json_bytes(manifest.as_dict()))
                        return LaunchableCandidate(request.candidate_sha256,request.target,manifest.kernel_name,
                            {role:sha256(value).hexdigest() for role,value in payloads.items()},manifest.canonical_sha256,payloads)
                environment = OpenCakeEnvironment(self.compiler,KernelBuilder(),workload=workload,case_id='primary',
                    authority_document={'lowering_route':schedule['lowering']})
                submission = CandidateSubmission.seal(environment.media_type,program.document_bytes)
                result = environment.build(submission)
                self.assertEqual(result.disposition,'launchable',result.feedback)
                self.assertFalse(result.launchable.is_program)
                self.assertEqual(result.launchable.candidate_sha256,submission.sha256)
                self.assertEqual(len(requests),1)
                self.assertIsNone(requests[0].tensor_abi)
                self.assertEqual(requests[0].target,program.target)
                self.assertEqual(requests[0].source,self.compiler.lower_program(program).lowerings[0].source.encode())

    def test_single_stage_projection_does_not_erase_a_public_binding(self):
        from open_cake_ir.evaluation.program import single_kernel_lowering
        from open_cake_ir.tasks.workloads import create_task
        _,source = create_task('softsign',backend='triton-b200',rows=2,columns=32)
        program = Program.from_schedule(frontend.parse(source).document)
        changed = program.document
        old = changed['inputs'][0]
        changed['inputs'][0] = 'public_x'
        changed['tensors']['public_x'] = changed['tensors'].pop(old)
        changed['stages'][0]['bindings'][old] = 'public_x'
        self.assertIsNone(single_kernel_lowering(self.compiler.lower_program(Program.from_dict(changed))))

    def test_single_stage_replay_checks_authored_program_source_and_launch(self):
        from open_cake_ir.lab.replay.artifacts import _replay_launchable_candidate
        from open_cake_ir.tasks.launch import parse_launch_manifest
        from open_cake_ir.tasks.workloads import create_task
        _,source = create_task('softsign',backend='triton-b200',rows=2,columns=32)
        program = Program.from_schedule(frontend.parse(source).document)
        candidate,_,_ = self.build(program.document)
        self.assertFalse(candidate.is_program)
        def replay(payloads):
            spec = parse_launch_manifest(json.loads(payloads['launch_manifest']))
            bound = LaunchableCandidate(candidate.candidate_sha256,candidate.target,spec.kernel_name,
                {role:sha256(value).hexdigest() for role,value in payloads.items()},spec.canonical_sha256,payloads)
            event = {'kind':'launchable_candidate_sealed','payload':{'turn':1,
                'candidate_sha256':bound.candidate_sha256,'candidate_record_sha256':bound.canonical_sha256,
                'objects':[{'role':role,'sha256':digest} for role,digest in bound.artifact_roles.items()]}}
            evidence = SimpleNamespace(read_object=lambda reference:payloads[reference['role']])
            return _replay_launchable_candidate(evidence,[event],turn=1,candidate_sha256=bound.candidate_sha256,
                arm='open_cake',manifest_parser=parse_launch_manifest,compiler_factory=lambda:self.compiler,
                authored_bytes=program.document_bytes)
        self.assertEqual(replay(candidate.artifact_payloads).canonical_sha256,candidate.canonical_sha256)
        with self.assertRaisesRegex(ValueError,'authored Program lowering'):
            replay({**candidate.artifact_payloads,'lowered_source':b'changed source'})
        manifest = json.loads(candidate.artifact_payloads['launch_manifest'])
        manifest['grid'][0] += 1
        with self.assertRaisesRegex(ValueError,'authored Program lowering'):
            replay({**candidate.artifact_payloads,'launch_manifest':canonical_json_bytes(manifest)})

    def test_missing_stage_result_is_rejected_by_common_oracle(self):
        candidate, workload, _ = self.build()
        receipt, _ = self.assay(candidate, workload, failing_stage='epilogue')
        self.assertFalse(receipt.correctness_passed)
        self.assertGreater(receipt.correctness['output_mismatches'], 0)

    def test_bundle_stage_substitution_and_missing_component_fail_at_sealing(self):
        candidate, _, _ = self.build()
        manifest, children, _ = program_components(candidate)
        for altered in ({'producer':children['epilogue'], 'epilogue':children['producer']},
                        {'producer':children['producer']}):
            payloads = {**candidate.artifact_payloads, 'program_bundle': pack_candidates(altered)}
            with self.assertRaises(ValueError):
                LaunchableCandidate(candidate.candidate_sha256, candidate.target, candidate.entry_point,
                    {k:sha256(v).hexdigest() for k,v in payloads.items()}, candidate.launch_spec_sha256, payloads)
        self.assertEqual(manifest.program.inputs, ('a','b','bias'))

    def test_modified_launch_or_incomplete_child_evidence_refuses(self):
        candidate, _, _ = self.build()
        manifest, children, _ = program_components(candidate)
        original = children['producer']
        for field, value in (('grid', [1,1,1]), ('block', [64,2,1]),
                             ('kernel_name', 'unbound_entry'), ('dynamic_shared_memory_bytes', 32),
                             ('hidden_null_pointer_parameters', 0)):
            payloads = dict(original.artifact_payloads)
            document = json.loads(payloads['launch_manifest']);document[field] = value
            payloads['launch_manifest'] = canonical_json_bytes(document)
            child = LaunchableCandidate(original.candidate_sha256, original.target, document['kernel_name'],
                {role:sha256(payload).hexdigest() for role,payload in payloads.items()},
                sha256(payloads['launch_manifest']).hexdigest(), payloads)
            altered = {**candidate.artifact_payloads, 'program_bundle': pack_candidates({**children, 'producer':child})}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'compiler metadata'):
                LaunchableCandidate(candidate.candidate_sha256,candidate.target,candidate.entry_point,
                    {role:sha256(payload).hexdigest() for role,payload in altered.items()},candidate.launch_spec_sha256,altered)
        payloads = {role:payload for role,payload in original.artifact_payloads.items() if role != 'ptx'}
        child = LaunchableCandidate(original.candidate_sha256,original.target,original.entry_point,
            {role:sha256(payload).hexdigest() for role,payload in payloads.items()},original.launch_spec_sha256,payloads)
        altered = {**candidate.artifact_payloads, 'program_bundle':pack_candidates({**children,'producer':child})}
        with self.assertRaisesRegex(ValueError,'build evidence'):
            LaunchableCandidate(candidate.candidate_sha256,candidate.target,candidate.entry_point,
                {role:sha256(payload).hexdigest() for role,payload in altered.items()},candidate.launch_spec_sha256,altered)

    def test_cohort_release_drops_tensors_but_keeps_correctness_arguments(self):
        import gc, weakref
        candidate, _, _ = self.build()
        loaded, manifest, tensor, _, _ = self.loaded(candidate)
        def arguments():
            return [tensor([0.]*math.prod(shape),shape,dtype) for _,shape,dtype,_ in manifest.tensor_abi]
        initial = arguments(); loaded.prepare_arguments(initial)
        try:
            for _ in range(3):
                fresh = arguments(); output = weakref.ref(fresh[-1]); loaded.prepare_arguments(fresh)
                loaded.release_arguments(fresh); del fresh; gc.collect()
                self.assertIsNone(output())
                self.assertEqual(len(loaded._prepared),1)
        finally: loaded.close(synchronize=lambda:None)

    def test_initial_preparation_failure_closes_every_loaded_module(self):
        candidate, _, _ = self.build()
        manifest, _, _ = program_components(candidate)
        closed = []
        class FakeTensor:
            device = 'cuda:0'
            def __init__(self, shape): self.shape = shape
            def reshape(self, shape): self.shape=shape; return self
        def full(shape, *args, **kw):
            if tuple(shape) == (2,8) and closed == ['modules_loaded']:
                raise MemoryError('intermediate allocation')
            return FakeTensor(shape)
        loads = []
        class Kernel:
            launch_calls = 0
            resources = {}
            closed = False
            def close(self, *, synchronize):
                synchronize(); closed.append('closed');self.closed=True
        def loader(*args):
            loads.append(Kernel())
            if len(loads)==2: closed.append('modules_loaded')
            return loads[-1]
        torch = SimpleNamespace(float32='fp32',bfloat16='bf16',
            tensor=lambda values, **kw:FakeTensor((len(values),)),full=full,
            cuda=SimpleNamespace(current_stream=lambda:SimpleNamespace(cuda_stream=0),synchronize=lambda:None))
        inputs = {'a':[0.]*16,'b':[0.]*64,'bias':[0.]*8}
        with patch.dict('sys.modules',{'torch':torch}), \
             patch.dict('open_cake_ir.evaluation.core._MODULE_LOADERS',{CodeObject.CUBIN:loader}):
            with self.assertRaisesRegex(MemoryError,'intermediate allocation'):
                LoadedTorchTensorCandidate(candidate,manifest,inputs,None)
        self.assertEqual(closed,['modules_loaded','closed','closed'])
        self.assertTrue(all(kernel.closed for kernel in loads))

    def test_changed_stream_unprepared_arguments_and_replaced_public_tensor_refuse(self):
        candidate, _, _ = self.build()
        loaded, manifest, tensor, calls, _ = self.loaded(candidate)
        args = [tensor([0.]*math.prod(shape),shape,dtype) for _,shape,dtype,_ in manifest.tensor_abi]
        try:
            with self.assertRaisesRegex(ValueError,'prepared'):
                loaded.launch(args, tensor_contract=manifest, stream='fixed-stream')
            loaded.prepare_arguments(args)
            with self.assertRaisesRegex(ValueError,'stream'):
                loaded.launch(args, tensor_contract=manifest, stream='other')
            args[-1] = tensor([0.]*16,(2,8),'bf16')
            with self.assertRaisesRegex(ValueError,'prepared'):
                loaded.launch(args, tensor_contract=manifest, stream='fixed-stream')
            self.assertFalse(calls)
        finally: loaded.close(synchronize=lambda:None)

    def test_replay_reports_unknown_authored_candidate_without_reading_artifacts(self):
        from unittest.mock import Mock
        from open_cake_ir.lab.replay.candidates import _replay_candidates
        from open_cake_ir.lab.replay.refusals import ReplayRefusal
        evidence = Mock()
        with self.assertRaisesRegex(ReplayRefusal, 'not submitted'):
            _replay_candidates(arm='open_cake', case_id='primary',
                events=[{'kind':'launchable_candidate_sealed','payload':{'turn':1,'candidate_sha256':'2'*64}}],
                evidence=evidence, fault_turn=None, faults=(), lock=None, manifest_parser=Mock(),
                protocol_sha256='3'*64, workload_sha256='1'*64,
                provider_candidates_by_turn={1:('4'*64,)}, provider_candidate_bytes={})
        evidence.read_object.assert_not_called()

    def test_program_profile_covers_each_dispatch_and_rejects_projection_drift(self):
        candidate, _, _ = self.build()
        manifest, children, _ = program_components(candidate)
        stages = [{'stage': stage.name, 'kernel_name': children[stage.name].entry_point} for stage in manifest.program.stages]
        values = (95,2,2,16,8,61.,24.,41.,37.,18.,3.)
        lines = ['"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"']
        for index, stage in enumerate(stages):
            for metric,value in zip(NCU_ATTRIBUTION_METRICS, values, strict=True):
                unit = '%' if 'pct' in metric else 'count'
                lines.append(f'"{index}","{stage["kernel_name"]}","{metric}","{unit}","{value}"')
        payload = build_ncu_program_profile(candidate_sha256=candidate.candidate_sha256, case_id='primary', stages=stages,
            ncu_version='CPU-fixture', ncu_executable_sha256='a'*64, stdout='\n'.join(lines).encode(), stderr=b'')
        profile = load_ncu_program_profile(payload, expected_candidate_sha256=candidate.candidate_sha256, expected_case_id='primary')
        self.assertEqual([row['stage'] for row in profile['stages']], ['producer','epilogue'])
        profile['stages'][0]['summary']['occupancy']['registers_per_thread'] = 999
        with self.assertRaises(ValueError):
            load_ncu_program_profile(canonical_json_bytes(profile), expected_candidate_sha256=candidate.candidate_sha256, expected_case_id='primary')
