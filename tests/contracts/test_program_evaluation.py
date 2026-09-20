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

from open_cake_ir.compiler import Compiler, Program
from open_cake_ir.evaluation.core import EvaluationProtocol, LaunchableCandidate
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
        profile['stages'][0]['summary']['resources']['registers_per_thread'] = 999
        with self.assertRaises(ValueError):
            load_ncu_program_profile(canonical_json_bytes(profile), expected_candidate_sha256=candidate.candidate_sha256, expected_case_id='primary')
