"""CPU contracts for native Program custody; fixtures are not executable GPU code."""
from dataclasses import dataclass
from contextlib import redirect_stdout
import importlib.util
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler, Program
from open_cake_ir.compiler.toolchain import TritonCompilation, triton_route
from open_cake_ir.evaluation.core import EvaluationProtocol
from open_cake_ir.evaluation.metax_benchmark import McptiDispatchBenchmark
from open_cake_ir.evaluation.program import program_components
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.lab import CandidateSubmission, OpenCakeEnvironment, TritonToolchainBuilder
from open_cake_ir.lab.build import build_program_candidate
from open_cake_ir.lab.faults import RunProtocolFault
from open_cake_ir.tasks.solx_fib import attention
from open_cake_ir.tasks.program_evaluation import PreparedProgramCase, evaluate_program_case
from open_cake_ir.evaluation.torch_tensor_inputs import check_cpu_tensor_inputs
from tests.contracts.test_ordered_launch_plan import document
import tests.contracts.test_program_evaluation as program_fixtures

ROOT = Path(__file__).resolve().parents[2]


class McaCompilationFixture:
    def compile(self, source, requirements):
        route = triton_route(requirements)
        artifacts = {role: b'CPU fixture; not executable' for role in route.artifact_roles}
        artifacts['source'] = source
        params = ', '.join(f'%arg{i}: !tt.ptr<f32>' for i in range(len(requirements['signature'])))
        artifacts['ttgir'] = f'tt.func public @fixture({params}) attributes {{}}'.encode()
        return TritonCompilation(source, requirements['target'], requirements['kernel_entry_point'], artifacts,
            requirements['compile_options']['num_warps'] * requirements['warp_size'], 0, 'CPU fixture', route.code_object.value)


def build(program, workload, compiler):
    builder = TritonToolchainBuilder(workload=workload, case_id='primary', isolated_compiler=McaCompilationFixture())
    submission = CandidateSubmission.seal(OpenCakeEnvironment.media_type, program.document_bytes)
    return build_program_candidate(compiler.lower_program(program), builder,
        candidate_sha256=submission.sha256, workload=workload, case_id='primary')


class NativeProgramCustody(unittest.TestCase):
    def test_cli_persists_fault_observations_without_publishing_a_receipt(self):
        spec = importlib.util.spec_from_file_location('qualify_tensor_program_fixture', ROOT / 'tools/qualify_tensor_program.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        payload = b'{"observed_tensors":{"out":{"bytes_base64":"AA=="}}}'
        def fail(args, result):
            raise RunProtocolFault('harness_fault', 'fixture teardown failed',
                                   artifact_payloads={'program_observation': payload})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result'
            argv = ['qualify_tensor_program.py', 'evaluate', '--built', str(Path(directory)/'built'),
                    '--case', 'primary', '--output', str(output)]
            with patch.object(sys, 'argv', argv), patch.object(module, 'evaluate', fail), redirect_stdout(StringIO()):
                self.assertEqual(module.main(), 1)
            self.assertEqual((output / 'fault-program_observation.bin').read_bytes(), payload)
            self.assertFalse(json.loads((output / 'result.json').read_text())['passed'])
            self.assertFalse((output / 'receipt.json').exists())

    def test_maca_pointer_signature_uses_the_comma_as_a_delimiter(self):
        from open_cake_ir.compiler.metax_toolchain import pointer_parameters
        for separator in (',', ', ', ' ,\n'):
            signature = f'tt.func public @k(%arg0: !tt.ptr<f32>{separator}%arg1: !tt.ptr<bf16>) attributes {{}}'
            with self.subTest(separator=separator):
                self.assertEqual(pointer_parameters(signature.encode()), 2)
        for argument in ('i32', '!tt.ptr<f32, 3>', '!tt.ptr<f32>garbage'):
            with self.subTest(argument=argument), self.assertRaises(ValueError):
                pointer_parameters(f'tt.func public @k(%arg0: {argument}) attributes {{}}'.encode())

    def test_maca_sealed_stages_use_the_existing_program_order_and_lifecycle(self):
        raw = document()
        raw['target'] = 'xcore1002'
        for stage in raw['stages']:
            stage['schedule']['target'] = 'xcore1002'
        program = Program.from_dict(raw)
        workload = program_fixtures.workload_for(program)
        candidate = build(program, workload, Compiler.load(ROOT, ROOT / 'compiler/revision.json'))
        loaded, manifest, tensor, calls, children = program_fixtures.ProgramEvaluationTests().loaded(candidate)
        arguments = [tensor([1, 2, 3, 4], (4,), 'fp32'), tensor([float('nan')]*4, (4,), 'fp32')]
        loaded.prepare_arguments(arguments)
        loaded.launch(arguments, tensor_contract=manifest, stream='fixed-stream')
        self.assertEqual(arguments[1].data, [3, 4, 5, 6])
        self.assertEqual(calls, ['first', 'second'])
        self.assertEqual(loaded.launch_calls, 2)
        loaded.close(synchronize=lambda: None)
        self.assertTrue(all(child.closed for child in children.values()))
        _, _, manifests = program_components(candidate)
        self.assertTrue(all(m.hidden_null_pointer_parameters == 0 for m in manifests.values()))
        with patch('open_cake_ir.evaluation.metax_benchmark.activity_collector') as collector:
            with self.assertRaisesRegex(ValueError, 'Program timing is not qualified'):
                McptiDispatchBenchmark(manifest, activity_library='not-loaded', l2_cache_bytes=8388608)
            collector.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec('torch'), 'tensor CPU contracts require Torch')
class TensorOracleReceipt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')
        cls.workload = WorkloadContract(attention.workload_document(next(iter(attention.TASKS)),
                                        variant='boundary', backend='triton-metax'))
        cls.candidate = build(attention.launch_plan(cls.workload), cls.workload, cls.compiler)
        cls.manifest = program_components(cls.candidate)[0]

    def test_cpu_preparation_is_outside_allocation_and_uses_the_original_tensor_oracle(self):
        with patch.dict('os.environ', {'METAL_BROKER_LOCK_FD': '123'}):
            with self.assertRaisesRegex(ValueError, 'precede GPU allocation'):
                PreparedProgramCase(self.workload, 'primary')
        case = PreparedProgramCase(self.workload, 'zeros')
        check_cpu_tensor_inputs(self.manifest, case.inputs)
        bad = dict(case.inputs)
        bad['q'] = bad['q'].float()
        with self.assertRaisesRegex(ValueError, 'dtype'):
            check_cpu_tensor_inputs(self.manifest, bad)

    def test_complete_ieee_outputs_input_effects_counts_and_cleanup_decide_the_receipt(self):
        prepared = PreparedProgramCase(self.workload, 'segmented')
        protocol = EvaluationProtocol('cpu-tensor-fixture', 'confirmatory', self.workload.canonical_sha256, 'segmented', 'none')
        @dataclass
        class Admission:
            target: str = 'xcore1002'
        owner = self
        class Loaded:
            def __init__(self, *args):
                self.loaded = SimpleNamespace(launch_calls=0, resources={'CPU_fixture': True}, closed=False)
            def launch(self):
                self.loaded.launch_calls += owner.manifest.kernels_per_call
            def snapshot(self):
                return {name: value.clone() for name, value in prepared.expected.items()}, {name: True for name in prepared.inputs}
            def close(self):
                self.loaded.closed = True
        with patch('open_cake_ir.tasks.program_evaluation.LoadedTorchTensorInputs', Loaded):
            receipt = evaluate_program_case(self.candidate, self.workload, protocol, Admission(), prepared=prepared)
        self.assertTrue(receipt.correctness_passed)
        self.assertEqual(receipt.kernel_calls, 4)
        self.assertEqual(receipt.artifact_payloads['timing_samples'], b'null')
        self.assertTrue(json.loads(receipt.artifact_payloads['launch_receipt'])['module_unloaded'])
        activity = {'source': 'CPU observer fixture', 'records': [1, 2, 3, 4]}
        def observe(launch):
            launch()
            return activity
        with patch('open_cake_ir.tasks.program_evaluation.LoadedTorchTensorInputs', Loaded):
            receipt = evaluate_program_case(self.candidate, self.workload, protocol, Admission(),
                                            prepared=prepared, observe=observe)
        self.assertEqual(json.loads(receipt.artifact_payloads['correctness_output'])['native_activity'], activity)
        self.assertEqual(receipt.kernel_calls, 4)
        with patch.object(Loaded, 'snapshot', side_effect=ValueError('snapshot failed')), patch(
                'open_cake_ir.tasks.program_evaluation.LoadedTorchTensorInputs', Loaded):
            with self.assertRaisesRegex(RunProtocolFault, 'snapshot failed') as failed:
                evaluate_program_case(self.candidate, self.workload, protocol, Admission(),
                                      prepared=prepared, observe=observe)
            self.assertEqual(json.loads(failed.exception.artifact_payloads['program_activity']), activity)
        with patch('open_cake_ir.tasks.program_evaluation.LoadedTorchTensorInputs', Loaded):
            with self.assertRaisesRegex(RunProtocolFault, 'exact stage count'):
                evaluate_program_case(self.candidate, self.workload, protocol, Admission(),
                                      prepared=prepared, observe=lambda launch: activity)
        with patch.object(Loaded, 'launch', lambda self: None), patch('open_cake_ir.tasks.program_evaluation.LoadedTorchTensorInputs', Loaded):
            with self.assertRaisesRegex(RunProtocolFault, 'exact stage count') as failed:
                evaluate_program_case(self.candidate, self.workload, protocol, Admission(), prepared=prepared)
            self.assertIn('program_observation', failed.exception.artifact_payloads)
            self.assertEqual(json.loads(failed.exception.artifact_payloads['program_launch'])['kernel_calls'], 0)
        def changed(self):
            return {name: value.clone() for name, value in prepared.expected.items()}, {name: False for name in prepared.inputs}
        with patch.object(Loaded, 'snapshot', changed), patch('open_cake_ir.tasks.program_evaluation.LoadedTorchTensorInputs', Loaded):
            receipt = evaluate_program_case(self.candidate, self.workload, protocol, Admission(), prepared=prepared)
            self.assertFalse(receipt.correctness_passed)
        with patch.object(Loaded, 'close', side_effect=RuntimeError('unload failed')), patch('open_cake_ir.tasks.program_evaluation.LoadedTorchTensorInputs', Loaded):
            with self.assertRaisesRegex(RunProtocolFault, 'unload failed') as failed:
                evaluate_program_case(self.candidate, self.workload, protocol, Admission(), prepared=prepared)
            raw = json.loads(failed.exception.artifact_payloads['program_observation'])
            self.assertEqual(set(raw['observed_tensors']), set(prepared.expected))
            self.assertTrue(all(raw['input_checks'].values()))
            self.assertFalse(json.loads(failed.exception.artifact_payloads['program_launch'])['module_unloaded'])
        with patch.object(Loaded, 'close', lambda self: None), patch('open_cake_ir.tasks.program_evaluation.LoadedTorchTensorInputs', Loaded):
            with self.assertRaisesRegex(RunProtocolFault, 'remain open') as failed:
                evaluate_program_case(self.candidate, self.workload, protocol, Admission(), prepared=prepared)
            self.assertIn('program_observation', failed.exception.artifact_payloads)
