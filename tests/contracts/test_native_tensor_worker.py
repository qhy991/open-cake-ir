"""Native inputs reach the existing worker and fresh-output cohort, CPU doubles."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler
from open_cake_ir.evaluation.core import LoadedTorchTensorCandidate, _same_tensor_inputs
from open_cake_ir.evaluation.program import program_components
from open_cake_ir.evaluation.torch_tensor_inputs import LoadedTorchTensorInputs
from open_cake_ir.evaluation.triton_metax import MetaxDeviceAdmission
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks import evaluate as worker, workloads
from open_cake_ir.tasks.solx_fib import attention
from tests.contracts.test_native_program_tensors import build

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires CPU Torch')
class NativeTensorWorker(unittest.TestCase):
    def test_input_equality_preserves_fp8_storage_dtype_shape_and_signed_zero(self):
        import torch
        values = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn)
        self.assertTrue(_same_tensor_inputs({'x': values}, {'x': values.clone()}))
        for change in ('byte', 'dtype', 'shape'):
            other = values.clone()
            if change == 'byte': other.view(torch.uint8)[0] = 128
            if change == 'dtype': other = other.view(torch.uint8)
            if change == 'shape': other = other.reshape(1, 256)
            with self.subTest(change=change): self.assertFalse(_same_tensor_inputs({'x': values}, {'x': other}))

    def test_all_case_worker_and_fresh_cohort_keep_original_native_inputs(self):
        import torch
        workload = WorkloadContract(attention.workload_document(next(iter(attention.TASKS)),
                                    variant='boundary', backend='triton-metax'))
        candidate = build(attention.launch_plan(workload), workload,
                          Compiler.load(ROOT, ROOT / 'compiler/revision.json'))
        manifest = program_components(candidate)[0]
        admission = MetaxDeviceAdmission('maca-123456789abc', 'xcore1002', 'xcore1002',
                                        'MetaX C550', 64, '0000:0f:00', '/opt/maca-3.5.3/lib/libmcruntime.so')
        with tempfile.TemporaryDirectory() as directory:
            policy = {'case_id': 'primary', 'validation_case_ids': list(workload.case_ids)}
            authority = worker._Authority({'purpose': 'confirmatory', 'evaluation_protocol': policy},
                Path(directory), None, workload, manifest, candidate, candidate.artifact_payloads,
                'primary', allocation_mode='local_serialized', timed_assay_available=False)
            with patch.object(workloads, 'materialize_case', side_effect=AssertionError('flat inputs used')), patch.object(
                    workloads, 'reference_outputs', side_effect=AssertionError('flat reference used')):
                authority = worker._prepare_local_tensor_work(authority, 'maca')
            self.assertEqual(set(authority.prepared_cases), set(workload.case_ids))
            expected = {case.inputs['q'].data_ptr(): case.expected for case in authority.prepared_cases.values()}
            modules = []
            def native_transport(candidate, manifest, inputs, admission):
                loaded = object.__new__(LoadedTorchTensorInputs)
                loaded.manifest, loaded.inputs = manifest, inputs
                dtypes = {'bf16': torch.bfloat16, 'fp32': torch.float32, 'int32': torch.int32}
                loaded.arguments = [inputs[name] if mode == 'input' else
                    torch.full(shape, torch.finfo(dtypes[dtype]).max, dtype=dtypes[dtype])
                    for name, shape, dtype, mode in manifest.tensor_abi]
                gold = expected[inputs['q'].data_ptr()]
                class Module:
                    launch_calls = 0
                    closed = False
                    resources = {'CPU_fixture': True}
                    def launch(self, arguments, **kwargs):
                        for (name, shape, dtype, mode), value in zip(manifest.tensor_abi, arguments):
                            if mode == 'output': value.copy_(torch.tensor(gold[name], dtype=value.dtype).reshape(shape))
                        self.launch_calls += manifest.kernels_per_call
                    def prepare_arguments(self, arguments): pass
                    def release_arguments(self, arguments): pass
                    def close(self, *, synchronize): synchronize(); self.closed = True
                loaded.loaded = Module(); modules.append(loaded.loaded)
                return loaded
            result = {'counters': {'module_loads': 0, 'kernel_calls': 0, 'preflight_calls': 0}}
            with patch('open_cake_ir.evaluation.torch_tensor_inputs.LoadedTorchTensorInputs', side_effect=native_transport), patch.object(
                    torch.cuda, 'current_stream', return_value=SimpleNamespace(cuda_stream=0)), patch.object(torch.cuda, 'synchronize'):
                worker._evaluate_untimed_validation_cases(authority, result, admission)
                self.assertTrue(result['receipt']['correctness_passed'])
                self.assertEqual(result['counters']['kernel_calls'], 20)
                self.assertTrue(all(module.closed for module in modules))
                primary = authority.prepared_cases['primary']
                loaded = LoadedTorchTensorCandidate(candidate, manifest, primary.inputs, admission)
                def benchmark(launch, *, dry_run_iters, repeat_iters, **kwargs):
                    for _ in range(dry_run_iters + repeat_iters): launch()
                    return [0.001] * repeat_iters
                samples, check = worker._fresh_tile_cohort(loaded, benchmark, workload, primary.inputs,
                    primary.expected, samples_per_cohort=2, route_calls_per_cohort=13)
                self.assertEqual(samples, [0.001, 0.001])
                self.assertTrue(check['passed'])
                self.assertEqual(check['checked_launches'], 13)
                self.assertEqual(loaded.loaded.launch_calls, 52)
                loaded.close()
            raw = json.loads((Path(directory) / 'correctness-output.json').read_text())
            self.assertEqual(len(raw['validation_cases']), 5)
