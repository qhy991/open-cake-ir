"""CPU protocol regressions for the shared Workload-tensor GPU worker."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from open_cake_ir.evaluation.core import (
    EvaluationProtocol, EvaluationReceipt, TensorLaunchManifest, TorchTensorLauncher,
    evaluate_tile_workload,
)
from open_cake_ir.evaluation.tile_workloads import materialize_case, reference_outputs
from open_cake_ir.evaluation.workload import WorkloadContract
from tests.contracts.test_native_triton_pairing import encoded
from tools import evaluate_flash_candidate as worker
from hashlib import sha256

ROOT = Path(__file__).resolve().parents[2]


class TileGpuWorkerTests(unittest.TestCase):
    def setUp(self):
        self.workload = WorkloadContract.load(ROOT / 'contracts/workloads/rmsnorm-fp32-v1.json')
        self.manifest = TensorLaunchManifest.for_workload(self.workload, 'tiny', target='sm_100a',
            kernel_name='fixture', grid=[1, 1, 1], block=[128, 1, 1],
            dynamic_shared_memory_bytes=0, hidden_null_pointer_parameters=2)
        from open_cake_ir.evaluation import LaunchableCandidate
        payloads = {'cubin': b'\x7fELFfixture', 'launch_manifest': encoded(self.manifest.as_dict())}
        self.candidate = LaunchableCandidate('a' * 64, 'sm_100a', 'fixture',
            {k: sha256(v).hexdigest() for k, v in payloads.items()}, self.manifest.canonical_sha256, payloads)
        from open_cake_ir.evaluation.cuda_driver import CudaDeviceAdmission
        self.admission = CudaDeviceAdmission('NVIDIA B200', (10, 0), 'fixture-uuid', 'gpuq-123456789abc', 'exclusive')

    def assay(self, *, fail_preflight=False, mutate_after_timing=False, timing_error=False):
        workload = self.workload
        instances = []

        class Loaded:
            def __init__(self, candidate, manifest, inputs, admission):
                self.loaded = SimpleNamespace(launch_calls=0, resources={})
                self.closed = False
                instances.append(self)

            def launch(self):
                self.loaded.launch_calls += 1

            def launch_tensors(self, candidate, manifest, inputs):
                self.launch()
                output = reference_outputs(workload, 'tiny', inputs)
                after = {k: list(v) for k, v in inputs.items()}
                if fail_preflight:
                    output['y'][0] += 1
                if mutate_after_timing and self.loaded.launch_calls > 1:
                    after['x'][0] += 1
                return output, after, {'candidate_sha256': candidate.candidate_sha256,
                    'kernel_calls': 1, 'fallback_calls': 0}

            def close(self):
                self.closed = True

        calls = []
        def measure(function, **options):
            calls.append(options)
            if timing_error:
                raise RuntimeError('fixture CUPTI failure')
            for _ in range(options['dry_run_iters'] + options['repeat_iters']):
                function()
            return [1.0] * options['repeat_iters']

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authority = SimpleNamespace(workload=workload, case_id='tiny', candidate=self.candidate,
                manifest=self.manifest, request={'purpose': 'confirmatory'}, request_root=root)
            result = worker._base_result(self.admission.broker_job_id)
            with mock.patch.object(worker, 'LoadedTorchTensorCandidate', Loaded), \
                 mock.patch.object(worker, 'StrictCuptiBenchmark', return_value=measure):
                if timing_error:
                    with self.assertRaisesRegex(RuntimeError, 'CUPTI failure'):
                        worker._evaluate_tile_candidate(authority, result, object(), self.admission, True)
                    self.assertTrue(instances[0].closed)
                    return
                worker._evaluate_tile_candidate(authority, result, object(), self.admission, True)
            raw = result['receipt']
            artifacts = {role: (root / path).read_bytes() for role, path in raw['artifacts'].items()}
            receipt = EvaluationReceipt(self.candidate.candidate_sha256, workload.canonical_sha256,
                'b' * 64, 'confirmatory', 'tiny', raw['correctness_passed'], raw['correctness'],
                raw['kernel_calls'], raw['fallback_calls'], sha256(artifacts['launch_receipt']).hexdigest(),
                raw['timing'], artifact_payloads=artifacts)
            self.assertTrue(instances[0].closed)
            if fail_preflight:
                self.assertFalse(receipt.correctness_passed)
                self.assertFalse(calls)
                self.assertEqual(result['counters']['kernel_calls'], 1)
            else:
                self.assertEqual(len(calls), 5)
                self.assertTrue(all(o['cold_l2_cache'] and not o['use_cuda_graph'] for o in calls))
                self.assertEqual(result['counters']['kernel_calls'], 182)
                self.assertEqual(result['counters']['timing_samples'], 125)
                self.assertTrue(receipt.timing['measurement_quality_passed'])
                self.assertEqual(receipt.correctness_passed, not mutate_after_timing)

    def test_worker_preflight_timing_postflight_forms_a_valid_common_receipt(self):
        self.assay()

    def test_wrong_preflight_does_not_enter_timing(self):
        self.assay(fail_preflight=True)

    def test_postflight_mutation_is_incorrect_not_a_timing_or_harness_fault(self):
        self.assay(mutate_after_timing=True)

    def test_timing_failure_closes_the_loaded_module(self):
        self.assay(timing_error=True)

    def test_unwritten_outputs_are_poisoned_and_fail_the_real_oracle(self):
        class Tensor:
            def __init__(self, values): self.values = list(values)
            def reshape(self, *_): return self
            def cpu(self): return self
            def tolist(self): return list(self.values)

        import math
        fake_torch = SimpleNamespace(float32='fp32', bfloat16='bf16', float16='fp16', int32='int32',
            tensor=lambda values, **_: Tensor(values),
            full=lambda shape, value, **_: Tensor([value] * math.prod(shape)),
            cuda=SimpleNamespace(current_stream=lambda: SimpleNamespace(cuda_stream=0), synchronize=lambda: None))
        driver = SimpleNamespace(launch_calls=0, closed=False, resources={})
        def launch(*_, **__): driver.launch_calls += 1
        def close(**_): driver.closed = True
        driver.launch = launch; driver.close = close
        protocol = EvaluationProtocol('poison-test', 'confirmatory', self.workload.canonical_sha256, 'tiny', 'none')
        with mock.patch.dict('sys.modules', {'torch': fake_torch}), \
             mock.patch('open_cake_ir.evaluation.cuda_driver.LoadedCudaCandidate.load', return_value=driver):
            result = evaluate_tile_workload(self.candidate, self.workload, protocol, TorchTensorLauncher(self.admission))
        self.assertFalse(result.correctness_passed)
        self.assertGreater(result.correctness['output_mismatches'], 0)
        self.assertTrue(driver.closed)


if __name__ == '__main__':
    unittest.main()
