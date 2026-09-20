"""Actual FP16 GEMM arrays and driver lists describe the same tensor values."""
from array import array
from types import SimpleNamespace
import unittest

from open_cake_ir.evaluation.core import LoadedTorchTensorCandidate, compare_tile_outputs
from open_cake_ir.evaluation.cuda_driver import CudaDeviceAdmission
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.workloads import create_task, materialize_case


class TensorInputSequences(unittest.TestCase):
    def test_actual_gemm_array_inputs_match_list_snapshots_without_losing_mutation_checks(self):
        document, _ = create_task('fib_gemm_n128_k2048', backend='triton-b300', rows=1, columns=128)
        workload = WorkloadContract(document)
        before = materialize_case(workload, 'zeros')
        after = {name: list(values) for name, values in before.items()}
        output = {'out': [0.0] * 128}
        self.assertIsInstance(before['a'], array)
        self.assertTrue(compare_tile_outputs(workload, before, output, output, after)[0])
        for change in ('value', 'sign', 'length', 'key'):
            observed = {name: list(values) for name, values in before.items()}
            if change == 'value': observed['a'][0] = 1.0
            elif change == 'sign': observed['a'][0] = -0.0
            elif change == 'length': observed['a'].pop()
            else: observed.pop('a')
            with self.subTest(change=change):
                self.assertFalse(compare_tile_outputs(workload, before, output, output, observed)[0])

    def test_loaded_assay_compares_contents_and_keeps_candidate_and_manifest_guards(self):
        loaded = object.__new__(LoadedTorchTensorCandidate)
        loaded.candidate = SimpleNamespace(canonical_sha256='candidate', candidate_sha256='source')
        loaded.manifest = SimpleNamespace(canonical_sha256='manifest', tensor_abi=())
        loaded.inputs = {'a': [1.0, -0.0]}
        loaded.arguments = []
        loaded.loaded = SimpleNamespace(launch_calls=0, closed=False, resources={})
        loaded.admission = CudaDeviceAdmission('NVIDIA B300', (10,3), 'fixture', 'gpuq-123456789abc', 'exclusive')
        def launch(): loaded.loaded.launch_calls += 1
        loaded.launch = launch
        loaded.snapshot = lambda: ({'out': [1.0]}, {'a': [1.0, -0.0]})
        _, _, receipt = loaded.launch_tensors(loaded.candidate, loaded.manifest, {'a': array('f', [1.0, -0.0])})
        self.assertEqual(receipt['kernel_calls'], 1)
        for candidate, manifest, inputs in (
            (SimpleNamespace(canonical_sha256='other'), loaded.manifest, loaded.inputs),
            (loaded.candidate, SimpleNamespace(canonical_sha256='other'), loaded.inputs),
            (loaded.candidate, loaded.manifest, {'a': array('f', [1.0, 0.0])}),
        ):
            with self.assertRaisesRegex(ValueError, 'input or candidate differs'):
                loaded.launch_tensors(candidate, manifest, inputs)
        self.assertEqual(loaded.loaded.launch_calls, 1)
