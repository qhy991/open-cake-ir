"""Actual FP16 GEMM arrays and driver lists describe the same tensor values."""
from array import array
from types import SimpleNamespace
import unittest

from open_cake_ir.evaluation.core import (
    LoadedTorchTensorCandidate, compare_tile_outputs, compare_tile_output_values, _same_tensor_inputs,
)
from open_cake_ir.evaluation.cuda_driver import CudaDeviceAdmission
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.workloads import create_task, materialize_case


class TensorInputSequences(unittest.TestCase):
    def test_output_only_entry_keeps_numerics_and_combined_entry_keeps_input_effects(self):
        workload = SimpleNamespace(document={'validation':{
            'comparison':'elementwise_atol_rtol','atol':0.0,'rtol':0.0}})
        expected = {'out':[1.0,-0.0]}
        for observed in (expected, {'out':[2.0,0.0]}, {'out':[float('nan'),0.0]}, {}, {'out':[1.0]}):
            with self.subTest(observed=observed):
                passed, metrics = compare_tile_output_values(workload,expected,observed)
                full, combined = compare_tile_outputs(workload,{'x':[1.0]},expected,observed,{'x':[1.0]})
                self.assertEqual(passed,full)
                self.assertEqual(combined,{**metrics,'inputs_unchanged':True})
                self.assertNotIn('inputs_unchanged',metrics)
        full,metrics = compare_tile_outputs(workload,{'x':[-0.0]},expected,expected,{'x':[0.0]})
        self.assertFalse(full)
        self.assertFalse(metrics['inputs_unchanged'])
        self.assertEqual(metrics['output_mismatches'],0)

    def test_native_double_arrays_preserve_bit_checks_and_nan_refusal(self):
        for before, after, expected in [
            ([1.0, -0.0], [1.0, -0.0], True),
            ([1.0, -0.0], [1.0, 0.0], False),
            ([1.0], [2.0], False), ([1.0], [1.0, 2.0], False),
            ([float('nan')], [float('nan')], False),
            ([float('inf')], [float('inf')], True),
        ]:
            with self.subTest(before=before, after=after):
                self.assertEqual(_same_tensor_inputs({'a':array('d',before)},
                                                    {'a':array('d',after)}), expected)

    def test_cpu_tensor_snapshot_retains_exact_values_without_python_scalar_lists(self):
        try:
            import torch
        except ImportError:
            self.skipTest('CPU Torch is required to exercise tensor snapshots')
        for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.int32]:
            with self.subTest(dtype=dtype):
                loaded = object.__new__(LoadedTorchTensorCandidate)
                loaded._native_inputs = None
                loaded.manifest = SimpleNamespace(tensor_abi=(('x',(2,),str(dtype),'input'),
                                                              ('out',(2,),str(dtype),'output')))
                x = torch.tensor([1,0],dtype=dtype)
                loaded.arguments = [x, x.clone()]
                _, after = loaded.snapshot()
                self.assertIsInstance(after['x'],array)
                self.assertEqual(after['x'].typecode,'d')
                self.assertTrue(_same_tensor_inputs({'x':array('d',[1,0])},after))
        loaded.arguments[0][0] = 2
        _, after = loaded.snapshot()
        self.assertFalse(_same_tensor_inputs({'x':array('d',[1,0])},after))

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
        loaded._native_inputs = None
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
