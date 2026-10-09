"""Original Bench authority, budget conservation and lease-before-oracle checks."""
import copy
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.tasks.c550_bench.binding import BENCH_COMMIT, BenchProblem, factory_scalar, validate_document, validate_input_view_observation
from open_cake_ir.tasks.c550_bench.plan import case_budget_plan
from open_cake_ir.tasks.c550_bench.workload import BenchWorkload
from open_cake_ir.evaluation.core import compare_tile_output_values
from open_cake_ir.serialization import canonical_json_bytes


ORACLE_NUMERICS = {'float32_matmul_precision': 'high', 'allow_tf32': True,
    'allow_fp16_reduced_precision_reduction': True,
    'allow_bf16_reduced_precision_reduction': True,
    'initialization': {'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE': '1'}}


def problem():
    def tensor(dtype):
        return SimpleNamespace(dtype=SimpleNamespace(value=dtype))
    definition = SimpleNamespace(
        reference="def run(positions, mask, scale):\n    return positions * scale\n",
        custom_inputs_entrypoint=None,
        inputs={"positions": tensor("int64"), "mask": tensor("bool"), "scale": tensor("float32")},
        outputs={"out": tensor("bfloat16")},
        get_input_shapes=lambda axes: {"positions": (axes["N"],), "mask": (axes["N"],), "scale": None},
        get_output_shapes=lambda axes: {"out": (axes["N"],)},
    )
    tolerance = SimpleNamespace(model_dump=lambda: {"max_atol": 1e-5, "max_rtol": .05,
                                                    "required_matched_ratio": .99, "max_error_cap": None})
    workloads = tuple(SimpleNamespace(uuid=f"original-{i}", axes={"N": i + 1}, tolerance=tolerance)
                      for i in range(16))
    raw = tuple({"uuid": row.uuid, "inputs": {"scale": {"type": "scalar", "value": .5}},
                 "tolerance": {"max_rtol": .05}} for row in workloads)
    api = SimpleNamespace(document=lambda name: {"seed": 200})
    return BenchProblem(Path("/private/original-bench"), "fixture", api, definition,
                        workloads, raw, Path("/private/original-bench/.data/fixture"))


class BenchBindingTest(unittest.TestCase):
    def test_original_comparator_receives_shallow_named_outputs_and_unchanged_sequences(self):
        original = problem()
        wanted, actual = object(), object()
        expected, observed = MappingProxyType({'out': wanted}), MappingProxyType({'out': actual})
        calls = []
        original.api.compare_outputs = lambda *args: calls.append(args) or {'passed': True}
        original.compare('original-0', expected, observed)
        converted_actual, converted_expected, definition, tolerance, axes = calls.pop()
        self.assertIs(type(converted_actual), dict)
        self.assertIs(type(converted_expected), dict)
        self.assertIs(converted_actual['out'], actual)
        self.assertIs(converted_expected['out'], wanted)
        self.assertIs(definition, original.definition)
        self.assertIs(tolerance, original.workloads[0].tolerance)
        self.assertIs(axes, original.workloads[0].axes)
        self.assertIsInstance(expected, MappingProxyType)
        sequence = (actual,)
        original.compare('original-0', wanted, sequence)
        self.assertIs(calls[0][0], sequence)
        self.assertIs(calls[0][1], wanted)

    def test_view_observation_requires_complete_original_cases_and_actual_identity(self):
        from open_cake_ir.compiler.target import declared_target
        original = problem()
        target = declared_target('xcore1002')
        job, pci = 'maca-123456abcdef', '0000:d7:00'
        observation = {'bench_commit': BENCH_COMMIT, 'task': original.task_id,
            'target': 'xcore1002', 'status': 'complete', 'scope': 'original_input_factory_metadata_only',
            'job_id': job, 'expected_pci': pci, 'cases': []}
        for row in original.workloads:
            observation['cases'].append({'workload_uuid': row.uuid, 'zero_copy_verified': True,
                'device_admission': {'broker_job_id': job, 'target': 'xcore1002', 'device_arch': 'xcore1002',
                    'device_name': target.device_names[0], 'warp_size': target.warp_size,
                    'pci_bus_id': pci, 'runtime_library': '/opt/maca/lib/libmcruntime.so', 'gpu_uuid': None},
                'input_views': {'positions': [0], 'mask': [0]},
                'metadata': {name: {'shape': [row.axes['N']], 'strides': [1],
                    'dtype': 'torch.' + dtype, 'physical_axes': [0]}
                    for name, dtype in [('positions', 'int64'), ('mask', 'bool')]}})
        self.assertEqual(len(validate_input_view_observation(original, observation)), 16)
        mutations = [
            lambda x: x['cases'].pop(),
            lambda x: x['cases'].reverse(),
            lambda x: x['cases'][1].update(workload_uuid=x['cases'][0]['workload_uuid']),
            lambda x: x.update(target='other'),
            lambda x: x['cases'][0]['device_admission'].update(broker_job_id='maca-abcdef123456'),
            lambda x: x['cases'][0]['device_admission'].update(pci_bus_id='0000:c2:00'),
            lambda x: x['cases'][0].update(zero_copy_verified=False),
            lambda x: x['cases'][0]['metadata']['positions'].update(dtype='torch.int32'),
            lambda x: x['cases'][0]['metadata']['positions'].update(shape=[2]),
            lambda x: x['cases'][0]['metadata']['positions'].update(shape=[True]),
            lambda x: x['cases'][0]['input_views'].update(positions=[0, 0]),
            lambda x: x['cases'][1]['metadata']['positions'].update(strides=[2]),
        ]
        for mutate in mutations:
            changed = copy.deepcopy(observation)
            mutate(changed)
            with self.assertRaises(ValueError):
                validate_input_view_observation(original, changed)

    def test_generic_workload_does_not_enter_target_oracle_or_registry_lookup(self):
        from open_cake_ir.evaluation.workload import WorkloadContract
        from open_cake_ir.tasks.evaluate import _prepare_target_tensor_work, _tensor_snapshot_options
        generic = WorkloadContract({'workload_id': 'program-fixture',
            'cases': [{'case_id': 'primary', 'shape': {'N': 1}, 'seed': 1, 'mode': 'fixture'}]})
        authority = SimpleNamespace(workload=generic)
        self.assertIs(_prepare_target_tensor_work(authority, None), authority)
        self.assertEqual(_tensor_snapshot_options(generic), {})

    def test_custom_scalar_uses_original_named_return_and_rejects_dynamic_values(self):
        original = problem()
        original.definition.custom_inputs_entrypoint = 'get_inputs'
        original.definition.reference = "def get_inputs():\n    scale = 0.1\n    return {'positions': None, 'mask': None, 'scale': scale}\n"
        original.raw_workloads[7]['inputs']['scale'] = {'type': 'custom'}
        document = original.workload_document('original-7', oracle_numerics=ORACLE_NUMERICS)
        self.assertEqual(document['semantics']['fixed_scalar_inputs']['scale']['value'], .1)
        original.definition.reference = "def get_inputs():\n    return {'scale': 1.0}\n"
        self.assertEqual(factory_scalar(original.definition, 'scale'), 1.)
        for source in [
            "def get_inputs():\n    scale = 0.1\n    if condition:\n        scale = 0.2\n    return {'scale': scale}\n",
            "def get_inputs():\n    return {'scale': random_value()}\n",
        ]:
            original.definition.reference = source
            with self.assertRaises(ValueError):
                factory_scalar(original.definition, 'scale')

    def test_input_permutation_is_explicit_in_candidate_shape_and_original_shape_survives(self):
        original = problem()
        original.definition.get_input_shapes = lambda axes: {'positions': (2, 3, 4), 'mask': (24,), 'scale': None}
        document = original.workload_document('original-7', oracle_numerics=ORACLE_NUMERICS, input_views={'positions': [1, 0, 2]})
        self.assertEqual(document['tensors']['positions']['shape'], [3, 2, 4])
        self.assertEqual(document['semantics']['original_tensor_shapes']['positions'], [2, 3, 4])
        with patch.object(BenchProblem, 'open', return_value=original):
            validate_document(document)
        for order in ([0, 0, 1], [0, 1], [True, 0, 2]):
            with self.assertRaises(ValueError):
                original.workload_document('original-7', oracle_numerics=ORACLE_NUMERICS, input_views={'positions': order})

    def test_rejected_nonfinite_statistics_do_not_become_serialization_faults(self):
        import json
        for statistic, encoded in [(float('inf'), 'Infinity'), (float('-inf'), '-Infinity'), (float('nan'), 'NaN')]:
            check = {"passed": False, "outputs": {"out": {"passed": False, "max_absolute_error": statistic}}}
            original = SimpleNamespace(compare=lambda *args: check)
            workload = BenchWorkload(problem().workload_document("original-7", oracle_numerics=ORACLE_NUMERICS))
            with patch("open_cake_ir.tasks.c550_bench.workload.problem_for", return_value=(original, "original-7")):
                passed, metrics = compare_tile_output_values(workload, {}, {})
            retained = json.loads(canonical_json_bytes(metrics))
            self.assertFalse(passed)
            self.assertEqual(retained['original_bench_check']['outputs']['out']['max_absolute_error'], encoded)
            self.assertEqual(retained['max_abs_error_coverage'], 'finite_original_statistics_only')

    def test_changed_external_reference_factory_or_input_spec_refuses_the_frozen_document(self):
        changes = [
            lambda p: setattr(p.definition, 'reference', 'def run(*args): return None'),
            lambda p: setattr(p.definition, 'custom_inputs_entrypoint', 'other_factory'),
            lambda p: p.raw_workloads[7]['inputs']['scale'].update(value=2.),
        ]
        for change in changes:
            original = problem()
            document = copy.deepcopy(original.workload_document('original-7', oracle_numerics=ORACLE_NUMERICS))
            change(original)
            with patch.object(BenchProblem, 'open', return_value=original):
                with self.assertRaisesRegex(ValueError, 'original ABI, scalar, oracle or tolerance'):
                    validate_document(document)

    def test_common_comparison_keeps_original_ratio_verdict_and_full_record(self):
        original = problem()
        check = {"passed": True, "outputs": {"out": {"passed": True,
                 "reason": "upstream_numeric", "max_absolute_error": .0625,
                 "max_relative_error": .0139}}}
        observed, expected = object(), object()
        calls = []
        def compare(uuid, wanted, actual):
            calls.append((uuid, wanted, actual))
            return check
        original = SimpleNamespace(compare=compare)
        workload = BenchWorkload(problem().workload_document("original-7", oracle_numerics=ORACLE_NUMERICS))
        with patch("open_cake_ir.tasks.c550_bench.workload.problem_for", return_value=(original, "original-7")):
            passed, metrics = compare_tile_output_values(workload, expected, observed)
        self.assertTrue(passed)
        self.assertEqual(calls, [("original-7", expected, observed)])
        self.assertEqual(metrics["original_bench_check"], check)
        self.assertEqual(metrics["comparison_unit"], "original_bench_output_contract")
        self.assertEqual(metrics["output_mismatches"], 0)

    def test_original_non_numeric_output_refusal_remains_rejected(self):
        original = SimpleNamespace(compare=lambda *args: {"passed": False, "reason": "output_names", "outputs": {}})
        workload = BenchWorkload(problem().workload_document("original-7", oracle_numerics=ORACLE_NUMERICS))
        with patch("open_cake_ir.tasks.c550_bench.workload.problem_for", return_value=(original, "original-7")):
            passed, metrics = compare_tile_output_values(workload, {}, {})
        self.assertFalse(passed)
        self.assertEqual(metrics["output_mismatches"], 1)
        self.assertEqual(metrics["original_bench_check"]["reason"], "output_names")

    def test_preserves_original_integer_mask_scalar_and_effective_tolerance(self):
        document = problem().workload_document("original-7", oracle_numerics=ORACLE_NUMERICS)
        self.assertEqual(document["tensors"]["positions"]["dtype"], "int64")
        self.assertEqual(document["tensors"]["mask"]["dtype"], "bool")
        self.assertEqual(document["tensors"]["out"]["shape"], [8])
        self.assertEqual(document["semantics"]["candidate_abi"]["inputs"], ["positions", "mask"])
        self.assertEqual(document["semantics"]["fixed_scalar_inputs"]["scale"]["value"], .5)
        self.assertEqual(document["validation"]["effective_tolerance"]["required_matched_ratio"], .99)
        self.assertNotIn("required_matched_ratio", document["validation"]["raw_tolerance"])

    def test_refuses_contract_edits_to_original_shape_dtype_scalar_or_tolerance(self):
        original = problem()
        mutations = [
            lambda doc: doc["tensors"]["positions"].update(shape=[2]),
            lambda doc: doc["tensors"]["positions"].update(dtype="int32"),
            lambda doc: doc["semantics"]["fixed_scalar_inputs"]["scale"].update(value=1.),
            lambda doc: doc["validation"]["effective_tolerance"].update(required_matched_ratio=.98),
            lambda doc: doc["oracle"].update(reference_source="def run(*inputs): return inputs[0]"),
            lambda doc: doc["semantics"]["original_input_specifications"]["scale"].update(value=.25),
        ]
        with patch.object(BenchProblem, "open", return_value=original):
            document = original.workload_document("original-7", oracle_numerics=ORACLE_NUMERICS)
            validate_document(document)
            for mutate in mutations:
                changed = copy.deepcopy(document)
                mutate(changed)
                with self.assertRaisesRegex(ValueError, "original ABI, scalar, oracle or tolerance"):
                    validate_document(changed)

    def test_no_lease_refuses_before_importing_torch_or_running_the_original_factory(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "local maca broker admission is missing"):
                problem().prepare_on_target("original-0", runtime_library="/not-loaded/libmcruntime.so", oracle_numerics=ORACLE_NUMERICS)

    def test_all_original_cases_share_three_hours_including_final_confirmation(self):
        plan = case_budget_plan(problem())
        allocations = [row["wall_time_seconds"] for row in plan["cases"]]
        self.assertEqual(allocations, [608] * 8 + [607] * 8)
        self.assertEqual(sum(allocations) + plan["final_confirmation_seconds"], 10800)
        self.assertEqual(len({row["workload_uuid"] for row in plan["cases"]}), 16)
        self.assertEqual(plan["final_check"]["expected_checks"], 160)
        self.assertTrue(all(row["run"] is None and row["candidate"] is None for row in plan["cases"]))

    def test_duplicate_case_cannot_take_another_case_budget(self):
        original = problem()
        duplicated = BenchProblem(original.root, original.task_id, original.api, original.definition,
                                  original.workloads[:-1] + (original.workloads[0],),
                                  original.raw_workloads, original.task_root)
        with self.assertRaisesRegex(ValueError, "distinct original"):
            case_budget_plan(duplicated)


if __name__ == "__main__":
    unittest.main()
