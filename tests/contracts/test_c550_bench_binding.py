"""Original Bench authority, budget conservation and lease-before-oracle checks."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.tasks.c550_bench.binding import BenchProblem, validate_document
from open_cake_ir.tasks.c550_bench.plan import case_budget_plan
from open_cake_ir.tasks.c550_bench.workload import BenchWorkload
from open_cake_ir.evaluation.core import compare_tile_output_values


def problem():
    def tensor(dtype):
        return SimpleNamespace(dtype=SimpleNamespace(value=dtype))
    definition = SimpleNamespace(
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
        workload = BenchWorkload(problem().workload_document("original-7"))
        with patch("open_cake_ir.tasks.c550_bench.workload.problem_for", return_value=(original, "original-7")):
            passed, metrics = compare_tile_output_values(workload, expected, observed)
        self.assertTrue(passed)
        self.assertEqual(calls, [("original-7", expected, observed)])
        self.assertEqual(metrics["original_bench_check"], check)
        self.assertEqual(metrics["comparison_unit"], "original_bench_output_contract")
        self.assertEqual(metrics["output_mismatches"], 0)

    def test_original_non_numeric_output_refusal_remains_rejected(self):
        original = SimpleNamespace(compare=lambda *args: {"passed": False, "reason": "output_names", "outputs": {}})
        workload = BenchWorkload(problem().workload_document("original-7"))
        with patch("open_cake_ir.tasks.c550_bench.workload.problem_for", return_value=(original, "original-7")):
            passed, metrics = compare_tile_output_values(workload, {}, {})
        self.assertFalse(passed)
        self.assertEqual(metrics["output_mismatches"], 1)
        self.assertEqual(metrics["original_bench_check"]["reason"], "output_names")

    def test_preserves_original_integer_mask_scalar_and_effective_tolerance(self):
        document = problem().workload_document("original-7")
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
        ]
        with patch.object(BenchProblem, "open", return_value=original):
            document = original.workload_document("original-7")
            validate_document(document)
            for mutate in mutations:
                changed = copy.deepcopy(document)
                mutate(changed)
                with self.assertRaisesRegex(ValueError, "original ABI, scalar, oracle or tolerance"):
                    validate_document(changed)

    def test_no_lease_refuses_before_importing_torch_or_running_the_original_factory(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "local maca broker admission is missing"):
                problem().prepare_on_target("original-0", runtime_library="/not-loaded/libmcruntime.so")

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
