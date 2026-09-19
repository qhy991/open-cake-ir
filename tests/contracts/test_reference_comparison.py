"""CPU checks of the development comparison's input and timing boundaries."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.compare_flashinfer_reference import input_spec
from open_cake_ir.tasks.evaluate import _fresh_tile_cohort
from open_cake_ir.tasks.workloads import create_task, materialize_case, reference_outputs
from open_cake_ir.evaluation.workload import WorkloadContract


class ReferenceComparisonTests(unittest.TestCase):
    def test_reference_spec_requires_exact_existing_task_and_regular_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("submission.py", "kernel.py"):
                (root / name).write_text("# reference fixture\n")
            spec = {"task": "fib_rmsnorm_h2048", "rows": 79, "columns": 2048, "target": "sm_103a"}
            def write(value):
                (root / "comparison.json").write_text(json.dumps(value))
            write(spec)
            self.assertEqual(input_spec(root), spec)
            for change in ({"target": "sm_100a"}, {"columns": 4096}, {"rows": True}, {"task": "rmsnorm"}):
                write({**spec, **change})
                with self.assertRaises(ValueError):
                    input_spec(root)
            write(spec)
            (root / "submission.py").unlink()
            (root / "submission.py").symlink_to(root / "kernel.py")
            with self.assertRaises(ValueError):
                input_spec(root)

    def test_common_cohort_checks_each_fresh_output_and_refuses_count_drift(self):
        document, _ = create_task("fib_rmsnorm_h2048", backend="triton-b300", rows=1, columns=2048)
        workload = WorkloadContract(document)
        values = materialize_case(workload, "primary")
        expected = reference_outputs(workload, "primary", values)
        class Loaded:
            def fresh_argument_sets(self, count):
                self.used = []
                return list(range(count))
            def launch(self, argument):
                self.used.append(argument)
            def snapshot(self, argument):
                return expected, values
        def bench(function, **kwargs):
            self.assertTrue(kwargs["cold_l2_cache"])
            self.assertFalse(kwargs["use_cuda_graph"])
            for _ in range(4):
                function()
            return [1.0, 1.1]
        loaded = Loaded()
        samples, check = _fresh_tile_cohort(loaded, bench, workload, values, expected,
            samples_per_cohort=2, route_calls_per_cohort=4)
        self.assertEqual(loaded.used, [0, 1, 2, 3])
        self.assertTrue(check["passed"])
        self.assertEqual(len(samples), 2)
        with self.assertRaisesRegex(RuntimeError, "invocation count differs"):
            _fresh_tile_cohort(loaded, bench, workload, values, expected,
                samples_per_cohort=2, route_calls_per_cohort=5)


if __name__ == "__main__":
    unittest.main()
