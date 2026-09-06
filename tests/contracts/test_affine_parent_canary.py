from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples/gpu/affine_parent_canary"))
import canary
import prepare


def complete_record():
    original = canary.inputs()
    return {"schema": "open-cake.affine-complete-output.v1", "task_id": canary.TASK_ID,
            "workload": canary.WORKLOAD, "shape": [2, 2, 2], "device_name": "NVIDIA B200",
            "compute_capability": [10, 0], "input_bits_before": original,
            "input_bits_after": copy.deepcopy(original), "expected_bits": canary.reference(),
            "actual_bits": canary.reference(), "guard_bits": [canary.GUARD_BITS] * 2,
            "abi_valid": True, "performance_measured": False}


class AffineParentCanaryTests(unittest.TestCase):
    def test_retained_gpu_output_recomputes_from_the_frozen_input_formula(self):
        data = ROOT / "docs/data/affine-parent-v43-b200-canary-20260906"
        checked = canary.verify(json.loads((data / "complete-output.json").read_text()))
        self.assertEqual(checked, json.loads((data / "verification.json").read_text()))
        self.assertEqual(checked["status"], "passed")

    def test_reference_matches_parent_formula_and_detects_channel_only_indexing(self):
        original = canary.inputs()
        expected = []
        wrong = []
        for i, x in enumerate(original["x"]):
            plane = i // 2
            a, s, b = (canary.number(v) for v in (x, original["scale"][plane], original["bias"][plane]))
            # The selected rational operations are all exactly representable.
            expected.append(canary.bits(a * s + b))
            channel = plane % 2
            wrong.append(canary.bits(a * canary.number(original["scale"][channel]) + canary.number(original["bias"][channel])))
        self.assertEqual(canary.reference(), expected)
        record = complete_record(); record["actual_bits"] = wrong
        self.assertEqual(canary.verify(record)["output_mismatch_count"], 4)
        self.assertEqual(canary.verify(complete_record())["status"], "passed")

    def test_wrong_output_mutated_input_guard_and_abi_each_fail(self):
        mutations = (
            lambda d: d["actual_bits"].__setitem__(0, d["actual_bits"][0] ^ 1),
            lambda d: d["input_bits_after"]["scale"].__setitem__(0, 0),
            lambda d: d["guard_bits"].__setitem__(1, 0),
            lambda d: d.update(abi_valid=False),
        )
        for mutate in mutations:
            record = complete_record(); mutate(record)
            self.assertEqual(canary.verify(record)["status"], "failed")

    def test_incomplete_or_relabelled_evidence_is_refused(self):
        mutations = (
            lambda d: d["actual_bits"].pop(),
            lambda d: d["input_bits_before"]["bias"].__setitem__(0, 0),
            lambda d: d["expected_bits"].__setitem__(0, 0),
            lambda d: d.update(performance_measured=True),
            lambda d: d.update(workload="other-shape"),
        )
        for mutate in mutations:
            record = complete_record(); mutate(record)
            with self.assertRaises(ValueError):
                canary.verify(record)

    def test_prepare_uses_pinned_schedule_and_one_correctness_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            contract = prepare.prepare(output, Path(sys.executable).absolute(), ROOT)
            task = json.loads((output / "task.json").read_text())
            self.assertEqual(contract["compiler_revision_id"], "open-cake-ir-sm100a-v43")
            self.assertEqual(contract["parent_case_id"], canary.PARENT)
            self.assertEqual(contract["input_bits"], canary.inputs())
            self.assertEqual(len(task["stages"]), 1)
            self.assertEqual(task["stages"][0]["kind"], "correctness")
            self.assertEqual(task["stages"][0]["resources"]["mode"], "shared")
            source = (output / "candidate/kernel.py").read_text()
            compile(source, "<affine-canary>", "exec")
            self.assertIn("fma.rn.f32", source)
            with self.assertRaises(FileExistsError):
                prepare.prepare(output, Path(sys.executable).absolute(), ROOT)
            self.assertEqual((output / "candidate/kernel.py").read_text(), source)


if __name__ == "__main__":
    unittest.main()
