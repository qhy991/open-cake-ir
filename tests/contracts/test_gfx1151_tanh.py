"""The measured gfx1151 tanh contract admits GELU and refuses foreign spellings."""
from copy import deepcopy
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.compiler.target import Target
from open_cake_ir.tasks.devices import tanh_contract
from open_cake_ir.tasks.workloads import create_task

ROOT = Path(__file__).resolve().parents[2]


class Gfx1151TanhTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_target_declares_only_its_measured_library(self):
        target = Target.load(ROOT / "compiler/targets/gfx1151.json")
        self.assertEqual(target.instruction_contracts, frozenset({"ocml.tanh.f32"}))
        self.assertEqual(tanh_contract("triton-gfx1151"), "ocml.tanh.f32")

    def test_both_gelu_starters_lower_without_substituting_their_mathematics(self):
        for task in ("gelu_tanh", "gelu_tanh_backward"):
            document, source = create_task(task, backend="triton-gfx1151", rows=2, columns=8)
            assessment = self.compiler.assess(frontend.parse(source).document)
            with self.subTest(task=task):
                self.assertTrue(assessment.lowering_eligible, assessment.findings)
                lowering = self.compiler.lower(assessment)
                self.assertEqual(lowering.target, "gfx1151")
                self.assertIn("libdevice.tanh(inner)", lowering.source)
                self.assertEqual(document["validation"]["all_cases_required"], True)

    def test_foreign_tanh_is_refused_by_target_instruction_admission(self):
        _, source = create_task("gelu_tanh", backend="triton-gfx1151", rows=2, columns=8)
        original = frontend.parse(source).document
        for contract in ("libdevice.tanh.f32", "metal.precise.tanh.f32"):
            document = deepcopy(original)
            for operation in document["operations"]:
                if operation["id"] == "tanh":
                    operation["parameters"]["instruction"]["contract"] = contract
            with self.subTest(contract=contract):
                assessment = self.compiler.assess(document)
                self.assertFalse(assessment.lowering_eligible)
                self.assertIn("TARGET_INSTRUCTION_UNSUPPORTED",
                              {finding.code for finding in assessment.findings})
