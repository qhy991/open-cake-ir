"""Task instruction selection reads the Target's typed declarations once."""
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from open_cake_ir.compiler.target import Target
from open_cake_ir.tasks import devices

ROOT = Path(__file__).resolve().parents[2]


class TaskInstructionAdmissionTests(unittest.TestCase):
    def target(self, names):
        target = Target.load(ROOT / "compiler/targets/gfx1151.json")
        return replace(target, instruction_contracts=frozenset(names))

    def test_one_target_declaration_is_sufficient_without_a_second_device_row(self):
        self.assertNotIn("tanh_contract", devices.BACKENDS["triton-gfx1151"])
        target = self.target(["ocml.tanh.f32"])
        with patch.object(Target, "load", return_value=target):
            self.assertEqual(devices.tanh_contract("triton-gfx1151"), "ocml.tanh.f32")

    def test_absent_or_unrelated_contracts_do_not_supply_tanh(self):
        for names in ([], ["ptx.fma.rn.f32"]):
            with self.subTest(names=names), patch.object(Target, "load", return_value=self.target(names)):
                with self.assertRaisesRegex(ValueError, "no admitted tanh"):
                    devices.tanh_contract("triton-gfx1151")

    def test_ambiguous_instruction_choice_is_not_silently_selected(self):
        target = self.target(["ocml.tanh.f32", "libdevice.tanh.f32"])
        with patch.object(Target, "load", return_value=target):
            with self.assertRaisesRegex(ValueError, "multiple FP32 tanh"):
                devices.tanh_contract("triton-gfx1151")
