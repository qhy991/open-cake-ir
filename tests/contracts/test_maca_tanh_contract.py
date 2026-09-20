"""A MACA math name has types and emission; only a Target can admit it."""
from dataclasses import replace
from pathlib import Path
import unittest

from open_cake_ir.compiler import frontend
from open_cake_ir.compiler.backends import triton
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import verify

ROOT = Path(__file__).resolve().parents[2]
SOURCE = '''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="tanh", target="xcore1002", backend="triton", entry_point="tanh")
def candidate(lm, x: cake.Tensor((8, 128), "fp32"), y: cake.Tensor((8, 128), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0])
    row = lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
        value = lm.load(x[row, :], id="load")
        output = lm.tanh(value, instruction={"contract": "maca.tanh.f32"}, id="tanh")
        lm.store(y[row, :], output, id="store")
'''


class MacaTanhContractTests(unittest.TestCase):
    def test_record_and_emission_do_not_admit_a_target_implicitly(self):
        document = frontend.parse(SOURCE).document
        schedule = Schedule.from_dict(document)
        unadmitted = replace(Target.load(ROOT / "compiler/targets/xcore1002.json"),
                             instruction_contracts=frozenset())
        self.assertIn("TARGET_INSTRUCTION_UNSUPPORTED",
                      [f.code for f in verify(schedule, unadmitted)])
        # Explicit CPU-only target fixture exercises the future declaration through
        # the same verifier/emitter; it claims no physical-device qualification.
        target = replace(unadmitted, instruction_contracts=frozenset({"maca.tanh.f32"}))
        self.assertFalse([f for f in verify(schedule, target) if f.blocks_acceptance])
        self.assertFalse(triton.preflight(schedule, target))
        emitted = triton.emit(schedule, target, entry_point="tanh")
        self.assertIn("from triton.language.extra import libdevice", emitted.source)
        self.assertIn("libdevice.tanh(", emitted.source)
        self.assertNotIn("ocml", emitted.source)

    def test_the_maca_name_does_not_admit_another_elementwise_operation(self):
        document = frontend.parse(SOURCE).document
        operation = next(op for op in document["operations"] if op["id"] == "tanh")
        operation["parameters"]["op"] = "exp"
        with self.assertRaisesRegex(ValueError, "instruction has no defined effect for exp"):
            Schedule.from_dict(document)

    def test_a_declared_maca_contract_still_requires_fp32_operands(self):
        target = replace(Target.load(ROOT / "compiler/targets/xcore1002.json"),
                         instruction_contracts=frozenset({"maca.tanh.f32"}))
        document = frontend.parse(SOURCE.replace('"fp32"', '"fp16"')).document
        findings = verify(Schedule.from_dict(document), target)
        self.assertIn("ELEMENTWISE_INSTRUCTION_DTYPE_DIFFERS", [f.code for f in findings])
        self.assertNotIn("TARGET_INSTRUCTION_UNSUPPORTED", [f.code for f in findings])
