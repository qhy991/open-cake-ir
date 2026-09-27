"""The B300 Triton FP16 dot admission is exact and retains typed refusals."""

from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler.backends.triton import emit
from open_cake_ir.compiler.ir import ContractKind, DType, Schedule, contract
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import verify

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "corpus/schedules/gemm-bias-b1-smoke-b300.json"
CONTRACT = "triton.dot.fp16_fp32"


def _fp16_document() -> dict:
    document = json.loads(SOURCE.read_text())
    document["schedule_id"] = "sm103a-fp16-dot-contract"
    for buffer in document["buffers"]:
        if buffer["dtype"] == "bf16":
            buffer["dtype"] = "fp16"
    mma = next(operation for operation in document["operations"]
               if operation["kind"] == "mma")
    mma["parameters"]["instruction"]["contract"] = CONTRACT
    return document


class Sm103aFp16DotTest(unittest.TestCase):
    def test_exact_target_lowers_typed_fp16_dot(self) -> None:
        b300 = Target.load(ROOT / "compiler/targets/sm_103a.json")
        b200 = Target.load(ROOT / "compiler/targets/sm_100a.json")
        record = contract(CONTRACT)
        self.assertIs(record.kind, ContractKind.MMA)
        self.assertEqual(record.operand_dtypes, frozenset({DType.FP16}))
        self.assertIs(record.accumulator, DType.FP32)
        self.assertIn(CONTRACT, b300.instruction_contracts)
        self.assertNotIn(CONTRACT, b200.instruction_contracts)

        schedule = Schedule.from_dict(_fp16_document())
        findings = verify(schedule, b300)
        self.assertFalse([finding for finding in findings if finding.blocks_lowering],
                         findings)
        source = emit(schedule, b300).source
        ast.parse(source)
        self.assertIn("tl.dot(", source)

        b200_document = _fp16_document()
        b200_document["target"] = "sm_100a"
        unsupported = [
            finding for finding in verify(Schedule.from_dict(b200_document), b200)
            if finding.code == "TARGET_INSTRUCTION_UNSUPPORTED"
        ]
        self.assertEqual(len(unsupported), 1)
        self.assertEqual(unsupported[0].path,
                         "operations[2].parameters.instruction.contract")

    def test_bf16_contract_cannot_disguise_fp16_operands(self) -> None:
        document = deepcopy(_fp16_document())
        mma = next(operation for operation in document["operations"]
                   if operation["kind"] == "mma")
        mma["parameters"]["instruction"]["contract"] = "triton.dot.bf16_fp32"
        findings = verify(Schedule.from_dict(document),
                          Target.load(ROOT / "compiler/targets/sm_103a.json"))
        mismatch = [finding for finding in findings
                    if finding.code == "MMA_OPERAND_DTYPE_DIFFERS"]
        self.assertEqual(len(mismatch), 2)
        self.assertEqual(
            {finding.path for finding in mismatch},
            {"operations[2].parameters.instruction.contract"},
        )
        self.assertTrue(any("'a_tile' is fp16" in finding.message
                            for finding in mismatch))
        self.assertTrue(any("'b_tile' is fp16" in finding.message
                            for finding in mismatch))


if __name__ == "__main__":
    unittest.main()
