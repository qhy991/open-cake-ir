"""Contracts for explicit Triton TF32 input precision over FP32 MMA operands."""

from __future__ import annotations

import ast
import copy
import json
import unittest
from pathlib import Path

from open_cake_ir.compiler.backends.cutedsl import preflight as cute_preflight
from open_cake_ir.compiler.backends.triton import emit, preflight as triton_preflight
from open_cake_ir.compiler.ir import Schedule, ScheduleParseError
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import FindingCategory, verify

ROOT = Path(__file__).resolve().parents[2]
TARGET_PATH = ROOT / "compiler/targets/sm_100a.json"
POSITIVE = ROOT / "corpus/schedules/fp32-tf32-mma-b1-smoke.json"
BF16_DRIFT = ROOT / "corpus/schedules/fp32-tf32-mma-b1-smoke-bf16-drift.json"
QSA_IEEE = ROOT / "corpus/schedules/qsa-score-topk-t32768.json"
CUTE_POSITIVE = ROOT / "corpus/schedules/flash-kmeans-assignment-full.json"
CONTRACT = "triton.dot.fp32_tf32"


def _successor_target() -> Target:
    document = json.loads(TARGET_PATH.read_text(encoding="utf-8"))
    if CONTRACT not in document["instruction_contracts"]:
        document["instruction_contracts"].append(CONTRACT)
    return Target.from_dict(document)


def _target_without_tf32() -> Target:
    document = json.loads(TARGET_PATH.read_text(encoding="utf-8"))
    if CONTRACT in document["instruction_contracts"]:
        document["instruction_contracts"].remove(CONTRACT)
    return Target.from_dict(document)


class Fp32Tf32MmaContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prior_target = _target_without_tf32()
        cls.successor_target = _successor_target()

    def test_target_admission_is_required_and_the_successor_is_lowerable(self) -> None:
        schedule = Schedule.load(POSITIVE)
        current = verify(schedule, self.prior_target)
        successor = verify(schedule, self.successor_target)

        unsupported = next(
            finding
            for finding in current
            if finding.code == "TARGET_INSTRUCTION_UNSUPPORTED"
        )
        self.assertEqual(
            unsupported.path,
            "operations[2].parameters.instruction.contract",
        )
        self.assertFalse([finding for finding in successor if finding.blocks_lowering])

    def test_lowering_names_tf32_input_precision_without_implicit_default(self) -> None:
        schedule = Schedule.load(POSITIVE)

        source = emit(schedule, self.successor_target).source

        ast.parse(source)
        self.assertIn(
            "accumulator = tl.dot(a_tile, tl.trans(b_tile), "
            'input_precision="tf32")',
            source,
        )
        self.assertNotIn('input_precision="ieee"', source)

    def test_fp32_ieee_contract_keeps_its_existing_precision_spelling(self) -> None:
        source = emit(Schedule.load(QSA_IEEE), self.successor_target).source

        self.assertIn('input_precision="ieee"', source)
        self.assertNotIn('input_precision="tf32"', source)

    def test_bf16_operand_drift_is_localized_to_the_instruction_contract(self) -> None:
        findings = verify(Schedule.load(BF16_DRIFT), self.successor_target)

        dtype_findings = [
            finding
            for finding in findings
            if finding.code == "MMA_OPERAND_DTYPE_DIFFERS"
        ]
        self.assertEqual(len(dtype_findings), 1)
        self.assertEqual(
            dtype_findings[0].path,
            "operations[2].parameters.instruction.contract",
        )
        self.assertIs(
            dtype_findings[0].category,
            FindingCategory.HARDWARE_CONFORMANCE,
        )
        self.assertIn("reads fp32", dtype_findings[0].message)

    def test_non_fp32_accumulator_declaration_and_buffer_are_rejected(self) -> None:
        declaration = json.loads(POSITIVE.read_text(encoding="utf-8"))
        declaration["operations"][2]["parameters"]["accumulator"] = "bf16"
        with self.assertRaisesRegex(ScheduleParseError, "accumulator must be fp32"):
            Schedule.from_dict(declaration)

        storage = json.loads(POSITIVE.read_text(encoding="utf-8"))
        next(
            buffer for buffer in storage["buffers"] if buffer["name"] == "accumulator"
        )["dtype"] = "bf16"
        findings = verify(Schedule.from_dict(storage), self.successor_target)
        accumulator = next(
            finding
            for finding in findings
            if finding.code == "MMA_ACCUMULATOR_DTYPE_DIFFERS"
        )
        self.assertEqual(
            accumulator.path,
            "operations[2].parameters.instruction.contract",
        )
        self.assertIn("accumulates in fp32", accumulator.message)

    def test_each_generated_backend_rejects_the_other_instruction_family(self) -> None:
        cute_document = json.loads(CUTE_POSITIVE.read_text(encoding="utf-8"))
        cute_mma = next(
            operation for operation in cute_document["operations"] if operation["kind"] == "mma"
        )
        placed_instruction = copy.deepcopy(cute_mma["parameters"]["instruction"])
        cute_mma["parameters"]["instruction"] = {"contract": CONTRACT}
        cute_findings = cute_preflight(
            Schedule.from_dict(cute_document), self.successor_target
        )
        self.assertEqual(
            [finding.code for finding in cute_findings],
            ["CUTE_MMA_INSTRUCTION_UNSUPPORTED"],
        )

        triton_document = json.loads(POSITIVE.read_text(encoding="utf-8"))
        triton_document["operations"][2]["parameters"]["instruction"] = placed_instruction
        triton_findings = triton_preflight(
            Schedule.from_dict(triton_document), self.successor_target
        )
        self.assertEqual(
            [finding.code for finding in triton_findings],
            ["TRITON_MMA_INSTRUCTION_UNSUPPORTED"],
        )

    def test_triton_contract_cannot_claim_operand_placement(self) -> None:
        document = json.loads(POSITIVE.read_text(encoding="utf-8"))
        drifted = copy.deepcopy(document)
        drifted["operations"][2]["parameters"]["instruction"]["cta_group"] = 1

        findings = verify(Schedule.from_dict(drifted), self.successor_target)

        placement = next(
            finding
            for finding in findings
            if finding.code == "MMA_PLACEMENT_UNSUPPORTED"
        )
        self.assertEqual(placement.path, "operations[2].parameters.instruction")


if __name__ == "__main__":
    unittest.main()
