"""Typed TMEM state transfer admission before a native emission qualification.

P1-P8 check: the author uses one copy operation over familiar Buffer edges; the
32x32b atom, 128-lane role, mbarrier and tensor operand placement are explicit;
construction and Verifier share the one typed meaning; the ordinary global store
keeps its canonical form; counterexamples pin the modeled domain. The atom models
register-to-TMEM BF16 packing and completion observed before tensor-core reuse.
No target or backend gains capability merely because the vocabulary can say it.
"""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import jsonschema

from open_cake_ir.compiler import Schedule, Target
from open_cake_ir.compiler.ir import ScheduleParseError
from open_cake_ir.compiler.schema import schedule_schema
from open_cake_ir.compiler.verifier import verify

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/tmem-state-mma-sm103a.json"


def document() -> dict:
    return json.loads(FIXTURE.read_text())


def target(*, admit_store: bool) -> Target:
    value = json.loads((ROOT / "compiler/targets/sm_103a.json").read_text())
    value["operation_kinds"] = [
        kind for kind in value["operation_kinds"] if kind != "tmem_store"
    ]
    if admit_store:
        value["operation_kinds"].append("tmem_store")
    return Target.from_dict(value)


def codes(value: dict, *, admit_store: bool = True) -> set[str]:
    schedule = Schedule.from_dict(value)
    return {finding.code for finding in verify(schedule, target(admit_store=admit_store))
            if finding.blocks_lowering}


class TmemStateContract(unittest.TestCase):
    def test_complete_contract_is_typed_but_undeclared_target_refuses(self):
        value = document()
        jsonschema.Draft202012Validator(schedule_schema()).validate(value)
        self.assertEqual(codes(value), set())
        self.assertIn("TARGET_OPERATION_UNSUPPORTED", codes(value, admit_store=False))
        self.assertEqual(value["operations"][1]["parameters"], {
            "destination_atom": {"op": "tcgen05.St32x32b", "repetition": 8}
        })

    def test_wrong_atom_and_storage_are_localized(self):
        value = document()
        value["operations"][1]["parameters"]["destination_atom"]["repetition"] = 16
        self.assertIn("TMEM_STORE_ATOM", codes(value))
        value = document()
        next(b for b in value["buffers"] if b["name"] == "a_tmem")["dtype"] = "fp32"
        self.assertIn("TMEM_STORE_CONTRACT", codes(value))
        value = document()
        next(b for b in value["buffers"] if b["name"] == "a_reg")["shape"] = [64, 64]
        self.assertIn("TMEM_STORE_CONTRACT", codes(value))

    def test_missing_128_lane_completion_is_rejected(self):
        value = document()
        value["roles"][0]["execution_groups"] = [0, 1]
        self.assertIn("TMEM_STORE_ROLE_WIDTH", codes(value))
        value = document()
        value["barriers"][0]["count"] = 3
        self.assertIn("TMEM_STORE_COMPLETION", codes(value))
        value = document()
        value["operations"][1].pop("signals")
        self.assertIn("TMEM_STORE_COMPLETION", codes(value))

    def test_tensor_mma_must_read_tmem_a_and_shared_b_after_store(self):
        value = document()
        value["operations"][3]["waits"] = ["b_ready"]
        self.assertIn("TMEM_STORE_CONSUMER_WAIT", codes(value))
        self.assertIn("OP_CROSS_ROLE_RACE", codes(value))
        value = document()
        next(b for b in value["buffers"] if b["name"] == "a_tmem")["space"] = "shared"
        self.assertIn("MMA_OPERAND_SOURCE_MISMATCH", codes(value))

    def test_incomplete_atom_is_refused_at_construction(self):
        value = document()
        value["operations"][1]["parameters"] = {}
        with self.assertRaises(ScheduleParseError):
            Schedule.from_dict(value)


if __name__ == "__main__":
    unittest.main()
