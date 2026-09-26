"""Typed TMEM state transfer admission before a native emission qualification.

P1-P8 check: the author uses one copy operation over familiar Buffer edges; the
32x32b atom, 128-lane role, mbarrier and tensor operand placement are explicit;
construction and Verifier share the one typed meaning; the ordinary global store
keeps its canonical form; counterexamples pin the modeled domain. The atom models
register-to-TMEM BF16 packing and completion observed before tensor-core reuse.
No target or backend gains capability merely because the vocabulary can say it.
"""
from __future__ import annotations

import copy
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


def carried_document() -> dict:
    """Two repeated GEMMs expose TMEM state carry without KDA-specific math."""
    value = document()
    value["schedule_id"] = "tmem-carried-two-chunk-witness"
    for buffer in value["buffers"]:
        if buffer["name"] == "b":
            buffer["shape"] = [128, 64]
    value["buffers"].append({
        "name": "next_reg", "space": "register", "dtype": "bf16",
        "shape": [128, 64], "mode": "scratch",
    })
    value["tile_loops"] = [{
        "name": "chunks", "iterator": "chunk_index", "buffer": "b",
        "dimension": 0, "tile": 64,
        "body": ["load_b", "mma", "read_acc", "cast_next", "store_update", "store_out"],
        "carried_buffers": ["a_tmem"],
        "range_options": {"num_stages": 1, "loop_unroll_factor": 1,
                          "flatten": False, "warp_specialize": True,
                          "disallow_acc_multi_buffer": True, "disable_licm": False},
    }]
    operations = value["operations"]
    operations.insert(-1, {
        "id": "cast_next", "kind": "cast", "role": "compute",
        "reads": ["dot"], "writes": ["next_reg"],
        "parameters": {"to": "bf16"}, "depends_on": ["read_acc"],
    })
    operations.insert(-1, {
        "id": "store_update", "kind": "tmem_store", "role": "compute",
        "reads": ["next_reg"], "writes": ["a_tmem"],
        "parameters": {"destination_atom": {"op": "tcgen05.St32x32b", "repetition": 8}},
        "signals": ["state_ready"], "depends_on": ["cast_next"],
    })
    next(a for a in value["access_maps"] if a["operation"] == "load_b")["indices"] = [
        {"source": "loop_tile", "name": "chunk_index"},
        {"source": "dimension", "dimension": 1},
    ]
    return value


def readable_carried_document() -> dict:
    value = carried_document()
    value["buffers"].append({
        "name": "state_copy", "space": "register", "dtype": "bf16",
        "shape": [128, 64], "mode": "scratch",
    })
    read = {
        "id": "read_state", "kind": "load", "role": "compute",
        "reads": ["a_tmem"], "writes": ["state_copy"],
        "waits": ["state_ready"], "depends_on": ["read_acc"],
        "parameters": {
            "movement": "tmem",
            "source_atom": {"op": "tcgen05.Ld32x32b", "repetition": 16},
        },
    }
    value["operations"].insert(
        next(i for i, op in enumerate(value["operations"]) if op["id"] == "cast_next"),
        read,
    )
    body = value["tile_loops"][0]["body"]
    body.insert(body.index("cast_next"), "read_state")
    value["barriers"][0]["consumers"].append("compute")
    return value


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

    def test_carried_tmem_has_one_initializer_and_one_ordered_update(self):
        value = carried_document()
        jsonschema.Draft202012Validator(schedule_schema()).validate(value)
        self.assertEqual(codes(value), set())
        self.assertEqual(Schedule.from_dict(value).tile_loops[0].carried_buffers,
                         ("a_tmem",))

    def test_carry_declaration_does_not_erase_old_writer_guards(self):
        value = carried_document()
        value["tile_loops"][0].pop("carried_buffers")
        self.assertIn("BUFFER_MULTIPLE_WRITERS", codes(value))
        self.assertIn("TMEM_STORE_COMPLETION_OWNERSHIP", codes(value))
        value = carried_document()
        value["tile_loops"][0]["carried_buffers"] = []
        with self.assertRaises(ScheduleParseError):
            Schedule.from_dict(value)

    def test_carried_tmem_rejects_wrong_writer_and_barrier_phase(self):
        value = carried_document()
        value["tile_loops"][0]["body"].remove("store_update")
        self.assertIn("CARRIED_TMEM_WRITER_SCOPE", codes(value))
        value = carried_document()
        update = next(op for op in value["operations"] if op["id"] == "store_update")
        update["signals"] = ["done"]
        self.assertIn("CARRIED_TMEM_BARRIER", codes(value))
        value = carried_document()
        extra = copy.deepcopy(next(op for op in value["operations"] if op["id"] == "store_update"))
        extra["id"] = "extra_update"
        value["operations"].insert(-1, extra)
        value["tile_loops"][0]["body"].insert(-1, "extra_update")
        self.assertIn("CARRIED_TMEM_WRITERS", codes(value))

    def test_update_must_follow_every_in_loop_state_reader(self):
        value = carried_document()
        operations = value["operations"]
        update = next(op for op in operations if op["id"] == "store_update")
        operations.remove(update)
        operations.insert(3, update)
        body = value["tile_loops"][0]["body"]
        body.remove("store_update")
        body.insert(1, "store_update")
        self.assertIn("CARRIED_TMEM_UPDATE_ORDER", codes(value))

    def test_bf16_read_observes_the_carried_state_phase(self):
        value = readable_carried_document()
        jsonschema.Draft202012Validator(schedule_schema()).validate(value)
        self.assertEqual(codes(value), set())
        value["operations"][5]["waits"] = []
        self.assertIn("CARRIED_TMEM_BARRIER", codes(value))

    def test_bf16_read_requires_matching_packed_words_and_dtype(self):
        value = readable_carried_document()
        value["operations"][5]["parameters"]["source_atom"]["repetition"] = 64
        self.assertIn("TMEM_LOAD_BF16_PACKING", codes(value))
        value = readable_carried_document()
        next(b for b in value["buffers"] if b["name"] == "state_copy")["dtype"] = "fp32"
        self.assertIn("TMEM_LOAD_CONTRACT", codes(value))
        value = readable_carried_document()
        body = value["tile_loops"][0]["body"]
        body.remove("read_state")
        body.insert(body.index("store_update") + 1, "read_state")
        operations = value["operations"]
        read = next(op for op in operations if op["id"] == "read_state")
        operations.remove(read)
        operations.insert(
            next(i for i, op in enumerate(operations) if op["id"] == "store_update") + 1,
            read,
        )
        self.assertIn("CARRIED_TMEM_UPDATE_ORDER", codes(value))


if __name__ == "__main__":
    unittest.main()
