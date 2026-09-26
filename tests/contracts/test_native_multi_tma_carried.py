"""Multiple carried B tiles keep one typed owner per TMA/MMA edge.

The source test pins admission and refusal. Four-transfer B300 correctness is
retained in F-2026-09-24-003; this new typed route still needs its own device
qualification before it can serve a complete KDA Schedule.
"""
from __future__ import annotations

import copy
import unittest
from pathlib import Path

from open_cake_ir.compiler import Compiler, Schedule, Target
from open_cake_ir.compiler.backends.native_cuda import preflight
from open_cake_ir.compiler.verifier import verify
from tests.contracts.test_tmem_state_contract import carried_document


ROOT = Path(__file__).resolve().parents[2]


def multi_document() -> dict:
    value = carried_document()
    value["schedule_id"] = "tmem-carried-two-b-tiles"
    value["lowering"]["entry_point"] = "cake_tmem_carried_two_b_tiles"

    by_buffer = {item["name"]: item for item in value["buffers"]}
    for original, successor in (("b", "b_other"), ("c", "c_other"),
                                ("b_stage", "b_other_stage"),
                                ("acc", "acc_other"), ("dot", "dot_other")):
        item = copy.deepcopy(by_buffer[original])
        item["name"] = successor
        if original == "b_stage":
            item["allocation"] = "b_other_smem"
        if original == "acc":
            item["byte_offset"] = 49152
        value["buffers"].append(item)
    value["allocations"].insert(1, {
        "name": "b_other_smem", "space": "shared", "size_bytes": 8192,
    })
    tensor = next(item for item in value["allocations"] if item["name"] == "tensor")
    tensor["size_bytes"] = 131072
    tensor["tensor_columns"] = 256
    value["outputs"].append("c_other")
    next(item for item in value["barriers"] if item["name"] == "b_ready")["count"] = 2
    value["barriers"].append({
        "name": "done_other", "count": 1, "producers": ["mma"],
        "consumers": ["compute"], "mechanism": "mbarrier",
    })

    original = {item["id"]: item for item in value["operations"]}
    load_other = copy.deepcopy(original["load_b"])
    load_other.update(id="load_b_other", reads=["b_other"],
                      writes=["b_other_stage"])
    mma_other = copy.deepcopy(original["mma"])
    mma_other.update(id="mma_other", reads=["a_tmem", "b_other_stage"],
                     writes=["acc_other"], signals=["done_other"],
                     depends_on=["store_tmem", "load_b_other"])
    read_other = copy.deepcopy(original["read_acc"])
    read_other.update(id="read_acc_other", reads=["acc_other"],
                      writes=["dot_other"], waits=["done_other"],
                      depends_on=["mma_other"])
    store_other = copy.deepcopy(original["store_out"])
    store_other.update(id="store_out_other", reads=["dot_other"],
                       writes=["c_other"], depends_on=["read_acc_other"])
    original["store_update"]["depends_on"].append("read_acc_other")
    value["operations"] = [
        original["load_state"], original["store_tmem"],
        original["load_b"], load_other, original["mma"], mma_other,
        original["read_acc"], read_other, original["cast_next"],
        original["store_update"], original["store_out"], store_other,
    ]
    value["tile_loops"][0]["body"] = [
        "load_b", "load_b_other", "mma", "mma_other", "read_acc",
        "read_acc_other", "cast_next", "store_update", "store_out",
        "store_out_other",
    ]
    load_access = copy.deepcopy(next(item for item in value["access_maps"]
                                     if item["operation"] == "load_b"))
    load_access.update(operation="load_b_other", buffer="b_other")
    store_access = copy.deepcopy(next(item for item in value["access_maps"]
                                      if item["operation"] == "store_out"))
    store_access.update(operation="store_out_other", buffer="c_other")
    value["access_maps"].extend((load_access, store_access))
    return value


class NativeMultiTmaCarried(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        cls.target = Target.load(ROOT / "compiler/targets/sm_103a.json")

    def test_two_staged_operands_are_admitted_and_emitted(self):
        value = multi_document()
        schedule = Schedule.from_dict(value)
        self.assertFalse([f for f in verify(schedule, self.target)
                          if f.blocks_lowering])
        self.assertEqual(preflight(schedule, self.target), ())
        assessment = self.compiler.assess(value)
        self.assertTrue(assessment.lowering_eligible, [
            (f.code, f.path) for f in assessment.findings if f.blocks_lowering
        ])
        lowering = self.compiler.lower(assessment)
        source = lowering.source
        self.assertIn("cake_init(bar1+stage, 2)", source)
        self.assertEqual(source.count("cake_expect(bar1+stage, 8192)"), 2)
        self.assertIn("// CAKE_OP: load_b_other", source)
        self.assertIn("// CAKE_OP: mma_other", source)
        self.assertIn("cake_commit(bar3)", source)

    def test_each_b_tile_needs_its_own_loop_indexed_source(self):
        value = multi_document()
        access = next(item for item in value["access_maps"]
                      if item["operation"] == "load_b_other")
        access["indices"][0] = {"source": "dimension", "dimension": 0}
        schedule = Schedule.from_dict(value)
        self.assertIn("NATIVE_CARRIED_MMA_DOMAIN",
                      {f.code for f in preflight(schedule, self.target)})

    def test_ready_arrivals_and_matching_source_extent_are_required(self):
        value = multi_document()
        next(item for item in value["barriers"] if item["name"] == "b_ready")["count"] = 1
        self.assertIn("NATIVE_PIPELINE_BARRIER",
                      {f.code for f in preflight(Schedule.from_dict(value), self.target)})
        value = multi_document()
        next(item for item in value["buffers"] if item["name"] == "b_other")["shape"][0] = 64
        self.assertIn("NATIVE_CARRIED_MMA_DOMAIN",
                      {f.code for f in preflight(Schedule.from_dict(value), self.target)})


if __name__ == "__main__":
    unittest.main()
