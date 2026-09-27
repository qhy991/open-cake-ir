"""Exact B300 inverse-MMA carried route and its intended refusal owners."""
from __future__ import annotations

from copy import deepcopy
import unittest

from jsonschema import Draft202012Validator

from open_cake_ir.compiler import Schedule
from open_cake_ir.compiler.backends import native_cuda
from open_cake_ir.compiler.schema import schedule_schema
from open_cake_ir.compiler.target import declared_target
from open_cake_ir.compiler.verifier import verify
from tests.contracts.test_native_kda_prepared_beta_bridge import document as base_document


_RENAMES = {
    "p": "inverse_b", "p_stage": "inverse_stage",
    "p_smem": "inverse_smem", "p_ready": "inverse_ready",
    "load_p": "load_inverse",
}


def _renamed(value):
    if isinstance(value, dict):
        return {key: _renamed(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_renamed(item) for item in value]
    return _RENAMES.get(value, value) if isinstance(value, str) else value


def inverse_document() -> dict:
    value = _renamed(deepcopy(base_document()))
    value["schedule_id"] = "native-k128-inverse-mma-h64-contract"
    value["lowering"]["entry_point"] = "cake_kda_h64_inverse_mma_contract"
    value["buffers"].extend([
        {"name": "rhs_bf", "space": "register", "dtype": "bf16",
         "shape": [128, 32], "mode": "scratch"},
        {"name": "rhs_tmem", "space": "tensor", "dtype": "bf16",
         "shape": [128, 32], "mode": "scratch", "allocation": "tensor",
         "byte_offset": 304 * 512},
        {"name": "inverse_acc", "space": "tensor", "dtype": "fp32",
         "shape": [128, 32], "mode": "scratch", "allocation": "tensor",
         "byte_offset": 320 * 512},
    ])
    value["barriers"] = [b for b in value["barriers"]
                         if b["name"] != "inverse_ready"]
    value["barriers"].extend([
        {"name": "rhs_ready", "count": 4, "producers": ["compute"],
         "consumers": ["mma"], "mechanism": "mbarrier"},
        {"name": "inverse_done", "count": 1, "producers": ["mma"],
         "consumers": ["compute"], "mechanism": "mbarrier"},
    ])
    value["pipelines"].append({"name": "inverse", "stages": 1})
    load = next(op for op in value["operations"] if op["id"] == "load_inverse")
    load.pop("signals")
    operations = value["operations"]
    position = next(i for i, op in enumerate(operations) if op["id"] == "solve")
    operations.pop(position)
    operations[position:position] = [
        {"id": "round_rhs", "kind": "cast", "role": "compute",
         "reads": ["rhs_beta"], "writes": ["rhs_bf"],
         "depends_on": ["apply_beta"], "parameters": {"to": "bf16"}},
        {"id": "publish_rhs", "kind": "tmem_store", "role": "compute",
         "reads": ["rhs_bf"], "writes": ["rhs_tmem"],
         "depends_on": ["round_rhs"], "signals": ["rhs_ready"],
         "parameters": {"destination_atom": {"op": "tcgen05.St32x32b",
                                              "repetition": 8}}},
        {"id": "mma_inverse", "kind": "mma", "role": "mma",
         "reads": ["rhs_tmem", "inverse_stage"],
         "writes": ["inverse_acc"],
         "depends_on": ["publish_rhs", "load_inverse"],
         "waits": ["rhs_ready"], "signals": ["inverse_done"],
         "pipeline": "inverse",
         "parameters": {"accumulator": "fp32", "tile_shape": [128, 32, 32],
                        "instruction": {
                            "contract": "tcgen05.mma.cta_group::1.kind::f16",
                            "shape": [128, 32, 16], "cta_group": 1,
                            "operand_source": "tensor",
                            "operand_major": ["k", "mn"]}}},
        {"id": "read_inverse", "kind": "load", "role": "compute",
         "reads": ["inverse_acc"], "writes": ["updates"],
         "depends_on": ["mma_inverse"], "waits": ["inverse_done"],
         "parameters": {"movement": "tmem",
                        "source_atom": {"op": "tcgen05.Ld32x32b",
                                        "repetition": 32}}},
    ]
    next(op for op in operations if op["id"] == "round_updates")[
        "depends_on"] = ["read_inverse"]
    body = value["tile_loops"][0]["body"]
    position = body.index("solve")
    body[position:position+1] = ["round_rhs", "publish_rhs",
                                 "mma_inverse", "read_inverse"]
    return value


def blocked_codes(value: dict) -> set[str]:
    schedule = Schedule.from_dict(value)
    target = declared_target("sm_103a")
    return {finding.code for finding in native_cuda.preflight(schedule, target)
            if finding.blocks_lowering}


class NativeKdaInverseMmaTest(unittest.TestCase):
    def test_exact_inverse_route_emits_qualified_k32_and_preserves_k128(self):
        value = inverse_document()
        Draft202012Validator(schedule_schema()).validate(value)
        schedule = Schedule.from_dict(value)
        target = declared_target("sm_103a")
        self.assertFalse([f for f in verify(schedule, target) if f.blocks_lowering])
        self.assertEqual(blocked_codes(value), set())
        emission = native_cuda.emit(schedule, target)
        self.assertEqual(emission.toolchain["grid"], [64, 1, 1])
        self.assertEqual(emission.toolchain["dynamic_shared_bytes"], 57488)
        self.assertIn("// CAKE_OP: mma_inverse", emission.source)
        self.assertIn("// CAKE_OP: round_rhs fused", emission.source)
        self.assertIn("(*tm3 + 304)", emission.source)
        self.assertIn("(*tm3 + 320)", emission.source)
        self.assertNotIn("// CAKE_OP: solve\n", emission.source)
        self.assertEqual(blocked_codes(base_document()), set())

    def test_rhs_barrier_must_wait_for_four_compute_warps(self):
        value = inverse_document()
        next(b for b in value["barriers"] if b["name"] == "rhs_ready")["count"] = 2
        self.assertIn("NATIVE_INVERSE_MMA_DOMAIN", blocked_codes(value))

    def test_mn_major_k32_is_exact_not_a_broad_target_inheritance(self):
        value = inverse_document()
        next(op for op in value["operations"] if op["id"] == "mma_inverse")[
            "parameters"]["instruction"]["operand_major"] = ["k", "k"]
        self.assertIn("NATIVE_INVERSE_MMA_DOMAIN", blocked_codes(value))

    def test_inverse_tmem_columns_must_not_alias_carried_state(self):
        value = inverse_document()
        next(b for b in value["buffers"] if b["name"] == "rhs_tmem")[
            "byte_offset"] = 0
        self.assertIn("NATIVE_INVERSE_MMA_DOMAIN", blocked_codes(value))

    def test_inverse_factor_has_exactly_one_mma_reader(self):
        value = inverse_document()
        next(op for op in value["operations"] if op["id"] == "mma_query")[
            "reads"][1] = "inverse_stage"
        self.assertIn("NATIVE_INVERSE_MMA_DOMAIN", blocked_codes(value))

    def test_same_warp_inverse_stage_never_invents_a_barrier(self):
        value = inverse_document()
        next(op for op in value["operations"] if op["id"] == "load_inverse")[
            "signals"] = ["rhs_ready"]
        self.assertIn("NATIVE_INVERSE_MMA_DOMAIN", blocked_codes(value))


if __name__ == "__main__":
    unittest.main()
