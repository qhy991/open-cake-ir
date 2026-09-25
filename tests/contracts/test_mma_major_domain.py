"""The declared B major mode owns the staged matrix's physical tile domain.

P1-P8: the existing instruction field says which axis is contiguous. This
change gives that declaration one typed shape consequence; it adds no inferred
layout, target capability, lowering body or cost claim. Native CUDA must still
qualify its own MN-major descriptor route before a Schedule can lower.
"""
from __future__ import annotations

import copy
from pathlib import Path
import unittest

from open_cake_ir.compiler import Schedule, Target
from open_cake_ir.compiler.verifier import verify

ROOT = Path(__file__).resolve().parents[2]


def document() -> dict:
    return {
        "schema_version": 2, "schedule_id": "mn-major-b-domain",
        "target": "sm_103a",
        "lowering": {"backend": "native_cuda", "entry_point": "cake_mn_major_b_domain"},
        "grid": [1, 1, 1], "metadata": {}, "roles": [
            {"name": "mma", "execution_groups": [0]}
        ], "allocations": [], "pipelines": [], "barriers": [], "tile_loops": [],
        "access_maps": [],
        "buffers": [
            {"name": "a", "space": "shared", "dtype": "bf16", "shape": [128, 128],
             "mode": "scratch", "swizzle": "swizzle_128b"},
            {"name": "b", "space": "shared", "dtype": "bf16", "shape": [128, 32],
             "mode": "scratch", "swizzle": "swizzle_64b"},
            {"name": "c", "space": "tensor", "dtype": "fp32", "shape": [128, 32],
             "mode": "scratch"},
        ],
        "operations": [{
            "id": "mma", "kind": "mma", "role": "mma", "reads": ["a", "b"],
            "writes": ["c"], "parameters": {
                "accumulator": "fp32", "tile_shape": [128, 32, 128],
                "instruction": {
                    "contract": "tcgen05.mma.cta_group::1.kind::f16",
                    "shape": [128, 32, 16], "cta_group": 1,
                    "operand_source": "shared", "operand_major": ["k", "mn"],
                },
            },
        }],
        "outputs": ["c"],
    }


def codes(value: dict) -> set[str]:
    schedule = Schedule.from_dict(value)
    target = Target.load(ROOT / "compiler/targets/sm_103a.json")
    return {item.code for item in verify(schedule, target)}


class MmaMajorDomain(unittest.TestCase):
    def test_mn_major_b_uses_physical_k_by_n(self):
        value = document()
        self.assertNotIn("MMA_INPUT_TILE_DOMAIN", codes(value))
        value = copy.deepcopy(value)
        next(b for b in value["buffers"] if b["name"] == "b")["shape"] = [32, 128]
        self.assertIn("MMA_INPUT_TILE_DOMAIN", codes(value))

    def test_k_major_keeps_physical_n_by_k(self):
        value = document()
        value["operations"][0]["parameters"]["instruction"]["operand_major"] = ["k", "k"]
        self.assertIn("MMA_INPUT_TILE_DOMAIN", codes(value))
        next(b for b in value["buffers"] if b["name"] == "b")["shape"] = [32, 128]
        self.assertNotIn("MMA_INPUT_TILE_DOMAIN", codes(value))

    def test_a_domain_is_not_silently_transposed(self):
        value = document()
        next(b for b in value["buffers"] if b["name"] == "a")["shape"] = [128, 32]
        self.assertIn("MMA_INPUT_TILE_DOMAIN", codes(value))


if __name__ == "__main__":
    unittest.main()
