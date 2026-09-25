"""One explicit register-tile transpose, including its analysis and lowering gates.

P1-P8: transpose is the familiar rank-two value operation, names its shape change
in the Schedule, has one canonical empty parameter object, and is legal only for
same-dtype register tiles. It makes no global-memory traffic or floating-point
arithmetic claim. Each Target must still declare the operation before emission.
"""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import jsonschema

from open_cake_ir.compiler import Schedule, Target
from open_cake_ir.compiler.backends.triton import emit, preflight
from open_cake_ir.compiler.ir import ScheduleParseError
from open_cake_ir.compiler.performance.work import work_bound
from open_cake_ir.compiler.schema import schedule_schema
from open_cake_ir.compiler.verifier import verify

ROOT = Path(__file__).resolve().parents[2]


def document() -> dict:
    return {
        "schema_version": 2,
        "schedule_id": "rank-two-register-transpose",
        "target": "sm_103a",
        "lowering": {"backend": "triton", "entry_point": "cake_register_transpose"},
        "metadata": {},
        "roles": [{"name": "compute", "execution_groups": [0, 1, 2, 3]}],
        "allocations": [], "pipelines": [], "barriers": [], "tile_loops": [],
        "program_map": {"axes": [
            {"name": "head", "axis": 0, "buffer": "input", "dimension": 0, "tile": 1}
        ]},
        "buffers": [
            {"name": "input", "space": "global", "dtype": "bf16", "shape": [1, 16, 32], "mode": "input"},
            {"name": "tile", "space": "register", "dtype": "bf16", "shape": [16, 32], "mode": "scratch"},
            {"name": "turned", "space": "register", "dtype": "bf16", "shape": [32, 16], "mode": "scratch"},
            {"name": "output", "space": "global", "dtype": "bf16", "shape": [1, 32, 16], "mode": "output"},
        ],
        "operations": [
            {"id": "read", "kind": "load", "role": "compute", "reads": ["input"],
             "writes": ["tile"], "parameters": {"movement": "global"}},
            {"id": "turn", "kind": "transpose", "role": "compute", "reads": ["tile"],
             "writes": ["turned"], "depends_on": ["read"], "parameters": {}},
            {"id": "write", "kind": "store", "role": "compute", "reads": ["turned"],
             "writes": ["output"], "depends_on": ["turn"],
             "parameters": {"coalesced": True}},
        ],
        "access_maps": [
            {"operation": "read", "buffer": "input", "boundary": "mask_tiled_axes",
             "indices": [{"source": "program", "name": "head"},
                         {"source": "dimension", "dimension": 1},
                         {"source": "dimension", "dimension": 2}]},
            {"operation": "write", "buffer": "output", "boundary": "mask_tiled_axes",
             "indices": [{"source": "program", "name": "head"},
                         {"source": "dimension", "dimension": 1},
                         {"source": "dimension", "dimension": 2}]},
        ],
        "outputs": ["output"],
    }


def target(*, admit: bool) -> Target:
    value = json.loads((ROOT / "compiler/targets/sm_103a.json").read_text())
    if admit:
        value["operation_kinds"].append("transpose")
    return Target.from_dict(value)


def codes(value: dict, *, admit: bool = True) -> set[str]:
    schedule = Schedule.from_dict(value)
    return {item.code for item in verify(schedule, target(admit=admit))
            if item.blocks_lowering}


class RegisterTileTranspose(unittest.TestCase):
    def test_typed_transpose_emits_one_visible_permutation(self):
        value = document()
        jsonschema.Draft202012Validator(schedule_schema()).validate(value)
        schedule = Schedule.from_dict(value)
        self.assertEqual(codes(value), set())
        self.assertEqual(preflight(schedule, target(admit=True)), ())
        source = emit(schedule, target(admit=True)).source
        self.assertIn("turned = tl.trans(tile)", source)
        self.assertIn("tl.store(", source)
        bound = work_bound(schedule)
        self.assertEqual(bound.flops, 0)
        self.assertEqual(bound.compulsory_read_bytes, 1024)
        self.assertEqual(bound.compulsory_written_bytes, 1024)
        self.assertIn("TARGET_OPERATION_UNSUPPORTED", codes(value, admit=False))

    def test_wrong_shape_dtype_and_storage_are_localized(self):
        for field, replacement, expected in (
            ("shape", [16, 32], "TRANSPOSE_SHAPE"),
            ("dtype", "fp32", "TRANSPOSE_DTYPE"),
            ("space", "shared", "TRANSPOSE_STORAGE"),
        ):
            with self.subTest(field=field):
                value = document()
                next(b for b in value["buffers"] if b["name"] == "turned")[field] = replacement
                self.assertIn(expected, codes(value))
                self.assertIn("TRITON_TRANSPOSE_CONTRACT",
                              {item.code for item in preflight(Schedule.from_dict(value), target(admit=True))})

    def test_parameters_have_one_canonical_spelling(self):
        value = document()
        value["operations"][1]["parameters"] = {"axes": [1, 0]}
        with self.assertRaises(ScheduleParseError):
            Schedule.from_dict(value)


if __name__ == "__main__":
    unittest.main()
