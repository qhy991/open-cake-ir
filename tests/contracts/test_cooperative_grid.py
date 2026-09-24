"""Cooperative persistence is a typed launch commitment, not a barrier."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import unittest
import jsonschema

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.backends import BACKENDS
from open_cake_ir.compiler.ir import LoweringBackend, Schedule, ScheduleParseError
from open_cake_ir.compiler.schema import schedule_schema
from open_cake_ir.compiler.target import Target, TargetParseError
from open_cake_ir.compiler.verifier import verify


ROOT = Path(__file__).resolve().parents[2]
SCHEDULE = ROOT / "corpus/schedules/rmsnorm-b128-persistent.json"
TARGET = ROOT / "compiler/targets/sm_100a.json"


def document() -> dict:
    value = json.loads(SCHEDULE.read_text())
    value["program_map"]["cooperative"] = True
    return value


class CooperativeGridContract(unittest.TestCase):
    def test_cooperation_requires_a_persistent_map_at_construction(self) -> None:
        value = document()
        value["program_map"]["persistent"] = False
        value["program_map"].pop("traversal")
        with self.assertRaisesRegex(ScheduleParseError, "cooperative requires persistent"):
            Schedule.from_dict(value)
        value = document()
        jsonschema.Draft202012Validator(schedule_schema()).validate(value)
        self.assertTrue(Schedule.from_dict(value).program_map.cooperative)
        original = json.loads(SCHEDULE.read_text())
        self.assertFalse(Schedule.from_dict(original).program_map.cooperative)

    def test_target_fact_is_explicit_and_tri_state(self) -> None:
        source = json.loads(TARGET.read_text())
        self.assertIsNone(Target.from_dict(source).cooperative_grid)
        for declared in (False, True):
            modified = deepcopy(source)
            modified["cooperative_grid"] = declared
            self.assertIs(Target.from_dict(modified).cooperative_grid, declared)
        for invalid in (None, 1, "true"):
            modified = deepcopy(source)
            modified["cooperative_grid"] = invalid
            with self.subTest(value=invalid), self.assertRaises(TargetParseError):
                Target.from_dict(modified)

    def test_undeclared_and_refused_targets_block_the_promise(self) -> None:
        schedule = Schedule.from_dict(document())
        unmodeled = Target.load(TARGET)
        missing = verify(schedule, unmodeled)
        self.assertIn("TARGET_COOPERATIVE_GRID_UNMODELED", [f.code for f in missing])
        self.assertTrue(next(f for f in missing if f.code == "TARGET_COOPERATIVE_GRID_UNMODELED").blocks_lowering)
        unsupported = replace(unmodeled, cooperative_grid=False)
        self.assertIn("TARGET_COOPERATIVE_GRID_UNSUPPORTED",
                      [f.code for f in verify(schedule, unsupported)])
        supported = replace(unmodeled, cooperative_grid=True)
        self.assertFalse(any(f.code.startswith("TARGET_COOPERATIVE_GRID")
                             for f in verify(schedule, supported)))
        assessment = Compiler.load(ROOT).assess(document())
        self.assertFalse(assessment.lowering_eligible)
        self.assertIn("TARGET_COOPERATIVE_GRID_UNMODELED", [f.code for f in assessment.findings])
        self.assertIn("BACKEND_COOPERATIVE_GRID_UNSUPPORTED", [f.code for f in assessment.findings])
        self.assertFalse(BACKENDS[LoweringBackend.TRITON].module.COOPERATIVE_GRID)
        self.assertTrue(all(type(backend.module.COOPERATIVE_GRID) is bool
                            for backend in BACKENDS.values()))


if __name__ == "__main__":
    unittest.main()
