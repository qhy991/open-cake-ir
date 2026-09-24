"""Ordered warp MMA partials keep their boundaries and require an owning emitter."""

import json
from pathlib import Path
import unittest

import jsonschema

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.backends import cutedsl, native_cuda, triton
from open_cake_ir.compiler.ir import Schedule, ScheduleParseError
from open_cake_ir.compiler.schema import schedule_schema
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import verify


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / "fixtures/native_tinygemm_partitioned_b1.json"
TARGET = Target.load(ROOT / "compiler/targets/sm_103a.json")


def document():
    return json.loads(FIXTURE.read_text())


def mma(doc):
    return next(op for op in doc["operations"] if op["kind"] == "mma")


class MmaKPartitions(unittest.TestCase):
    def test_complete_partitioned_schedule_is_typed_and_only_native_route_emits_it(self):
        jsonschema.validate(document(), schedule_schema())
        schedule = Schedule.from_dict(document())
        parameters = next(op.parameters for op in schedule.operations if op.kind.value == "mma")
        self.assertEqual(parameters.k_partitions,
                         ((0, 256), (256, 512), (512, 768), (768, 1024)))
        self.assertEqual(parameters.contribution_ranges, ((0, 1024),))
        self.assertNotIn("MMA_PARTITION_STORAGE_MISMATCH", {f.code for f in verify(schedule, TARGET)})
        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        assessment = compiler.assess(document())
        self.assertTrue(assessment.accepted)
        self.assertTrue(assessment.lowering_eligible)
        self.assertFalse(native_cuda.preflight(schedule, TARGET))
        self.assertIn("TRITON_MMA_K_PARTITIONS_UNSUPPORTED",
                      {f.code for f in triton.preflight(schedule, TARGET)})
        self.assertIn("CUTE_MMA_K_PARTITIONS_UNSUPPORTED",
                      {f.code for f in cutedsl.requirements(schedule)})

    def test_partitions_are_complete_contiguous_and_not_a_second_k_ranges_spelling(self):
        for partitions in (
            [[0, 256]], [[0, 256], [257, 1024]], [[0, 256], [255, 1024]],
            [[0, 256], [256, 1000]],
        ):
            with self.subTest(partitions=partitions):
                changed = document()
                mma(changed)["parameters"]["k_partitions"] = partitions
                with self.assertRaisesRegex(ScheduleParseError, "k_partitions"):
                    Schedule.from_dict(changed)
        changed = document()
        mma(changed)["parameters"]["k_ranges"] = [[0, 256]]
        with self.assertRaisesRegex(ScheduleParseError, "excludes k_ranges"):
            Schedule.from_dict(changed)

    def test_partition_groups_and_atom_boundaries_are_verified(self):
        changed = document()
        changed["roles"][0]["execution_groups"] = [0, 1, 2]
        self.assertIn("MMA_PARTITION_GROUP_MISMATCH",
                      {f.code for f in verify(Schedule.from_dict(changed), TARGET)})
        changed = document()
        mma(changed)["parameters"]["k_partitions"] = [
            [0, 255], [255, 512], [512, 768], [768, 1024]
        ]
        self.assertIn("MMA_PARTITION_ATOM_MISMATCH",
                      {f.code for f in verify(Schedule.from_dict(changed), TARGET)})

    def test_shared_staging_requires_a_register_result(self):
        changed = document()
        next(buffer for buffer in changed["buffers"] if buffer["name"] == "acc")["space"] = "shared"
        self.assertIn("MMA_PARTITION_STORAGE_MISMATCH",
                      {f.code for f in verify(Schedule.from_dict(changed), TARGET)})
