"""Contracts for architecture-neutral Target schema v2.

Lowering routes name mechanisms, not vendors.  These tests therefore keep the existing
``triton`` route and constrain only the hardware facts a gfx1151 Target owns: wave width,
workgroup resources, and the deliberate absence of CUDA and tensor-memory identities.
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.ir import MemorySpace, Schedule
from open_cake_ir.compiler.target import Target, TargetParseError
from open_cake_ir.compiler.verifier import verify


ROOT = Path(__file__).resolve().parents[2]
SM100A_TARGET = ROOT / "compiler/targets/sm_100a.json"
RMSNORM = ROOT / "corpus/schedules/rmsnorm-b8-smoke.json"
TENSOR_SCHEDULE = ROOT / "corpus/schedules/flash-kmeans-assignment-full.json"


def _canonical_sha256(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _gfx1151_target() -> dict[str, object]:
    return {
        "schema_version": 2,
        "target_id": "gfx1151",
        "architecture": "gfx1151",
        "execution_group_width": 32,
        "memory_spaces": ["global", "shared", "register"],
        "operation_kinds": ["load", "reduce", "elementwise", "store"],
        "resource_limits": {
            "maximum_threads_per_workgroup": 1024,
            "maximum_threadgroup_memory_bytes": 65536,
            "maximum_grid": {"x": 2147483647, "y": 65535, "z": 65535},
        },
        "instruction_contracts": [],
        "synchronization_contracts": [],
        "citations": [
            {
                "kind": "device_observation",
                "source": "gfx1151 contract fixture",
            }
        ],
    }


class TargetV2ContractTests(unittest.TestCase):
    def test_gfx1151_owns_wave_width_without_inventing_cuda_or_tensor_facts(self) -> None:
        document = _gfx1151_target()

        target = Target.from_dict(document)

        self.assertEqual(target.schema_version, 2)
        self.assertEqual(target.execution_group_width, 32)
        self.assertEqual(target.warp_size, 32)
        self.assertIsNone(target.register_budget_group_width)
        self.assertIsNone(target.warps_per_warpgroup)
        self.assertIsNone(target.compute_capability)
        self.assertEqual(target.device_names, ())
        self.assertEqual(
            target.resource_limits.maximum_threads_per_workgroup,
            1024,
        )
        self.assertEqual(
            target.resource_limits.maximum_execution_groups_per_workgroup,
            32,
        )
        self.assertEqual(
            target.resource_limits.maximum_threadgroup_memory_bytes,
            65536,
        )
        self.assertIsNone(target.resource_limits.maximum_tensor_memory_bytes)
        self.assertIsNone(target.resource_limits.capacity(MemorySpace.TENSOR))
        self.assertNotIn("compute_capability", document)
        self.assertNotIn(
            "maximum_tensor_memory_bytes",
            document["resource_limits"],
        )

    def test_tensor_memory_limit_is_required_exactly_when_the_space_exists(self) -> None:
        missing = _gfx1151_target()
        missing["memory_spaces"].append("tensor")

        with self.assertRaisesRegex(TargetParseError, "tensor.*limit"):
            Target.from_dict(missing)

        declared = copy.deepcopy(missing)
        declared["resource_limits"]["maximum_tensor_memory_bytes"] = 262144
        target = Target.from_dict(declared)
        self.assertEqual(
            target.resource_limits.capacity(MemorySpace.TENSOR),
            262144,
        )

        invented = _gfx1151_target()
        invented["resource_limits"]["maximum_tensor_memory_bytes"] = 262144
        with self.assertRaisesRegex(TargetParseError, "without tensor memory"):
            Target.from_dict(invented)

    def test_execution_group_width_is_required_and_must_tile_the_workgroup(self) -> None:
        missing = _gfx1151_target()
        missing.pop("execution_group_width")
        with self.assertRaisesRegex(TargetParseError, "execution_group_width"):
            Target.from_dict(missing)

        partial = _gfx1151_target()
        partial["execution_group_width"] = 48
        with self.assertRaisesRegex(TargetParseError, "whole execution groups"):
            Target.from_dict(partial)

    def test_schema_v2_compute_capability_is_optional_not_synthesized(self) -> None:
        absent = Target.from_dict(_gfx1151_target())
        declared = _gfx1151_target()
        declared["compute_capability"] = [11, 5]

        self.assertIsNone(absent.compute_capability)
        self.assertEqual(Target.from_dict(declared).compute_capability, (11, 5))

    def test_gfx1151_refuses_a_role_register_budget_it_cannot_issue(self) -> None:
        target = Target.from_dict(_gfx1151_target())
        document = json.loads(RMSNORM.read_text(encoding="utf-8"))
        document["target"] = "gfx1151"
        document["roles"][0]["registers_per_thread"] = 64

        findings = verify(Schedule.from_dict(document), target)
        unsupported = tuple(
            item
            for item in findings
            if item.code == "ROLE_REGISTERS_TARGET_UNSUPPORTED"
        )

        self.assertEqual(len(unsupported), 1)
        self.assertTrue(unsupported[0].blocks_lowering)

    def test_missing_tensor_hardware_is_reported_without_inventing_a_capacity(self) -> None:
        target = Target.from_dict(_gfx1151_target())
        document = json.loads(TENSOR_SCHEDULE.read_text(encoding="utf-8"))
        document["target"] = "gfx1151"

        findings = verify(Schedule.from_dict(document), target)
        codes = {item.code for item in findings}

        self.assertIn("TARGET_MEMORY_SPACE_UNSUPPORTED", codes)
        self.assertNotIn("TARGET_TENSOR_MEMORY_LIMIT", codes)
        self.assertNotIn("TARGET_TENSOR_COLUMN_LIMIT", codes)


class CompilerTargetCompatibilityTests(unittest.TestCase):
    def test_compiler_loads_v2_without_cuda_fields_and_keeps_triton_mechanism_neutral(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target_path = root / "compiler/targets/gfx1151.json"
            target_path.parent.mkdir(parents=True)
            target_document = _gfx1151_target()
            target_path.write_text(json.dumps(target_document), encoding="utf-8")
            corpus_path = root / "corpus/manifest.json"
            corpus_path.parent.mkdir(parents=True)
            corpus_path.write_text("{}\n", encoding="utf-8")
            revision = {
                "schema_version": 1,
                "revision_id": "open-cake-ir-target-v2-contract",
                "state": "draft",
                "target_definitions": {
                    "gfx1151": {
                        "path": "compiler/targets/gfx1151.json",
                        "canonical_sha256": _canonical_sha256(target_document),
                    }
                },
                "corpus_manifest": "corpus/manifest.json",
                "calibration_coverage": [],
            }
            revision_path = root / "compiler/revision.json"
            revision_path.write_text(json.dumps(revision), encoding="utf-8")

            compiler = Compiler.load(root, revision_path)
            schedule = json.loads(RMSNORM.read_text(encoding="utf-8"))
            schedule["target"] = "gfx1151"
            assessment = compiler.assess(schedule)

        self.assertEqual(assessment.target, "gfx1151")
        self.assertEqual(assessment.route.backend.value, "triton")
        self.assertTrue(assessment.accepted, assessment.findings)
        self.assertTrue(assessment.lowering_eligible, assessment.findings)

    def test_schema_v1_b200_target_and_corpus_lowering_digests_remain_unchanged(self) -> None:
        target = Target.load(SM100A_TARGET)
        self.assertEqual(target.schema_version, 1)
        self.assertEqual(target.compute_capability, (10, 0))
        self.assertEqual(target.execution_group_width, 32)
        self.assertEqual(target.warp_size, 32)
        self.assertEqual(target.register_budget_group_width, 4)
        self.assertEqual(target.warps_per_warpgroup, 4)
        self.assertEqual(target.resource_limits.maximum_tensor_memory_bytes, 262144)

        report = Compiler.load(ROOT, ROOT / "compiler/revision.json").check_corpus()
        self.assertTrue(report.passed)
        self.assertEqual(
            report.case_count,
            len(json.loads((ROOT / "corpus/manifest.json").read_text())["cases"]),
        )
        self.assertTrue(
            all(
                case.observed_lowering_source_sha256
                == case.expected_lowering_source_sha256
                for case in report.cases
            )
        )


if __name__ == "__main__":
    unittest.main()
