"""Contracts for the finite Apple Metal generated backend Adapter."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from open_cake_ir.compiler import Compiler, CompilerError
from open_cake_ir.compiler.emit_metal import emit, preflight
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import Target


ROOT = Path(__file__).resolve().parents[2]
TARGET = Target.load(ROOT / "compiler/targets/apple_gpu_family9.json")
GATHER = ROOT / "corpus/schedules/indexed-gather-b8-metal-family9.json"
GATHER_DRIFT = ROOT / "corpus/schedules/indexed-gather-b8-metal-family9-load-drift.json"
COMBINE = ROOT / "corpus/schedules/kda-weighted-combine-b8-metal-family9.json"
COMBINE_DRIFT = (
    ROOT / "corpus/schedules/kda-weighted-combine-b8-metal-family9-role-drift.json"
)


def _schedule(path: Path) -> Schedule:
    return Schedule.from_dict(json.loads(path.read_text(encoding="utf-8")))


class MetalPreflightTest(unittest.TestCase):
    def test_two_primitive_graphs_are_admitted_without_a_workload_registry(
        self,
    ) -> None:
        for path in (GATHER, COMBINE):
            with self.subTest(path=path.name):
                self.assertEqual(preflight(_schedule(path), TARGET), ())

    def test_backend_commitment_drifts_are_findings(self) -> None:
        expected = {
            GATHER_DRIFT: "METAL_OPERATION_CONTRACT_UNSUPPORTED",
            COMBINE_DRIFT: "METAL_EXECUTION_GROUPS_UNSUPPORTED",
        }
        for path, code in expected.items():
            with self.subTest(path=path.name):
                self.assertIn(
                    code,
                    {item.code for item in preflight(_schedule(path), TARGET)},
                )

    def test_buffer_order_is_part_of_the_lowering_abi(self) -> None:
        document = json.loads(GATHER.read_text(encoding="utf-8"))
        document["buffers"][0], document["buffers"][1] = (
            document["buffers"][1],
            document["buffers"][0],
        )

        findings = preflight(Schedule.from_dict(document), TARGET)

        self.assertIn(
            "METAL_OPERATION_GRAPH_UNSUPPORTED",
            {item.code for item in findings},
        )

    def test_architecture_mismatch_cannot_fall_back(self) -> None:
        document = json.loads(
            (ROOT / "compiler/targets/apple_gpu_family9.json").read_text(
                encoding="utf-8"
            )
        )
        document["architecture"] = "another_architecture"
        target = Target.from_dict(document)

        findings = preflight(_schedule(GATHER), target)

        self.assertIn(
            "METAL_ARCHITECTURE_UNSUPPORTED",
            {item.code for item in findings},
        )


class MetalEmissionTest(unittest.TestCase):
    def test_gather_derives_binding_and_four_simdgroup_launch(self) -> None:
        schedule = _schedule(GATHER)

        emission = emit(schedule, TARGET)

        self.assertEqual(
            emission.toolchain["buffer_order"],
            ["expert_rows", "expert_ids", "row_ids", "gathered_rows"],
        )
        self.assertEqual(emission.toolchain["threadgroups_per_grid"], [8, 1, 1])
        self.assertEqual(emission.toolchain["threads_per_threadgroup"], [128, 1, 1])
        self.assertEqual(emission.toolchain["threadgroup_memory_bytes"], 0)
        self.assertIn("expert >= 0", emission.source)
        self.assertIn("row >= 0", emission.source)
        for operation in schedule.operations:
            self.assertEqual(emission.source.count(f"CAKE_OP:{operation.op_id}"), 1)

    def test_weighted_combine_uses_one_simdgroup_and_fixed_route_order(self) -> None:
        schedule = _schedule(COMBINE)

        emission = emit(schedule, TARGET)

        self.assertEqual(
            emission.toolchain["buffer_order"],
            ["expert_rows", "expert_ids", "row_ids", "route_weights", "output"],
        )
        self.assertEqual(emission.toolchain["threads_per_threadgroup"], [32, 1, 1])
        self.assertIn("for (uint route = 0; route < ROUTES; ++route)", emission.source)
        self.assertIn("combined += weighted_rows[route]", emission.source)
        self.assertIn(
            "output[token * FEATURES + feature] = bfloat(combined)", emission.source
        )
        for operation in schedule.operations:
            self.assertEqual(emission.source.count(f"CAKE_OP:{operation.op_id}"), 1)

    def test_emission_is_deterministic(self) -> None:
        schedule = _schedule(COMBINE)

        first = emit(schedule, TARGET)
        second = emit(schedule, TARGET)

        self.assertEqual(first, second)

    @unittest.skipUnless(
        sys.platform == "darwin" and shutil.which("xcrun") is not None,
        "requires the Apple Metal command-line toolchain",
    )
    def test_both_sources_compile_and_link_as_metal32(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            air_paths: list[Path] = []
            for path in (GATHER, COMBINE):
                emission = emit(_schedule(path), TARGET)
                source = output / f"{path.stem}.metal"
                air = output / f"{path.stem}.air"
                source.write_text(emission.source, encoding="utf-8")
                subprocess.run(
                    [
                        "xcrun",
                        "-sdk",
                        "macosx",
                        "metal",
                        *emission.toolchain["compiler_flags"],
                        "-c",
                        str(source),
                        "-o",
                        str(air),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                air_paths.append(air)
            metallib = output / "operators.metallib"
            subprocess.run(
                [
                    "xcrun",
                    "-sdk",
                    "macosx",
                    "metallib",
                    *(str(path) for path in air_paths),
                    "-o",
                    str(metallib),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(metallib.read_bytes()[:4], b"MTLB")


class MetalCompilerIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_assess_and_lower_use_the_generated_backend_seam(self) -> None:
        for path, expected_threads in ((GATHER, 128), (COMBINE, 32)):
            with self.subTest(path=path.name):
                assessment = self.compiler.assess_file(path)
                self.assertTrue(assessment.accepted)
                self.assertTrue(assessment.lowering_eligible)
                self.assertEqual(assessment.findings, ())

                lowering = self.compiler.lower(assessment)
                self.assertTrue(lowering.generated)
                self.assertEqual(lowering.route.backend.value, "metal")
                self.assertEqual(
                    lowering.toolchain_requirements["threads_per_threadgroup"],
                    [expected_threads, 1, 1],
                )
                self.assertEqual(
                    set(lowering.source_map),
                    {operation.op_id for operation in _schedule(path).operations},
                )

    def test_drift_is_refused_before_lowering(self) -> None:
        for path in (GATHER_DRIFT, COMBINE_DRIFT):
            with self.subTest(path=path.name):
                assessment = self.compiler.assess_file(path)
                self.assertTrue(assessment.accepted)
                self.assertFalse(assessment.lowering_eligible)
                with self.assertRaisesRegex(CompilerError, "not lowering eligible"):
                    self.compiler.lower(assessment)

    def test_schedule_id_cannot_escape_its_source_comment(self) -> None:
        document = json.loads(GATHER.read_text(encoding="utf-8"))
        document["schedule_id"] = "innocent\n#error INJECTED"

        assessment = self.compiler.assess(document)
        self.assertTrue(assessment.accepted)
        self.assertTrue(assessment.lowering_eligible)
        lowering = self.compiler.lower(assessment)

        self.assertIn(r"innocent\n#error INJECTED", lowering.source)
        self.assertNotIn("\n#error INJECTED", lowering.source)

    def test_ranking_refuses_mixed_targets(self) -> None:
        sm100 = self.compiler.assess_file(
            ROOT / "corpus/schedules/indexed-gather-b8-smoke.json"
        )
        apple = self.compiler.assess_file(GATHER)

        with self.assertRaisesRegex(CompilerError, "share one Target"):
            self.compiler.rank([sm100, apple])


if __name__ == "__main__":
    unittest.main()
