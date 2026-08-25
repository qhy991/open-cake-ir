"""Contract tests for the first generated Apple Metal Lowering.

The probe proves that one BF16 atom exists.  These tests hold the Compiler to the larger
claim this slice makes: an exact Target/profile Adapter derives a complete fixed-shape
Flash-KMeans kernel and its launch requirements from Schedule decisions, without a CUDA
fallback or an Apple performance Calibration.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from open_cake_ir.compiler import Compiler, CompilerError
from open_cake_ir.compiler.emit_metal import constraints, emit
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import Target, TargetParseError


ROOT = Path(__file__).resolve().parents[2]
SM_SCHEDULE = ROOT / "corpus/schedules/flash-kmeans-b32-smoke-v2.json"
APPLE_TARGET_PATH = ROOT / "compiler/targets/apple_gpu_family9.json"
APPLE_TARGET = Target.load(APPLE_TARGET_PATH)
METAL_CONTRACT = "metal.simdgroup_mma.m8n8k8.bf16_bf16_fp32"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _apple_variant(
    *,
    simdgroups: int = 4,
    num_stages: int = 1,
    token_tile: int = 256,
    centroid_tile: int = 64,
) -> dict:
    document = copy.deepcopy(json.loads(SM_SCHEDULE.read_text(encoding="utf-8")))
    document["schedule_id"] = f"apple-fkm-sg{simdgroups}-stages{num_stages}"
    document["target"] = "apple_gpu_family9"
    document["roles"][0]["warps"] = list(range(simdgroups))
    for axis in document["program_map"]["axes"]:
        if axis["name"] == "token_block":
            axis["tile"] = token_tile
    document["tile_loops"][0]["tile"] = centroid_tile
    document["tile_loops"][0]["range_options"].update(
        {
            "num_stages": num_stages,
            "loop_unroll_factor": 1,
            "flatten": False,
            "warp_specialize": False,
            "disallow_acc_multi_buffer": False,
            "disable_licm": False,
        }
    )
    for operation in document["operations"]:
        if operation["kind"] == "mma":
            operation["parameters"]["tile_shape"] = [
                token_tile,
                centroid_tile,
                128,
            ]
            operation["parameters"]["instruction"] = {
                "contract": METAL_CONTRACT,
                "shape": [8, 8, 8],
            }
    buffers = {item["name"]: item for item in document["buffers"]}
    buffers["token_tile"]["shape"] = [token_tile, 128]
    buffers["centroid_tile"]["shape"] = [centroid_tile, 128]
    buffers["norm_tile"]["shape"] = [centroid_tile]
    buffers["best_index_tile"]["shape"] = [token_tile]
    for name in ("cross", "scaled_cross", "distance_tile"):
        buffers[name]["shape"] = [token_tile, centroid_tile]
    return document


def _draft(*, calibration_coverage: list[object] | None = None) -> dict:
    current = json.loads((ROOT / "compiler/revision.json").read_text(encoding="utf-8"))
    target = json.loads(APPLE_TARGET_PATH.read_text(encoding="utf-8"))
    current["target_definitions"]["apple_gpu_family9"] = {
        "path": "compiler/targets/apple_gpu_family9.json",
        "canonical_sha256": _canonical_sha256(target),
    }
    current["calibration_coverage"] = calibration_coverage or []
    return current


class TargetV2Test(unittest.TestCase):
    def test_architecture_contract_needs_no_cuda_identity(self) -> None:
        document = json.loads(APPLE_TARGET_PATH.read_text(encoding="utf-8"))

        self.assertNotIn("device_names", document)
        self.assertNotIn("compute_capability", document)
        self.assertEqual(APPLE_TARGET.execution_group_width, 32)
        self.assertEqual(
            APPLE_TARGET.resource_limits.maximum_threads_per_workgroup, 1024
        )
        self.assertEqual(
            APPLE_TARGET.resource_limits.maximum_threadgroup_memory_bytes, 32768
        )
        self.assertIsNone(APPLE_TARGET.resource_limits.maximum_grid)
        self.assertIsNone(APPLE_TARGET.occupancy)

    def test_target_parse_errors_do_not_leak_schedule_exceptions(self) -> None:
        document = json.loads(APPLE_TARGET_PATH.read_text(encoding="utf-8"))
        document["target_id"] = ""

        with self.assertRaises(TargetParseError):
            Target.from_dict(document)


class MetalEmissionTest(unittest.TestCase):
    def test_missing_target_synchronization_contract_fails_closed(self) -> None:
        document = json.loads(APPLE_TARGET_PATH.read_text(encoding="utf-8"))
        document["synchronization_contracts"] = []
        target = Target.from_dict(document)

        findings = constraints(Schedule.from_dict(_apple_variant()), target)

        self.assertIn(
            "METAL_SYNCHRONIZATION_CONTRACT_UNSUPPORTED",
            {finding.code for finding in findings},
        )

    def test_missing_target_threadgroup_memory_contract_fails_closed(self) -> None:
        document = json.loads(APPLE_TARGET_PATH.read_text(encoding="utf-8"))
        document["memory_spaces"].remove("shared")
        target = Target.from_dict(document)

        findings = constraints(Schedule.from_dict(_apple_variant()), target)

        self.assertIn(
            "METAL_THREADGROUP_MEMORY_CONTRACT_UNSUPPORTED",
            {finding.code for finding in findings},
        )

    def test_mismatched_target_architecture_fails_closed(self) -> None:
        document = json.loads(APPLE_TARGET_PATH.read_text(encoding="utf-8"))
        document["architecture"] = "blackwell"
        target = Target.from_dict(document)

        findings = constraints(Schedule.from_dict(_apple_variant()), target)

        self.assertIn(
            "METAL_ARCHITECTURE_UNSUPPORTED",
            {finding.code for finding in findings},
        )

    def test_derived_grid_respects_an_optional_target_limit(self) -> None:
        document = json.loads(APPLE_TARGET_PATH.read_text(encoding="utf-8"))
        document["resource_limits"]["maximum_grid"] = {"x": 1, "y": 31, "z": 1}
        target = Target.from_dict(document)

        findings = constraints(Schedule.from_dict(_apple_variant()), target)

        self.assertIn("METAL_GRID_LIMIT", {finding.code for finding in findings})

    def test_changed_target_instruction_semantics_fail_closed(self) -> None:
        document = json.loads(APPLE_TARGET_PATH.read_text(encoding="utf-8"))
        document["instruction_contracts"][0]["placement"] = "explicit"
        target = Target.from_dict(document)

        findings = constraints(Schedule.from_dict(_apple_variant()), target)

        self.assertIn(
            "METAL_TARGET_INSTRUCTION_CONTRACT_UNSUPPORTED",
            {finding.code for finding in findings},
        )

    def test_schedule_decisions_generate_the_msl_and_launch_contract(self) -> None:
        schedule = Schedule.from_dict(_apple_variant())

        emission = emit(schedule, APPLE_TARGET, entry_point="cake_flash_kmeans_assign")

        self.assertIn("simdgroup_bfloat8x8", emission.source)
        self.assertIn("simdgroup_multiply_accumulate", emission.source)
        self.assertIn("constexpr uint TOKEN_TILE = 256u", emission.source)
        self.assertIn("constexpr uint CENTROID_TILE = 64u", emission.source)
        self.assertIn("constexpr uint SIMD_GROUPS = 4u", emission.source)
        self.assertIn("candidate_index < best_index", emission.source)
        self.assertLess(
            emission.source.index("CAKE_OP:load_tokens"),
            emission.source.index("for (uint centroid_macro"),
        )
        self.assertIn("token_fragments[FEATURES / ATOM_K]", emission.source)
        for operation in schedule.operations:
            self.assertIn(f"CAKE_OP:{operation.op_id}", emission.source)
        self.assertIn("CAKE_KERNEL_END", emission.source)
        self.assertEqual(
            emission.toolchain,
            {
                "threadgroups_per_grid": [2, 32, 1],
                "threads_per_threadgroup": [128, 1, 1],
                "threadgroup_memory_bytes": 1024,
                "language_standard": "metal3.2",
                "compiler_flags": [
                    "-std=metal3.2",
                    "-fmetal-math-mode=safe",
                    "-ffp-contract=off",
                ],
            },
        )

    def test_four_to_eight_simdgroups_changes_source_and_work_division(self) -> None:
        four = emit(Schedule.from_dict(_apple_variant(simdgroups=4)), APPLE_TARGET)
        eight = emit(Schedule.from_dict(_apple_variant(simdgroups=8)), APPLE_TARGET)

        self.assertNotEqual(four.source, eight.source)
        self.assertEqual(four.toolchain["threads_per_threadgroup"], [128, 1, 1])
        self.assertEqual(eight.toolchain["threads_per_threadgroup"], [256, 1, 1])
        self.assertIn("constexpr uint SIMD_GROUPS = 8u", eight.source)

    def test_macro_tiles_change_source_constants_and_grid(self) -> None:
        wide = emit(
            Schedule.from_dict(_apple_variant(token_tile=256, centroid_tile=64)),
            APPLE_TARGET,
        )
        narrow = emit(
            Schedule.from_dict(_apple_variant(token_tile=128, centroid_tile=32)),
            APPLE_TARGET,
        )

        self.assertNotEqual(wide.source, narrow.source)
        self.assertEqual(narrow.constants["TOKEN_TILE"], 128)
        self.assertEqual(narrow.constants["CENTROID_TILE"], 32)
        self.assertEqual(narrow.toolchain["threadgroups_per_grid"], [4, 32, 1])

    @unittest.skipUnless(
        sys.platform == "darwin" and shutil.which("xcrun"),
        "requires the Apple Metal command-line toolchain",
    )
    def test_emitted_source_compiles_and_links_on_metal(self) -> None:
        emission = emit(Schedule.from_dict(_apple_variant()), APPLE_TARGET)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "kernel.metal"
            air = root / "kernel.air"
            metallib = root / "kernel.metallib"
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
            subprocess.run(
                ["xcrun", "-sdk", "macosx", "metallib", str(air), "-o", str(metallib)],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(metallib.read_bytes()[:4], b"MTLB")


class ExactRouteTest(unittest.TestCase):
    def _compiler(self, revision: dict) -> Compiler:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "revision.json"
        path.write_text(json.dumps(revision), encoding="utf-8")
        return Compiler.load(ROOT, path)

    def test_unsupported_range_option_is_a_finding_before_emission(self) -> None:
        compiler = self._compiler(_draft())

        assessment = compiler.assess(_apple_variant(num_stages=2))

        self.assertTrue(assessment.accepted)
        self.assertFalse(assessment.lowering_eligible)
        self.assertIn(
            "METAL_RANGE_OPTION_UNSUPPORTED",
            {finding.code for finding in assessment.findings},
        )
        with self.assertRaisesRegex(CompilerError, "not lowering eligible"):
            compiler.lower(assessment)

    def test_nondivisible_centroid_tile_is_refused_before_it_can_read_a_tail(self) -> None:
        compiler = self._compiler(_draft())

        assessment = compiler.assess(_apple_variant(centroid_tile=24))

        self.assertTrue(assessment.accepted)
        self.assertFalse(assessment.lowering_eligible)
        self.assertIn(
            "METAL_TILE_LOOP_UNSUPPORTED",
            {finding.code for finding in assessment.findings},
        )

    def test_unimplemented_access_and_operation_commitments_fail_closed(self) -> None:
        compiler = self._compiler(_draft())
        variants: list[tuple[str, dict, str]] = []

        access = _apple_variant()
        access["access_maps"][0]["indices"][:2] = reversed(
            access["access_maps"][0]["indices"][:2]
        )
        variants.append(("access map", access, "METAL_ACCESS_MAP_UNSUPPORTED"))

        movement = _apple_variant()
        movement["operations"][0]["parameters"]["movement"] = "tma"
        variants.append(("load movement", movement, "METAL_LOAD_CONTRACT_UNSUPPORTED"))

        store = _apple_variant()
        store["operations"][-1]["parameters"]["coalesced"] = False
        variants.append(("store", store, "METAL_STORE_CONTRACT_UNSUPPORTED"))

        dependency = _apple_variant()
        dependency["operations"][6]["depends_on"] = ["distance"]
        variants.append(
            (
                "dependency",
                dependency,
                "METAL_OPERATION_TOPOLOGY_UNSUPPORTED",
            )
        )

        scale = _apple_variant()
        scale["operations"][4]["parameters"]["broadcast_axis"] = 0
        variants.append(
            ("scale broadcast", scale, "METAL_REDUCTION_SEMANTICS_UNSUPPORTED")
        )

        instruction = _apple_variant()
        instruction["operations"][3]["parameters"]["instruction"]["cta_group"] = 1
        variants.append(
            ("instruction placement", instruction, "METAL_MMA_CONTRACT_UNSUPPORTED")
        )

        for label, document, expected in variants:
            with self.subTest(label=label):
                assessment = compiler.assess(document)
                self.assertFalse(assessment.lowering_eligible)
                self.assertIn(expected, {item.code for item in assessment.findings})

    def test_adapter_constraints_are_total_for_a_low_rank_abi(self) -> None:
        compiler = self._compiler(_draft())
        document = _apple_variant()
        next(item for item in document["buffers"] if item["name"] == "tokens")[
            "shape"
        ] = [32, 512]

        assessment = compiler.assess(document)

        self.assertFalse(assessment.lowering_eligible)
        self.assertIn("METAL_ABI_UNSUPPORTED", {item.code for item in assessment.findings})

    def test_an_understood_profile_never_falls_back_to_its_cuda_adapter(self) -> None:
        compiler = self._compiler(_draft())
        document = _apple_variant()
        document["metadata"]["profile"] = "flash_kmeans_assignment_full"

        assessment = compiler.assess(document)

        self.assertFalse(assessment.lowering_eligible)
        self.assertIn(
            "LOWERING_TARGET_PROFILE_UNSUPPORTED",
            {finding.code for finding in assessment.findings},
        )

    def test_calibration_coverage_is_an_exact_target_profile_pair(self) -> None:
        compiler = self._compiler(
            _draft(
                calibration_coverage=[
                    {"target": "sm_100a", "profile": "flash_kmeans_b32_smoke"}
                ]
            )
        )

        sm = compiler.assess_file(SM_SCHEDULE)
        apple = compiler.assess(_apple_variant())

        self.assertTrue(sm.calibration_available)
        self.assertFalse(apple.calibration_available)

    def test_ranking_refuses_a_mixed_target_set(self) -> None:
        compiler = self._compiler(_draft())
        sm = compiler.assess_file(SM_SCHEDULE)
        apple = compiler.assess(_apple_variant())

        for assessments in ([sm, apple], [apple, sm]):
            with self.subTest(first=assessments[0].target):
                with self.assertRaisesRegex(CompilerError, "share one Target"):
                    compiler.rank(assessments)


if __name__ == "__main__":
    unittest.main()
