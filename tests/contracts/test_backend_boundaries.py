"""Current backend ownership, retired syntax and direct-emission refusals."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import get_args, get_origin, get_type_hints
from unittest.mock import patch

from open_cake_ir.compiler import Compiler, EmitError, Finding, FindingSeverity
from open_cake_ir.compiler.backends import BACKENDS, BackendModule, common, cutedsl, metal, triton
from open_cake_ir.compiler.diagnostics import Finding as DiagnosticFinding
from open_cake_ir.compiler.ir import DType, LoweringBackend, Schedule, ScheduleParseError
from open_cake_ir.compiler.target import Target

ROOT = Path(__file__).resolve().parents[2]


def document(name: str) -> dict:
    return json.loads((ROOT / "corpus/schedules" / (name + ".json")).read_text())


class BackendBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        cls.target = Target.load(ROOT / "compiler/targets/sm_100a.json")

    def test_public_backend_return_annotations_share_the_canonical_finding_type(self):
        self.assertIs(Finding, DiagnosticFinding)
        self.assertIs(get_type_hints(common.refusal)["return"], DiagnosticFinding)
        entries = [("common.vocabulary_findings", common.vocabulary_findings)]
        for entry in ("requirements", "preflight"):
            entries.append(("BackendModule." + entry, getattr(BackendModule, entry)))
            entries.extend((backend.value + "." + entry, getattr(descriptor.module, entry))
                           for backend, descriptor in BACKENDS.items())
        for name, entry in entries:
            with self.subTest(entry=name):
                annotation = get_type_hints(entry)["return"]
                self.assertIs(get_origin(annotation), tuple)
                self.assertEqual(get_args(annotation), (DiagnosticFinding, Ellipsis))

    def test_retired_asset_documents_are_structural_refusals_without_a_route(self):
        for name in ("tinygemm2-stage4-split-k", "tinygemm2-stage4-split-k-reduction-drift"):
            with self.subTest(name=name):
                original = document(name)
                with self.assertRaisesRegex(ScheduleParseError, "schedule.lowering.backend is unsupported"):
                    Schedule.from_dict(original)
                assessment = self.compiler.assess(original)
                self.assertFalse(assessment.accepted)
                self.assertFalse(assessment.lowering_eligible)
                self.assertIsNone(assessment.route)
                self.assertEqual(assessment.guidance, ())
                self.assertEqual(
                    [(f.code, f.path, f.blocks_acceptance, f.blocks_lowering) for f in assessment.findings],
                    [("SCHEDULE_STRUCTURE", "schedule.lowering.backend", True, True)],
                )
                with self.assertRaisesRegex(ValueError, "not lowering eligible"):
                    self.compiler.lower(assessment)

    def test_generic_resource_fixture_preserves_derived_analysis(self):
        fixture = document("tinygemm2-stage4-split-k")
        # This historical resource fixture names register-only warp MMA while
        # placing its operands and accumulator in shared memory. Keep its
        # derived analysis inspectable without accepting that hardware mismatch.
        fixture["lowering"]["backend"] = "cutlass_cute_dsl"
        assessment = self.compiler.assess(fixture)
        self.assertFalse(assessment.accepted)
        self.assertIn("MMA_OPERAND_SOURCE_MISMATCH", [f.code for f in assessment.findings])
        self.assertFalse(assessment.lowering_eligible)
        self.assertEqual(assessment.analysis["grid"], (64, 1, 1))
        self.assertEqual(assessment.analysis["total_warps"], 12)
        self.assertEqual(assessment.analysis["operation_counts"],
                         {"epilogue": 1, "load": 3, "mma": 1, "reduce": 1})

    def test_static_inventory_covers_exact_current_backend_vocabulary(self):
        self.assertEqual(set(BACKENDS), set(LoweringBackend))
        with self.assertRaises(TypeError):
            BACKENDS[LoweringBackend.TRITON] = BACKENDS[LoweringBackend.METAL]
        for backend in ("checked_cuda_asset", "cuda", "ptx"):
            mutated = document("fma-b8-smoke")
            mutated["lowering"]["backend"] = backend
            with self.subTest(backend=backend), self.assertRaises(ScheduleParseError):
                Schedule.from_dict(mutated)

    def test_triton_public_pointer_spelling_matches_emitted_compile_signatures(self):
        expected = {DType.BF16: "*bf16", DType.FP16: "*fp16", DType.FP32: "*fp32",
                    DType.FP8_E4M3: "*fp8e4nv", DType.INT32: "*i32"}
        for dtype, spelling in expected.items():
            with self.subTest(dtype=dtype):
                fixture = document("cast-b8-smoke")
                for buffer in fixture["buffers"]:
                    if buffer["name"] in {"x", "x_tile"}:
                        buffer["dtype"] = dtype.value
                emission = triton.emit(Schedule.from_dict(fixture), self.target)
                self.assertEqual(triton.pointer_type(dtype), spelling)
                self.assertEqual(emission.toolchain["signature"]["x"], spelling)
        for invalid in (None, "fp32", "int64", 32):
            with self.subTest(invalid=invalid), self.assertRaises(EmitError):
                triton.pointer_type(invalid)

    def test_backend_vocabulary_refusal_has_the_same_owner_at_both_entrypoints(self):
        for module, name, target_path in (
            (triton, "fma-b8-smoke", "sm_100a"),
            (cutedsl, "flash-kmeans-assignment-full", "sm_100a"),
            (metal, "metal-elementwise-odd", "apple_gpu_family8"),
        ):
            fixture = document(name)
            schedule = Schedule.from_dict(fixture)
            target = Target.load(ROOT / "compiler/targets" / (target_path + ".json"))
            # Remove one currently required representation from the backend's
            # actual declared support, exercising its shared refusal boundary.
            for attribute, removed, code in (
                ("SUPPORTED_DTYPES", schedule.buffers[0].dtype, "BACKEND_DTYPE_UNEMITTABLE"),
                ("SUPPORTED_OPERATION_KINDS", schedule.operations[0].kind, "BACKEND_OPERATION_UNEMITTABLE"),
            ):
                with self.subTest(module=module.__name__, attribute=attribute), patch.object(
                    module, attribute, getattr(module, attribute) - {removed}
                ):
                    direct = [f for f in module.preflight(schedule, target) if f.code == code]
                    assessment = self.compiler.assess(fixture)
                    projected = [f for f in assessment.findings if f.code == code]
                    self.assertEqual(direct, projected)
                    self.assertTrue(direct)
                    self.assertTrue(assessment.accepted)
                    self.assertFalse(assessment.lowering_eligible)
                    with self.assertRaises(EmitError):
                        module.emit(schedule, target)

    def test_argmin_restriction_is_shared_by_direct_and_compiler_emission(self):
        fixture = document("flash-kmeans-b32-warp-specialized-argmin")
        direct = triton.preflight(Schedule.from_dict(fixture), self.target)
        finding = next(f for f in direct if f.code == "TRITON_WARP_SPECIALIZED_ARGMIN_UNSUPPORTED")
        self.assertIsInstance(finding, Finding)
        self.assertFalse(finding.blocks_acceptance)
        self.assertTrue(finding.blocks_lowering)
        self.assertIn(finding, self.compiler.assess(fixture).findings)
        with self.assertRaisesRegex(EmitError, "value-and-index argmin"):
            triton.emit(Schedule.from_dict(fixture), self.target)

    def test_metal_owns_its_advisory_report_and_direct_emit_does_not_block_it(self):
        fixture = document("metal-elementwise-odd")
        schedule = Schedule.from_dict(fixture)
        target = Target.load(ROOT / "compiler/targets/apple_gpu_family8.json")
        report = next(f for f in metal.preflight(schedule, target) if f.code == "METAL_SIMD_EXECUTION")
        self.assertIs(report.severity, FindingSeverity.REPORT)
        self.assertFalse(report.blocks_acceptance)
        self.assertFalse(report.blocks_lowering)
        self.assertIn("unmodeled", report.message)
        self.assertIn(report, self.compiler.assess(fixture).findings)
        self.assertTrue(metal.emit(schedule, target).source)
