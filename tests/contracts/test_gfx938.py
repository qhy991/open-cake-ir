"""The exact gfx938 target, its wave64 arithmetic and its AMDGCN lowering route.

Everything here runs on the host without a DCU. The on-device evidence that the emitted
program compiles to an hsaco and matches a reference across five input distributions is
retained separately and cited by F-2026-09-14-001; a passing test file is not that
evidence and does not stand in for it.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.backends import triton
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import CodeObject, Target, TargetParseError, Vendor
from open_cake_ir.compiler.toolchain import (
    TritonCompilation,
    _parse_amdgcn_resources,
    inspect_triton_resources,
    route_for_code_object,
    triton_route,
)

ROOT = Path(__file__).resolve().parents[2]


def _document(name: str) -> dict:
    return json.loads((ROOT / "corpus/schedules" / f"{name}.json").read_text(encoding="utf-8"))


def _route(target_id: str):
    """The route from the compile contract the emitter writes for one declared Target.

    Offline compilation never opens a Target document, so `triton_route` reads the
    code object, architecture and lane width the emitter carried over from the Target
    it held; this is that contract for a Target of this checkout.
    """
    target = Target.load(ROOT / "compiler/targets" / f"{target_id}.json")
    return triton_route({"target": target_id, **triton.target_route_facts(target)})


class TargetDocumentTest(unittest.TestCase):
    """The declared hardware facts, and the two spellings the parser refuses."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.document = json.loads(
            (ROOT / "compiler/targets/gfx938.json").read_text(encoding="utf-8")
        )
        cls.target = Target.from_dict(cls.document)

    def test_the_declared_width_and_vendor_belong_to_the_document(self) -> None:
        # Hygon, not AMD: the hardware reports vendor C-3000 and device type HCU, and
        # the amdgcn-amd-amdhsa triple in its object is the ABI, not the manufacturer.
        self.assertIs(self.target.vendor, Vendor.HYGON)
        self.assertEqual(self.target.warp_size, 64)
        # No CUDA capability, and no borrowed warpgroup rule against a 64-lane wavefront.
        self.assertIsNone(self.target.compute_capability)
        self.assertIsNone(self.target.warps_per_warpgroup)

    def test_the_cta_budget_agrees_with_the_declared_lane_width(self) -> None:
        limits = self.target.resource_limits
        self.assertEqual(limits.maximum_threads_per_cta, 1024)
        self.assertEqual(
            limits.maximum_warps_per_cta * self.target.warp_size,
            limits.maximum_threads_per_cta,
        )
        self.assertEqual(limits.maximum_shared_memory_bytes, 65536)
        # No tensor memory space is declared, so the limit is unmodeled rather than zero.
        self.assertIsNone(limits.maximum_tensor_memory_bytes)

    def test_the_per_multiprocessor_facts_were_read_from_a_device(self) -> None:
        kinds = {citation["kind"] for citation in self.document["citations"]}
        self.assertIn("device_observation", kinds)
        self.assertEqual(
            self.document["occupancy"],
            {
                "multiprocessor_count": 64,
                "registers_per_multiprocessor": 131072,
                "shared_memory_per_multiprocessor_bytes": 65536,
                "maximum_threads_per_multiprocessor": 2560,
            },
        )
        occupancy = self.target.occupancy
        assert occupancy is not None
        self.assertEqual(
            occupancy.maximum_threads_per_multiprocessor % self.target.warp_size, 0
        )

    def test_no_peak_is_declared_without_a_calibration(self) -> None:
        self.assertNotIn("peak", self.document)
        self.assertIsNone(self.target.peak)

    def test_a_non_nvidia_document_carries_no_cuda_capability(self) -> None:
        with self.assertRaises(TargetParseError):
            Target.from_dict({**self.document, "compute_capability": [9, 3]})

    def test_the_declared_width_is_not_a_default(self) -> None:
        document = {k: v for k, v in self.document.items() if k != "warp_size"}
        with self.assertRaises(TargetParseError):
            Target.from_dict(document)


class Wave64ArithmeticTest(unittest.TestCase):
    """One Schedule, two targets: the lane width is what separates the verdicts."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_thirty_two_slots_fit_a_cuda_cta_and_overrun_a_wave64_one(self) -> None:
        wide = _document("gfx938-wave64-thread-extent-refusal")
        refused = self.compiler.assess(wide)
        codes = {finding.code for finding in refused.findings}
        self.assertFalse(refused.accepted)
        self.assertIn("TARGET_THREAD_LIMIT", codes)
        self.assertIn("TARGET_WARP_LIMIT", codes)
        self.assertTrue(
            any("2048 threads" in finding.message for finding in refused.findings),
            [finding.message for finding in refused.findings],
        )

        control = {**wide, "schedule_id": "control-32-slots-sm100a", "target": "sm_100a"}
        accepted = self.compiler.assess(control)
        self.assertTrue(accepted.accepted, [f.message for f in accepted.findings])
        self.assertTrue(accepted.lowering_eligible)

    def test_the_accepted_schedule_launches_sixty_four_lane_slots(self) -> None:
        document = _document("gfx938-rmsnorm-b8-smoke")
        assessment = self.compiler.assess(document)
        self.assertTrue(assessment.accepted, [f.message for f in assessment.findings])
        self.assertTrue(assessment.lowering_eligible)
        schedule = Schedule.from_dict(document)
        target = Target.load(ROOT / "compiler/targets/gfx938.json")
        self.assertEqual(schedule.total_warp_extent * target.warp_size, 256)
        # The residency report counts those 256 threads, not 128.
        self.assertTrue(
            any("256 of 2560 threads" in finding.message for finding in assessment.findings),
            [finding.message for finding in assessment.findings],
        )


class TritonAdmissionTest(unittest.TestCase):
    """What this backend emits for, and the one budget it refuses to pretend to hold."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        cls.target = Target.load(ROOT / "compiler/targets/gfx938.json")

    def test_a_declared_register_budget_is_refused_rather_than_dropped(self) -> None:
        assessment = self.compiler.assess(_document("gfx938-register-budget-refusal"))
        codes = {finding.code for finding in assessment.findings}
        self.assertIn("TRITON_AMDGCN_REGISTER_BUDGET_UNENFORCEABLE", codes)
        self.assertFalse(assessment.lowering_eligible)
        # The same budget is enforceable on CUDA and carries no such refusal there.
        cuda = self.compiler.assess(_document("rmsnorm-b8-smoke"))
        self.assertNotIn(
            "TRITON_AMDGCN_REGISTER_BUDGET_UNENFORCEABLE",
            {finding.code for finding in cuda.findings},
        )

    def test_the_backend_admits_the_exact_target_and_the_compiler_refuses_metal(self) -> None:
        """The backend declares the objects it emits; the Compiler holds that against
        the object the Target declares it runs. Neither keeps a table of target ids."""
        document = _document("gfx938-rmsnorm-b8-smoke")
        self.assertEqual(triton.preflight(Schedule.from_dict(document), self.target), ())
        self.assertEqual(triton.CODE_OBJECTS, {CodeObject.CUBIN, CodeObject.HSACO})
        assessment = self.compiler.assess({**document, "target": "apple_gpu_family9"})
        refusals = [f for f in assessment.findings if f.code == "BACKEND_TARGET_UNSUPPORTED"]
        self.assertEqual(len(refusals), 1, [f.code for f in assessment.findings])
        self.assertEqual(
            refusals[0].message,
            "the triton backend emits ['cubin', 'hsaco'] and the 'apple_gpu_family9' "
            "target runs 'metal_binary_archive'")
        self.assertFalse(assessment.lowering_eligible)

    def test_the_route_reads_the_document_it_is_handed_and_keeps_no_copy(self) -> None:
        """A drifted document is followed, not caught: the backend has no second table
        of ids, architectures or widths that could disagree with the Target.

        Built with `replace` rather than a document, because a document naming an
        unknown architecture is refused by the parser first and would prove nothing
        about what the backend reads.
        """
        schedule = Schedule.from_dict(_document("gfx938-rmsnorm-b8-smoke"))
        for field, value, key in (("target_id", "gfx942", "triton_arch"),
                                  ("warp_size", 32, "warp_size")):
            with self.subTest(field=field):
                drifted = replace(self.target, **{field: value})
                self.assertEqual(triton.preflight(schedule, drifted), ())
                self.assertEqual(triton.target_route_facts(drifted)[key], value)
        # Only the code object decides admission, and the emitter refuses a foreign one
        # by its name rather than by the vendor or architecture beside it.
        foreign = replace(self.target, code_object=CodeObject.METAL_BINARY_ARCHIVE)
        with self.assertRaisesRegex(Exception, "emits no 'metal_binary_archive' code object"):
            triton.target_route_facts(foreign)


class TritonRouteTest(unittest.TestCase):
    """Neither vendor's artifact names, target text nor scratch fields are assumed."""

    def test_the_amdgcn_route_names_its_own_artifacts(self) -> None:
        route = _route("gfx938")
        self.assertEqual(route.gpu_backend, "hip")
        self.assertIs(route.code_object, CodeObject.HSACO)
        self.assertEqual(route.architecture, "gfx938")
        self.assertEqual(route.warp_size, 64)
        self.assertEqual(route.binary_role, "hsaco")
        self.assertEqual(route.text_role, "amdgcn")
        self.assertNotIn("ptx", route.artifact_roles)
        self.assertNotIn("cubin", route.artifact_roles)
        # HIPOptions defines no global scratch field, so nothing reads one.
        self.assertEqual(route.scratch_fields, ("profile_scratch_size",))

    def test_the_route_and_the_target_document_cannot_drift(self) -> None:
        """Offline compilation never opens a Target document, so the facts it needs ride
        the compile contract the emitter wrote from the Target; there is no second copy."""
        target = Target.load(ROOT / "compiler/targets/gfx938.json")
        route = _route(target.target_id)
        self.assertEqual(route.warp_size, target.warp_size)
        self.assertEqual(route.architecture, target.target_id)
        self.assertEqual(route.code_object.value, target.code_object.value)

    def test_the_cuda_route_is_unchanged(self) -> None:
        for target, architecture in (("sm_100a", 100), ("sm_103a", 103)):
            with self.subTest(target=target):
                route = _route(target)
                self.assertEqual(route.gpu_backend, "cuda")
                self.assertIs(route.code_object, CodeObject.CUBIN)
                self.assertEqual(route.architecture, architecture)
                self.assertEqual(route.warp_size, 32)
                self.assertEqual(route.binary_role, "cubin")
                self.assertEqual(route.text_role, "ptx")
                self.assertEqual(
                    route.scratch_fields, ("global_scratch_size", "profile_scratch_size")
                )

    def test_the_cuda_inspector_refuses_an_amdgcn_compilation(self) -> None:
        """It has no CUBIN to read, and says so instead of failing on a missing key."""
        compilation = TritonCompilation(
            source=b"# lowered\n", target="gfx938", entry_point="k",
            artifacts={"amdgcn": b"", "hsaco": b""},
            threads_per_cta=256, dynamic_shared_bytes=0, compiler_version="3.6.0",
            code_object="hsaco",
        )
        with self.assertRaises(ValueError) as raised:
            inspect_triton_resources(compilation, "/nonexistent/cuobjdump")
        self.assertIn("inspect_amdgcn_resources", str(raised.exception))

    def test_a_contract_that_names_no_route_is_refused_not_stepped_down(self) -> None:
        """No key is defaulted: a contract without the object, the architecture or the
        width is a differing contract, never a CUDA compilation at 32 lanes."""
        complete = {"target": "gfx938", **triton.target_route_facts(
            Target.load(ROOT / "compiler/targets/gfx938.json"))}
        self.assertEqual(triton_route(complete).architecture, "gfx938")
        for missing in ("target", "code_object", "triton_arch", "warp_size"):
            with self.subTest(missing=missing):
                with self.assertRaisesRegex(ValueError, "Triton compile contract differs"):
                    triton_route({k: v for k, v in complete.items() if k != missing})
        for changed in ({"target": ""}, {"target": None}, {"warp_size": 0},
                        {"warp_size": "64"}, {"triton_arch": 938},
                        {"code_object": "cubin", "triton_arch": "gfx938"}):
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(ValueError, "Triton compile contract differs"):
                    triton_route({**complete, **changed})
        # A declared object Triton does not emit is refused by its name.
        with self.assertRaisesRegex(ValueError, "no 'metal_binary_archive' code object"):
            route_for_code_object("metal_binary_archive")
        with self.assertRaisesRegex(ValueError, "no 'elf' code object"):
            route_for_code_object("elf")

    def test_the_emitted_target_line_must_name_the_exact_isa(self) -> None:
        import re

        pattern = _route("gfx938").target_pattern
        for line in (
            b"amdhsa.target:   'amdgcn-amd-amdhsa--gfx938:xnack-'",
            b"amdhsa.target:   'amdgcn-amd-amdhsa--gfx938:sramecc+:xnack-'",
            b"amdhsa.target:   'amdgcn-amd-amdhsa--gfx938'",
            # Unquoted, because the emitter quotes only what YAML makes it quote: a
            # feature suffix contains a colon and a bare target name does not. Measured
            # on this toolchain, which writes gfx938:xnack- quoted and gfx1151 bare.
            b"amdhsa.target:   amdgcn-amd-amdhsa--gfx938",
        ):
            with self.subTest(line=line):
                self.assertIsNotNone(re.search(pattern, line, re.MULTILINE))
        for line in (
            b"amdhsa.target:   'amdgcn-amd-amdhsa--gfx9380:xnack-'",
            b"amdhsa.target:   amdgcn-amd-amdhsa--gfx9380",
            b"amdhsa.target:   'amdgcn-amd-amdhsa--gfx942:xnack-'",
            b"amdhsa.target:   'amdgcn-amd-amdhsa--gfx938:xnack'",
        ):
            with self.subTest(line=line):
                self.assertIsNone(re.search(pattern, line, re.MULTILINE))


# One kernel's .amdgpu_metadata note, as the DTK 26.04 AMDGPU backend wrote it for the
# lowered gfx938-rmsnorm-b8-smoke Schedule. Trimmed to the fields the parser reads plus
# enough neighbours to keep the shape honest.
_METADATA = """\t.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .max_flat_workgroup_size: 256
    .name:           _cake_gfx938_rmsnorm_b8_smoke_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     47
    .sgpr_spill_count: 0
    .symbol:         _cake_gfx938_rmsnorm_b8_smoke_kernel.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     134
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   'amdgcn-amd-amdhsa--gfx938:xnack-'
amdhsa.version:
  - 1
  - 2
...
"""


class AmdgcnResourceParseTest(unittest.TestCase):
    """The compiler's own note replaces cuobjdump, and refuses the same ambiguities."""

    ENTRY = "_cake_gfx938_rmsnorm_b8_smoke_kernel"

    def test_it_reads_the_per_lane_allocation(self) -> None:
        self.assertEqual(
            _parse_amdgcn_resources(_METADATA, self.ENTRY),
            {
                "registers_per_thread": 134,
                "static_shared_bytes": 0,
                "local_bytes": 0,
                "stack_bytes": 0,
            },
        )

    def test_the_scalar_count_is_not_folded_into_a_per_thread_number(self) -> None:
        """.sgpr_count is per wavefront; reading it as per thread would be 47 too many."""
        self.assertEqual(
            _parse_amdgcn_resources(_METADATA, self.ENTRY)["registers_per_thread"], 134
        )

    def test_a_note_naming_another_kernel_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            _parse_amdgcn_resources(_METADATA, "some_other_kernel")

    def test_a_two_kernel_note_is_refused_rather_than_guessed(self) -> None:
        doubled = _METADATA.replace(
            "amdhsa.target:",
            "  - .agpr_count:     0\n    .name:           second_kernel\namdhsa.target:",
            1,
        )
        with self.assertRaises(ValueError):
            _parse_amdgcn_resources(doubled, self.ENTRY)

    def test_a_missing_field_is_refused(self) -> None:
        for label in (".vgpr_count", ".group_segment_fixed_size",
                      ".private_segment_fixed_size"):
            with self.subTest(label=label):
                stripped = "\n".join(
                    line for line in _METADATA.splitlines()
                    if not line.strip().startswith(label + ":")
                )
                with self.assertRaises(ValueError):
                    _parse_amdgcn_resources(stripped, self.ENTRY)


class LaunchManifestTargetTest(unittest.TestCase):
    """A sealed launch reads limits every Target declares, not a CUDA capability.

    `CudaKernelSpec.from_dict` resolved its target through `compiler.target.cuda_target`,
    which decodes an sm_1xxa capability first. Grid, block and shared-memory limits are
    not CUDA's -- every Target document declares them -- so a gfx938 manifest was refused
    as `unsupported exact CUDA target`: a vendor's words for a caller that named no
    vendor. This is the same defect AGENTS.md records for `executable_role` in this layer.
    """

    def test_a_gfx938_launch_seals_against_its_own_declared_limits(self) -> None:
        from open_cake_ir.evaluation.cuda_manifest import CudaKernelSpec
        spec = CudaKernelSpec.from_dict({
            "target": "gfx938", "kernel_name": "cake_rmsnorm",
            "grid": [128, 1, 1], "block": [256, 1, 1],
            "dynamic_shared_memory_bytes": 0, "hidden_null_pointer_parameters": 1,
        })
        self.assertEqual((spec.target, spec.block_threads), ("gfx938", 256))
        # 16 wave64 warps is what the document declares, and 1088 threads is past it.
        with self.assertRaises(ValueError):
            CudaKernelSpec.from_dict({
                "target": "gfx938", "kernel_name": "cake_rmsnorm",
                "grid": [128, 1, 1], "block": [1088, 1, 1],
                "dynamic_shared_memory_bytes": 0, "hidden_null_pointer_parameters": 1,
            })

    def test_a_target_no_document_declares_is_refused_without_naming_a_vendor(self) -> None:
        from open_cake_ir.compiler.target import TargetParseError
        from open_cake_ir.evaluation.cuda_manifest import _launch_target
        for absent in ("sm_999a", "gfx1", "apple_gpu_family99"):
            with self.subTest(absent=absent):
                with self.assertRaises(TargetParseError) as raised:
                    _launch_target(absent)
                self.assertIn(absent, str(raised.exception))
                self.assertNotIn("CUDA", str(raised.exception))

    def test_a_target_id_cannot_name_a_file_outside_the_target_directory(self) -> None:
        from open_cake_ir.compiler.target import TargetParseError
        from open_cake_ir.evaluation.cuda_manifest import _launch_target
        for escape in ("../source_set", "/etc/passwd", "sm_103a/../../secret", "", None, 938):
            with self.subTest(escape=escape):
                with self.assertRaises(TargetParseError):
                    _launch_target(escape)


class MeasurementCoverageTest(unittest.TestCase):
    """Where the DCU stops today, stated as a test rather than found by running one."""

    def study(self, backend: str):
        from open_cake_ir.tasks.normalization.study import study_template
        from open_cake_ir.tasks.workloads import create_task
        from open_cake_ir.evaluation.workload import WorkloadContract
        import json, tempfile
        from pathlib import Path as _Path
        document, source = create_task("silu", backend=backend, rows=2, columns=8)
        directory = _Path(tempfile.mkdtemp())
        workload_path = directory / "workload.json"
        starter = directory / "starter.py"
        workload_path.write_text(json.dumps(document))
        starter.write_text(source)
        root = _Path(__file__).resolve().parents[2]
        return study_template(root, WorkloadContract(document), workload_path, starter,
                              harness="claude-code", model="m", effort="high", turns=2,
                              token_budget=12000)

    def test_the_dcu_study_names_its_own_timer_and_its_own_call_count(self) -> None:
        """Not CUPTI's, which is what falling through produced on a machine without it.

        The source was withheld until something measured the device: roctracer's
        per-dispatch device time, read against rocprofv2's own reading of the same kernel
        and shape. The call count travels with it -- CUPTI spends six calls on its
        calibration callbacks and this source spends none.
        """
        from open_cake_ir.evaluation.paired import PAIRED_HIP_KIND, paired_protocol
        policy = self.study("triton-dcu")["evaluation_protocol"]
        self.assertEqual(policy["paired_timing"]["kind"], PAIRED_HIP_KIND)
        self.assertEqual(policy["search_evaluation"], "correctness_then_paired_hip_dispatch")
        self.assertNotIn("cupti", policy["search_evaluation"])
        self.assertNotIn("measurement_coverage", policy)
        self.assertEqual(paired_protocol(policy).route_calls_per_cohort, 11 + 25)

    def test_a_backend_with_no_named_timer_still_states_the_limitation(self) -> None:
        """The behaviour that carried the DCU before it had a source, kept for the next one.

        A timing source is minted by measuring, so a backend arrives without one. What it
        must not do is inherit another target's timer: it says so, and its Study carries a
        coverage limitation instead of a paired assay.
        """
        from unittest.mock import patch
        from open_cake_ir.tasks import devices
        untimed = {**devices.BACKENDS["triton-dcu"], "timing_source": None}
        with patch.dict(devices.BACKENDS, {"triton-dcu": untimed}):
            policy = self.study("triton-dcu")["evaluation_protocol"]
        self.assertNotIn("paired_timing", policy)
        self.assertEqual(policy["measurement_coverage"]["timed_assay"], "unavailable")
        self.assertIn("gfx938", policy["measurement_coverage"]["reason"])
        for stage in ("search_evaluation", "confirmatory_evaluation"):
            self.assertNotIn("cupti", policy[stage])
            self.assertNotIn("metal", policy[stage])

    def test_a_b300_study_still_names_cupti(self) -> None:
        policy = self.study("triton-b300")["evaluation_protocol"]
        self.assertEqual(policy["search_evaluation"], "correctness_then_paired_cupti")
        self.assertNotIn("measurement_coverage", policy)

    def test_the_hidden_pointer_count_comes_from_the_kernel_not_a_constant(self) -> None:
        """Measured on the DCU, by segfault, after a correctness run had already passed.

        An rmsnorm over three tensors emits a kernel declaring five pointer arguments and
        a 40-byte kernarg segment. Deriving the hidden count from the route's scratch
        fields gave one, the launcher passed four addresses, and the kernel read its fifth
        out of uninitialized kernarg memory. One launch survived that -- this kernel never
        dereferences its scratch -- which is why a passing correctness run did not catch
        it and a second launch did.
        """
        from open_cake_ir.evaluation.triton_hip import amdgcn_kernarg_pointers
        from open_cake_ir.lab.build import _hidden_pointers

        def metadata(pointers: int, segment: int | None = None) -> bytes:
            """The shape this emitter writes, including the duplicate it writes twice."""
            entries = "\n".join(
                "      - .address_space: global\n"
                f"        .offset:         {index * 8}\n"
                "        .size:           8\n"
                "        .value_kind:     global_buffer"
                for index in range(pointers))
            size = 8 * pointers if segment is None else segment
            return ("\t.amdgpu_metadata\n---\namdhsa.kernels:\n  - .args:\n"
                    f"{entries}\n    .group_segment_fixed_size: 0\n"
                    f"    .kernarg_segment_size: {size}\n    .name:           k\n"
                    # The emitter repeats every argument under `.unfolded_args`, and a
                    # count over the whole block counted each one twice.
                    f"    .unfolded_args:\n{entries}\n    .wavefront_size: 64\n...\n"
                    "\t.end_amdgpu_metadata\n").encode()

        self.assertEqual(amdgcn_kernarg_pointers(metadata(5)), 5)
        # Three tensors and five declared pointers means two hidden, not the one the
        # route's single scratch field would have given.
        self.assertEqual(
            _hidden_pointers(_route("gfx938"), {"amdgcn": metadata(5)}, 3), 2)
        self.assertEqual(
            _hidden_pointers(_route("gfx1151"), {"amdgcn": metadata(4)}, 2), 2)
        # The CUDA route keeps the literal every retained CUDA manifest replays through.
        self.assertEqual(_hidden_pointers(_route("sm_103a"), {}, 3), 2)

    def test_a_kernel_that_is_not_pointers_alone_is_refused_rather_than_counted(self) -> None:
        from open_cake_ir.evaluation.triton_hip import amdgcn_kernarg_pointers
        from open_cake_ir.lab.build import _hidden_pointers

        mixed = ("\t.amdgpu_metadata\n---\namdhsa.kernels:\n  - .args:\n"
                 "      - .address_space: global\n        .offset:         0\n"
                 "        .size:           8\n        .value_kind:     global_buffer\n"
                 "      - .offset:         8\n        .size:           4\n"
                 "        .value_kind:     by_value\n"
                 "    .kernarg_segment_size: 12\n...\n\t.end_amdgpu_metadata\n").encode()
        with self.assertRaisesRegex(ValueError, "packs pointer arguments alone"):
            amdgcn_kernarg_pointers(mixed)
        # And a kernel declaring fewer pointers than the case binds tensors is refused
        # rather than yielding a negative hidden count.
        fine = ("\t.amdgpu_metadata\n---\namdhsa.kernels:\n  - .args:\n"
                "      - .address_space: global\n        .offset:         0\n"
                "        .size:           8\n        .value_kind:     global_buffer\n"
                "    .kernarg_segment_size: 8\n...\n\t.end_amdgpu_metadata\n").encode()
        with self.assertRaisesRegex(ValueError, "fewer than"):
            _hidden_pointers(_route("gfx938"), {"amdgcn": fine}, 3)


class Gfx938DeclaredContracts(unittest.TestCase):
    """What this one Target declares, pinned where its other facts are."""

    def test_gfx938_declares_the_contraction_and_contracts_its_evidence_covers(self) -> None:
        target = Target.load(ROOT / "compiler/targets/gfx938.json")
        self.assertIn("mma", {kind.value for kind in target.operation_kinds})
        self.assertEqual(
            sorted(target.instruction_contracts),
            ["triton.dot.fp16_fp32", "triton.dot.fp8e4m3_fp32"])

    def test_the_two_gfx938_contracts_read_the_dtypes_that_were_measured(self) -> None:
        from open_cake_ir.compiler.ir import ContractKind, DType, contract
        for name, operand in (("triton.dot.fp16_fp32", DType.FP16),
                              ("triton.dot.fp8e4m3_fp32", DType.FP8_E4M3)):
            with self.subTest(contract=name):
                record = contract(name)
                self.assertIs(record.kind, ContractKind.MMA)
                self.assertEqual(record.operand_dtypes, frozenset({operand}))
                self.assertIs(record.accumulator, DType.FP32)
        # The block-scaled sibling is a different instruction and stays distinct.
        self.assertNotEqual("triton.dot.fp8e4m3_fp32",
                            "triton.dot.fp8e4m3_block_scale_fp32")

    def test_a_triton_target_admits_no_contract_its_route_cannot_emit(self) -> None:
        """Otherwise the Target admits by name what the only backend then refuses."""
        from open_cake_ir.compiler.backends.triton import _TRITON_MMA_CONTRACTS
        target = Target.load(ROOT / "compiler/targets/gfx938.json")
        self.assertEqual(
            sorted(set(target.instruction_contracts) - set(_TRITON_MMA_CONTRACTS)), [])


if __name__ == "__main__":
    unittest.main()
