from __future__ import annotations

import copy
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from open_cake_ir.evaluation import triton_hip as hip
from open_cake_ir.evaluation.rocprofv3 import (
    Rocprofv3KernelTraceExpectation, find_kernel_trace_csv, iteration_label,
    project_iteration_durations,
    parse_kernel_stats_csv, parse_kernel_trace_csv, parse_results_json,
    validate_cross_output_agreement,
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


KERNEL = "_cake_llama_rmsnorm_mul_b8_n512_d128_kernel"
HEADER = (
    "Kind",
    "Agent_Id",
    "Queue_Id",
    "Stream_Id",
    "Thread_Id",
    "Dispatch_Id",
    "Kernel_Id",
    "Kernel_Name",
    "Correlation_Id",
    "Start_Timestamp",
    "End_Timestamp",
    "LDS_Block_Size",
    "Scratch_Size",
    "VGPR_Count",
    "Accum_VGPR_Count",
    "SGPR_Count",
    "Workgroup_Size_X",
    "Workgroup_Size_Y",
    "Workgroup_Size_Z",
    "Grid_Size_X",
    "Grid_Size_Y",
    "Grid_Size_Z",
)


def _trace_row(
    *,
    kernel: str = KERNEL,
    dispatch: int = 1,
    start: int = 100,
    end: int = 120,
    lds: int = 0,
) -> list[object]:
    return [
        "KERNEL_DISPATCH",
        "Agent 1",
        1,
        0,
        99,
        dispatch,
        1,
        kernel,
        dispatch,
        start,
        end,
        lds,
        0,
        16,
        0,
        128,
        256,
        1,
        1,
        16384,
        8,
        1,
    ]


def _csv_bytes(header: tuple[str, ...], rows: list[list[object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n", quoting=csv.QUOTE_ALL)
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode()


def _expectation() -> Rocprofv3KernelTraceExpectation:
    return Rocprofv3KernelTraceExpectation(
        kernel_name=KERNEL,
        dispatch_count=1,
        workgroup_size=(256, 1, 1),
        grid_size=(16384, 8, 1),
    )


def _stats(calls: int = 1, duration: int = 20) -> bytes:
    return _csv_bytes(
        (
            "Name",
            "Calls",
            "TotalDurationNs",
            "AverageNs",
            "Percentage",
            "MinNs",
            "MaxNs",
            "StdDev",
        ),
        [[KERNEL, calls, duration, float(duration), 100.0, duration, duration, 0.0]],
    )


def _results_json(
    *, kernel: str = KERNEL, dispatches: int = 1, group_segment_size: int = 0
) -> bytes:
    records = []
    for index in range(dispatches):
        records.append(
            {
                "kind": 11,
                "operation": 2,
                "thread_id": 99,
                "start_timestamp": 100 + index * 30,
                "end_timestamp": 120 + index * 30,
                "correlation_id": {"internal": index + 1, "external": 0},
                "stream_id": {"handle": 0},
                "dispatch_info": {
                    "kernel_id": 1,
                    "dispatch_id": index + 1,
                    "agent_id": {"handle": 53062},
                    "queue_id": {"handle": 1},
                    "grid_size": {"x": 16384, "y": 8, "z": 1},
                    "workgroup_size": {"x": 256, "y": 1, "z": 1},
                    "group_segment_size": group_segment_size,
                    "private_segment_size": 0,
                },
            }
        )
    value = {
        "rocprofiler-sdk-tool": [
            {
                "metadata": {"pid": 1, "init_time": 1, "fini_time": 2},
                "kernel_symbols": [
                    {"kernel_id": 0},
                    {
                        "kernel_id": 1,
                        "kernel_name": f"{kernel}.kd",
                        "demangled_kernel_name": f"{kernel}.kd",
                        "formatted_kernel_name": kernel,
                        "truncated_kernel_name": f"{kernel}.kd",
                        "code_object_id": 1,
                        "arch_vgpr_count": 16,
                        "accum_vgpr_count": 0,
                        "sgpr_count": 128,
                        "group_segment_size": group_segment_size,
                        "private_segment_size": 0,
                        "kernarg_segment_size": 40,
                        "kernarg_segment_alignment": 16,
                    }
                ],
                "buffer_records": {"kernel_dispatch": records},
            }
        ]
    }
    return _canonical_bytes(value)


class Rocprofv3ProjectionTests(unittest.TestCase):
    def test_three_raw_surfaces_agree_without_projecting_duration(self) -> None:
        trace = parse_kernel_trace_csv(
            _csv_bytes(
                HEADER,
                [
                    _trace_row(kernel="unrelated_torch_kernel", dispatch=7),
                    _trace_row(dispatch=8, start=130, end=160),
                ],
            ),
            _expectation(),
        )
        stats = parse_kernel_stats_csv(
            _stats(), kernel_name=KERNEL, expected_calls=1
        )
        result_json = parse_results_json(_results_json(), _expectation())

        validate_cross_output_agreement(trace, stats, result_json)
        self.assertEqual(trace["dispatch_count"], 1)
        self.assertEqual(trace["non_target_dispatch_count"], 1)
        self.assertEqual(trace["launch"]["workgroup_size"], [256, 1, 1])
        self.assertEqual(trace["launch"]["grid_size"], [16384, 8, 1])
        self.assertEqual(trace["resources"]["vgpr_count"], 16)
        self.assertFalse(trace["resources"]["occupancy_derived"])
        self.assertFalse(trace["timestamps_projected"])
        self.assertFalse(stats["duration_values_projected"])
        shifted_trace = parse_kernel_trace_csv(
            _csv_bytes(
                HEADER,
                [
                    _trace_row(
                        kernel="unrelated_torch_kernel",
                        dispatch=70,
                        start=1_000_000,
                        end=1_000_001,
                    ),
                    _trace_row(
                        dispatch=80, start=2_000_000, end=9_000_000
                    ),
                ],
            ),
            _expectation(),
        )
        self.assertEqual(trace, shifted_trace)
        changed_duration = parse_kernel_stats_csv(
            _stats(duration=999999), kernel_name=KERNEL, expected_calls=1
        )
        self.assertEqual(stats, changed_duration)
        rounded_lds_trace = parse_kernel_trace_csv(
            _csv_bytes(HEADER, [_trace_row(lds=512)]), _expectation()
        )
        exact_group_json = parse_results_json(
            _results_json(group_segment_size=32), _expectation()
        )
        validate_cross_output_agreement(
            rounded_lds_trace, stats, exact_group_json
        )
        self.assertEqual(
            rounded_lds_trace["resources"]["lds_allocation_block_bytes"], 512
        )
        self.assertEqual(exact_group_json["resources"]["group_segment_size"], 32)

    def test_trace_rejects_wrong_count_geometry_or_timestamp(self) -> None:
        failures = {
            "count": [_trace_row(), _trace_row(dispatch=2, start=130, end=140)],
            "geometry": [_trace_row()],
            "timestamp": [_trace_row(start=121, end=120)],
        }
        for label, rows in failures.items():
            with self.subTest(label=label):
                expectation = _expectation()
                if label == "geometry":
                    expectation = Rocprofv3KernelTraceExpectation(
                        kernel_name=KERNEL,
                        dispatch_count=1,
                        workgroup_size=(128, 1, 1),
                        grid_size=(16384, 8, 1),
                    )
                with self.assertRaises(ValueError):
                    parse_kernel_trace_csv(_csv_bytes(HEADER, rows), expectation)

    def test_stats_and_json_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Calls"):
            parse_kernel_stats_csv(
                _stats(calls=2), kernel_name=KERNEL, expected_calls=1
            )
        with self.assertRaisesRegex(ValueError, "dispatch count"):
            parse_results_json(
                _results_json(kernel="other", dispatches=1), _expectation()
            )
        trace = parse_kernel_trace_csv(
            _csv_bytes(HEADER, [_trace_row()]), _expectation()
        )
        stats = parse_kernel_stats_csv(
            _stats(), kernel_name=KERNEL, expected_calls=1
        )
        changed = copy.deepcopy(parse_results_json(_results_json(), _expectation()))
        changed["resources"]["vgpr_count"] = 32
        with self.assertRaisesRegex(ValueError, "disagree"):
            validate_cross_output_agreement(trace, stats, changed)


ASSEMBLY = b"""
.amdhsa_kernel _fixture
  .amdhsa_group_segment_fixed_size 256
  .amdhsa_private_segment_fixed_size 64
  .amdhsa_kernarg_size 40
  .amdhsa_wavefront_size32 1
  .amdhsa_uses_dynamic_stack 0
  .amdhsa_next_free_vgpr 117
  .amdhsa_next_free_sgpr 39
  .amdhsa_shared_vgpr_count 0
  .amdhsa_workgroup_processor_mode 1
.end_amdhsa_kernel
"""


class HipArtifactContractTests(unittest.TestCase):
    # extract_artifacts/artifact_records are gone with the module-level HIP loader; the
    # hsaco ELF check now lives in loaders.check_launch_authority, exercised through
    # LoadedHipModuleCandidate.load in test_hip_driver. What stays here is the AMDGCN
    # resource domain this module still reads.
    def test_the_resource_record_reads_the_amdgcn_domain(self) -> None:
        resources = hip.amdgcn_resource_record(ASSEMBLY)
        self.assertEqual(resources["vgpr_count"], 117)
        self.assertEqual(resources["lds_bytes_per_workgroup"], 256)
        self.assertEqual(resources["scratch_bytes_per_workitem"], 64)
        self.assertEqual(resources["wave_size"], 32)
        self.assertIs(resources["occupancy_derived"], False)

    def test_the_declared_wave_mode_is_read_not_assumed(self) -> None:
        """wavefront_size32 selects a mode; it does not gate the reader.

        `.amdhsa_wavefront_size32 0` is a wave64 RDNA kernel and refusing it described one
        mode as the only one. A CDNA-class kernel omits the directive entirely -- measured
        on gfx938, whose 43 directives include none of the three RDNA-only ones -- and is
        wave64 by its ISA.
        """
        wave32 = hip.amdgcn_resource_record(ASSEMBLY)
        self.assertEqual(wave32["wave_size"], 32)
        wave64 = hip.amdgcn_resource_record(
            ASSEMBLY.replace(b".amdhsa_wavefront_size32 1", b".amdhsa_wavefront_size32 0"))
        self.assertEqual(wave64["wave_size"], 64)
        cdna = ASSEMBLY
        for directive in (b"  .amdhsa_wavefront_size32 1\n",
                          b"  .amdhsa_shared_vgpr_count 0\n",
                          b"  .amdhsa_workgroup_processor_mode 1\n"):
            cdna = cdna.replace(directive, b"")
        record = hip.amdgcn_resource_record(cdna)
        self.assertEqual(record["wave_size"], 64)
        # Absence is reported as absence, not as a value the kernel never declared.
        self.assertIsNone(record["shared_vgpr_count"])
        self.assertIsNone(record["workgroup_processor_mode"])
        self.assertEqual(record["vgpr_count"], wave32["vgpr_count"])

    def test_ambiguous_or_malformed_resource_blocks_are_refused(self) -> None:
        for payload in (
            b"\xff", ASSEMBLY + ASSEMBLY,
            ASSEMBLY.replace(b".end_amdhsa_kernel", b""),
            # Dropping a directive every AMDGCN kernel declares is still a refusal.
            ASSEMBLY.replace(b"  .amdhsa_kernarg_size 40\n", b""),
            ASSEMBLY.replace(b".amdhsa_next_free_vgpr 117", b".amdhsa_next_free_vgpr -1"),
            ASSEMBLY.replace(b".amdhsa_uses_dynamic_stack 0", b".amdhsa_uses_dynamic_stack 2"),
            ASSEMBLY.replace(b".amdhsa_next_free_vgpr 117", b".amdhsa_next_free_vgpr 1.5"),
            ASSEMBLY.replace(b".end_amdhsa_kernel", b".amdhsa_next_free_vgpr 117\n.end_amdhsa_kernel"),
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                hip.amdgcn_resource_record(payload)


class ExactHipTargetAdmissionTests(unittest.TestCase):
    def requirements(self) -> dict[str, object]:
        return {
            "target": "gfx1151", "binary_role": "hsaco", "assembly_role": "amdgcn",
            "triton_target": {"backend": "hip", "arch": "gfx1151", "warp_size": 32},
        }

    def modules(self) -> dict[str, object]:
        properties = SimpleNamespace(gcnArchName="gfx1151", warp_size=32)
        torch = SimpleNamespace(version=SimpleNamespace(hip="7.2.1", cuda=None), cuda=SimpleNamespace(
            is_available=Mock(return_value=True), device_count=Mock(return_value=1),
            get_device_properties=Mock(return_value=properties),
        ))
        return {
            "torch": torch, "triton": SimpleNamespace(),
            "triton.runtime.driver": SimpleNamespace(driver=SimpleNamespace(active=SimpleNamespace(
                get_current_target=Mock(return_value=SimpleNamespace(backend="hip", arch="gfx1151", warp_size=32)),
            ))),
        }

    def test_exact_lowering_target_is_observed_without_runtime_target_policy(self) -> None:
        modules = self.modules()
        with patch.object(hip.importlib, "import_module", side_effect=modules.__getitem__):
            torch, triton, properties = hip.admit_exact_hip(self.requirements())
        self.assertIs(torch, modules["torch"])
        self.assertIs(triton, modules["triton"])
        self.assertEqual(properties.gcnArchName, "gfx1151")

    def test_malformed_lowering_is_refused_before_import_or_device_query(self) -> None:
        for mutate in (
            lambda r: r.__setitem__("target", "other"),
            lambda r: r.__setitem__("binary_role", "cubin"),
            lambda r: r["triton_target"].__setitem__("backend", "cuda"),
            lambda r: r["triton_target"].__setitem__("warp_size", True),
            lambda r: r["triton_target"].__setitem__("warp_size", "32"),
            lambda r: r.__setitem__("triton_target", []),
        ):
            requirements = self.requirements()
            mutate(requirements)
            with patch.object(hip.importlib, "import_module") as imported:
                with self.assertRaises(ValueError):
                    hip.admit_exact_hip(requirements)
                imported.assert_not_called()

    def test_cuda_or_mixed_torch_is_refused_before_device_queries(self) -> None:
        for hip_version, cuda_version in ((None, "13.0"), ("7.2.1", "13.0")):
            modules = self.modules()
            modules["torch"].version = SimpleNamespace(hip=hip_version, cuda=cuda_version)
            with patch.object(hip.importlib, "import_module", side_effect=modules.__getitem__):
                with self.assertRaisesRegex(RuntimeError, "ROCm PyTorch"):
                    hip.admit_exact_hip(self.requirements())
            modules["torch"].cuda.is_available.assert_not_called()

    def test_device_count_runtime_target_and_properties_must_match_exactly(self) -> None:
        for kind in ("count", "boolean_count", "arch", "wave", "properties", "property_wave"):
            modules = self.modules()
            torch = modules["torch"]
            runtime_target = modules["triton.runtime.driver"].driver.active.get_current_target.return_value
            if kind in ("count", "boolean_count"):
                torch.cuda.device_count.return_value = 2 if kind == "count" else True
            elif kind == "arch":
                runtime_target.arch = "gfx1100"
            elif kind == "wave":
                runtime_target.warp_size = 64
            elif kind == "properties":
                # A different ISA, not the same one naming its features: gfx1151:xnack- is
                # the device the Target describes and is admitted below.
                torch.cuda.get_device_properties.return_value.gcnArchName = "gfx11510"
            else:
                torch.cuda.get_device_properties.return_value.warp_size = "32"
            with self.subTest(kind=kind), patch.object(hip.importlib, "import_module", side_effect=modules.__getitem__):
                with self.assertRaises(RuntimeError):
                    hip.admit_exact_hip(self.requirements())


class Rocprofv3MalformedBoundaryTests(unittest.TestCase):
    def test_malformed_expectations_are_value_errors(self) -> None:
        for name, count, workgroup in ((7, 1, (1, 1, 1)), (KERNEL, True, (1, 1, 1)), (KERNEL, 1, None)):
            with self.assertRaises(ValueError):
                Rocprofv3KernelTraceExpectation(name, count, workgroup, (1, 1, 1))

    def test_csv_extra_and_missing_fields_are_not_silently_ignored(self) -> None:
        for row in (_trace_row() + ["extra"], _trace_row()[:-1]):
            with self.assertRaisesRegex(ValueError, "row fields"):
                parse_kernel_trace_csv(_csv_bytes(HEADER, [row]), _expectation())
        with self.assertRaises(ValueError):
            parse_kernel_stats_csv(_stats(), kernel_name=KERNEL, expected_calls=True)

    def test_cross_output_boundary_rejects_malformed_projections(self) -> None:
        for trace in (None, {}, {"kernel_name": KERNEL, "dispatch_count": True}):
            with self.assertRaises(ValueError):
                validate_cross_output_agreement(trace, {}, {})

    def test_duplicate_json_key_and_boolean_dispatch_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            parse_results_json(b'{"rocprofiler-sdk-tool":[],"rocprofiler-sdk-tool":[]}', _expectation())
        document = json.loads(_results_json())
        document["rocprofiler-sdk-tool"][0]["buffer_records"]["kernel_dispatch"][0]["dispatch_info"]["dispatch_id"] = True
        with self.assertRaises(ValueError):
            parse_results_json(_canonical_bytes(document), _expectation())

    def test_raw_output_symlink_is_refused_even_with_another_regular_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "1_kernel_trace.csv").write_text("fixture")
            (root / "2_kernel_trace.csv").symlink_to(root / "1_kernel_trace.csv")
            with self.assertRaisesRegex(ValueError, "custody"):
                find_kernel_trace_csv(root)
            alias = root / "alias"
            alias.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "custody"):
                find_kernel_trace_csv(alias)


if __name__ == "__main__":
    unittest.main()


MARKER_HEADER = ("Domain", "Function", "Process_Id", "Thread_Id", "Correlation_Id",
                 "Start_Timestamp", "End_Timestamp")


def _marker_rows(spans):
    return _csv_bytes(MARKER_HEADER, [
        ["MARKER_CORE_RANGE_API", label, 99, 99, index, start, end]
        for index, (label, start, end) in enumerate(spans)])


class Rocprofv3TimedAssayTests(unittest.TestCase):
    """The timed assay the resource projection deliberately does not make.

    That projection states `duration_used_for_timing_or_promotion: False` for its own
    callers. These durations are a separate, named entry point, so both statements stay
    true at once.
    """

    def _cohort(self, iterations=3, gap=5_000_000):
        """One dispatch per iteration, each inside its own range, plus one outside."""

        kernel_rows, spans = [], []
        for index in range(iterations):
            base = 1000 + index * gap
            kernel_rows.append(_trace_row(dispatch=index + 1, start=base + 10,
                                          end=base + 10 + (index + 1) * 1_000_000))
            spans.append((iteration_label("search", index), base, base + gap - 1))
        # Warm-up: a dispatch of the same kernel before any range opens.
        kernel_rows.insert(0, _trace_row(dispatch=99, start=1, end=5))
        return _csv_bytes(HEADER, kernel_rows), _marker_rows(spans)

    def test_each_iteration_is_the_sum_of_the_dispatches_inside_its_own_range(self):
        kernel, marker = self._cohort()
        projection = project_iteration_durations(
            kernel, marker, _expectation(), cohort="search", iterations=3)
        self.assertEqual(projection["kind"], "rocprofv3_iteration_duration_projection_v1")
        self.assertEqual(projection["samples_ms"], [1.0, 2.0, 3.0])
        self.assertEqual(projection["dispatches_per_iteration"], [1, 1, 1])
        self.assertTrue(projection["duration_used_for_timing_or_promotion"])
        # The warm-up dispatch lies outside every range and is attributed to nothing.
        self.assertNotIn(0.000004, projection["samples_ms"])

    def test_work_a_candidate_does_in_a_second_kernel_is_counted_and_named(self):
        """Excluded from the latency, but not silent: dropping it makes a split
        candidate read as faster than it is."""

        kernel_rows = [
            _trace_row(dispatch=1, start=1010, end=1_001_010),
            _trace_row(kernel="second_kernel_of_this_candidate", dispatch=2,
                       start=1_100_000, end=1_900_000),
        ]
        marker = _marker_rows([(iteration_label("search", 0), 1000, 1_999_999)])
        projection = project_iteration_durations(
            _csv_bytes(HEADER, kernel_rows), marker, _expectation(),
            cohort="search", iterations=1)
        self.assertEqual(projection["samples_ms"], [1.0])
        self.assertEqual(projection["non_target_dispatches_per_iteration"], [1])
        self.assertEqual(projection["non_target_dispatch_count"], 1)
        self.assertIn("second kernel", projection["interval"])

    def test_a_dispatch_of_another_kernel_inside_the_range_is_not_counted(self):
        kernel_rows = [
            _trace_row(dispatch=1, start=1010, end=1_001_010),
            _trace_row(kernel="unrelated_torch_kernel", dispatch=2, start=1_100_000,
                       end=1_900_000),
        ]
        marker = _marker_rows([(iteration_label("search", 0), 1000, 1_999_999)])
        projection = project_iteration_durations(
            _csv_bytes(HEADER, kernel_rows), marker, _expectation(),
            cohort="search", iterations=1)
        self.assertEqual(projection["samples_ms"], [1.0])
        self.assertEqual(projection["dispatches_per_iteration"], [1])
        self.assertEqual(projection["non_target_dispatch_count"], 1)

    def test_an_iteration_with_no_dispatch_is_refused_rather_than_measured_as_zero(self):
        kernel = _csv_bytes(HEADER, [_trace_row(dispatch=1, start=1010, end=1_001_010)])
        marker = _marker_rows([(iteration_label("search", 0), 1000, 1_999_999),
                               (iteration_label("search", 1), 2_000_000, 2_999_999)])
        with self.assertRaisesRegex(ValueError, "not a zero"):
            project_iteration_durations(kernel, marker, _expectation(),
                                        cohort="search", iterations=2)

    def test_a_declared_iteration_with_no_range_is_refused(self):
        kernel, marker = self._cohort(iterations=2)
        with self.assertRaisesRegex(ValueError, "holds no range for iteration"):
            project_iteration_durations(kernel, marker, _expectation(),
                                        cohort="search", iterations=3)

    def test_another_cohort_s_ranges_do_not_satisfy_this_one(self):
        kernel, marker = self._cohort(iterations=2)
        with self.assertRaisesRegex(ValueError, "holds no range for iteration"):
            project_iteration_durations(kernel, marker, _expectation(),
                                        cohort="confirmatory", iterations=2)

    def test_a_marker_trace_without_an_open_cake_range_is_refused(self):
        kernel, _ = self._cohort(iterations=1)
        foreign = _marker_rows([("someone_elses_range", 1000, 1999)])
        with self.assertRaisesRegex(ValueError, "no open-cake iteration range"):
            project_iteration_durations(kernel, foreign, _expectation(),
                                        cohort="search", iterations=1)

    def test_overlapping_iteration_ranges_are_refused_rather_than_double_counted(self):
        """A dispatch inside two ranges would be charged to both, making each look slower
        while the pair looks consistent. The refusal was added without a test."""

        kernel = _csv_bytes(HEADER, [_trace_row(dispatch=1, start=1010, end=1_001_010)])
        overlapping = _marker_rows([
            (iteration_label("search", 0), 1000, 3_000_000),
            (iteration_label("search", 1), 2_000_000, 4_000_000),
        ])
        with self.assertRaisesRegex(ValueError, "overlap"):
            project_iteration_durations(kernel, overlapping, _expectation(),
                                        cohort="search", iterations=2)

    def test_ranges_that_merely_touch_are_not_overlapping(self):
        """The boundary case the refusal must not swallow: one range ends where the next
        begins, which is what a back-to-back cohort actually produces."""

        kernel = _csv_bytes(HEADER, [
            _trace_row(dispatch=1, start=1010, end=1_001_010),
            _trace_row(dispatch=2, start=2_000_010, end=2_002_010),
        ])
        touching = _marker_rows([
            (iteration_label("search", 0), 1000, 2_000_000),
            (iteration_label("search", 1), 2_000_000, 3_000_000),
        ])
        projection = project_iteration_durations(kernel, touching, _expectation(),
                                                 cohort="search", iterations=2)
        self.assertEqual(projection["dispatches_per_iteration"], [1, 1])

    def test_the_label_is_closed_over_its_own_separator(self):
        self.assertEqual(iteration_label("search", 2), "OPENCAKE|search|2")
        for cohort, index in (("a|b", 0), ("", 0), ("search", -1)):
            with self.assertRaises(ValueError):
                iteration_label(cohort, index)
