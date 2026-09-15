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
    Rocprofv3KernelTraceExpectation, find_kernel_trace_csv,
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
    def payloads(self) -> dict[str, bytes]:
        return {
            **{role: b"intermediate" for role in hip.ARTIFACT_ROLES},
            "amdgcn": ASSEMBLY, "hsaco": b"\x7fELFfixture",
        }

    def test_all_six_roles_preserve_binary_bytes_and_resource_domain(self) -> None:
        payloads = self.payloads()
        compiled = SimpleNamespace(asm={**payloads, "ttir": "intermediate"})
        self.assertEqual(hip.extract_artifacts(compiled), payloads)
        records = hip.artifact_records(payloads)
        self.assertEqual(set(records), set(hip.ARTIFACT_ROLES))
        resources = hip.amdgcn_resource_record(payloads["amdgcn"])
        self.assertEqual(resources["vgpr_count"], 117)
        self.assertEqual(resources["lds_bytes_per_workgroup"], 256)
        self.assertEqual(resources["scratch_bytes_per_workitem"], 64)
        self.assertEqual(resources["wave_size"], 32)
        self.assertIs(resources["occupancy_derived"], False)

    def test_missing_mixed_empty_or_wrong_binary_roles_are_refused(self) -> None:
        for label, mutate in (
            ("missing", lambda p: p.pop("amdgcn")),
            ("cuda", lambda p: p.__setitem__("cubin", b"cuda")),
            ("empty", lambda p: p.__setitem__("ttir", b"")),
            ("text_binary", lambda p: p.__setitem__("hsaco", "\x7fELF")),
            ("invalid_binary", lambda p: p.__setitem__("hsaco", b"not ELF")),
            ("wrong_type", lambda p: p.__setitem__("source", None)),
        ):
            payloads = self.payloads()
            mutate(payloads)
            with self.subTest(label=label), self.assertRaises((ValueError, RuntimeError)):
                hip.extract_artifacts(SimpleNamespace(asm=payloads))
        with self.assertRaises(ValueError):
            hip.extract_artifacts(SimpleNamespace(asm=None))
        with self.assertRaises(ValueError):
            hip.artifact_records({"hsaco": b"\x7fELFfixture"})

    def test_ambiguous_malformed_or_non_wave32_resource_blocks_are_refused(self) -> None:
        for payload in (
            b"\xff", ASSEMBLY + ASSEMBLY,
            ASSEMBLY.replace(b".end_amdhsa_kernel", b""),
            ASSEMBLY.replace(b".amdhsa_wavefront_size32 1", b".amdhsa_wavefront_size32 0"),
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
                torch.cuda.get_device_properties.return_value.gcnArchName = "gfx1151:xnack-"
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
