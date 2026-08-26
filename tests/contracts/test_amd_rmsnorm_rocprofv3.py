from __future__ import annotations

import copy
import csv
import io
import json
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples/gpu"))

import rmsnorm_amd_rocprofv3 as runner  # noqa: E402
import rmsnorm_amd_search as search_runner  # noqa: E402
from open_cake_ir.evaluation.amd_rmsnorm_search import (  # noqa: E402
    LEAF_TIMING_WIN,
    derive_confirmatory_decision,
    derive_noise_decision,
    derive_profiled_diagnosis,
    materialize_candidates,
)
from open_cake_ir.evaluation.rocprofv3 import (  # noqa: E402
    Rocprofv3KernelTraceExpectation,
    parse_kernel_stats_csv,
    parse_kernel_trace_csv,
    parse_results_json,
    validate_cross_output_agreement,
)
from tests.contracts.test_amd_rmsnorm_search import (  # noqa: E402
    ContractFixture,
    _canonical_bytes,
    _measurements,
    _noise_measurements,
)


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


class Rocprofv3RunnerContractTests(unittest.TestCase):
    def test_attribution_reaches_supervised_profiler_and_retains_timeout(self) -> None:
        fixture = ContractFixture(self)
        contract = fixture.load()
        parent_result = {
            "selected_candidate": {
                "candidate_id": "r1-w1",
                "amdgcn_resources": {"kernel_name": KERNEL},
            },
            "baseline": {"amdgcn_resources": {"kernel_name": KERNEL}},
        }
        handoff = {
            "path": "/evidence/parent",
            "manifest_sha256": "a" * 64,
            "result_sha256": "b" * 64,
        }
        admission = SimpleNamespace(
            device_monitor={"path": "/exact/amd-smi"},
            profilers=({"kind": "rocprofv3", "path": "/exact/rocprofv3"},),
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "profile"
            timeout = runner.SupervisedProcessTimeout(b"partial-out", b"partial-err")
            with (
                patch.object(
                    runner,
                    "_load_timing_handoff",
                    return_value=(parent_result, handoff),
                ),
                patch.object(
                    search_runner,
                    "_admit_search_host",
                    return_value=(admission, {"returncode": 0}),
                ),
                patch.object(
                    search_runner,
                    "_amd_smi",
                    return_value={"returncode": 0},
                ),
                patch.object(
                    search_runner, "_no_foreign_processes", return_value=True
                ),
                patch.object(
                    runner,
                    "git_state",
                    return_value={"revision": "fixed", "tree_clean": True},
                ),
                patch.object(runner, "run_supervised", side_effect=timeout) as run,
                self.assertRaises(runner.SupervisedProcessTimeout),
            ):
                runner._run_attribution(
                    project_root=fixture.root,
                    contract=contract,
                    executor=SimpleNamespace(),
                    parent_root=Path("/evidence/parent"),
                    artifact_dir=artifact,
                )
            run.assert_called_once()
            self.assertEqual(
                (artifact / "arms/candidate/rocprofv3.stdout.bin").read_bytes(),
                b"partial-out",
            )
            failure = json.loads((artifact / "failure.json").read_text())
            self.assertEqual(failure["status"], "PROFILE_INCOMPLETE")
            self.assertIsNone(failure["parent_timing_decision_unchanged"])
            self.assertFalse(failure["parent_evidence_written_by_profiler"])

    def test_attribution_root_cannot_nest_inside_parent_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "timing"
            parent.mkdir()
            before = tuple(parent.iterdir())
            with self.assertRaisesRegex(ValueError, "disjoint"):
                runner._require_disjoint_evidence_roots(
                    parent.resolve(), parent / "profile"
                )
            self.assertEqual(tuple(parent.iterdir()), before)

    def test_profiled_diagnosis_requires_both_checked_arms(self) -> None:
        contract = ContractFixture(self).load()
        trace = parse_kernel_trace_csv(
            _csv_bytes(HEADER, [_trace_row()]), _expectation()
        )
        stats = parse_kernel_stats_csv(
            _stats(), kernel_name=KERNEL, expected_calls=1
        )
        result_json = parse_results_json(_results_json(), _expectation())
        checked = {
            "kernel_trace": trace,
            "kernel_stats": stats,
            "results_json": result_json,
            "cross_output_agreement": True,
        }
        receipts = {
            arm: {
                "kind": "open_cake_gfx1151_rmsnorm_profile_replay_v1",
                "status": "PROFILE_REPLAY_COMPLETE",
                "arm": arm,
                "search_contract_sha256": contract.canonical_sha256,
                "artifact_identity": {"kernel_name": KERNEL},
                "target_dispatch_count": 1,
                "profiled_launch_correctness": {
                    "passed": True,
                    "inputs_unchanged": True,
                    "fallback_calls": 0,
                },
                "performance_measured": False,
                "timing_samples": 0,
                "timing_used_for_decision": False,
                "performance_decision": None,
                "promotion_authorized": False,
                "fallback_calls": 0,
            }
            for arm in ("candidate", "baseline")
        }
        diagnosis = derive_profiled_diagnosis(
            contract,
            no_profiler_status=LEAF_TIMING_WIN,
            arm_receipts=receipts,
            checked_projections={arm: checked for arm in receipts},
        )
        self.assertEqual(
            diagnosis.reason_code, "LEAF_WIN_READY_FOR_AITER_COMPARISON"
        )
        incomplete = dict(receipts)
        incomplete.pop("baseline")
        with self.assertRaisesRegex(ValueError, "both frozen arms"):
            derive_profiled_diagnosis(
                contract,
                no_profiler_status=LEAF_TIMING_WIN,
                arm_receipts=incomplete,
                checked_projections={"candidate": checked},
            )

    def test_selects_profiler_by_kind_not_position(self) -> None:
        admission = SimpleNamespace(
            profilers=(
                {"kind": "rocprof", "path": "/legacy"},
                {"kind": "rocprofv3", "path": "/exact"},
            )
        )
        self.assertEqual(runner._profile_tool(admission)["path"], "/exact")
        with self.assertRaises(ValueError):
            runner._profile_tool(SimpleNamespace(profilers=admission.profilers[:1]))

    def test_command_uses_independent_raw_and_replay_paths(self) -> None:
        contract = ContractFixture(self).load()
        command = runner._rocprofv3_command(
            profiler="/exact/rocprofv3",
            python="/exact/python",
            project_root=Path("/checkout"),
            contract_path=Path("/checkout/search.json"),
            parent_root=Path("/evidence/timing"),
            attribution_root=Path("/evidence/profile"),
            arm="candidate",
            protocol=contract.attribution,
        )
        self.assertEqual(command[0], "/exact/rocprofv3")
        self.assertEqual(
            command[command.index("--kernel-trace") + 1],
            "true" if contract.attribution.trace == "kernel" else "false",
        )
        self.assertEqual(
            command[command.index("--stats") + 1],
            "true" if contract.attribution.stats else "false",
        )
        format_index = command.index("--output-format")
        self.assertEqual(
            tuple(command[format_index + 1 : format_index + 3]),
            contract.attribution.output_formats,
        )
        self.assertEqual(command[command.index("--output-file") + 1], "candidate")
        self.assertEqual(
            command[command.index("--output-directory") + 1],
            "/evidence/profile/arms/candidate/raw",
        )
        self.assertNotIn("--kernel-include-regex", command)
        separator = command.index("--")
        self.assertEqual(command[separator + 1], "/exact/python")
        self.assertEqual(command[-2:], ["--profile-child-arm", "candidate"])

    def test_parent_handoff_replays_raw_abba_and_rejects_tampering(self) -> None:
        fixture = ContractFixture(self)
        contract = fixture.load()
        candidate = materialize_candidates(contract)[0]
        count = contract.confirmatory.timing.samples_per_cohort
        measurements = _measurements(contract, [0.9] * count, [1.0] * count)
        decision = derive_confirmatory_decision(contract, measurements)
        noise_count = contract.noise.timing.samples_per_cohort
        noise_measurements = _noise_measurements(
            contract, [1.0] * noise_count, [1.0] * noise_count
        )
        noise_decision = derive_noise_decision(contract, noise_measurements)
        source = {"revision": "fixed-revision", "tree_clean": True}
        result = {
            "schema_version": 1,
            "kind": "open_cake_gfx1151_llama_rmsnorm_search_v2",
            "status": LEAF_TIMING_WIN,
            "search_id": contract.search_id,
            "search_contract_sha256": contract.canonical_sha256,
            "compiler_revision": {
                "revision_id": contract.compiler_revision_id,
                "canonical_sha256": contract.compiler_sha256,
            },
            "executor": {
                "executor_id": contract.executor_id,
                "canonical_sha256": contract.executor_sha256,
            },
            "workload": {
                "workload_id": "llama-rmsnorm-mul-fp32-independent-v2",
                "canonical_sha256": contract.workload_sha256,
                "case_ids": ["seeded_random", "reduction_rsqrt_stress"],
            },
            "source_custody": source,
            "selected_candidate": {
                "candidate_id": candidate.candidate_id,
                "row_tile": candidate.row_tile,
                "num_warps": candidate.num_warps,
                "schedule_sha256": candidate.canonical_sha256,
                "source_sha256": "a" * 64,
                "hsaco_sha256": "b" * 64,
                "amdgcn_resources": {"kernel_name": KERNEL},
            },
            "baseline": {
                "row_tile": 64,
                "num_warps": 4,
                "schedule_sha256": "c" * 64,
                "source_sha256": "d" * 64,
                "hsaco_sha256": "e" * 64,
                "amdgcn_resources": {"kernel_name": KERNEL},
            },
            "screening": {"selected_candidate_id": candidate.candidate_id},
            "noise": {
                "preflight_correctness": {
                    "passed": True,
                    "inputs_unchanged": True,
                },
                "measurements": noise_measurements,
                "observation": search_runner._observation_document(noise_decision),
                "postflight_correctness": {
                    "passed": True,
                    "inputs_unchanged": True,
                },
                "passed": True,
            },
            "confirmatory": {
                "preflight_correctness": {
                    "candidate": {"passed": True, "inputs_unchanged": True},
                    "baseline": {"passed": True, "inputs_unchanged": True},
                },
                "measurements": measurements,
                "observation": search_runner._observation_document(decision),
                "postflight_correctness": {
                    "candidate": {"passed": True, "inputs_unchanged": True},
                    "baseline": {"passed": True, "inputs_unchanged": True},
                },
            },
            "performance_measured": True,
            "profiler_evidence_collected": False,
            "promotion_authorized": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write_parent(value: dict[str, object]) -> None:
                measurements_path = root / "confirmatory/measurements.json"
                measurements_path.parent.mkdir(parents=True, exist_ok=True)
                measurements_path.write_bytes(
                    _canonical_bytes(value["confirmatory"]["measurements"]) + b"\n"
                )
                noise_path = root / "noise/measurements.json"
                noise_path.parent.mkdir(parents=True, exist_ok=True)
                noise_path.write_bytes(
                    _canonical_bytes(value["noise"]["measurements"]) + b"\n"
                )
                (root / "result.json").write_bytes(_canonical_bytes(value) + b"\n")
                records = []
                for path in (measurements_path, noise_path, root / "result.json"):
                    payload = path.read_bytes()
                    records.append(
                        {
                            "path": path.relative_to(root).as_posix(),
                            "sha256": sha256(payload).hexdigest(),
                            "size_bytes": len(payload),
                        }
                    )
                manifest = {
                    "schema_version": 1,
                    "kind": "open_cake_gfx1151_rmsnorm_search_manifest_v2",
                    "files": sorted(records, key=lambda item: item["path"]),
                }
                (root / "manifest.json").write_bytes(
                    _canonical_bytes(manifest) + b"\n"
                )

            write_parent(result)
            with patch.object(runner, "git_state", return_value=source):
                observed, handoff = runner._load_timing_handoff(
                    project_root=fixture.root,
                    contract=contract,
                    evidence_root=root,
                )
            self.assertEqual(observed["status"], LEAF_TIMING_WIN)
            self.assertEqual(len(handoff["manifest_sha256"]), 64)

            changed = copy.deepcopy(result)
            changed["confirmatory"]["measurements"][0]["arms"]["candidate"][
                "samples_ms"
            ][0] = 0.1
            write_parent(changed)
            with (
                patch.object(runner, "git_state", return_value=source),
                self.assertRaisesRegex(ValueError, "measurements differ"),
            ):
                runner._load_timing_handoff(
                    project_root=fixture.root,
                    contract=contract,
                    evidence_root=root,
                )

            changed_noise = copy.deepcopy(result)
            for measurement in changed_noise["noise"]["measurements"]:
                measurement["arms"]["baseline_a"]["samples_ms"] = [
                    0.9
                ] * noise_count
                measurement["arms"]["baseline_a"][
                    "summary"
                ] = search_runner.summarize_cohort([0.9] * noise_count)
            write_parent(changed_noise)
            with (
                patch.object(runner, "git_state", return_value=source),
                self.assertRaisesRegex(ValueError, "noise decision does not replay"),
            ):
                runner._load_timing_handoff(
                    project_root=fixture.root,
                    contract=contract,
                    evidence_root=root,
                )


if __name__ == "__main__":
    unittest.main()
