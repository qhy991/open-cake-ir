#!/usr/bin/env python3
"""Collect post-decision rocprofv3 path/resource evidence for gfx1151 RMSNorm."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Mapping, cast

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

import rmsnorm_amd_search as search  # noqa: E402
from open_cake_ir.tasks.workloads import load_workload  # noqa: E402
from open_cake_ir.evaluation import (  # noqa: E402
    WorkloadContract,
)
from open_cake_ir.tasks.amd.rmsnorm import (  # noqa: E402
    rmsnorm_metrics,
)
from open_cake_ir.tasks.amd.rmsnorm_search import (  # noqa: E402
    LEAF_TIMING_WIN,
    AmdRmsNormAttributionProtocol,
    AmdRmsNormCandidate,
    AmdRmsNormSearchContract,
    derive_confirmatory_decision,
    derive_noise_decision,
    derive_profiled_diagnosis,
    materialize_candidates,
)
from open_cake_ir.evaluation.rocprofv3 import (  # noqa: E402
    Rocprofv3KernelTraceExpectation,
    find_kernel_stats_csv,
    find_kernel_trace_csv,
    find_results_json,
    parse_kernel_stats_csv,
    parse_kernel_trace_csv,
    parse_results_json,
    validate_cross_output_agreement,
)
from open_cake_ir.evaluation.triton_hip import (  # noqa: E402
    admit_exact_hip,
    amdgcn_resource_record,
    artifact_records,
    canonical_json_bytes,
    git_state,
    require_object,
    resolve_new_external_directory,
)
from open_cake_ir.lab import ExecutorRevision  # noqa: E402
from open_cake_ir.lab.process import (  # noqa: E402
    SupervisedProcessOutputLimit,
    SupervisedProcessTimeout,
    run_supervised,
    sanitized_environment,
)


DEFAULT_CONTRACT = search.DEFAULT_CONTRACT
_ARMS = ("candidate", "baseline")
_PROFILE_TIMEOUT_SECONDS = 300


def _strict_json_bytes(payload: bytes, context: str) -> object:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{context} contains duplicate key {key!r}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> object:
        raise ValueError(f"{context} contains non-finite number {value}")

    try:
        return json.loads(
            payload,
            object_pairs_hook=object_pairs,
            parse_constant=invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not valid UTF-8 JSON") from error


def _canonical_value(payload: bytes, context: str) -> object:
    value = _strict_json_bytes(payload, context)
    if payload != canonical_json_bytes(value) + b"\n":
        raise ValueError(f"{context} is not canonical JSON")
    return value


def _canonical_document(path: Path) -> Mapping[str, object]:
    value = _canonical_value(path.read_bytes(), str(path))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain an object")
    return cast(Mapping[str, object], value)


def _sha(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _existing_external_directory(project_root: Path, value: Path) -> Path:
    unresolved = value.absolute()
    if unresolved.is_symlink():
        raise ValueError("parent evidence directory custody differs")
    path = unresolved.resolve(strict=True)
    if not path.is_dir() or path == project_root or project_root in path.parents:
        raise ValueError("parent evidence directory must be outside the checkout")
    return path


def _require_disjoint_evidence_roots(parent_root: Path, artifact_dir: Path) -> None:
    parent_root = parent_root.resolve(strict=True)
    artifact_absolute = artifact_dir.absolute()
    artifact_dir = artifact_absolute.parent.resolve(strict=True) / artifact_absolute.name
    if (
        parent_root == artifact_dir
        or parent_root in artifact_dir.parents
        or artifact_dir in parent_root.parents
    ):
        raise ValueError("attribution and parent evidence roots must be disjoint")


def _verify_manifest(
    evidence_root: Path,
    *,
    expected_kind: str,
    required_paths: tuple[str, ...],
) -> tuple[dict[str, Mapping[str, object]], dict[str, bytes], str]:
    """Verify this external handoff once; its exact bytes select the profiled arm."""

    manifest_path = evidence_root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("parent evidence manifest custody differs")
    manifest_payload = manifest_path.read_bytes()
    manifest_value = _canonical_value(manifest_payload, str(manifest_path))
    if not isinstance(manifest_value, Mapping):
        raise ValueError("evidence manifest must contain an object")
    manifest = cast(Mapping[str, object], manifest_value)
    if (
        set(manifest) != {"schema_version", "kind", "files"}
        or manifest.get("schema_version") != 1
        or manifest.get("kind") != expected_kind
        or not isinstance(manifest.get("files"), list)
    ):
        raise ValueError("parent evidence manifest differs")
    records: dict[str, Mapping[str, object]] = {}
    payloads: dict[str, bytes] = {}
    for index, value in enumerate(cast(list[object], manifest["files"])):
        if not isinstance(value, Mapping) or set(value) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError(f"parent manifest files[{index}] differs")
        relative_value = value.get("path")
        if not isinstance(relative_value, str):
            raise ValueError("parent manifest path differs")
        relative = PurePosixPath(relative_value)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in relative_value
            or relative_value in records
            or relative_value == "manifest.json"
        ):
            raise ValueError("parent manifest path is unsafe or duplicated")
        expected_size = value.get("size_bytes")
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
        ):
            raise ValueError("parent manifest size differs")
        expected_sha = _sha(value.get("sha256"), "parent manifest sha256")
        path = evidence_root.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file():
            raise ValueError("parent manifest file custody differs")
        payload = path.read_bytes()
        if len(payload) != expected_size or sha256(payload).hexdigest() != expected_sha:
            raise ValueError("parent manifest file bytes differ")
        records[relative_value] = cast(Mapping[str, object], value)
        payloads[relative_value] = payload
    observed = {
        path.relative_to(evidence_root).as_posix()
        for path in evidence_root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if observed != set(records):
        raise ValueError("parent evidence file set differs from manifest")
    if any(path not in records for path in required_paths):
        raise ValueError("evidence manifest lacks required authority")
    return records, payloads, sha256(manifest_payload).hexdigest()


def _all_correctness_passed(value: object, context: str) -> bool:
    document = require_object(value, context)
    return set(document) == set(_ARMS) and all(
        require_object(document[arm], f"{context}.{arm}").get("passed") is True
        and require_object(document[arm], f"{context}.{arm}").get(
            "inputs_unchanged"
        )
        is True
        for arm in _ARMS
    )


def _correctness_passed(value: object, context: str) -> bool:
    document = require_object(value, context)
    return (
        document.get("passed") is True
        and document.get("inputs_unchanged") is True
    )


def _load_timing_handoff(
    *,
    project_root: Path,
    contract: AmdRmsNormSearchContract,
    evidence_root: Path,
) -> tuple[Mapping[str, object], dict[str, object]]:
    records, payloads, manifest_sha256 = _verify_manifest(
        evidence_root,
        expected_kind="open_cake_gfx1151_rmsnorm_search_manifest_v2",
        required_paths=(
            "result.json",
            "noise/measurements.json",
            "confirmatory/measurements.json",
        ),
    )
    result_record = records["result.json"]
    result_value = _canonical_value(payloads["result.json"], "parent result")
    if not isinstance(result_value, Mapping):
        raise ValueError("parent result must contain an object")
    result = cast(Mapping[str, object], result_value)
    measurements = _canonical_value(
        payloads["confirmatory/measurements.json"],
        "parent confirmatory measurements",
    )
    noise_measurements = _canonical_value(
        payloads["noise/measurements.json"],
        "parent baseline noise measurements",
    )
    if result.get("kind") != "open_cake_gfx1151_llama_rmsnorm_search_v2":
        raise ValueError("parent result kind differs")
    if (
        result.get("status") != LEAF_TIMING_WIN
        or result.get("profiler_evidence_collected") is not False
        or result.get("promotion_authorized") is not False
        or result.get("performance_measured") is not True
    ):
        raise ValueError("parent result is not an unprofiled leaf timing win")
    if (
        result.get("search_id") != contract.search_id
        or result.get("search_contract_sha256") != contract.canonical_sha256
    ):
        raise ValueError("parent search authority differs")
    if result.get("compiler_revision") != {
        "revision_id": contract.compiler_revision_id,
        "canonical_sha256": contract.compiler_sha256,
    } or result.get("executor") != {
        "executor_id": contract.executor_id,
        "canonical_sha256": contract.executor_sha256,
    }:
        raise ValueError("parent Compiler or Executor authority differs")
    workload = require_object(result.get("workload"), "parent workload")
    if workload.get("canonical_sha256") != contract.workload_sha256:
        raise ValueError("parent Workload authority differs")
    source = git_state(project_root)
    if result.get("source_custody") != source or source["tree_clean"] is not True:
        raise ValueError("parent and current source custody differ")
    noise = require_object(result.get("noise"), "parent baseline noise")
    if set(noise) != {
        "preflight_correctness",
        "measurements",
        "observation",
        "postflight_correctness",
        "passed",
    } or noise.get("measurements") != noise_measurements:
        raise ValueError("parent raw baseline noise measurements differ")
    if not _correctness_passed(
        noise.get("preflight_correctness"), "parent noise preflight"
    ) or not _correctness_passed(
        noise.get("postflight_correctness"), "parent noise postflight"
    ):
        raise ValueError("parent baseline noise correctness differs")
    noise_decision = derive_noise_decision(contract, noise_measurements)
    if (
        not noise_decision.passed
        or noise.get("passed") is not True
        or search._observation_document(noise_decision) != noise.get("observation")
    ):
        raise ValueError("parent baseline noise decision does not replay")
    confirmatory = require_object(result.get("confirmatory"), "parent confirmatory")
    if set(confirmatory) != {
        "preflight_correctness",
        "measurements",
        "observation",
        "postflight_correctness",
    } or confirmatory.get("measurements") != measurements:
        raise ValueError("parent raw confirmatory measurements differ")
    if not _all_correctness_passed(
        confirmatory.get("preflight_correctness"), "parent preflight"
    ) or not _all_correctness_passed(
        confirmatory.get("postflight_correctness"), "parent postflight"
    ):
        raise ValueError("parent confirmatory correctness differs")
    decision = derive_confirmatory_decision(contract, measurements)
    if (
        decision.status != result["status"]
        or search._observation_document(decision) != confirmatory.get("observation")
    ):
        raise ValueError("parent terminal timing decision does not replay")

    selected = require_object(result.get("selected_candidate"), "selected candidate")
    selected_id = selected.get("candidate_id")
    specifications = {
        value.candidate_id: value for value in materialize_candidates(contract)
    }
    if selected_id not in specifications:
        raise ValueError("parent selected candidate is outside the frozen domain")
    specification = specifications[cast(str, selected_id)]
    if (
        selected.get("row_tile") != specification.row_tile
        or selected.get("num_warps") != specification.num_warps
        or require_object(result.get("screening"), "parent screening").get(
            "selected_candidate_id"
        )
        != selected_id
    ):
        raise ValueError("parent selected geometry differs")
    baseline = require_object(result.get("baseline"), "parent baseline")
    if baseline.get("row_tile") != 64 or baseline.get("num_warps") != 4:
        raise ValueError("parent baseline geometry differs")
    for label, value in (("candidate", selected), ("baseline", baseline)):
        for name in ("schedule_sha256", "source_sha256", "hsaco_sha256"):
            _sha(value.get(name), f"parent {label}.{name}")
        resources = require_object(value.get("amdgcn_resources"), f"parent {label}")
        if not isinstance(resources.get("kernel_name"), str) or not resources[
            "kernel_name"
        ]:
            raise ValueError(f"parent {label} kernel name differs")

    return result, {
        "path": str(evidence_root),
        "manifest_sha256": manifest_sha256,
        "result_sha256": result_record["sha256"],
    }


def _profile_tool(admission: object) -> Mapping[str, object]:
    record = search._profiler_record(admission.profilers, "rocprofv3")
    if record is None:
        raise ValueError("Executor must admit exactly one rocprofv3 executable")
    return record


def _arm_expected(result: Mapping[str, object], arm: str) -> Mapping[str, object]:
    key = "selected_candidate" if arm == "candidate" else "baseline"
    return require_object(result.get(key), f"parent {arm}")


def _arm_schedule(
    *,
    project_root: Path,
    contract: AmdRmsNormSearchContract,
    candidates: tuple[AmdRmsNormCandidate, ...],
    result: Mapping[str, object],
    arm: str,
) -> tuple[str, int, int, dict[str, object]]:
    expected = _arm_expected(result, arm)
    if arm == "candidate":
        selected_id = cast(str, expected["candidate_id"])
        matches = [value for value in candidates if value.candidate_id == selected_id]
        if len(matches) != 1:
            raise ValueError("profile candidate selection differs")
        candidate = matches[0]
        return (
            candidate.candidate_id,
            candidate.row_tile,
            candidate.num_warps,
            candidate.document,
        )
    baseline, _ = search._canonical_document(project_root / search.BASELINE_SCHEDULE)
    return "baseline", 64, 4, baseline


def _inputs_unchanged(
    cases: Mapping[str, object], snapshots: Mapping[str, tuple[object, ...]], torch: object
) -> bool:
    return all(
        all(
            bool(torch.equal(value, snapshot))
            for value, snapshot in zip(material.inputs, snapshots[case_id], strict=True)
        )
        for case_id, material in cases.items()
    )


def _profile_child(
    *,
    project_root: Path,
    contract: AmdRmsNormSearchContract,
    compiler: object,
    executor: ExecutorRevision,
    candidates: tuple[AmdRmsNormCandidate, ...],
    parent_root: Path,
    artifact_dir: Path,
    arm: str,
) -> tuple[int, dict[str, object]]:
    evidence = search._Evidence(
        artifact_dir,
        manifest_kind="open_cake_gfx1151_rmsnorm_profile_replay_manifest_v1",
    )
    runtimes: list[object] = []
    stage = "parent_handoff"
    try:
        result, handoff = _load_timing_handoff(
            project_root=project_root,
            contract=contract,
            evidence_root=parent_root,
        )
        evidence.json("parent-handoff.json", handoff)
        stage = "host_admission"
        executor.admit_hip_host()
        candidate_id, row_tile, num_warps, schedule_document = _arm_schedule(
            project_root=project_root,
            contract=contract,
            candidates=candidates,
            result=result,
            arm=arm,
        )
        baseline_document, _ = search._canonical_document(
            project_root / search.BASELINE_SCHEDULE
        )
        requirements = require_object(
            compiler.lower(compiler.assess(baseline_document)).toolchain_requirements,
            "baseline.toolchain",
        )
        torch, _, _ = admit_exact_hip(requirements)
        stage = "workload_materialization"
        workload = load_workload(contract.workload_path)
        cases = search._case_materials(workload, torch)
        snapshots = {
            case_id: tuple(value.clone() for value in material.inputs)
            for case_id, material in cases.items()
        }
        stage = "compile_only"
        runtime = search._compile_candidate(
            compiler=compiler,
            candidate_id=candidate_id,
            row_tile=row_tile,
            num_warps=num_warps,
            schedule=schedule_document,
            cases=cases,
            torch=torch,
            evidence=evidence,
            relative_root="kernel",
            compile_only=True,
        )
        runtimes.append(runtime)
        expected = _arm_expected(result, arm)
        observed_resources = amdgcn_resource_record(runtime.artifacts["amdgcn"])
        observed_artifacts = artifact_records(runtime.artifacts)
        if (
            runtime.assessment.schedule_sha256 != expected["schedule_sha256"]
            or runtime.lowering.source_sha256 != expected["source_sha256"]
            or observed_artifacts["hsaco"]["sha256"] != expected["hsaco_sha256"]
            or observed_resources["kernel_name"]
            != require_object(expected["amdgcn_resources"], "parent resources")[
                "kernel_name"
            ]
            or runtime.launch_calls != 0
        ):
            raise ValueError("profile replay artifact identity differs")

        stage = "profiled_launch_correctness"
        case_id = contract.attribution.profile_case_id
        material = cases[case_id]
        for _ in range(contract.attribution.profile_launches):
            compiled = runtime.launch(case_id, material)
            if artifact_records(search.extract_artifacts(compiled)) != observed_artifacts:
                raise RuntimeError("profiled launch artifact identity differs")
        torch.cuda.synchronize()
        metrics = rmsnorm_metrics(workload, runtime.outputs[case_id], material.reference)
        output = runtime.outputs[case_id]
        profiled = {
            "case_id": case_id,
            "launches": contract.attribution.profile_launches,
            "passed": bool(metrics["passed"]),
            "output_shape": list(output.shape),
            "output_dtype": str(output.dtype),
            "output_contiguous": bool(output.is_contiguous()),
            "inputs_unchanged": _inputs_unchanged(cases, snapshots, torch),
            "metrics": metrics,
            "fallback_calls": 0,
        }
        evidence.json("profiled-launch-correctness.json", profiled)
        if (
            not profiled["passed"]
            or profiled["output_shape"] != list(material.shape)
            or profiled["output_dtype"] != "torch.float32"
            or not profiled["output_contiguous"]
            or not profiled["inputs_unchanged"]
            or int(metrics["nonfinite_output_count"]) != 0
        ):
            raise RuntimeError("profiled launch correctness failed")

        expected_dispatches = contract.attribution.profile_launches
        if runtime.launch_calls != expected_dispatches:
            raise RuntimeError("profile target launch count differs")
        workgroup = (num_warps * 32, 1, 1)
        grid = tuple(
            int(runtime.grid[index]) * workgroup[index] for index in range(3)
        )
        receipt = {
            "schema_version": 1,
            "kind": "open_cake_gfx1151_rmsnorm_profile_replay_v1",
            "status": "PROFILE_REPLAY_COMPLETE",
            "arm": arm,
            "candidate_id": candidate_id,
            "search_contract_sha256": contract.canonical_sha256,
            "parent_handoff": handoff,
            "artifact_identity": {
                "schedule_sha256": runtime.assessment.schedule_sha256,
                "source_sha256": runtime.lowering.source_sha256,
                "hsaco_sha256": observed_artifacts["hsaco"]["sha256"],
                "kernel_name": observed_resources["kernel_name"],
            },
            "target_dispatch_count": runtime.launch_calls,
            "expected_trace_launch": {
                "workgroup_size": list(workgroup),
                "grid_size": list(grid),
            },
            "profiled_launch_correctness": profiled,
            "fallback_calls": 0,
            "purpose": "path_and_runtime_resource_attribution",
            "performance_measured": False,
            "timing_samples": 0,
            "timing_used_for_decision": False,
            "performance_decision": None,
            "promotion_authorized": False,
        }
        evidence.json("result.json", receipt)
        evidence.manifest()
        return 0, receipt
    except BaseException as error:
        failure = {
            "schema_version": 1,
            "kind": "open_cake_gfx1151_rmsnorm_profile_replay_failure_v1",
            "status": "PROFILE_REPLAY_FAILED",
            "arm": arm,
            "failed_stage": stage,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "performance_conclusion_authorized": False,
            "parent_evidence_written_by_profiler": False,
        }
        if not (artifact_dir / "failure.json").exists():
            evidence.json("failure.json", failure)
            evidence.manifest()
        raise
    finally:
        for runtime in runtimes:
            runtime.cleanup()


def _rocprofv3_command(
    *,
    profiler: str,
    python: str,
    project_root: Path,
    contract_path: Path,
    parent_root: Path,
    attribution_root: Path,
    arm: str,
    protocol: AmdRmsNormAttributionProtocol,
) -> list[str]:
    return [
        profiler,
        f"--{protocol.trace}-trace",
        "true",
        "--stats",
        str(protocol.stats).lower(),
        "--output-format",
        *protocol.output_formats,
        "--output-directory",
        str(attribution_root / "arms" / arm / "raw"),
        "--output-file",
        arm,
        "--",
        python,
        str(Path(__file__).resolve()),
        "--project-root",
        str(project_root),
        "--contract",
        str(contract_path),
        "--profile-from",
        str(parent_root),
        "--artifact-dir",
        str(attribution_root / "arms" / arm / "replay"),
        "--profile-child-arm",
        arm,
    ]


def _validate_arm_receipt(
    *,
    arm: str,
    receipt: Mapping[str, object],
    expected: Mapping[str, object],
    contract: AmdRmsNormSearchContract,
    parent_handoff: Mapping[str, object],
) -> None:
    identity = require_object(receipt.get("artifact_identity"), "profile identity")
    resources = require_object(expected.get("amdgcn_resources"), "parent resources")
    if (
        receipt.get("kind") != "open_cake_gfx1151_rmsnorm_profile_replay_v1"
        or receipt.get("status") != "PROFILE_REPLAY_COMPLETE"
        or receipt.get("arm") != arm
        or receipt.get("candidate_id")
        != (expected.get("candidate_id") if arm == "candidate" else "baseline")
        or receipt.get("search_contract_sha256") != contract.canonical_sha256
        or receipt.get("parent_handoff") != parent_handoff
        or receipt.get("performance_measured") is not False
        or receipt.get("timing_samples") != 0
        or receipt.get("timing_used_for_decision") is not False
        or receipt.get("performance_decision") is not None
        or receipt.get("promotion_authorized") is not False
        or receipt.get("fallback_calls") != 0
        or identity.get("schedule_sha256") != expected.get("schedule_sha256")
        or identity.get("source_sha256") != expected.get("source_sha256")
        or identity.get("hsaco_sha256") != expected.get("hsaco_sha256")
        or identity.get("kernel_name") != resources.get("kernel_name")
    ):
        raise ValueError(f"{arm} profile replay receipt differs")


def _run_attribution(
    *,
    project_root: Path,
    contract: AmdRmsNormSearchContract,
    executor: ExecutorRevision,
    parent_root: Path,
    artifact_dir: Path,
) -> tuple[int, dict[str, object]]:
    evidence = search._Evidence(
        artifact_dir,
        manifest_kind="open_cake_gfx1151_rmsnorm_rocprofv3_manifest_v1",
    )
    stage = "parent_handoff"
    try:
        result, handoff = _load_timing_handoff(
            project_root=project_root,
            contract=contract,
            evidence_root=parent_root,
        )
        source = git_state(project_root)
        evidence.json(
            "attempt-authority.json",
            {
                "schema_version": 1,
                "search_id": contract.search_id,
                "search_contract_sha256": contract.canonical_sha256,
                "parent_handoff": handoff,
                "source_custody": source,
            },
        )
        stage = "host_and_profiler_admission"
        host_admission, process_initial = search._admit_search_host(executor)
        evidence.json("amd-smi-process-initial.json", process_initial)
        profiler = _profile_tool(host_admission)
        python = str(Path(sys.executable).absolute())
        receipts: dict[str, object] = {}
        projections: dict[str, object] = {}
        for arm in contract.attribution.arm_order:
            stage = f"{arm}_process_admission"
            process = search._amd_smi(
                str(host_admission.device_monitor["path"]), ["process"]
            )
            evidence.json(f"arms/{arm}/amd-smi-process-initial.json", process)
            if not search._no_foreign_processes(process):
                raise RuntimeError("gfx1151 has another or unobservable compute process")
            expected = _arm_expected(result, arm)
            kernel_name = cast(
                str,
                require_object(expected["amdgcn_resources"], "parent resources")[
                    "kernel_name"
                ],
            )
            arm_root = artifact_dir / "arms" / arm
            command = _rocprofv3_command(
                profiler=cast(str, profiler["path"]),
                python=python,
                project_root=project_root,
                contract_path=contract.path,
                parent_root=parent_root,
                attribution_root=artifact_dir,
                arm=arm,
                protocol=contract.attribution,
            )
            command_record = {
                "argv": command,
                "profiler_authority": dict(profiler),
            }
            evidence.json(f"arms/{arm}/command.json", command_record)
            stage = f"{arm}_rocprofv3"
            try:
                completed = run_supervised(
                    command,
                    cwd=project_root,
                    timeout_seconds=_PROFILE_TIMEOUT_SECONDS,
                    environment=sanitized_environment(),
                )
            except (SupervisedProcessTimeout, SupervisedProcessOutputLimit) as error:
                evidence.bytes(f"arms/{arm}/rocprofv3.stdout.bin", error.stdout)
                evidence.bytes(f"arms/{arm}/rocprofv3.stderr.bin", error.stderr)
                raise
            evidence.bytes(f"arms/{arm}/rocprofv3.stdout.bin", completed.stdout)
            evidence.bytes(f"arms/{arm}/rocprofv3.stderr.bin", completed.stderr)
            if completed.returncode != 0:
                raise RuntimeError(f"rocprofv3 {arm} exited {completed.returncode}")
            _, replay_payloads, replay_manifest_sha256 = _verify_manifest(
                arm_root / "replay",
                expected_kind=(
                    "open_cake_gfx1151_rmsnorm_profile_replay_manifest_v1"
                ),
                required_paths=("result.json",),
            )
            replay_value = _canonical_value(
                replay_payloads["result.json"], f"{arm} profile replay result"
            )
            if not isinstance(replay_value, Mapping):
                raise ValueError(f"{arm} profile replay result must contain an object")
            replay = cast(Mapping[str, object], replay_value)
            _validate_arm_receipt(
                arm=arm,
                receipt=replay,
                expected=expected,
                contract=contract,
                parent_handoff=handoff,
            )
            raw_path = find_kernel_trace_csv(arm_root / "raw")
            raw = raw_path.read_bytes()
            expected_launch = require_object(
                replay.get("expected_trace_launch"), "expected trace launch"
            )
            projection = parse_kernel_trace_csv(
                raw,
                Rocprofv3KernelTraceExpectation(
                    kernel_name=kernel_name,
                    dispatch_count=cast(int, replay["target_dispatch_count"]),
                    workgroup_size=cast(
                        tuple[int, int, int],
                        tuple(expected_launch["workgroup_size"]),
                    ),
                    grid_size=cast(
                        tuple[int, int, int], tuple(expected_launch["grid_size"])
                    ),
                ),
            )
            stats_projection = parse_kernel_stats_csv(
                find_kernel_stats_csv(arm_root / "raw").read_bytes(),
                kernel_name=kernel_name,
                expected_calls=cast(int, replay["target_dispatch_count"]),
            )
            json_projection = parse_results_json(
                find_results_json(arm_root / "raw").read_bytes(),
                Rocprofv3KernelTraceExpectation(
                    kernel_name=kernel_name,
                    dispatch_count=cast(int, replay["target_dispatch_count"]),
                    workgroup_size=cast(
                        tuple[int, int, int],
                        tuple(expected_launch["workgroup_size"]),
                    ),
                    grid_size=cast(
                        tuple[int, int, int], tuple(expected_launch["grid_size"])
                    ),
                ),
            )
            validate_cross_output_agreement(
                projection, stats_projection, json_projection
            )
            checked = {
                "replay_manifest_sha256": replay_manifest_sha256,
                "kernel_trace": projection,
                "kernel_stats": stats_projection,
                "results_json": json_projection,
                "cross_output_agreement": True,
            }
            evidence.json(f"arms/{arm}/projection.json", checked)
            receipts[arm] = replay
            projections[arm] = checked
        stage = "terminal_parent_handoff"
        terminal_parent, terminal_handoff = _load_timing_handoff(
            project_root=project_root,
            contract=contract,
            evidence_root=parent_root,
        )
        if terminal_parent != result or terminal_handoff != handoff:
            raise ValueError("parent timing handoff changed during attribution")
        diagnosis = derive_profiled_diagnosis(
            contract,
            no_profiler_status=LEAF_TIMING_WIN,
            arm_receipts=receipts,
            checked_projections=projections,
        ).document()
        terminal = {
            "schema_version": 1,
            "kind": "open_cake_gfx1151_rmsnorm_rocprofv3_attribution_v1",
            "status": "ATTRIBUTION_COMPLETE",
            "search_id": contract.search_id,
            "search_contract_sha256": contract.canonical_sha256,
            "parent_status": LEAF_TIMING_WIN,
            "parent_handoff": handoff,
            "profiler": dict(profiler),
            "arm_order": list(contract.attribution.arm_order),
            "arms": receipts,
            "checked_projections": projections,
            "profiler_evidence_collected": True,
            "purpose": "path_and_runtime_resource_attribution",
            "performance_measured": False,
            "timing_samples": 0,
            "timing_used_for_decision": False,
            "performance_decision": None,
            "parent_timing_decision_unchanged": True,
            "parent_evidence_written_by_profiler": False,
            "promotion_authorized": False,
            "llama_cpp_build_claim": False,
            "llama_cpp_e2e_claim": False,
            "diagnosis": diagnosis,
        }
        evidence.json("result.json", terminal)
        evidence.manifest()
        return 0, terminal
    except BaseException as error:
        failure = {
            "schema_version": 1,
            "kind": "open_cake_gfx1151_rmsnorm_rocprofv3_failure_v1",
            "status": "PROFILE_INCOMPLETE",
            "failed_stage": stage,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "parent_timing_decision_unchanged": None,
            "parent_evidence_written_by_profiler": False,
            "performance_conclusion_authorized": False,
            "promotion_authorized": False,
        }
        if not (artifact_dir / "failure.json").exists():
            evidence.json("failure.json", failure)
            evidence.manifest()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--profile-from", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--profile-child-arm",
        choices=_ARMS,
        help=argparse.SUPPRESS,
    )
    arguments = parser.parse_args()

    root = arguments.project_root.resolve(strict=True)
    contract_path = arguments.contract or root / DEFAULT_CONTRACT
    contract = AmdRmsNormSearchContract.load(root, contract_path)
    compiler, executor, candidates, _ = search._prepare(root, contract)
    try:
        parent_root = _existing_external_directory(root, arguments.profile_from)
        artifact_dir = resolve_new_external_directory(root, arguments.artifact_dir)
        _require_disjoint_evidence_roots(parent_root, artifact_dir)
    except ValueError as error:
        parser.error(str(error))
    if arguments.profile_child_arm is None:
        exit_code, result = _run_attribution(
            project_root=root,
            contract=contract,
            executor=executor,
            parent_root=parent_root,
            artifact_dir=artifact_dir,
        )
    else:
        exit_code, result = _profile_child(
            project_root=root,
            contract=contract,
            compiler=compiler,
            executor=executor,
            candidates=candidates,
            parent_root=parent_root,
            artifact_dir=artifact_dir,
            arm=arguments.profile_child_arm,
        )
    sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
