#!/usr/bin/env python3
"""Run the bounded Cake-IR llama RMSNorm+Mul search on exact gfx1151."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import traceback
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping, cast

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler  # noqa: E402
from open_cake_ir.evaluation import (  # noqa: E402
    PairedTimingProtocol,
    WorkloadContract,
    generate_rmsnorm_case,
    rmsnorm_metrics,
    rmsnorm_oracle,
    summarize_cohort,
)
from open_cake_ir.evaluation.amd_rmsnorm_search import (  # noqa: E402
    INCONCLUSIVE_MEASUREMENT_QUALITY,
    LEAF_TIMING_WIN,
    AmdRmsNormCandidate,
    AmdRmsNormSearchContract,
    diagnose_terminal_decision,
    derive_confirmatory_decision,
    derive_noise_decision,
    materialize_candidates,
)
from open_cake_ir.evaluation.triton_hip import (  # noqa: E402
    admit_exact_hip,
    amdgcn_resource_record,
    artifact_records,
    canonical_json_bytes,
    extract_artifacts,
    git_state,
    load_generated_module,
    require_object,
    resolve_new_external_directory,
    write_new_json,
)
from open_cake_ir.lab import ExecutorRevision, HipHostAdmission  # noqa: E402


DEFAULT_CONTRACT = (
    "contracts/calibrations/llama-rmsnorm-mul-gfx1151-one-row-search-v2.json"
)
BASELINE_SCHEDULE = "corpus/schedules/llama-rmsnorm-mul-b8-gfx1151-r64-w4.json"


@dataclass
class _CaseMaterial:
    inputs: tuple[object, ...]
    reference: object
    shape: tuple[int, int, int]
    input_snapshots: tuple[object, ...]
    input_data_ptrs: tuple[int, ...]


@dataclass
class _RuntimeCandidate:
    candidate_id: str
    row_tile: int
    num_warps: int
    schedule: dict[str, object]
    assessment: object
    lowering: object
    kernel: object
    generated_directory: object
    constants: dict[str, object]
    options: dict[str, object]
    grid: tuple[int, int, int]
    compiled: object
    artifacts: dict[str, bytes]
    outputs: dict[str, object]
    launch_calls: int = 0

    def launch(self, case_id: str, material: _CaseMaterial) -> object:
        compiled = self.kernel.run(
            *material.inputs,
            self.outputs[case_id],
            **self.constants,
            **self.options,
            grid=self.grid,
            warmup=False,
        )
        self.launch_calls += 1
        return compiled

    def cleanup(self) -> None:
        self.generated_directory.cleanup()


class _Evidence:
    def __init__(
        self,
        root: Path,
        *,
        manifest_kind: str = "open_cake_gfx1151_rmsnorm_search_manifest_v2",
    ) -> None:
        if not manifest_kind:
            raise ValueError("evidence manifest kind must be non-empty")
        root.mkdir(mode=0o700)
        self.root = root
        self.events = root / "events.jsonl"
        self.manifest_kind = manifest_kind

    def json(self, relative: str, value: object) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        write_new_json(path, value)
        return path

    def bytes(self, relative: str, payload: bytes) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
        return path

    def event(self, event: str, payload: Mapping[str, object]) -> None:
        record = {"schema_version": 1, "event": event, "payload": dict(payload)}
        with self.events.open("ab") as stream:
            stream.write(canonical_json_bytes(record) + b"\n")

    def manifest(self) -> dict[str, object]:
        records = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.name == "manifest.json":
                continue
            payload = path.read_bytes()
            records.append(
                {
                    "path": path.relative_to(self.root).as_posix(),
                    "sha256": sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
            )
        document = {
            "schema_version": 1,
            "kind": self.manifest_kind,
            "files": records,
        }
        self.json("manifest.json", document)
        return document


def _assessment_document(assessment: object) -> dict[str, object]:
    return {
        "schedule_id": assessment.schedule_id,
        "schedule_sha256": assessment.schedule_sha256,
        "target": assessment.target,
        "route": (
            None
            if assessment.route is None
            else {
                "backend": assessment.route.backend.value,
                "entry_point": assessment.route.entry_point,
            }
        ),
        "accepted": assessment.accepted,
        "lowering_eligible": assessment.lowering_eligible,
        "calibration_available": assessment.calibration_available,
        "findings": [
            {
                "code": item.code,
                "path": item.path,
                "message": item.message,
                "blocks_acceptance": item.blocks_acceptance,
                "blocks_lowering": item.blocks_lowering,
            }
            for item in assessment.findings
        ],
    }


def _canonical_document(path: Path) -> tuple[dict[str, object], str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value, sha256(canonical_json_bytes(value)).hexdigest()


def _amd_smi(executable: str, arguments: list[str]) -> dict[str, object]:
    completed = subprocess.run(
        [executable, *arguments, "--json"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    result: dict[str, object] = {
        "available": True,
        "executable": executable,
        "arguments": arguments,
        "returncode": completed.returncode,
        "stdout_sha256": sha256(completed.stdout).hexdigest(),
        "stderr": completed.stderr.decode("utf-8", errors="replace"),
    }
    try:
        result["document"] = json.loads(completed.stdout)
    except json.JSONDecodeError:
        result["stdout"] = completed.stdout.decode("utf-8", errors="replace")
    return result


def _no_foreign_processes(snapshot: Mapping[str, object]) -> bool:
    if snapshot.get("available") is not True or snapshot.get("returncode") != 0:
        return False
    document = snapshot.get("document")
    if not isinstance(document, list) or len(document) != 1:
        return False
    gpu = document[0]
    if not isinstance(gpu, Mapping):
        return False
    processes = gpu.get("process_list")
    return (
        isinstance(processes, list)
        and len(processes) == 1
        and isinstance(processes[0], Mapping)
        and processes[0].get("process_info") == "No running processes detected"
    )


def _admit_search_host(
    executor: ExecutorRevision,
) -> tuple[HipHostAdmission, dict[str, object]]:
    admission = executor.admit_hip_host()
    monitor_path = str(admission.device_monitor["path"])
    process_initial = _amd_smi(monitor_path, ["process"])
    if not _no_foreign_processes(process_initial):
        raise RuntimeError("gfx1151 has another or unobservable compute process")
    return admission, process_initial


def _profiler_record(
    profilers: tuple[Mapping[str, object], ...], kind: str
) -> dict[str, object] | None:
    matches = [value for value in profilers if value.get("kind") == kind]
    return dict(matches[0]) if len(matches) == 1 else None


def _runtime_document(torch: object, triton: object, properties: object) -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "torch_hip": str(torch.version.hip),
        "triton": importlib.metadata.version("triton"),
        "device_name": str(properties.name),
        "gcn_arch_name": str(properties.gcnArchName),
        "warp_size": int(properties.warp_size),
        "multiprocessor_count": int(properties.multi_processor_count),
        "total_memory_bytes": int(properties.total_memory),
        "timer": "torch.cuda.Event backed by HIP events",
    }


def _case_materials(
    workload: WorkloadContract, torch: object
) -> dict[str, _CaseMaterial]:
    result: dict[str, _CaseMaterial] = {}
    for case_id in workload.case_ids:
        case = workload.case(case_id)
        shape_value = require_object(case["shape"], "workload.case.shape")
        shape = tuple(int(shape_value[name]) for name in ("B", "N", "D"))
        inputs = generate_rmsnorm_case(workload, case_id, device="cuda")
        reference = rmsnorm_oracle(workload, *inputs)
        if tuple(inputs[0].shape) != shape:
            raise RuntimeError("Workload materialization shape differs")
        snapshots = tuple(value.clone() for value in inputs)
        pointers = tuple(int(value.data_ptr()) for value in inputs)
        result[case_id] = _CaseMaterial(
            inputs=inputs,
            reference=reference,
            shape=cast(tuple[int, int, int], shape),
            input_snapshots=snapshots,
            input_data_ptrs=pointers,
        )
    torch.cuda.synchronize()
    return result


def _artifact_suffix(role: str) -> str:
    return "bin" if role == "hsaco" else "txt"


def _compile_candidate(
    *,
    compiler: Compiler,
    candidate_id: str,
    row_tile: int,
    num_warps: int,
    schedule: dict[str, object],
    cases: Mapping[str, _CaseMaterial],
    torch: object,
    evidence: _Evidence,
    relative_root: str,
    compile_only: bool = False,
) -> _RuntimeCandidate:
    root = f"{relative_root}/{candidate_id}"
    evidence.json(f"{root}/schedule.json", schedule)
    assessment = compiler.assess(schedule)
    evidence.json(f"{root}/assessment.json", _assessment_document(assessment))
    if not assessment.lowering_eligible:
        raise ValueError(
            "candidate is not lowering eligible: "
            + ",".join(item.code for item in assessment.findings)
        )
    lowering = compiler.lower(assessment)
    requirements = require_object(lowering.toolchain_requirements, "lowering.toolchain")
    if (
        requirements.get("target") != "gfx1151"
        or requirements.get("compiler") != "triton"
        or requirements.get("binary_role") != "hsaco"
        or requirements.get("assembly_role") != "amdgcn"
        or set(require_object(requirements.get("signature"), "signature"))
        != {"x", "gamma", "y"}
    ):
        raise ValueError("candidate lowering toolchain differs")
    module, generated_directory = load_generated_module(lowering)
    try:
        kernel = getattr(module, str(requirements["kernel_entry_point"]), None)
        if kernel is None:
            raise RuntimeError("generated Triton kernel is missing")
        constants = dict(
            require_object(requirements["compile_constants"], "compile_constants")
        )
        options = dict(require_object(requirements["compile_options"], "compile_options"))
        grid_value = requirements["grid"]
        if not isinstance(grid_value, (list, tuple)) or len(grid_value) != 3:
            raise ValueError("candidate grid differs")
        grid = tuple(int(item) for item in grid_value)
        outputs = {
            case_id: torch.empty(material.shape, dtype=torch.float32, device="cuda")
            for case_id, material in cases.items()
        }
        first_case_id = next(iter(cases))
        compiled = kernel.run(
            *cases[first_case_id].inputs,
            outputs[first_case_id],
            **constants,
            **options,
            grid=grid,
            warmup=compile_only,
        )
        torch.cuda.synchronize()
        if compiled is None:
            raise RuntimeError("Triton launch returned no compiled kernel")
        artifacts = extract_artifacts(compiled)
        evidence.bytes(f"{root}/generated.py", lowering.source.encode())
        for role, payload in artifacts.items():
            evidence.bytes(
                f"{root}/kernel.{role}.{_artifact_suffix(role)}", payload
            )
        evidence.json(
            f"{root}/build.json",
            {
                "source_sha256": lowering.source_sha256,
                "entry_point": lowering.route.entry_point,
                "kernel_name": str(compiled.metadata.name),
                "grid": list(grid),
                "block": [
                    num_warps * int(require_object(
                        requirements["triton_target"], "triton_target"
                    )["warp_size"]), 1, 1
                ],
                "shared_memory_bytes": int(compiled.metadata.shared),
                "amdgcn_resources": amdgcn_resource_record(
                    artifacts["amdgcn"]
                ),
                "artifacts": artifact_records(artifacts),
            },
        )
        return _RuntimeCandidate(
            candidate_id=candidate_id,
            row_tile=row_tile,
            num_warps=num_warps,
            schedule=schedule,
            assessment=assessment,
            lowering=lowering,
            kernel=kernel,
            generated_directory=generated_directory,
            constants=constants,
            options=options,
            grid=cast(tuple[int, int, int], grid),
            compiled=compiled,
            artifacts=artifacts,
            outputs=outputs,
            launch_calls=0 if compile_only else 1,
        )
    except BaseException:
        generated_directory.cleanup()
        raise


def _inputs_unchanged(material: _CaseMaterial, torch: object) -> bool:
    return all(
        int(value.data_ptr()) == pointer and bool(torch.equal(value.view(torch.uint8), snapshot.view(torch.uint8)))
        for value, snapshot, pointer in zip(
            material.inputs,
            material.input_snapshots,
            material.input_data_ptrs,
            strict=True,
        )
    )


def _correctness(
    candidate: _RuntimeCandidate,
    workload: WorkloadContract,
    cases: Mapping[str, _CaseMaterial],
    torch: object,
) -> dict[str, object]:
    records = []
    expected_artifacts = artifact_records(candidate.artifacts)
    for case_id, material in cases.items():
        # Correctness must observe this invocation's complete output. Initialization
        # stays here, outside the separate paired timing launch path.
        candidate.outputs[case_id].fill_(float("nan"))
        compiled = candidate.launch(case_id, material)
        torch.cuda.synchronize()
        if artifact_records(extract_artifacts(compiled)) != expected_artifacts:
            raise RuntimeError("correctness cases used different compiled artifacts")
        metrics = rmsnorm_metrics(
            workload, candidate.outputs[case_id], material.reference
        )
        # An unwritten NaN has no finite error magnitude. Keep actual mismatch and
        # nonfinite counts, while retaining the rejection in strict JSON evidence.
        for key in ("max_abs_error", "max_rel_error"):
            if key in metrics and not math.isfinite(metrics[key]):
                metrics[key] = None
        inputs_unchanged = _inputs_unchanged(material, torch)
        output = candidate.outputs[case_id]
        output_contract = {
            "shape_matches": tuple(output.shape) == material.shape,
            "dtype_matches": output.dtype == torch.float32,
            "contiguous": bool(output.is_contiguous()),
            "aliases_input": int(output.data_ptr()) in material.input_data_ptrs,
        }
        records.append(
            {
                "case_id": case_id,
                **metrics,
                "inputs_unchanged": inputs_unchanged,
                "output_contract": output_contract,
                "passed": bool(metrics["passed"])
                and inputs_unchanged
                and output_contract["shape_matches"]
                and output_contract["dtype_matches"]
                and output_contract["contiguous"]
                and not output_contract["aliases_input"],
            }
        )
    return {
        "passed": all(bool(item["passed"]) for item in records),
        "inputs_unchanged": all(
            bool(item["inputs_unchanged"]) for item in records
        ),
        "cases": records,
        "fallback_calls": 0,
    }


def _flush_l2(torch: object, buffer: object) -> None:
    buffer.add_(1.0)
    torch.cuda.synchronize()


def _event_sample_ms(
    torch: object,
    candidate: _RuntimeCandidate,
    case_id: str,
    material: _CaseMaterial,
    flush: object,
    launches: int,
) -> float:
    _flush_l2(torch, flush)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(launches):
        candidate.launch(case_id, material)
    end.record()
    end.synchronize()
    elapsed = float(start.elapsed_time(end)) / launches
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError("HIP event timing sample is not finite and positive")
    return elapsed


def _warmup(
    torch: object,
    candidate: _RuntimeCandidate,
    case_id: str,
    material: _CaseMaterial,
    launches: int,
) -> None:
    for _ in range(launches):
        candidate.launch(case_id, material)
    torch.cuda.synchronize()


def _screen(
    *,
    contract: AmdRmsNormSearchContract,
    candidates: Mapping[str, _RuntimeCandidate],
    cases: Mapping[str, _CaseMaterial],
    torch: object,
    flush: object,
    evidence: _Evidence,
) -> tuple[str | None, dict[str, object]]:
    protocol = contract.screening
    case_id = protocol.case_id
    material = cases[case_id]
    ids = sorted(candidates)
    if not ids:
        result = {
            "case_id": case_id,
            "orders": [],
            "candidates": {},
            "quality_qualified_candidate_ids": [],
            "selected_candidate_id": None,
            "selection_scope": "fixed_four_candidate_one_row_domain_had_no_correctness_survivor",
        }
        evidence.json("screening/result.json", result)
        return None, result
    raw: dict[str, list[list[float]]] = {candidate_id: [] for candidate_id in ids}
    orders = []
    for round_index in range(protocol.rounds_per_candidate):
        offset = round_index % len(ids)
        order = ids[offset:] + ids[:offset]
        if round_index % 2:
            order = list(reversed(order))
        orders.append(order)
        for candidate_id in order:
            candidate = candidates[candidate_id]
            _warmup(
                torch,
                candidate,
                case_id,
                material,
                protocol.warmup_launches,
            )
            samples = [
                _event_sample_ms(
                    torch,
                    candidate,
                    case_id,
                    material,
                    flush,
                    protocol.launches_per_sample,
                )
                for _ in range(protocol.samples_per_round)
            ]
            raw[candidate_id].append(samples)
            evidence.event(
                "screening_cohort",
                {
                    "candidate_id": candidate_id,
                    "round_index": round_index,
                    "samples_ms": samples,
                    "summary": summarize_cohort(samples),
                },
            )
    summaries: dict[str, object] = {}
    qualified = []
    for candidate_id, cohorts in raw.items():
        pooled = [sample for cohort in cohorts for sample in cohort]
        summary = summarize_cohort(pooled)
        cohort_summaries = [summarize_cohort(cohort) for cohort in cohorts]
        quality = (
            float(summary["cv"]) <= protocol.maximum_cv
            and all(
                float(item["cv"]) <= protocol.maximum_cv
                for item in cohort_summaries
            )
        )
        summaries[candidate_id] = {
            "cohorts_ms": cohorts,
            "cohort_summaries": cohort_summaries,
            "pooled_summary": summary,
            "measurement_quality_passed": quality,
        }
        if quality:
            qualified.append(candidate_id)
    selected = (
        min(
            qualified,
            key=lambda candidate_id: cast(
                Mapping[str, object], summaries[candidate_id]
            )["pooled_summary"]["median_ms"],
        )
        if qualified
        else None
    )
    result = {
        "case_id": case_id,
        "orders": orders,
        "candidates": summaries,
        "quality_qualified_candidate_ids": qualified,
        "selected_candidate_id": selected,
        "selection_scope": "fastest_quality_qualified_member_of_fixed_four_candidate_one_row_domain",
    }
    evidence.json("screening/result.json", result)
    return selected, result


def _measure_paired(
    *,
    timing: PairedTimingProtocol,
    warmup_launches_per_cohort: int,
    launches_per_sample: int,
    arms: Mapping[str, _RuntimeCandidate],
    case_id: str,
    cases: Mapping[str, _CaseMaterial],
    torch: object,
    flush: object,
    evidence: _Evidence,
    event_kind: str,
) -> list[dict[str, object]]:
    material = cases[case_id]
    measurements: list[dict[str, object]] = []
    if set(arms) != set(timing.arms):
        raise ValueError("paired runtime arm set differs")
    for pair_index, order in enumerate(timing.pair_order):
        records: dict[str, object] = {}
        for position, arm in enumerate(order):
            runtime = arms[arm]
            _warmup(
                torch,
                runtime,
                case_id,
                material,
                warmup_launches_per_cohort,
            )
            samples = [
                _event_sample_ms(
                    torch,
                    runtime,
                    case_id,
                    material,
                    flush,
                    launches_per_sample,
                )
                for _ in range(timing.samples_per_cohort)
            ]
            records[arm] = {
                "position": position,
                "samples_ms": samples,
                "summary": summarize_cohort(samples),
                "route_calls": timing.route_calls_per_cohort,
            }
        measurement = {
            "pair_index": pair_index,
            "order": list(order),
            "arms": records,
        }
        measurements.append(measurement)
        evidence.event(event_kind, cast(Mapping[str, object], measurement))
    return measurements


def _noise(
    *,
    contract: AmdRmsNormSearchContract,
    baseline: _RuntimeCandidate,
    cases: Mapping[str, _CaseMaterial],
    torch: object,
    flush: object,
    evidence: _Evidence,
) -> tuple[object, list[dict[str, object]]]:
    protocol = contract.noise
    measurements = _measure_paired(
        timing=protocol.timing,
        warmup_launches_per_cohort=protocol.warmup_launches_per_cohort,
        launches_per_sample=protocol.launches_per_sample,
        arms={"baseline_a": baseline, "baseline_b": baseline},
        case_id=protocol.case_id,
        cases=cases,
        torch=torch,
        flush=flush,
        evidence=evidence,
        event_kind="baseline_noise_pair",
    )
    decision = derive_noise_decision(contract, measurements)
    evidence.json("noise/measurements.json", measurements)
    return decision, measurements


def _confirm(
    *,
    contract: AmdRmsNormSearchContract,
    candidate: _RuntimeCandidate,
    baseline: _RuntimeCandidate,
    cases: Mapping[str, _CaseMaterial],
    torch: object,
    flush: object,
    evidence: _Evidence,
) -> tuple[object, list[dict[str, object]]]:
    protocol = contract.confirmatory
    measurements = _measure_paired(
        timing=protocol.timing,
        warmup_launches_per_cohort=protocol.warmup_launches_per_cohort,
        launches_per_sample=protocol.launches_per_sample,
        arms={"candidate": candidate, "baseline": baseline},
        case_id=contract.screening.case_id,
        cases=cases,
        torch=torch,
        flush=flush,
        evidence=evidence,
        event_kind="confirmatory_pair",
    )
    decision = derive_confirmatory_decision(contract, measurements)
    evidence.json("confirmatory/measurements.json", measurements)
    return decision, measurements


def _observation_document(decision: object) -> dict[str, object]:
    observation = decision.observation
    return {
        "measurement_quality_passed": observation.measurement_quality_passed,
        "pair_wins": dict(observation.pair_wins),
        "tied_pairs": observation.tied_pairs,
        "pooled_sample_counts": dict(observation.pooled_sample_counts),
        "pooled_medians_ms": dict(observation.pooled_medians_ms),
        "speedup": observation.speedup,
        "classification": observation.classification,
    }


def _failure_class(stage: str) -> str:
    if stage == "source_custody":
        return "CUSTODY_BLOCKED"
    if stage == "static_filter":
        return "AUTHORITY_BLOCKED"
    if stage in {"host_and_process_admission", "runtime_admission"}:
        return "ENVIRONMENT_BLOCKED"
    if stage.endswith("correctness_rejected"):
        return "CORRECTNESS_REJECTED"
    return "HARNESS_FAULT"


def _filter_document(
    compiler: Compiler,
    candidates: tuple[AmdRmsNormCandidate, ...],
) -> dict[str, object]:
    assessments = tuple(compiler.assess(item.document) for item in candidates)
    if any(not item.lowering_eligible for item in assessments):
        raise ValueError("the frozen search domain contains a non-lowerable candidate")
    scored, withheld = compiler.rank(assessments)
    expected_withheld = tuple(item.schedule_id for item in assessments)
    if scored or withheld != expected_withheld:
        raise ValueError("first gfx1151 search requires explicit ranking abstention")
    return {
        "construction_passed_count": len(candidates),
        "verifier_passed_count": sum(item.accepted for item in assessments),
        "lowering_eligible_count": sum(
            item.lowering_eligible for item in assessments
        ),
        "candidate_findings": {
            candidate.candidate_id: {
                "schedule_id": assessment.schedule_id,
                "schedule_sha256": assessment.schedule_sha256,
                "accepted": assessment.accepted,
                "lowering_eligible": assessment.lowering_eligible,
                "calibration_available": assessment.calibration_available,
                "finding_codes": [item.code for item in assessment.findings],
            }
            for candidate, assessment in zip(candidates, assessments)
        },
        "ranking_attempted": True,
        "ranking_applied": False,
        "ranking_abstained_reason": "gfx1151_calibration_unavailable",
        "scored_candidates": [],
        "withheld_candidate_ids": [item.candidate_id for item in candidates],
        "withheld_schedule_ids": list(withheld),
        "gpu_survivor_policy": "empirical_calibration_sweep_all_verifier_survivors",
    }


def _prepare(
    root: Path, contract: AmdRmsNormSearchContract
) -> tuple[Compiler, ExecutorRevision, tuple[AmdRmsNormCandidate, ...], dict[str, object]]:
    compiler = Compiler.load(root, contract.compiler_path)
    gate = compiler.check_corpus()
    if (
        gate.compiler_revision_id != contract.compiler_revision_id
        or gate.compiler_revision_sha256 != contract.compiler_sha256
        or compiler.state != "released"
        or not gate.passed
    ):
        raise ValueError("search Compiler authority differs")
    executor = ExecutorRevision.load(root, contract.executor_path)
    if (
        executor.executor_id != contract.executor_id
        or executor.canonical_sha256 != contract.executor_sha256
        or executor.document["schema_version"] != 2
        or executor.document["host_environment"].get("runtime_kind") != "hip"
    ):
        raise ValueError("search Executor authority differs")
    candidates = materialize_candidates(contract)
    filter_document = _filter_document(compiler, candidates)
    prepared = {
        "schema_version": 1,
        "kind": "open_cake_gfx1151_llama_rmsnorm_search_v2",
        "status": "prepared",
        "search_id": contract.search_id,
        "search_contract_sha256": contract.canonical_sha256,
        "compiler": {
            "revision_id": gate.compiler_revision_id,
            "canonical_sha256": gate.compiler_revision_sha256,
        },
        "executor": {
            "executor_id": executor.executor_id,
            "canonical_sha256": executor.canonical_sha256,
        },
        "candidate_count": len(candidates),
        "candidate_ids": [item.candidate_id for item in candidates],
        "filter": filter_document,
        "gpu_submitted": False,
        "performance_measured": False,
    }
    return compiler, executor, candidates, prepared


def _run(
    *,
    root: Path,
    contract: AmdRmsNormSearchContract,
    compiler: Compiler,
    executor: ExecutorRevision,
    candidate_specs: tuple[AmdRmsNormCandidate, ...],
    artifact_dir: Path,
) -> tuple[int, dict[str, object]]:
    source = git_state(root)
    evidence = _Evidence(artifact_dir)
    runtimes: list[_RuntimeCandidate] = []
    authority = {
        "schema_version": 1,
        "search_id": contract.search_id,
        "target": "gfx1151",
        "search_contract": {
            "path": contract.path.relative_to(root).as_posix(),
            "canonical_sha256": contract.canonical_sha256,
        },
        "compiler": {
            "path": contract.compiler_path.relative_to(root).as_posix(),
            "revision_id": contract.compiler_revision_id,
            "canonical_sha256": contract.compiler_sha256,
        },
        "executor": {
            "path": contract.executor_path.relative_to(root).as_posix(),
            "executor_id": contract.executor_id,
            "canonical_sha256": contract.executor_sha256,
        },
        "workload": {
            "path": contract.workload_path.relative_to(root).as_posix(),
            "canonical_sha256": contract.workload_sha256,
        },
    }
    evidence.json("attempt-authority.json", authority)
    evidence.json("protocol.json", json.loads(contract.path.read_text(encoding="utf-8")))
    evidence.json("source-custody.json", source)
    stage = "source_custody"
    try:
        if not bool(source["tree_clean"]):
            raise RuntimeError("a clean Git tree is required for retained timing")
        stage = "static_filter"
        filter_document = _filter_document(compiler, candidate_specs)
        evidence.json("filter.json", filter_document)
        stage = "host_and_process_admission"
        host_admission, process_initial = _admit_search_host(executor)
        monitor_path = str(host_admission.device_monitor["path"])
        evidence.json("amd-smi-process-initial.json", process_initial)
        stage = "runtime_admission"
        baseline_document, _ = _canonical_document(root / BASELINE_SCHEDULE)
        baseline_assessment = compiler.assess(baseline_document)
        baseline_lowering = compiler.lower(baseline_assessment)
        baseline_requirements = require_object(
            baseline_lowering.toolchain_requirements, "baseline.toolchain"
        )
        torch, triton, properties = admit_exact_hip(baseline_requirements)
        runtime = _runtime_document(torch, triton, properties)
        evidence.json("runtime.json", runtime)
        evidence.json(
            "runtime-admission.json",
            {
                "schema_version": 1,
                "admitted": True,
                "executor": dict(executor.reference),
                "lowering_target": dict(
                    require_object(
                        baseline_requirements["triton_target"], "triton_target"
                    )
                ),
                "host": {
                    "torch_hip_version": host_admission.torch_hip_version,
                    "visible_device_count": host_admission.visible_device_count,
                    "device_monitor": dict(host_admission.device_monitor),
                    "profilers": [
                        dict(value) for value in host_admission.profilers
                    ],
                },
                "observed_runtime": runtime,
            },
        )
        evidence.event("run_started", {"search_id": contract.search_id})
        stage = "workload_materialization"
        workload = WorkloadContract.load(contract.workload_path)
        cases = _case_materials(workload, torch)
        l2_elements = contract.screening.l2_flush_bytes // 4
        flush = torch.zeros(l2_elements, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()

        stage = "baseline_correctness"
        baseline = _compile_candidate(
            compiler=compiler,
            candidate_id="baseline-r64-w4",
            row_tile=64,
            num_warps=4,
            schedule=baseline_document,
            cases=cases,
            torch=torch,
            evidence=evidence,
            relative_root="candidates",
        )
        runtimes.append(baseline)
        baseline_correctness = _correctness(baseline, workload, cases, torch)
        evidence.json(
            "candidates/baseline-r64-w4/correctness.json", baseline_correctness
        )
        if not baseline_correctness["passed"]:
            stage = "baseline_correctness_rejected"
            raise RuntimeError("baseline correctness failed")

        profiler_record = _profiler_record(host_admission.profilers, "rocprofv3")
        profiler = (
            str(profiler_record["path"]) if profiler_record is not None else None
        )
        stage = "baseline_noise"
        noise_decision, noise_measurements = _noise(
            contract=contract,
            baseline=baseline,
            cases=cases,
            torch=torch,
            flush=flush,
            evidence=evidence,
        )
        stage = "baseline_noise_postflight_correctness"
        noise_postflight = _correctness(baseline, workload, cases, torch)
        noise_record = {
            "preflight_correctness": baseline_correctness,
            "measurements": noise_measurements,
            "observation": _observation_document(noise_decision),
            "postflight_correctness": noise_postflight,
            "passed": bool(noise_decision.passed),
        }
        evidence.json("noise/result.json", noise_record)
        if not noise_postflight["passed"]:
            stage = "baseline_noise_postflight_correctness_rejected"
            raise RuntimeError("baseline noise postflight correctness failed")
        if not noise_decision.passed:
            result = {
                "schema_version": 1,
                "kind": "open_cake_gfx1151_llama_rmsnorm_search_v2",
                "status": INCONCLUSIVE_MEASUREMENT_QUALITY,
                "search_id": contract.search_id,
                "search_contract_sha256": contract.canonical_sha256,
                "source_custody": source,
                "filter": filter_document,
                "noise": noise_record,
                "candidate_timing_started": False,
                "performance_measured": True,
                "profiler_tooling_available": profiler is not None,
                "profiler_evidence_collected": False,
                "promotion_authorized": False,
                "llama_cpp_e2e_claim": False,
                "diagnosis": diagnose_terminal_decision(
                    INCONCLUSIVE_MEASUREMENT_QUALITY,
                    profiler_evidence_collected=False,
                ).document(),
            }
            evidence.json("result.json", result)
            evidence.manifest()
            return 3, result

        survivors: dict[str, _RuntimeCandidate] = {}
        dispositions: dict[str, object] = {}
        stage = "candidate_correctness"
        for spec in candidate_specs:
            runtime = _compile_candidate(
                compiler=compiler,
                candidate_id=spec.candidate_id,
                row_tile=spec.row_tile,
                num_warps=spec.num_warps,
                schedule=spec.document,
                cases=cases,
                torch=torch,
                evidence=evidence,
                relative_root="candidates",
            )
            runtimes.append(runtime)
            correctness = _correctness(runtime, workload, cases, torch)
            evidence.json(
                f"candidates/{spec.candidate_id}/correctness.json", correctness
            )
            if not correctness["inputs_unchanged"]:
                stage = "candidate_correctness_rejected"
                raise RuntimeError(
                    f"candidate {spec.candidate_id} mutated a Workload input"
                )
            if correctness["passed"]:
                survivors[spec.candidate_id] = runtime
                disposition = "correctness_qualified"
            else:
                disposition = "CORRECTNESS_REJECTED"
            dispositions[spec.candidate_id] = {
                "status": disposition,
                "correctness": correctness,
            }
            evidence.event(
                "candidate_disposition",
                {
                    "candidate_id": spec.candidate_id,
                    **cast(Mapping[str, object], dispositions[spec.candidate_id]),
                },
            )
        evidence.json("candidate-dispositions.json", dispositions)

        stage = "screening"
        selected_id, screening = _screen(
            contract=contract,
            candidates=survivors,
            cases=cases,
            torch=torch,
            flush=flush,
            evidence=evidence,
        )
        if not all(_inputs_unchanged(material, torch) for material in cases.values()):
            stage = "screening_correctness_rejected"
            raise RuntimeError("screening mutated a Workload input")
        if selected_id is None:
            result = {
                "schema_version": 1,
                "kind": "open_cake_gfx1151_llama_rmsnorm_search_v2",
                "status": INCONCLUSIVE_MEASUREMENT_QUALITY,
                "search_id": contract.search_id,
                "search_contract_sha256": contract.canonical_sha256,
                "source_custody": source,
                "correctness_qualified_candidate_count": len(survivors),
                "screening": screening,
                "noise": noise_record,
                "filter": filter_document,
                "performance_measured": True,
                "profiler_tooling_available": profiler is not None,
                "profiler_evidence_collected": False,
                "promotion_authorized": False,
                "llama_cpp_e2e_claim": False,
                "diagnosis": diagnose_terminal_decision(
                    INCONCLUSIVE_MEASUREMENT_QUALITY,
                    profiler_evidence_collected=False,
                ).document(),
            }
            evidence.json("result.json", result)
            evidence.manifest()
            return 3, result

        selected_spec = next(
            item for item in candidate_specs if item.candidate_id == selected_id
        )
        stage = "confirmatory_compilation"
        confirm_candidate = _compile_candidate(
            compiler=compiler,
            candidate_id="candidate",
            row_tile=selected_spec.row_tile,
            num_warps=selected_spec.num_warps,
            schedule=selected_spec.document,
            cases=cases,
            torch=torch,
            evidence=evidence,
            relative_root="confirmatory",
        )
        confirm_baseline = _compile_candidate(
            compiler=compiler,
            candidate_id="baseline",
            row_tile=64,
            num_warps=4,
            schedule=baseline_document,
            cases=cases,
            torch=torch,
            evidence=evidence,
            relative_root="confirmatory",
        )
        runtimes.extend((confirm_candidate, confirm_baseline))
        stage = "confirmatory_preflight_correctness"
        preflight = {
            "candidate": _correctness(confirm_candidate, workload, cases, torch),
            "baseline": _correctness(confirm_baseline, workload, cases, torch),
        }
        evidence.json("confirmatory/preflight-correctness.json", preflight)
        if not all(bool(value["passed"]) for value in preflight.values()):
            stage = "confirmatory_preflight_correctness_rejected"
            raise RuntimeError("confirmatory preflight correctness failed")
        stage = "confirmatory_timing"
        decision, measurements = _confirm(
            contract=contract,
            candidate=confirm_candidate,
            baseline=confirm_baseline,
            cases=cases,
            torch=torch,
            flush=flush,
            evidence=evidence,
        )
        stage = "confirmatory_postflight_correctness"
        postflight = {
            "candidate": _correctness(confirm_candidate, workload, cases, torch),
            "baseline": _correctness(confirm_baseline, workload, cases, torch),
        }
        evidence.json("confirmatory/postflight-correctness.json", postflight)
        if not all(bool(value["passed"]) for value in postflight.values()):
            stage = "confirmatory_postflight_correctness_rejected"
            raise RuntimeError("confirmatory postflight correctness failed")

        stage = "terminal_evidence"
        observation = _observation_document(decision)
        profiler_available = profiler is not None
        result = {
            "schema_version": 1,
            "kind": "open_cake_gfx1151_llama_rmsnorm_search_v2",
            "status": decision.status,
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
                "workload_id": workload.workload_id,
                "canonical_sha256": workload.canonical_sha256,
                "case_ids": list(workload.case_ids),
            },
            "source_custody": source,
            "selected_candidate": {
                "candidate_id": selected_id,
                "row_tile": selected_spec.row_tile,
                "num_warps": selected_spec.num_warps,
                "schedule_sha256": confirm_candidate.assessment.schedule_sha256,
                "source_sha256": confirm_candidate.lowering.source_sha256,
                "hsaco_sha256": artifact_records(confirm_candidate.artifacts)["hsaco"][
                    "sha256"
                ],
                "amdgcn_resources": amdgcn_resource_record(
                    confirm_candidate.artifacts["amdgcn"]
                ),
            },
            "baseline": {
                "route": {
                    "backend": baseline.assessment.route.backend.value,
                    "entry_point": baseline.assessment.route.entry_point,
                },
                "row_tile": 64,
                "num_warps": 4,
                "schedule_sha256": confirm_baseline.assessment.schedule_sha256,
                "source_sha256": confirm_baseline.lowering.source_sha256,
                "hsaco_sha256": artifact_records(confirm_baseline.artifacts)["hsaco"][
                    "sha256"
                ],
                "amdgcn_resources": amdgcn_resource_record(
                    confirm_baseline.artifacts["amdgcn"]
                ),
            },
            "candidate_dispositions": dispositions,
            "filter": filter_document,
            "screening": screening,
            "noise": noise_record,
            "confirmatory": {
                "preflight_correctness": preflight,
                "measurements": measurements,
                "observation": observation,
                "postflight_correctness": postflight,
            },
            "performance_measured": True,
            "timer_scope": "direct Triton JIT kernel on current HIP stream",
            "fallback_calls": 0,
            "profiler_tooling_available": profiler_available,
            "profiler_executable": profiler,
            "profiler_authority": profiler_record,
            "profiler_evidence_collected": False,
            "leaf_timing_claim": decision.status == LEAF_TIMING_WIN,
            "promotion_authorized": False,
            "llama_cpp_build_claim": False,
            "llama_cpp_e2e_claim": False,
            "diagnosis": diagnose_terminal_decision(
                decision.status,
                profiler_evidence_collected=False,
            ).document(),
        }
        evidence.json(
            "amd-smi-metric-final.json", _amd_smi(monitor_path, ["metric"])
        )
        evidence.json("result.json", result)
        evidence.event(
            "run_completed",
            {
                "status": decision.status,
                "selected_candidate_id": selected_id,
                "speedup": observation["speedup"],
            },
        )
        evidence.manifest()
        exit_code = 0 if decision.status == LEAF_TIMING_WIN else 2
        return exit_code, result
    except BaseException as error:
        failure_class = _failure_class(stage)
        failure = {
            "schema_version": 1,
            "kind": "open_cake_gfx1151_llama_rmsnorm_search_failure_v2",
            "status": failure_class,
            "failure_class": failure_class,
            "failed_stage": stage,
            "authority": authority,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "performance_conclusion_authorized": False,
        }
        if not (artifact_dir / "failure.json").exists():
            evidence.json("failure.json", failure)
            evidence.event("run_failed", {"error": failure["error"]})
            if not (artifact_dir / "manifest.json").exists():
                evidence.manifest()
        raise
    finally:
        for runtime in runtimes:
            runtime.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    root = arguments.project_root.resolve(strict=True)
    contract_path = arguments.contract or root / DEFAULT_CONTRACT
    contract = AmdRmsNormSearchContract.load(root, contract_path)
    compiler, executor, candidates, prepared = _prepare(root, contract)
    artifact_dir: Path | None = None
    if arguments.artifact_dir is not None:
        try:
            artifact_dir = resolve_new_external_directory(root, arguments.artifact_dir)
        except ValueError as error:
            parser.error(str(error))
    if arguments.prepare_only:
        result = prepared
        exit_code = 0
    else:
        if artifact_dir is None:
            parser.error("--artifact-dir is required unless --prepare-only is used")
        exit_code, result = _run(
            root=root,
            contract=contract,
            compiler=compiler,
            executor=executor,
            candidate_specs=candidates,
            artifact_dir=artifact_dir,
        )
    payload = canonical_json_bytes(result) + b"\n"
    if arguments.output is not None:
        output = arguments.output.absolute()
        output = output.parent.resolve(strict=True) / output.name
        if output.exists() or output.is_symlink():
            parser.error("--output must be a new path")
        with output.open("xb") as stream:
            stream.write(payload)
    sys.stdout.buffer.write(payload)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
