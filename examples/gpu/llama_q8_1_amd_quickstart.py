#!/usr/bin/env python3
"""Prepare or validate the live llama.cpp Q8_1 producer on exact gfx1151."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import sys
import traceback
from hashlib import sha256
from pathlib import Path
from typing import Mapping, cast


ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler  # noqa: E402
from open_cake_ir.evaluation import (  # noqa: E402
    WorkloadContract,
    materialize_q4_mmvq_case,
    q4_mmvq_reference,
)
from open_cake_ir.evaluation.llama_q4_mmvq import (  # noqa: E402
    Q8_1_WORKSPACE_BYTES,
)
from open_cake_ir.evaluation.triton_hip import (  # noqa: E402
    artifact_records,
    extract_artifacts,
    git_state,
    load_generated_module,
    admit_exact_hip,
    require_object,
    resolve_new_external_directory,
    write_new_json,
)
from open_cake_ir.lab import ExecutorRevision  # noqa: E402


SCHEDULE = "corpus/schedules/packed-q8_1-producer-gfx1151.json"
WORKLOAD = "contracts/workloads/llama-q4_0-q8_1-mmvq-f32-v1.json"
RUNNER_SOURCE = "examples/gpu/llama_q8_1_amd_quickstart.py"
RESULT_KIND = "open_cake_gfx1151_llama_q8_1_producer_quickstart_v1"
_EXECUTOR_ID = re.compile(r"open-cake-ir-gfx1151-v[1-9][0-9]*")
_SAFE_CASE_ID = re.compile(r"[a-z][a-z0-9_]*")
_EXPECTED_CASE_IDS = (
    "seeded_random",
    "stored_sum_correction_stress",
    "zero_scale",
    "roundf_boundary",
    "xor_tree_sum",
    "partial_rounding",
    "q4_zero_scale",
)
_EXPECTED_REQUIREMENTS = {
    "source_language": "python",
    "compiler": "triton",
    "target": "gfx1151",
    "kernel_entry_point": "_cake_packed_q8_1_producer_gfx1151_kernel",
    "signature": {"activation": "*fp32", "q8_workspace": "*u8"},
    "compile_constants": {
        "N_ACTIVATION_TILE": 32,
        "D_Q8_WORKSPACE_0": 16,
        "D_Q8_WORKSPACE_1": 36,
        "BLOCK_ACTIVATION_TILE": 512,
    },
    "compile_options": {"num_warps": 8},
    "grid": [1, 1, 1],
    "triton_target": {"backend": "hip", "arch": "gfx1151", "warp_size": 32},
    "binary_role": "hsaco",
    "assembly_role": "amdgcn",
}


def _object(value: object, context: str) -> Mapping[str, object]:
    return require_object(value, context)


def _relative(root: Path, path: Path, context: str) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"{context} must be inside the checkout") from error


def _validate_lowering(requirements: Mapping[str, object]) -> None:
    if dict(requirements) != _EXPECTED_REQUIREMENTS:
        raise ValueError("Q8_1 producer lowering requirements differ")


def _assessment_summary(assessment: object) -> dict[str, object]:
    route = assessment.route
    return {
        "schedule_id": assessment.schedule_id,
        "schedule_sha256": assessment.schedule_sha256,
        "target": assessment.target,
        "route": (
            None
            if route is None
            else {"backend": route.backend.value, "entry_point": route.entry_point}
        ),
        "accepted": assessment.accepted,
        "lowering_eligible": assessment.lowering_eligible,
        "calibration_available": assessment.calibration_available,
        "findings": [
            {"code": item.code, "path": item.path, "message": item.message}
            for item in assessment.findings
        ],
    }


def _prepare(
    project_root: Path, revision_path: Path
) -> tuple[Compiler, object, WorkloadContract, dict[str, object]]:
    schedule_path = project_root / SCHEDULE
    workload_path = project_root / WORKLOAD
    workload = WorkloadContract.load(workload_path)
    if (
        workload.document.get("operator") != "llama_q4_0_q8_1_mmvq_f32"
        or workload.case_ids != _EXPECTED_CASE_IDS
    ):
        raise ValueError("Q8_1 producer Workload authority differs")
    schedule = _object(
        json.loads(schedule_path.read_text(encoding="utf-8")), "schedule"
    )
    metadata = _object(schedule.get("metadata"), "schedule.metadata")
    if metadata.get("workload_contract_sha256") != workload.canonical_sha256:
        raise ValueError("Q8_1 producer Schedule is not bound to the frozen Workload")

    compiler = Compiler.load(project_root, revision_path)
    assessment = compiler.assess_file(schedule_path)
    summary: dict[str, object] = {
        "schema_version": 1,
        "kind": RESULT_KIND,
        "status": "rejected",
        "scope": {
            "kernel": "live_fp32_to_q8_1",
            "workspace_bytes_per_case": Q8_1_WORKSPACE_BYTES,
            "mmvq_consumer_evaluated": False,
            "workload_claim_complete": False,
            "performance_measured": False,
            "promotion_authorized": False,
        },
        "compiler_revision": {
            "path": _relative(project_root, revision_path, "Compiler Revision"),
            "revision_id": assessment.compiler_revision_id,
            "canonical_sha256": assessment.compiler_revision_sha256,
            "state": compiler.state,
        },
        "assessment": _assessment_summary(assessment),
        "workload": {
            "path": WORKLOAD,
            "workload_id": workload.workload_id,
            "canonical_sha256": workload.canonical_sha256,
            "case_ids": list(workload.case_ids),
        },
        "evaluation": {
            "gpu_submitted": False,
            "producer_correctness_passed": False,
            "kernel_calls": 0,
            "fallback_calls": 0,
            "performance_measured": False,
        },
    }
    if not assessment.lowering_eligible:
        return compiler, None, workload, summary
    if (
        assessment.target != "gfx1151"
        or assessment.route is None
        or assessment.route.backend.value != "triton"
        or assessment.route.entry_point != "cake_packed_q8_1_producer_gfx1151"
        or assessment.calibration_available
    ):
        raise ValueError("Q8_1 producer requires the exact uncalibrated gfx1151 route")
    lowering = compiler.lower(assessment)
    requirements = _object(lowering.toolchain_requirements, "lowering.toolchain")
    _validate_lowering(requirements)
    summary["status"] = "prepared"
    summary["lowering"] = {
        "generated": lowering.generated,
        "entry_point": lowering.route.entry_point,
        "source_sha256": lowering.source_sha256,
        "source_bytes": len(lowering.source.encode("utf-8")),
        "toolchain_requirements": dict(requirements),
    }
    return compiler, lowering, workload, summary


def _admit_released_compiler_lock(
    project_root: Path,
    revision_path: Path,
    compiler: Compiler,
) -> object:
    lock = (project_root / "compiler/revision.lock.json").resolve(strict=True)
    if revision_path != lock or compiler.state != "released":
        raise ValueError("Q8_1 GPU execution requires the released Compiler lock")
    gate = compiler.check_corpus()
    if not gate.passed:
        raise ValueError("Q8_1 GPU execution requires a passing released Corpus Gate")
    return gate


def _admit_executor_contract(executor: ExecutorRevision) -> None:
    document = executor.document
    if (
        document.get("schema_version") != 2
        or _EXECUTOR_ID.fullmatch(executor.executor_id) is None
        or _object(document.get("host_environment"), "Executor host").get(
            "runtime_kind"
        )
        != "hip"
    ):
        raise ValueError("Q8_1 GPU execution requires an exact released gfx1151 Executor")
    source_paths = {
        str(_object(value, "Executor source").get("path"))
        for value in cast(tuple[object, ...], document["sources"])
    }
    if RUNNER_SOURCE not in source_paths:
        raise ValueError("gfx1151 Executor does not own the Q8_1 runner")


def _workspace_metrics(
    observed: bytes,
    expected: bytes,
    *,
    activation_unchanged: bool,
) -> dict[str, object]:
    mismatch_count = sum(
        left != right for left, right in zip(observed, expected)
    ) + abs(len(observed) - len(expected))
    full_byte_exact = (
        len(observed) == Q8_1_WORKSPACE_BYTES
        and len(expected) == Q8_1_WORKSPACE_BYTES
        and mismatch_count == 0
    )
    padding_records_zero = (
        len(observed) == Q8_1_WORKSPACE_BYTES
        and observed[36:] == bytes(15 * 36)
    )
    first_mismatch = next(
        (
            index
            for index, (left, right) in enumerate(
                zip(observed, expected)
            )
            if left != right
        ),
        None,
    )
    if first_mismatch is None and len(observed) != len(expected):
        first_mismatch = min(len(observed), len(expected))
    return {
        "passed": full_byte_exact and activation_unchanged and padding_records_zero,
        "activation_unchanged": activation_unchanged,
        "padding_records_zero": padding_records_zero,
        "q8_workspace_byte_exact": full_byte_exact,
        "q8_workspace_bytes_compared": Q8_1_WORKSPACE_BYTES,
        "q8_workspace_observed_bytes": len(observed),
        "q8_workspace_mismatch_count": mismatch_count,
        "first_mismatch_byte": first_mismatch,
    }


def _write_case_artifacts(
    evidence_root: Path,
    case_id: str,
    observed: bytes,
    reference: bytes,
) -> dict[str, dict[str, object]]:
    if _SAFE_CASE_ID.fullmatch(case_id) is None or case_id not in _EXPECTED_CASE_IDS:
        raise ValueError("Q8_1 evidence case id differs")
    if len(observed) != Q8_1_WORKSPACE_BYTES or len(reference) != Q8_1_WORKSPACE_BYTES:
        raise ValueError("Q8_1 evidence requires complete 576-byte workspaces")
    records: dict[str, dict[str, object]] = {}
    for role, payload in (("observed", observed), ("reference", reference)):
        name = f"case-{case_id}.{role}-q8_1.bin"
        path = evidence_root / name
        with path.open("xb") as stream:
            stream.write(payload)
        records[role] = {
            "path": name,
            "sha256": sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    return records


def _tensor_bytes(value: object) -> bytes:
    flat = value.detach().cpu().reshape(-1).tolist()
    return bytes(int(item) for item in flat)


def _write_manifest(root: Path) -> None:
    files = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.name == "manifest.json":
            continue
        payload = path.read_bytes()
        files.append(
            {
                "path": path.name,
                "sha256": sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    write_new_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "kind": "open_cake_gfx1151_q8_1_evidence_manifest_v1",
            "files": files,
        },
    )


def _failure_class(stage: str) -> str:
    if stage in {"compiler_admission", "executor_source_custody", "git_custody"}:
        return "AUTHORITY_BLOCKED"
    if stage in {"executor_host_admission", "runtime_admission"}:
        return "ENVIRONMENT_BLOCKED"
    return "HARNESS_FAULT"


def _run_live_impl(
    project_root: Path,
    revision_path: Path,
    compiler: Compiler,
    executor: ExecutorRevision,
    lowering: object,
    workload: WorkloadContract,
    summary: dict[str, object],
    attempt: dict[str, str],
    evidence_root: Path,
) -> int:
    requirements = _object(lowering.toolchain_requirements, "lowering.toolchain")
    _validate_lowering(requirements)
    attempt["stage"] = "compiler_admission"
    _admit_released_compiler_lock(project_root, revision_path, compiler)
    attempt["stage"] = "executor_source_custody"
    _admit_executor_contract(executor)
    attempt["stage"] = "git_custody"
    source = git_state(project_root)
    if not bool(source["tree_clean"]):
        raise RuntimeError("a clean Git tree is required for retained AMD correctness")
    summary["source_custody"] = source
    attempt["stage"] = "executor_host_admission"
    host = executor.admit_hip_host()
    attempt["stage"] = "runtime_admission"
    torch, _triton, properties = admit_exact_hip(requirements)
    summary["executor"] = {
        **dict(executor.reference),
        "host_admitted": True,
        "torch_hip_version": host.torch_hip_version,
        "device_monitor": dict(host.device_monitor),
        "profilers": [dict(value) for value in host.profilers],
    }

    attempt["stage"] = "compilation_and_correctness"
    module, generated_directory = load_generated_module(lowering)
    artifact_payloads: dict[str, bytes] | None = None
    artifact_index: dict[str, dict[str, object]] | None = None
    compiled = None
    cases: list[dict[str, object]] = []
    try:
        kernel_name = str(requirements["kernel_entry_point"])
        kernel = getattr(module, kernel_name, None)
        if kernel is None:
            raise RuntimeError("generated Q8_1 Triton kernel is missing")
        constants = dict(_object(requirements["compile_constants"], "compile_constants"))
        options = dict(_object(requirements["compile_options"], "compile_options"))
        grid = tuple(int(value) for value in cast(list[object], requirements["grid"]))
        for case_id in workload.case_ids:
            material = materialize_q4_mmvq_case(workload, case_id)
            reference = q4_mmvq_reference(workload, material)
            activation = torch.tensor(
                material.activation,
                dtype=torch.float32,
                device="cuda",
            )
            activation_before = activation.detach().clone()
            output = torch.full(
                (16, 36),
                0xA5,
                dtype=torch.uint8,
                device="cuda",
            )
            observed_compiled = kernel.run(
                activation,
                output,
                **constants,
                **options,
                grid=grid,
                warmup=False,
            )
            torch.cuda.synchronize()
            if observed_compiled is None:
                raise RuntimeError("Triton launch returned no compiled Q8_1 kernel")
            observed_artifacts = extract_artifacts(observed_compiled)
            observed_index = artifact_records(observed_artifacts)
            if artifact_payloads is None:
                artifact_payloads = observed_artifacts
                artifact_index = observed_index
                compiled = observed_compiled
            elif observed_index != artifact_index:
                raise RuntimeError("the seven Q8_1 cases used different artifacts")
            observed = _tensor_bytes(output)
            activation_unchanged = bool(torch.equal(activation, activation_before))
            case_artifacts = _write_case_artifacts(
                evidence_root,
                case_id,
                observed,
                reference.q8_1_workspace,
            )
            cases.append(
                {
                    "case_id": case_id,
                    **_workspace_metrics(
                        observed,
                        reference.q8_1_workspace,
                        activation_unchanged=activation_unchanged,
                    ),
                    "artifacts": case_artifacts,
                }
            )
        if tuple(case["case_id"] for case in cases) != _EXPECTED_CASE_IDS:
            raise RuntimeError("the frozen Q8_1 case set was not fully executed")
        assert compiled is not None
        assert artifact_payloads is not None
        assert artifact_index is not None
    finally:
        generated_directory.cleanup()

    passed = all(bool(case["passed"]) for case in cases)
    summary["status"] = (
        "producer_correctness_passed" if passed else "producer_correctness_rejected"
    )
    summary["runtime"] = {
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "torch_hip": str(torch.version.hip),
        "triton": importlib.metadata.version("triton"),
        "triton_target": dict(
            _object(requirements["triton_target"], "triton_target")
        ),
        "device_name": str(properties.name),
        "gcn_arch_name": str(properties.gcnArchName),
        "warp_size": int(properties.warp_size),
        "multiprocessor_count": int(properties.multi_processor_count),
        "total_memory_bytes": int(properties.total_memory),
    }
    summary["build"] = {
        "kernel_name": str(compiled.metadata.name),
        "grid": list(cast(list[object], requirements["grid"])),
        "shared_memory_bytes": int(compiled.metadata.shared),
        "artifacts": artifact_index,
    }
    launch_receipt = {
        "schema_version": 1,
        "target": "gfx1151",
        "kernel_name": str(compiled.metadata.name),
        "grid": list(cast(list[object], requirements["grid"])),
        "block": [
            int(_object(requirements["compile_options"], "compile_options")["num_warps"])
            * int(_object(requirements["triton_target"], "triton_target")["warp_size"]),
            1,
            1,
        ],
        "dynamic_shared_memory_bytes": int(compiled.metadata.shared),
        "hsaco_sha256": artifact_index["hsaco"]["sha256"],
        "case_count": len(cases),
        "kernel_calls": len(cases),
        "fallback_calls": 0,
        "producer_launches_per_case": 1,
        "post_launch_synchronized": True,
    }
    summary["evaluation"] = {
        "gpu_submitted": True,
        "producer_correctness_passed": passed,
        "cases": cases,
        "kernel_calls": len(cases),
        "fallback_calls": 0,
        "launch_receipt": launch_receipt,
        "performance_measured": False,
        "mmvq_consumer_evaluated": False,
        "workload_claim_complete": False,
        "promotion_authorized": False,
    }

    attempt["stage"] = "artifact_retention"
    (evidence_root / "generated.py").write_text(lowering.source, encoding="utf-8")
    for role, payload in artifact_payloads.items():
        suffix = "bin" if role == "hsaco" else "txt"
        (evidence_root / f"kernel.{role}.{suffix}").write_bytes(payload)
    write_new_json(evidence_root / "case-results.json", cases)
    summary["evidence_root"] = str(evidence_root)
    write_new_json(evidence_root / "result.json", summary)
    return 0 if passed else 2


def _run_live(
    project_root: Path,
    revision_path: Path,
    compiler: Compiler,
    executor: ExecutorRevision,
    lowering: object,
    workload: WorkloadContract,
    summary: dict[str, object],
    evidence_root: Path,
) -> int:
    evidence_root.mkdir(mode=0o700)
    authority = {
        "schema_version": 1,
        "kind": RESULT_KIND,
        "compiler_revision": summary["compiler_revision"],
        "executor": dict(executor.reference),
        "assessment": summary["assessment"],
        "workload": summary["workload"],
        "scope": summary["scope"],
    }
    write_new_json(evidence_root / "attempt-authority.json", authority)
    attempt = {"stage": "compiler_admission"}
    try:
        result = _run_live_impl(
            project_root,
            revision_path,
            compiler,
            executor,
            lowering,
            workload,
            summary,
            attempt,
            evidence_root,
        )
        _write_manifest(evidence_root)
        return result
    except BaseException as error:
        failure_class = _failure_class(attempt["stage"])
        failure = {
            "schema_version": 1,
            "kind": "open_cake_gfx1151_llama_q8_1_failure_v1",
            "status": failure_class,
            "failure_class": failure_class,
            "failed_stage": attempt["stage"],
            "authority": authority,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "gpu_result_authorized": False,
            "performance_conclusion_authorized": False,
        }
        if not (evidence_root / "failure.json").exists():
            write_new_json(evidence_root / "failure.json", failure)
        if not (evidence_root / "manifest.json").exists():
            _write_manifest(evidence_root)
        raise


def _existing_file(value: Path, context: str) -> Path:
    if value.is_symlink():
        raise ValueError(f"{context} custody differs")
    path = value.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{context} custody differs")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--revision", type=Path)
    parser.add_argument("--executor", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    project_root = arguments.project_root.resolve(strict=True)
    revision_argument = arguments.revision
    if revision_argument is None:
        revision_argument = project_root / (
            "compiler/revision.json"
            if arguments.prepare_only
            else "compiler/revision.lock.json"
        )
    try:
        revision_path = _existing_file(revision_argument, "Compiler Revision")
    except ValueError as error:
        parser.error(str(error))

    executor: ExecutorRevision | None = None
    if arguments.executor is not None:
        try:
            executor = ExecutorRevision.load(
                project_root,
                _existing_file(arguments.executor, "Executor Revision"),
            )
            _admit_executor_contract(executor)
        except ValueError as error:
            parser.error(str(error))
    elif not arguments.prepare_only:
        parser.error("--executor is required for GPU execution")

    evidence_root: Path | None = None
    if arguments.evidence_root is not None:
        try:
            evidence_root = resolve_new_external_directory(
                project_root, arguments.evidence_root
            )
        except ValueError as error:
            parser.error(str(error))
    elif not arguments.prepare_only:
        parser.error("--evidence-root is required for GPU execution")

    output: Path | None = None
    if arguments.output is not None:
        candidate = arguments.output.absolute()
        if candidate.is_symlink():
            parser.error("--output must be a new path")
        output = candidate.parent.resolve(strict=True) / candidate.name
        if output.exists():
            parser.error("--output must be a new path")

    compiler, lowering, workload, summary = _prepare(project_root, revision_path)
    if executor is not None:
        summary["executor"] = {**dict(executor.reference), "host_admitted": False}
    if arguments.prepare_only:
        exit_code = 0 if summary["status"] == "prepared" else 2
    elif lowering is None:
        exit_code = 2
    else:
        assert executor is not None
        assert evidence_root is not None
        exit_code = _run_live(
            project_root,
            revision_path,
            compiler,
            executor,
            lowering,
            workload,
            summary,
            evidence_root,
        )
    payload = json.dumps(summary, sort_keys=True, ensure_ascii=False) + "\n"
    if output is not None:
        with output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        output.chmod(0o644)
    sys.stdout.write(payload)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
