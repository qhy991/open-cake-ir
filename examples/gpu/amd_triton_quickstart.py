#!/usr/bin/env python3
"""Prepare or run one correctness-only Open Cake Triton path on exact gfx1151."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import traceback
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable, Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler  # noqa: E402
from open_cake_ir.evaluation import (  # noqa: E402
    WorkloadContract,
    generate_rmsnorm_case,
    generate_swiglu_case,
    rmsnorm_metrics,
    rmsnorm_oracle,
    swiglu_metrics,
    swiglu_oracle,
)
from open_cake_ir.evaluation.triton_hip import (  # noqa: E402
    admit_exact_hip,
    artifact_bytes,
    git_state,
    load_generated_module,
    require_object,
)
from open_cake_ir.lab import ExecutorRevision  # noqa: E402


@dataclass(frozen=True)
class QuickstartSpec:
    name: str
    result_kind: str
    schedule: str
    workload: str
    generate: Callable[..., tuple[object, ...]]
    oracle: Callable[..., object]
    metrics: Callable[..., dict[str, object]]


_SPECS = {
    "swiglu": QuickstartSpec(
        name="swiglu",
        result_kind="open_cake_gfx1151_swiglu_quickstart_v2",
        schedule="corpus/schedules/swiglu-b8-smoke-gfx1151.json",
        workload="contracts/workloads/swiglu-fp32-v1.json",
        generate=generate_swiglu_case,
        oracle=swiglu_oracle,
        metrics=swiglu_metrics,
    ),
    "llama_rmsnorm": QuickstartSpec(
        name="llama_rmsnorm",
        result_kind="open_cake_gfx1151_llama_rmsnorm_mul_quickstart_v2",
        schedule="corpus/schedules/llama-rmsnorm-mul-b8-gfx1151-r64-w4.json",
        workload="contracts/workloads/llama-rmsnorm-mul-fp32-v2.json",
        generate=generate_rmsnorm_case,
        oracle=rmsnorm_oracle,
        metrics=rmsnorm_metrics,
    ),
}


def _object(value: object, context: str) -> Mapping[str, object]:
    return require_object(value, context)


def _artifact_bytes(value: object, role: str) -> bytes:
    return artifact_bytes(value, role)


def _git_state(project_root: Path) -> dict[str, object]:
    return git_state(project_root)


def _write_new_json(path: Path, value: object) -> None:
    with path.open("xb") as stream:
        stream.write(
            json.dumps(value, sort_keys=True, ensure_ascii=False).encode() + b"\n"
        )


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
    _write_new_json(
        root / "manifest.json",
        {"schema_version": 1, "kind": "open_cake_amd_quickstart_manifest_v1", "files": files},
    )


def _failure_class(stage: str) -> str:
    if stage in {"compiler_admission", "source_custody"}:
        return "AUTHORITY_BLOCKED"
    if stage in {"executor_host_admission", "runtime_admission"}:
        return "ENVIRONMENT_BLOCKED"
    return "HARNESS_FAULT"


def _prepare(
    project_root: Path,
    revision_path: Path,
    schedule_path: Path,
    workload_path: Path,
    spec: QuickstartSpec,
) -> tuple[Compiler, object, object, WorkloadContract, dict[str, object]]:
    workload = WorkloadContract.load(workload_path)
    default_workload = WorkloadContract.load(project_root / spec.workload)
    if workload.document.get("operator") != default_workload.document.get("operator"):
        raise ValueError("AMD quickstart Workload operator differs")
    schedule = _object(
        json.loads(schedule_path.read_text(encoding="utf-8")), "schedule"
    )
    metadata = _object(schedule["metadata"], "schedule.metadata")
    if metadata.get("workload_contract_sha256") != workload.canonical_sha256:
        raise ValueError("AMD Schedule is not bound to the frozen Workload")
    compiler = Compiler.load(project_root, revision_path)
    assessment = compiler.assess_file(schedule_path)
    summary: dict[str, object] = {
        "schema_version": 1,
        "kind": spec.result_kind,
        "status": "rejected",
        "compiler_revision": {
            "revision_id": assessment.compiler_revision_id,
            "canonical_sha256": assessment.compiler_revision_sha256,
        },
        "assessment": {
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
                }
                for item in assessment.findings
            ],
        },
        "workload": {
            "workload_id": workload.workload_id,
            "canonical_sha256": workload.canonical_sha256,
            "case_ids": list(workload.case_ids),
        },
        "evaluation": {
            "gpu_submitted": False,
            "performance_measured": False,
        },
    }
    if not assessment.lowering_eligible:
        return compiler, assessment, None, workload, summary
    if (
        assessment.target != "gfx1151"
        or assessment.route is None
        or assessment.route.backend.value != "triton"
        or assessment.calibration_available
    ):
        raise ValueError("AMD quickstart requires the exact uncalibrated gfx1151 route")
    lowering = compiler.lower(assessment)
    requirements = _object(lowering.toolchain_requirements, "lowering.toolchain")
    if (
        requirements.get("compiler") != "triton"
        or requirements.get("target") != "gfx1151"
        or requirements.get("binary_role") != "hsaco"
        or requirements.get("assembly_role") != "amdgcn"
    ):
        raise ValueError("AMD quickstart lowering toolchain differs")
    summary["status"] = "prepared"
    summary["lowering"] = {
        "generated": lowering.generated,
        "entry_point": lowering.route.entry_point,
        "source_sha256": lowering.source_sha256,
        "source_bytes": len(lowering.source.encode()),
        "toolchain_requirements": dict(requirements),
    }
    return compiler, assessment, lowering, workload, summary


def _admit_runtime(requirements: Mapping[str, object]) -> tuple[object, object, object]:
    return admit_exact_hip(requirements)


def _admit_released_compiler(compiler: Compiler) -> object:
    if compiler.state != "released":
        raise ValueError("AMD GPU execution requires a released Compiler")
    gate = compiler.check_corpus()
    if not gate.passed:
        raise ValueError("AMD GPU execution requires a passing released Corpus Gate")
    return gate


def _load_generated(lowering: object) -> tuple[object, object]:
    return load_generated_module(lowering)


def _run_gpu_impl(
    project_root: Path,
    compiler: Compiler,
    executor: ExecutorRevision,
    lowering: object,
    workload: WorkloadContract,
    spec: QuickstartSpec,
    summary: dict[str, object],
    attempt: dict[str, str],
    *,
    artifact_dir: Path,
) -> int:
    requirements = _object(lowering.toolchain_requirements, "lowering.toolchain")
    attempt["stage"] = "compiler_admission"
    _admit_released_compiler(compiler)
    attempt["stage"] = "source_custody"
    source = _git_state(project_root)
    if not bool(source["tree_clean"]):
        raise RuntimeError("a clean Git tree is required for retained AMD correctness")
    summary["source_custody"] = source
    attempt["stage"] = "executor_host_admission"
    host_admission = executor.admit_hip_host()
    attempt["stage"] = "runtime_admission"
    torch, triton, properties = _admit_runtime(requirements)
    summary["executor"] = {
        **dict(executor.reference),
        "host_admitted": True,
        "torch_hip_version": host_admission.torch_hip_version,
        "device_monitor": dict(host_admission.device_monitor),
        "profilers": [dict(value) for value in host_admission.profilers],
    }
    attempt["stage"] = "compilation_and_correctness"
    module, generated_directory = _load_generated(lowering)
    try:
        kernel_name = str(requirements["kernel_entry_point"])
        kernel = getattr(module, kernel_name, None)
        if kernel is None:
            raise RuntimeError("generated Triton kernel is missing")
        constants = dict(_object(requirements["compile_constants"], "compile_constants"))
        options = dict(_object(requirements["compile_options"], "compile_options"))
        grid_value = requirements["grid"]
        if not isinstance(grid_value, (list, tuple)) or len(grid_value) != 3:
            raise ValueError("Triton grid differs")
        grid = tuple(int(item) for item in grid_value)
        cases: list[dict[str, object]] = []
        compiled = None
        artifact_payloads: dict[str, bytes] | None = None
        for case_id in workload.case_ids:
            case = workload.case(case_id)
            case_shape = _object(case["shape"], "workload.case.shape")
            shape = tuple(int(case_shape[name]) for name in ("B", "N", "D"))
            inputs = spec.generate(workload, case_id, device="cuda")
            if not isinstance(inputs, tuple) or not inputs:
                raise RuntimeError("Workload generator returned no input tuple")
            first = inputs[0]
            if tuple(first.shape) != shape:
                raise RuntimeError("Workload input shape differs from its case")
            if any(not bool(torch.isfinite(value).all().item()) for value in inputs):
                raise RuntimeError("Workload inputs must be finite before oracle")
            input_snapshots = tuple(value.clone() for value in inputs)
            input_pointers = tuple(int(value.data_ptr()) for value in inputs)
            reference = spec.oracle(workload, *inputs)
            output = torch.empty(shape, dtype=torch.float32, device="cuda")
            observed_compiled = kernel.run(
                *inputs,
                output,
                **constants,
                **options,
                grid=grid,
                warmup=False,
            )
            torch.cuda.synchronize()
            if observed_compiled is None:
                raise RuntimeError("Triton launch returned no compiled kernel")
            observed_payloads = {
                role: _artifact_bytes(observed_compiled.asm[role], role)
                for role in ("source", "ttir", "ttgir", "llir", "amdgcn", "hsaco")
            }
            if not observed_payloads["hsaco"].startswith(b"\x7fELF"):
                raise RuntimeError("Triton did not produce an ELF HSACO")
            if artifact_payloads is None:
                artifact_payloads = observed_payloads
                compiled = observed_compiled
            elif {
                role: sha256(payload).hexdigest()
                for role, payload in observed_payloads.items()
            } != {
                role: sha256(payload).hexdigest()
                for role, payload in artifact_payloads.items()
            }:
                raise RuntimeError("the two correctness cases used different artifacts")
            inputs_unchanged = all(
                int(value.data_ptr()) == pointer and bool(torch.equal(value.view(torch.uint8), snapshot.view(torch.uint8)))
                for value, pointer, snapshot in zip(
                    inputs, input_pointers, input_snapshots, strict=True
                )
            )
            metrics = spec.metrics(workload, output, reference)
            metrics["inputs_unchanged"] = inputs_unchanged
            metrics["passed"] = bool(metrics["passed"]) and inputs_unchanged
            cases.append({"case_id": case_id, **metrics})
        assert compiled is not None and artifact_payloads is not None
    finally:
        generated_directory.cleanup()

    passed = all(bool(case["passed"]) for case in cases)
    artifact_records = {
        role: {"sha256": sha256(payload).hexdigest(), "size_bytes": len(payload)}
        for role, payload in sorted(artifact_payloads.items())
    }
    summary["status"] = "passed" if passed else "correctness_rejected"
    summary["runtime"] = {
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "torch_hip": str(torch.version.hip),
        "triton": importlib.metadata.version("triton"),
        "triton_target": dict(_object(requirements["triton_target"], "triton_target")),
        "device_name": str(properties.name),
        "gcn_arch_name": str(properties.gcnArchName),
        "warp_size": int(properties.warp_size),
        "multiprocessor_count": int(properties.multi_processor_count),
        "total_memory_bytes": int(properties.total_memory),
    }
    summary["build"] = {
        "kernel_name": str(compiled.metadata.name),
        "grid": list(grid),
        "shared_memory_bytes": int(compiled.metadata.shared),
        "artifacts": artifact_records,
    }
    launch_receipt = {
        "schema_version": 1,
        "target": "gfx1151",
        "kernel_name": str(compiled.metadata.name),
        "grid": list(grid),
        "block": [
            int(options["num_warps"])
            * int(_object(requirements["triton_target"], "triton_target")["warp_size"]),
            1,
            1,
        ],
        "dynamic_shared_memory_bytes": int(compiled.metadata.shared),
        "hsaco_sha256": artifact_records["hsaco"]["sha256"],
        "kernel_calls": len(cases),
        "fallback_calls": 0,
        "post_launch_synchronized": True,
    }
    launch_receipt_sha256 = sha256(
        json.dumps(
            launch_receipt,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    summary["evaluation"] = {
        "gpu_submitted": True,
        "correctness_passed": passed,
        "cases": cases,
        "kernel_calls": len(cases),
        "fallback_calls": 0,
        "launch_receipt": launch_receipt,
        "launch_receipt_sha256": launch_receipt_sha256,
        "performance_measured": False,
        "calibration_available": False,
        "scientific_or_speedup_claim": False,
    }

    attempt["stage"] = "artifact_retention"
    (artifact_dir / "generated.py").write_text(lowering.source, encoding="utf-8")
    for role, payload in artifact_payloads.items():
        suffix = "bin" if role == "hsaco" else "txt"
        (artifact_dir / f"kernel.{role}.{suffix}").write_bytes(payload)
    summary["artifact_directory"] = str(artifact_dir)
    _write_new_json(artifact_dir / "result.json", summary)
    return 0 if passed else 2


def _run_gpu(
    project_root: Path,
    compiler: Compiler,
    executor: ExecutorRevision,
    lowering: object,
    workload: WorkloadContract,
    spec: QuickstartSpec,
    summary: dict[str, object],
    *,
    artifact_dir: Path,
) -> int:
    artifact_dir.mkdir(mode=0o700)
    authority = {
        "schema_version": 1,
        "kind": summary["kind"],
        "compiler_revision": summary["compiler_revision"],
        "executor": dict(executor.reference),
        "assessment": summary["assessment"],
        "workload": summary["workload"],
    }
    _write_new_json(artifact_dir / "attempt-authority.json", authority)
    attempt = {"stage": "compiler_admission"}
    try:
        result = _run_gpu_impl(
            project_root,
            compiler,
            executor,
            lowering,
            workload,
            spec,
            summary,
            attempt,
            artifact_dir=artifact_dir,
        )
        _write_manifest(artifact_dir)
        return result
    except BaseException as error:
        failure_class = _failure_class(attempt["stage"])
        failure = {
            "schema_version": 1,
            "kind": "open_cake_amd_quickstart_failure_v1",
            "status": failure_class,
            "failure_class": failure_class,
            "failed_stage": attempt["stage"],
            "authority": authority,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "gpu_result_authorized": False,
            "performance_conclusion_authorized": False,
        }
        if not (artifact_dir / "failure.json").exists():
            _write_new_json(artifact_dir / "failure.json", failure)
        if not (artifact_dir / "manifest.json").exists():
            _write_manifest(artifact_dir)
        raise


def main(*, default_operator: str | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--operator",
        choices=sorted(_SPECS),
        default=default_operator,
        required=default_operator is None,
    )
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--revision", type=Path)
    parser.add_argument("--executor", type=Path)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    spec = _SPECS[arguments.operator]

    project_root = arguments.project_root.resolve(strict=True)
    revision_path = (
        arguments.revision.resolve(strict=True)
        if arguments.revision is not None
        else project_root / (
            "compiler/revision.json" if arguments.prepare_only
            else "compiler/revision.lock.json"
        )
    )
    schedule_path = (
        arguments.schedule.resolve(strict=True)
        if arguments.schedule is not None
        else project_root / spec.schedule
    )
    workload_path = (
        arguments.workload.resolve(strict=True)
        if arguments.workload is not None
        else project_root / spec.workload
    )
    executor: ExecutorRevision | None = None
    if arguments.executor is not None:
        try:
            executor = ExecutorRevision.load(
                project_root, arguments.executor.resolve(strict=True)
            )
            if (
                executor.document["schema_version"] != 2
                or executor.document["host_environment"].get("runtime_kind") != "hip"
            ):
                raise ValueError("Executor is not an exact HIP authority")
        except ValueError as error:
            parser.error(str(error))
    elif not arguments.prepare_only:
        parser.error("--executor is required for GPU execution")
    artifact_dir: Path | None = None
    if arguments.artifact_dir is not None:
        candidate = arguments.artifact_dir.absolute()
        artifact_dir = candidate.parent.resolve(strict=True) / candidate.name
        if artifact_dir.exists() or artifact_dir.is_symlink():
            parser.error("--artifact-dir must be a new path")
        if artifact_dir == project_root or project_root in artifact_dir.parents:
            parser.error("--artifact-dir must be outside the checkout")
    elif not arguments.prepare_only:
        parser.error("--artifact-dir is required for GPU execution")
    output: Path | None = None
    if arguments.output is not None:
        candidate = arguments.output.absolute()
        output = candidate.parent.resolve(strict=True) / candidate.name
        if output.exists() or output.is_symlink():
            parser.error("--output must be a new path")

    compiler, assessment, lowering, workload, summary = _prepare(
        project_root, revision_path, schedule_path, workload_path, spec
    )
    if executor is not None:
        summary["executor"] = {
            **dict(executor.reference),
            "host_admitted": False,
        }
    if arguments.prepare_only:
        exit_code = 0 if summary["status"] == "prepared" else 2
    elif lowering is None:
        exit_code = 2
    else:
        assert executor is not None
        assert artifact_dir is not None
        exit_code = _run_gpu(
            project_root,
            compiler,
            executor,
            lowering,
            workload,
            spec,
            summary,
            artifact_dir=artifact_dir,
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
