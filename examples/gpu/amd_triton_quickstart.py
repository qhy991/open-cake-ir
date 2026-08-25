#!/usr/bin/env python3
"""Prepare or run one correctness-only Open Cake Triton path on exact gfx1151."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
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


@dataclass(frozen=True)
class QuickstartSpec:
    name: str
    result_kind: str
    schedule: str
    workload: str
    entry_point: str
    generate: Callable[..., tuple[object, ...]]
    oracle: Callable[..., object]
    metrics: Callable[..., dict[str, object]]


_SPECS = {
    "swiglu": QuickstartSpec(
        name="swiglu",
        result_kind="open_cake_gfx1151_swiglu_quickstart_v2",
        schedule="corpus/schedules/swiglu-b8-smoke-gfx1151.json",
        workload="contracts/workloads/swiglu-fp32-v1.json",
        entry_point="cake_swiglu_b8_smoke_gfx1151",
        generate=generate_swiglu_case,
        oracle=swiglu_oracle,
        metrics=swiglu_metrics,
    ),
    "llama_rmsnorm": QuickstartSpec(
        name="llama_rmsnorm",
        result_kind="open_cake_gfx1151_llama_rmsnorm_mul_quickstart_v2",
        schedule="corpus/schedules/llama-rmsnorm-mul-b8-gfx1151-r64-w4.json",
        workload="contracts/workloads/llama-rmsnorm-mul-fp32-v2.json",
        entry_point="cake_llama_rmsnorm_mul_gfx1151_r64_w4",
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


def _prepare(
    project_root: Path,
    revision_path: Path,
    schedule_path: Path,
    workload_path: Path,
    spec: QuickstartSpec,
) -> tuple[Compiler, object, object, WorkloadContract, dict[str, object]]:
    workload = WorkloadContract.load(workload_path)
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
        or assessment.route.entry_point != spec.entry_point
        or assessment.calibration_available
    ):
        raise ValueError("AMD quickstart requires the exact uncalibrated gfx1151 route")
    lowering = compiler.lower(assessment)
    requirements = _object(lowering.toolchain_requirements, "lowering.toolchain")
    if (
        requirements.get("compiler") != "triton"
        or requirements.get("target") != "gfx1151"
        or requirements.get("binary_role") != "hsaco"
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


def _load_generated(lowering: object) -> tuple[object, object]:
    return load_generated_module(lowering)


def _run_gpu(
    project_root: Path,
    lowering: object,
    workload: WorkloadContract,
    spec: QuickstartSpec,
    summary: dict[str, object],
    *,
    artifact_dir: Path | None,
) -> int:
    requirements = _object(lowering.toolchain_requirements, "lowering.toolchain")
    torch, triton, properties = _admit_runtime(requirements)
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
            reference = spec.oracle(workload, *inputs)
            metrics = spec.metrics(workload, output, reference)
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
    summary["source_custody"] = _git_state(project_root)
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

    if artifact_dir is not None:
        artifact_dir.mkdir(mode=0o700)
        (artifact_dir / "generated.py").write_text(lowering.source, encoding="utf-8")
        for role, payload in artifact_payloads.items():
            suffix = "bin" if role == "hsaco" else "txt"
            (artifact_dir / f"kernel.{role}.{suffix}").write_bytes(payload)
        summary["artifact_directory"] = str(artifact_dir)
        result = json.dumps(summary, sort_keys=True, ensure_ascii=False) + "\n"
        (artifact_dir / "result.json").write_text(result, encoding="utf-8")
    return 0 if passed else 2


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
        else project_root / "compiler/revision.lock.json"
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
    artifact_dir: Path | None = None
    if arguments.artifact_dir is not None:
        candidate = arguments.artifact_dir.absolute()
        artifact_dir = candidate.parent.resolve(strict=True) / candidate.name
        if artifact_dir.exists() or artifact_dir.is_symlink():
            parser.error("--artifact-dir must be a new path")
        if artifact_dir == project_root or project_root in artifact_dir.parents:
            parser.error("--artifact-dir must be outside the checkout")
    output: Path | None = None
    if arguments.output is not None:
        candidate = arguments.output.absolute()
        output = candidate.parent.resolve(strict=True) / candidate.name
        if output.exists() or output.is_symlink():
            parser.error("--output must be a new path")

    _, assessment, lowering, workload, summary = _prepare(
        project_root, revision_path, schedule_path, workload_path, spec
    )
    if arguments.prepare_only:
        exit_code = 0 if summary["status"] == "prepared" else 2
    elif lowering is None:
        exit_code = 2
    else:
        exit_code = _run_gpu(
            project_root,
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
