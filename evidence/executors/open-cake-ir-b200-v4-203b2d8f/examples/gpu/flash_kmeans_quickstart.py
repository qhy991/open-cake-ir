#!/usr/bin/env python3
"""Prepare or run the educational Open Cake Flash-KMeans B200 smoke path."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Mapping, cast

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler  # noqa: E402
from open_cake_ir.evaluation import (  # noqa: E402
    CudaLaunchManifest,
    CudaTensorContract,
    EvaluationProtocol,
    LaunchableCandidate,
    LaunchObservation,
    WorkloadContract,
    evaluate_flash_kmeans,
    launch_candidate_once,
    observe_exclusive_b200,
)
from open_cake_ir.lab import (  # noqa: E402
    CandidateSubmission,
    ExecutorRevision,
    OpenCakeEnvironment,
    TritonToolchainBuilder,
)


def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return cast(Mapping[str, object], value)


def _summary(project_root: Path, schedule_path: Path, case_id: str) -> dict[str, object]:
    compiler = Compiler.load(project_root, project_root / "compiler/revision.lock.json")
    assessment = compiler.assess_file(schedule_path)
    findings = [
        {
            "code": finding.code,
            "path": finding.path,
            "message": finding.message,
        }
        for finding in assessment.findings
    ]
    summary: dict[str, object] = {
        "schema_version": 1,
        "kind": "open_cake_gpu_quickstart_v1",
        "educational_only": True,
        "status": "rejected",
        "compiler_revision": {
            "revision_id": assessment.compiler_revision_id,
            "canonical_sha256": assessment.compiler_revision_sha256,
        },
        "assessment": {
            "schedule_id": assessment.schedule_id,
            "schedule_sha256": assessment.schedule_sha256,
            "target": assessment.target,
            "profile": assessment.profile,
            "accepted": assessment.accepted,
            "lowering_eligible": assessment.lowering_eligible,
            "findings": findings,
        },
        "evaluation": {"gpu_submitted": False},
    }
    if not assessment.lowering_eligible:
        return summary

    workload = WorkloadContract.load(
        project_root / "contracts/workloads/flash-kmeans-assign-v2.json"
    )
    case = workload.case(case_id)
    schedule = _object(json.loads(assessment.schedule_bytes), "schedule")
    metadata = _object(schedule["metadata"], "schedule.metadata")
    if metadata.get("workload_contract_sha256") != workload.canonical_sha256:
        raise ValueError("quickstart Schedule is not bound to the declared Workload")
    lowering = compiler.lower(assessment)
    summary["status"] = "prepared"
    summary["workload"] = {
        "workload_id": workload.workload_id,
        "canonical_sha256": workload.canonical_sha256,
        "case_id": case_id,
        "shape": case["shape"],
    }
    summary["lowering"] = {
        "entry_point": lowering.entry_point,
        "source_sha256": lowering.source_sha256,
        "source_bytes": len(lowering.source.encode()),
    }
    return summary


class _QuickstartLauncher:
    def __init__(
        self,
        cubin: bytes,
        manifest: CudaLaunchManifest,
        tensor_contract: CudaTensorContract,
    ) -> None:
        self._cubin = cubin
        self._manifest = manifest
        self._tensor_contract = tensor_contract
        self.launch_receipt: dict[str, object] | None = None

    def launch(
        self,
        candidate: LaunchableCandidate,
        tokens: object,
        centroids: object,
        centroid_sq: object,
    ) -> LaunchObservation:
        torch = __import__("torch")
        output = torch.empty(
            (self._tensor_contract.batch, self._tensor_contract.tokens),
            dtype=torch.int32,
            device="cuda",
        )
        receipt = launch_candidate_once(
            candidate,
            self._cubin,
            self._manifest,
            (tokens, centroids, centroid_sq, output),
            tensor_contract=self._tensor_contract,
            stream=torch.cuda.current_stream().cuda_stream,
            synchronize=torch.cuda.synchronize,
        )
        self.launch_receipt = asdict(receipt)
        receipt_bytes = json.dumps(
            self.launch_receipt,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return LaunchObservation(
            output=output,
            kernel_calls=receipt.kernel_calls,
            fallback_calls=receipt.fallback_calls,
            launch_receipt_sha256=sha256(receipt_bytes).hexdigest(),
        )


def _run_gpu(
    project_root: Path,
    schedule_path: Path,
    case_id: str,
    summary: dict[str, object],
    executor_path: Path,
) -> int:
    executor = ExecutorRevision.load(project_root, executor_path)
    executor.admit_host()
    admission = observe_exclusive_b200()
    compiler = Compiler.load(project_root, project_root / "compiler/revision.lock.json")
    workload = WorkloadContract.load(
        project_root / "contracts/workloads/flash-kmeans-assign-v2.json"
    )
    schedule_bytes = schedule_path.read_bytes()
    environment = OpenCakeEnvironment(
        compiler,
        TritonToolchainBuilder(),
        authority_document={"schedule_profile": "flash_kmeans_b32_smoke"},
        workload=workload,
        case_id=case_id,
    )
    result = environment.build(
        CandidateSubmission.seal(
            "application/vnd.open-cake.schedule+json",
            schedule_bytes,
        )
    )
    if result.launchable is None:
        summary["status"] = "build_rejected"
        summary["build"] = {"feedback": dict(result.feedback)}
        return 2

    candidate = result.launchable
    manifest = CudaLaunchManifest.from_dict(
        json.loads(candidate.artifact_payloads["launch_manifest"])
    )
    shape = _object(workload.case(case_id)["shape"], "workload.case.shape")
    tensor_contract = CudaTensorContract(
        int(shape["B"]),
        int(shape["N"]),
        int(shape["K"]),
        int(shape["D"]),
    )
    launcher = _QuickstartLauncher(
        candidate.artifact_payloads["cubin"],
        manifest,
        tensor_contract,
    )
    protocol = EvaluationProtocol(
        protocol_id="open-cake-ir-b200-teaching-smoke-v1",
        purpose="confirmatory",
        workload_sha256=workload.canonical_sha256,
        case_id=case_id,
        timing="none",
    )
    receipt = evaluate_flash_kmeans(
        candidate,
        workload,
        protocol,
        launcher,
        device="cuda",
    )
    launch_receipt = launcher.launch_receipt
    if launch_receipt is None:
        raise RuntimeError("quickstart launch receipt is missing")
    summary["status"] = "passed" if receipt.correctness_passed else "correctness_rejected"
    summary["build"] = {
        "target": candidate.target,
        "cubin_sha256": candidate.artifact_roles["cubin"],
        "cubin_size_bytes": len(candidate.artifact_payloads["cubin"]),
    }
    summary["gpu"] = {
        "name": admission.device_name,
        "compute_capability": list(admission.compute_capability),
        "mode": admission.mode,
        "gpuq_admission_observed": True,
    }
    summary["evaluation"] = {
        "gpu_submitted": True,
        "case_id": receipt.case_id,
        "correctness_passed": receipt.correctness_passed,
        "metrics": dict(receipt.correctness),
        "kernel_calls": receipt.kernel_calls,
        "fallback_calls": receipt.fallback_calls,
        "module_unloaded": launch_receipt["module_unloaded"],
        "evaluation_receipt_sha256": receipt.canonical_sha256,
        "performance_measured": False,
        "scientific_claim_authorized": False,
    }
    return 0 if receipt.correctness_passed else 2


def _current_executor_path(project_root: Path) -> Path:
    inventory = _object(
        json.loads(
            (project_root / "inventory/EXECUTOR_REVISIONS.json").read_text(
                encoding="utf-8"
            )
        ),
        "executor inventory",
    )
    current = _object(inventory["current"], "executor inventory.current")
    path = current.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError("current Executor path differs")
    return (project_root / path).resolve(strict=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--case-id", default="b32_smoke")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--executor", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    project_root = arguments.project_root.resolve(strict=True)
    schedule_path = (
        arguments.schedule.resolve(strict=True)
        if arguments.schedule is not None
        else project_root / "examples/gpu/flash-kmeans-b32-smoke.json"
    )
    summary = _summary(project_root, schedule_path, arguments.case_id)
    if arguments.prepare_only:
        exit_code = 0 if summary["status"] == "prepared" else 2
    else:
        executor_path = (
            arguments.executor.resolve(strict=True)
            if arguments.executor is not None
            else _current_executor_path(project_root)
        )
        exit_code = _run_gpu(
            project_root,
            schedule_path,
            arguments.case_id,
            summary,
            executor_path,
        )
    payload = json.dumps(summary, sort_keys=True, ensure_ascii=False) + "\n"
    if arguments.output is not None:
        output = arguments.output.absolute()
        output.parent.resolve(strict=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        output.chmod(0o644)
    sys.stdout.write(payload)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
