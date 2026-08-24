#!/usr/bin/env python3
"""Evaluate one sealed Flash-KMeans CUBIN on an already allocated exclusive B200."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path, PurePosixPath
from typing import Mapping, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.evaluation import (  # noqa: E402
    CudaLaunchManifest,
    CudaTensorContract,
    LaunchableCandidate,
    LoadedCudaCandidate,
    StrictCuptiBenchmark,
    WorkloadContract,
    assignment_raw_sha256,
    classify_flash_kmeans_output,
    flash_kmeans_oracle,
    generate_flash_kmeans_case,
    observe_exclusive_b200,
    summarize_cohort,
)
from open_cake_ir.lab.executor import ExecutorRevision  # noqa: E402


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return cast(Mapping[str, object], value)


def _input_path(root: Path, value: object, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} path differs")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise ValueError(f"{context} path is unsafe")
    path = (root / value).resolve(strict=True)
    if root not in path.parents or path.is_symlink() or not path.is_file():
        raise ValueError(f"{context} custody differs")
    return path


def _write_new(path: Path, value: object) -> None:
    with path.open("xb") as stream:
        stream.write(_canonical_json_bytes(value))


def _base_result(job_id: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "job_id": job_id,
        "mode": "exclusive",
        "admitted": False,
        "error": None,
        "failure_class": None,
        "counters": {
            "compiler_invocations": 0,
            "module_loads": 0,
            "preflight_calls": 0,
            "kernel_calls": 0,
            "timing_samples": 0,
            "fallback_calls": 0,
        },
        "receipt": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request_path = args.request.resolve(strict=True)
    request_root = request_path.parent
    request = _object(json.loads(request_path.read_text(encoding="utf-8")), "request")
    job_id = os.environ.get("GPUQ_JOB_ID", "gpuq-000000000000")
    result = _base_result(job_id)
    try:
        executor_ref = _object(request.get("executor_revision"), "request.executor_revision")
        if set(executor_ref) != {"path", "canonical_sha256", "executor_id"}:
            raise ValueError("worker Executor reference fields differ")
        executor = ExecutorRevision.load(ROOT, ROOT / str(executor_ref["path"]))
        if (
            executor.canonical_sha256 != executor_ref["canonical_sha256"]
            or executor.executor_id != executor_ref["executor_id"]
        ):
            raise ValueError("worker Executor Revision differs")
        helper = executor.admit_host()
        try:
            admission = observe_exclusive_b200()
        except ValueError:
            result["error"] = "gpu_admission_differs"
            _write_new(args.output, result)
            return 0
        job_id = admission.broker_job_id
        result["job_id"] = job_id
        result["admitted"] = True
        torch = __import__("torch")
        artifact_paths = _object(request.get("artifact_paths"), "request.artifact_paths")
        artifact_roles = _object(request.get("artifact_roles"), "request.artifact_roles")
        payloads = {
            role: _input_path(request_root, path, f"artifact.{role}").read_bytes()
            for role, path in artifact_paths.items()
        }
        if set(payloads) != set(artifact_roles) or any(
            __import__("hashlib").sha256(payload).hexdigest() != artifact_roles[role]
            for role, payload in payloads.items()
        ):
            raise ValueError("candidate artifact bytes differ")
        workload_path = Path(str(request["workload_path"])).resolve(strict=True)
        workload = WorkloadContract.load(workload_path)
        if workload.canonical_sha256 != request["workload_sha256"]:
            raise ValueError("worker Workload bytes differ")
        manifest = CudaLaunchManifest.from_dict(
            json.loads(payloads["launch_manifest"])
        )
        candidate = LaunchableCandidate(
            candidate_sha256=str(request["candidate_sha256"]),
            target=str(request["target"]),
            entry_point=str(request["entry_point"]),
            artifact_roles=dict(artifact_roles),
            launch_spec_sha256=str(request["launch_spec_sha256"]),
            artifact_payloads=payloads,
        )
        if candidate.canonical_sha256 != request["candidate_record_sha256"]:
            raise ValueError("worker Candidate record differs")
        purpose = str(request["purpose"])
        if purpose == "attribution":
            # An attribution assay carries profiler evidence and no timing, and this
            # evaluator produces the opposite. Without this the receipt is built, the
            # contract refuses its artifact roles, and the run faults on a shape error
            # far from the thing that is actually missing.
            raise ValueError(
                "this evaluator produces timing, not profiler evidence; a Study that "
                "declares attribution_evaluation needs an evaluator that does"
            )
        case_id = str(request["case_id"])
        case = workload.case(case_id)
        shape = _object(case["shape"], "workload.case.shape")
        contract = CudaTensorContract(
            int(shape["B"]), int(shape["N"]), int(shape["K"]), int(shape["D"])
        )
        tokens, centroids = generate_flash_kmeans_case(workload, case_id, device="cuda")
        centroids_fp32 = centroids.to(torch.float32)
        centroid_sq = (centroids_fp32 * centroids_fp32).sum(
            dim=-1, dtype=torch.float32
        ).contiguous()
        output = torch.empty(
            (contract.batch, contract.tokens), dtype=torch.int32, device="cuda"
        )
        loaded = LoadedCudaCandidate.load(
            candidate,
            payloads["cubin"],
            manifest,
            admission,
        )
        cast(dict[str, int], result["counters"])["module_loads"] = 1
        try:
            arguments = (tokens, centroids, centroid_sq, output)
            loaded.launch(arguments, tensor_contract=contract, stream=torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            cast(dict[str, int], result["counters"])["preflight_calls"] = 1
            oracle = flash_kmeans_oracle(workload, tokens, centroids, case_id=case_id)
            passed, metrics = classify_flash_kmeans_output(
                workload, tokens, centroids, output, oracle, case_id=case_id
            )
            cohorts: list[list[float]] = []
            timing = None
            if passed:
                strict_cupti = StrictCuptiBenchmark(helper)

                def launch() -> None:
                    loaded.launch(
                        arguments,
                        tensor_contract=contract,
                        stream=torch.cuda.current_stream().cuda_stream,
                    )

                for _ in range(5):
                    samples = [
                        float(value)
                        for value in strict_cupti(
                            launch,
                            dry_run_iters=11,
                            repeat_iters=25,
                            cold_l2_cache=True,
                            use_cuda_graph=False,
                        )
                    ]
                    if len(samples) != 25:
                        raise ValueError("worker CUPTI sample count differs")
                    cohorts.append(samples)
                quality = all(
                    summarize_cohort(samples)["cv"] <= 0.05 for samples in cohorts
                )
                pooled = statistics.median(value for cohort in cohorts for value in cohort)
                timing = {
                    "measurement_quality_passed": quality,
                    "pooled_median_ms": pooled,
                    "cohort_count": 5,
                    "samples_per_cohort": 25,
                }
            correctness_path = request_root / "correctness-output.json"
            launch_path = request_root / "launch-receipt.json"
            timing_path = request_root / "timing-samples.json"
            output_sha, output_size = assignment_raw_sha256(output)
            _write_new(
                correctness_path,
                {
                    "metrics": metrics,
                    "output_sha256": output_sha,
                    "output_size_bytes": output_size,
                },
            )
            _write_new(
                launch_path,
                {
                    "job_id": job_id,
                    "gpu_uuid": admission.gpu_uuid,
                    "candidate_sha256": candidate.candidate_sha256,
                    "correctness_launches": 1,
                    "fallback_calls": 0,
                },
            )
            _write_new(
                timing_path,
                {"cohorts_ms": cohorts}
                if passed
                else {"not_measured": "correctness_rejected"},
            )
            cast(dict[str, int], result["counters"])["kernel_calls"] = loaded.launch_calls
            cast(dict[str, int], result["counters"])["timing_samples"] = (
                125 if passed else 0
            )
            result["receipt"] = {
                "correctness_passed": passed,
                "correctness": metrics,
                "kernel_calls": 1,
                "fallback_calls": 0,
                "timing": timing,
                "artifacts": {
                    "correctness_output": correctness_path.name,
                    "launch_receipt": launch_path.name,
                    "timing_samples": timing_path.name,
                },
            }
        finally:
            loaded.close(synchronize=torch.cuda.synchronize)
    except Exception as error:
        result["error"] = "evaluator_failed"
        result["failure_class"] = type(error).__name__
        result["receipt"] = None
    _write_new(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
