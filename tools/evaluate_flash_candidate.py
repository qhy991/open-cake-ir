#!/usr/bin/env python3
"""Evaluate one sealed CUBIN on an already allocated exclusive B200.

The historical command path retains Flash-KMeans and the explicit Workload tensor ABI.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Mapping, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.evaluation import (  # noqa: E402
    CudaDeviceAdmission,
    CudaLaunchManifest,
    CudaTensorContract,
    LaunchableCandidate,
    LoadedCudaCandidate,
    StrictCuptiBenchmark,
    WorkloadContract,
    assignment_raw_sha256,
    build_ncu_attribution_profile,
    classify_flash_kmeans_output,
    flash_kmeans_oracle,
    generate_flash_kmeans_case,
    NCU_ATTRIBUTION_METRICS,
    observe_exclusive_b200,
    summarize_cohort,
)
from open_cake_ir.lab.executor import ExecutorRevision  # noqa: E402
from open_cake_ir.evaluation.core import (  # noqa: E402
    EvaluationProtocol, LoadedTorchTensorCandidate, TensorLaunchManifest,
    compare_tile_outputs, evaluate_tile_workload, parse_launch_manifest,
)
from open_cake_ir.evaluation.tile_workloads import materialize_case, reference_outputs  # noqa: E402
from open_cake_ir.lab.process import (  # noqa: E402
    SupervisedProcessOutputLimit,
    SupervisedProcessTimeout,
    run_supervised,
    sanitized_environment,
)


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


@dataclass(frozen=True)
class _Authority:
    request: Mapping[str, object]
    request_root: Path
    executor: ExecutorRevision
    workload: WorkloadContract
    manifest: CudaLaunchManifest | TensorLaunchManifest
    candidate: LaunchableCandidate
    payloads: Mapping[str, bytes]
    case_id: str


def _load_authority(request_path: Path) -> _Authority:
    request_root = request_path.parent
    request = _object(json.loads(request_path.read_text(encoding="utf-8")), "request")
    executor_ref = _object(request.get("executor_revision"), "request.executor_revision")
    if set(executor_ref) != {"path", "canonical_sha256", "executor_id"}:
        raise ValueError("worker Executor reference fields differ")
    executor = ExecutorRevision.load(ROOT, ROOT / str(executor_ref["path"]))
    if (
        executor.canonical_sha256 != executor_ref["canonical_sha256"]
        or executor.executor_id != executor_ref["executor_id"]
    ):
        raise ValueError("worker Executor Revision differs")
    artifact_paths = _object(request.get("artifact_paths"), "request.artifact_paths")
    artifact_roles = _object(request.get("artifact_roles"), "request.artifact_roles")
    payloads = {
        role: _input_path(request_root, path, f"artifact.{role}").read_bytes()
        for role, path in artifact_paths.items()
    }
    if set(payloads) != set(artifact_roles) or any(
        sha256(payload).hexdigest() != artifact_roles[role]
        for role, payload in payloads.items()
    ):
        raise ValueError("candidate artifact bytes differ")
    workload = WorkloadContract.load(Path(str(request["workload_path"])).resolve(strict=True))
    if workload.canonical_sha256 != request["workload_sha256"]:
        raise ValueError("worker Workload bytes differ")
    manifest = parse_launch_manifest(json.loads(payloads["launch_manifest"]))
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
    purpose = request.get("purpose")
    if purpose not in {"search", "confirmatory", "attribution"}:
        raise ValueError("worker Evaluation purpose differs")
    case_id = str(request["case_id"])
    workload.case(case_id)
    if isinstance(manifest, TensorLaunchManifest):
        manifest.check_workload(workload, case_id)
    return _Authority(
        request,
        request_root,
        executor,
        workload,
        manifest,
        candidate,
        payloads,
        case_id,
    )


def _evaluate_tile_candidate(authority, result, helper, admission, collect_timing):
    """Use the common oracle and one loaded module across correctness and timing."""
    inputs = materialize_case(authority.workload, authority.case_id)
    loaded = LoadedTorchTensorCandidate(authority.candidate, authority.manifest, inputs, admission)
    counters = result['counters']
    counters['module_loads'] = 1
    # Attribution's child supplies correctness; the parent adds the profiler assay.
    purpose = 'confirmatory' if authority.request['purpose'] == 'attribution' else authority.request['purpose']
    protocol = EvaluationProtocol('workload-tensor-worker-correctness', purpose,
        authority.workload.canonical_sha256, authority.case_id, 'none')
    cohorts = []
    timed_checks = []
    try:
        preflight = evaluate_tile_workload(authority.candidate, authority.workload, protocol, loaded)
        counters['preflight_calls'] = 1
        passed = preflight.correctness_passed
        metrics = dict(preflight.correctness)
        timing = None
        correctness_calls = 1
        if collect_timing and passed:
            strict_cupti = StrictCuptiBenchmark(helper)
            expected = reference_outputs(authority.workload, authority.case_id, inputs)
            for _ in range(5):
                # The retained FlashInfer helper makes six untimed estimation calls,
                # then eleven warmups and twenty-five timed calls. Prepare every
                # output before entering it; the callback contains only the CUBIN.
                arguments = loaded.fresh_argument_sets(6 + 11 + 25)
                used = 0
                def launch_fresh():
                    nonlocal used
                    if used >= len(arguments):
                        raise RuntimeError('CUPTI invocation budget exceeded; output reuse is forbidden')
                    loaded.launch(arguments[used])
                    used += 1
                samples = [float(value) for value in strict_cupti(launch_fresh,
                    dry_run_iters=11, repeat_iters=25, cold_l2_cache=True, use_cuda_graph=False)]
                cohorts.append(samples)
                if len(samples) != 25:
                    raise ValueError('worker CUPTI sample count differs')
                if used != len(arguments):
                    raise RuntimeError('retained CUPTI helper invocation count differs')
                check = {'checked_launches': used, 'passed': True, 'output_mismatches': 0,
                         'max_abs_error': 0.0, 'inputs_unchanged': True}
                for values in arguments:
                    observed, after = loaded.snapshot(values)
                    correct, observation = compare_tile_outputs(authority.workload, inputs, expected, observed, after)
                    check['passed'] = check['passed'] and correct
                    check['output_mismatches'] += observation['output_mismatches']
                    check['max_abs_error'] = max(check['max_abs_error'], observation['max_abs_error'])
                    check['inputs_unchanged'] = check['inputs_unchanged'] and observation['inputs_unchanged']
                timed_checks.append(check)
                passed = passed and check['passed']
                metrics['output_mismatches'] += check['output_mismatches']
                metrics['max_abs_error'] = max(metrics['max_abs_error'], check['max_abs_error'])
                metrics['inputs_unchanged'] = metrics['inputs_unchanged'] and check['inputs_unchanged']
            postflight = evaluate_tile_workload(authority.candidate, authority.workload, protocol, loaded)
            correctness_calls += 1
            passed = passed and postflight.correctness_passed
            metrics['output_mismatches'] += postflight.correctness['output_mismatches']
            metrics['max_abs_error'] = max(metrics['max_abs_error'], postflight.correctness['max_abs_error'])
            metrics['inputs_unchanged'] = metrics['inputs_unchanged'] and postflight.correctness['inputs_unchanged']
            timing = {
                'measurement_quality_passed': all(summarize_cohort(s)['cv'] <= 0.05 for s in cohorts),
                'pooled_median_ms': statistics.median(v for s in cohorts for v in s),
                'cohort_count': 5, 'samples_per_cohort': 25,
            }
        correctness_path = authority.request_root / 'correctness-output.json'
        launch_path = authority.request_root / 'launch-receipt.json'
        _write_new(correctness_path, {'passed': passed, 'metrics': metrics,
            'preflight': dict(preflight.correctness), 'correctness_launches': correctness_calls,
            'timed_output_checks': timed_checks})
        _write_new(launch_path, {'job_id': admission.broker_job_id, 'gpu_uuid': admission.gpu_uuid,
            'candidate_sha256': authority.candidate.candidate_sha256,
            'correctness_launches': correctness_calls, 'fallback_calls': 0,
            'resources': loaded.loaded.resources})
        artifacts = {'correctness_output': correctness_path.name, 'launch_receipt': launch_path.name}
        if collect_timing:
            timing_path = authority.request_root / 'timing-samples.json'
            _write_new(timing_path, {'cohorts_ms': cohorts} if cohorts else {'not_measured': 'correctness_rejected'})
            artifacts['timing_samples'] = timing_path.name
        # The common receipt describes the final correctness launch; counters and
        # the raw launch artifact retain the separate preflight and timing work.
        result['receipt'] = {'correctness_passed': passed, 'correctness': metrics,
            'kernel_calls': 1, 'fallback_calls': 0, 'timing': timing, 'artifacts': artifacts}
    finally:
        counters['kernel_calls'] = loaded.loaded.launch_calls
        counters['timing_samples'] = sum(len(s) for s in cohorts)
        loaded.close()


def _evaluate_candidate(
    authority: _Authority,
    result: dict[str, object],
    *,
    collect_timing: bool,
    admission: CudaDeviceAdmission | None = None,
) -> None:
    helper = authority.executor.admit_host()
    if admission is None:
        try:
            admission = observe_exclusive_b200()
        except ValueError:
            result["error"] = "gpu_admission_differs"
            return
    result["job_id"] = admission.broker_job_id
    result["admitted"] = True
    torch = __import__("torch")
    if (
        torch.cuda.device_count() != 1
        or torch.cuda.get_device_name(0) != admission.device_name
        or torch.cuda.get_device_capability(0) != admission.compute_capability
        or str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
        != admission.gpu_uuid
    ):
        raise ValueError("profile child CUDA device differs from parent admission")
    if isinstance(authority.manifest, TensorLaunchManifest):
        _evaluate_tile_candidate(authority, result, helper, admission, collect_timing)
        return
    case = authority.workload.case(authority.case_id)
    shape = _object(case["shape"], "workload.case.shape")
    contract = CudaTensorContract(
        int(shape["B"]), int(shape["N"]), int(shape["K"]), int(shape["D"])
    )
    tokens, centroids = generate_flash_kmeans_case(
        authority.workload, authority.case_id, device="cuda"
    )
    centroids_fp32 = centroids.to(torch.float32)
    centroid_sq = (centroids_fp32 * centroids_fp32).sum(
        dim=-1, dtype=torch.float32
    ).contiguous()
    output = torch.empty(
        (contract.batch, contract.tokens), dtype=torch.int32, device="cuda"
    )
    loaded = LoadedCudaCandidate.load(
        authority.candidate,
        authority.payloads["cubin"],
        authority.manifest,
        admission,
    )
    counters = cast(dict[str, int], result["counters"])
    counters["module_loads"] = 1
    try:
        arguments = (tokens, centroids, centroid_sq, output)
        loaded.launch(
            arguments,
            tensor_contract=contract,
            stream=torch.cuda.current_stream().cuda_stream,
        )
        torch.cuda.synchronize()
        counters["preflight_calls"] = 1
        oracle = flash_kmeans_oracle(
            authority.workload, tokens, centroids, case_id=authority.case_id
        )
        passed, metrics = classify_flash_kmeans_output(
            authority.workload,
            tokens,
            centroids,
            output,
            oracle,
            case_id=authority.case_id,
        )
        cohorts: list[list[float]] = []
        timing = None
        if collect_timing and passed:
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
            timing = {
                "measurement_quality_passed": quality,
                "pooled_median_ms": statistics.median(
                    value for cohort in cohorts for value in cohort
                ),
                "cohort_count": 5,
                "samples_per_cohort": 25,
            }
        correctness_path = authority.request_root / "correctness-output.json"
        launch_path = authority.request_root / "launch-receipt.json"
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
                "job_id": admission.broker_job_id,
                "gpu_uuid": admission.gpu_uuid,
                "candidate_sha256": authority.candidate.candidate_sha256,
                "correctness_launches": 1,
                "fallback_calls": 0,
            },
        )
        artifacts = {
            "correctness_output": correctness_path.name,
            "launch_receipt": launch_path.name,
        }
        if collect_timing:
            timing_path = authority.request_root / "timing-samples.json"
            _write_new(
                timing_path,
                {"cohorts_ms": cohorts}
                if passed
                else {"not_measured": "correctness_rejected"},
            )
            artifacts["timing_samples"] = timing_path.name
        counters["kernel_calls"] = loaded.launch_calls
        counters["timing_samples"] = 125 if collect_timing and passed else 0
        result["receipt"] = {
            "correctness_passed": passed,
            "correctness": metrics,
            "kernel_calls": 1,
            "fallback_calls": 0,
            "timing": timing,
            "artifacts": artifacts,
        }
    finally:
        loaded.close(synchronize=torch.cuda.synchronize)


def _forward_profile_output(stdout: bytes, stderr: bytes) -> None:
    sys.stdout.buffer.write(stdout)
    sys.stdout.buffer.flush()
    sys.stderr.buffer.write(stderr)
    sys.stderr.buffer.flush()


def _profile_candidate(
    authority: _Authority,
    request_path: Path,
    result: dict[str, object],
) -> None:
    profiler = authority.executor.admit_profiler()
    try:
        admission = observe_exclusive_b200()
    except ValueError:
        result["error"] = "gpu_admission_differs"
        return
    result["job_id"] = admission.broker_job_id
    result["admitted"] = True
    admission_path = authority.request_root / "profile-admission.json"
    _write_new(
        admission_path,
        {
            "schema_version": 1,
            "device_name": admission.device_name,
            "compute_capability": list(admission.compute_capability),
            "gpu_uuid": admission.gpu_uuid,
            "broker_job_id": admission.broker_job_id,
            "mode": admission.mode,
        },
    )
    child_result_path = authority.request_root / "profile-child-result.json"
    command = [
        str(profiler["path"]),
        "--csv",
        "--metrics",
        ",".join(NCU_ATTRIBUTION_METRICS),
        "--target-processes",
        "all",
        "--kernel-name-base",
        "function",
        "--kernel-name",
        authority.candidate.entry_point,
        "--print-kernel-base",
        "function",
        "--launch-count",
        "1",
        "--replay-mode",
        "kernel",
        sys.executable,
        str(Path(__file__).resolve()),
        "--profile-child",
        "--profile-admission",
        str(admission_path),
        "--request",
        str(request_path),
        "--output",
        str(child_result_path),
    ]
    try:
        completed = run_supervised(
            command,
            cwd=ROOT,
            environment=sanitized_environment(),
            timeout_seconds=900,
            maximum_output_bytes=16 * 1024 * 1024,
        )
    except (SupervisedProcessTimeout, SupervisedProcessOutputLimit) as error:
        _forward_profile_output(error.stdout, error.stderr)
        raise
    if completed.returncode != 0:
        _forward_profile_output(completed.stdout, completed.stderr)
        raise RuntimeError(f"NCU exited {completed.returncode}")
    try:
        if not child_result_path.is_file() or child_result_path.is_symlink():
            raise RuntimeError("profile child produced no result")
        child = _object(
            json.loads(child_result_path.read_bytes()), "profile child result"
        )
        if set(child) != set(result) or child.get("schema_version") != 1:
            raise ValueError("profile child result fields differ")
        result.update(child)
        if child.get("admitted") is not True or child.get("error") is not None:
            if child.get("error") is not None:
                _forward_profile_output(completed.stdout, completed.stderr)
            return
        receipt = _object(child.get("receipt"), "profile child receipt")
        artifacts = _object(receipt.get("artifacts"), "profile child artifacts")
        if (
            set(receipt)
            != {
                "correctness_passed",
                "correctness",
                "kernel_calls",
                "fallback_calls",
                "timing",
                "artifacts",
            }
            or receipt.get("timing") is not None
            or set(artifacts) != {"correctness_output", "launch_receipt"}
        ):
            raise ValueError("profile child receipt differs")
        if receipt.get("correctness_passed") is not True:
            raise ValueError("profiled launch did not pass the external oracle")
        profile_path = authority.request_root / "profile.json"
        profile_payload = build_ncu_attribution_profile(
            candidate_sha256=authority.candidate.candidate_sha256,
            case_id=authority.case_id,
            kernel_name=authority.candidate.entry_point,
            ncu_version=str(profiler["version"]),
            ncu_executable_sha256=str(profiler["sha256"]),
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        with profile_path.open("xb") as stream:
            stream.write(profile_payload)
        result_receipt = cast(dict[str, object], result["receipt"])
        result_artifacts = cast(dict[str, str], result_receipt["artifacts"])
        result_artifacts["profile"] = profile_path.name
    except Exception:
        _forward_profile_output(completed.stdout, completed.stderr)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--profile-admission", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    request_path = args.request.resolve(strict=True)
    result = _base_result(os.environ.get("GPUQ_JOB_ID", "gpuq-000000000000"))
    try:
        authority = _load_authority(request_path)
        purpose = str(authority.request["purpose"])
        if args.profile_child:
            if purpose != "attribution" or args.profile_admission is None:
                raise ValueError("profile child requires attribution purpose")
            admission_path = _input_path(
                authority.request_root,
                args.profile_admission.name,
                "profile admission",
            )
            admission_document = _object(
                json.loads(admission_path.read_bytes()), "profile admission"
            )
            if set(admission_document) != {
                "schema_version",
                "device_name",
                "compute_capability",
                "gpu_uuid",
                "broker_job_id",
                "mode",
            } or admission_document.get("schema_version") != 1:
                raise ValueError("profile admission fields differ")
            capability = admission_document["compute_capability"]
            if not isinstance(capability, list) or len(capability) != 2:
                raise ValueError("profile admission capability differs")
            admission = CudaDeviceAdmission(
                str(admission_document["device_name"]),
                (int(capability[0]), int(capability[1])),
                str(admission_document["gpu_uuid"]),
                str(admission_document["broker_job_id"]),
                str(admission_document["mode"]),
            )
            _evaluate_candidate(
                authority,
                result,
                collect_timing=False,
                admission=admission,
            )
        elif purpose == "attribution":
            if args.profile_admission is not None:
                raise ValueError("profile admission is internal-only")
            _profile_candidate(authority, request_path, result)
        else:
            _evaluate_candidate(authority, result, collect_timing=True)
    except Exception as error:
        result["error"] = "evaluator_failed"
        result["failure_class"] = type(error).__name__
        result["receipt"] = None
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
    _write_new(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
