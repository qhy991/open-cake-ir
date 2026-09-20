#!/usr/bin/env python3
"""Evaluate one sealed artifact on its admitted exact backend.

The existing worker retains CUDA assays and observes Metal binary archives through
the same common Evaluation receipt and broker boundary.
"""

from __future__ import annotations

from open_cake_ir.serialization import canonical_json_bytes as _canonical_json_bytes

import argparse
import json
import os
import statistics
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, cast

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from open_cake_ir.tasks.workloads import load_workload

from open_cake_ir.evaluation import CudaDeviceAdmission, LaunchableCandidate, LoadedCudaCandidate, WorkloadContract, build_ncu_attribution_profile, NCU_ATTRIBUTION_METRICS, summarize_cohort
from open_cake_ir.tasks.flash_kmeans.cuda_manifest import CudaLaunchManifest
from open_cake_ir.tasks.flash_kmeans.cuda import CudaTensorContract
from open_cake_ir.evaluation.benchmark import StrictCuptiBenchmark
from open_cake_ir.tasks.flash_kmeans.workload import assignment_raw_sha256, classify_flash_kmeans_output, flash_kmeans_oracle, generate_flash_kmeans_case
from open_cake_ir.lab.executor import ExecutorRevision
from open_cake_ir.compiler.target import CodeObject
from open_cake_ir.evaluation.admission import observe_exclusive_cuda, observe_local_cuda
from open_cake_ir.evaluation.attempts import job_mode
from open_cake_ir.evaluation.platforms import ExecutionPlatform, PLATFORMS, platform_for, platform_for_paired_kind
from open_cake_ir.evaluation.core import EvaluationProtocol, LoadedTorchTensorCandidate, TensorLaunchManifest, compare_tile_outputs, _same_tensor_inputs
from open_cake_ir.tasks.tiles.evaluation import evaluate_tile_workload, evaluate_tile_validation_case
from open_cake_ir.tasks.launch import parse_launch_manifest
from open_cake_ir.evaluation.metal_manifest import MetalTensorLaunchManifest
from open_cake_ir.tasks.workloads import materialize_case, reference_outputs
from open_cake_ir.lab.process import SupervisedProcessOutputLimit, SupervisedProcessTimeout, sanitized_environment
from open_cake_ir.evaluation.paired import (
    METAL_KINDS, paired_protocol, paired_summary, candidate_identity, validation_case_ids,
    candidate_from_identity, validate_pair_candidates,
)
from open_cake_ir.tasks.devices import allocation_mode




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


_PROFILE_OUTPUT_OWNER: tuple[int, int] | None = None
_BROKER_ALLOCATION: dict | None = None


def _write_new(path: Path, value: object) -> None:
    from open_cake_ir.lab.ncu_process import write_new
    # Attach allocation provenance to device observations, not to the closed
    # worker result envelope. Legacy local-broker evidence remains unchanged.
    if (_BROKER_ALLOCATION is not None and isinstance(value, dict)
            and value.get("job_id") == _BROKER_ALLOCATION["job_id"] and "counters" not in value):
        value = {**value, "broker_allocation": _BROKER_ALLOCATION}
        if "allocation_mode" in value:
            value["allocation_mode"] = "exclusive"
    write_new(path, _canonical_json_bytes(value), _PROFILE_OUTPUT_OWNER)


def _base_result(job_id: str) -> dict[str, object]:
    # The prefix names the allocator that issued the job, and each allocator has one
    # mode; the placeholder the worker starts with is the cluster allocator's.
    return {
        "schema_version": 1,
        "job_id": job_id,
        "mode": job_mode(job_id),
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
    manifest: CudaLaunchManifest | TensorLaunchManifest | MetalTensorLaunchManifest
    candidate: LaunchableCandidate
    payloads: Mapping[str, bytes]
    case_id: str
    baseline: LaunchableCandidate | None = None
    # What the Study says about whether a latency can be reported for this run.
    timed_assay_available: bool = True
    # How the Study's device is reached, as the device registry row the Study was admitted
    # against declares it: the cluster allocator's exclusive lease or the local broker.
    # The worker admits under that allocator and no other (D6); an authority that states
    # none is refused at admission rather than admitted under a default.
    allocation_mode: str | None = None


def _load_authority(request_path: Path) -> _Authority:
    request_root = request_path.parent
    request = _object(json.loads(request_path.read_text(encoding="utf-8")), "request")
    executor = ExecutorRevision.load_reference(ROOT, request.get("executor_revision"), "request.executor_revision")
    from open_cake_ir.lab.bindings import load_compiler_reference
    dependency = load_compiler_reference(ROOT, request.get("compiler_revision"), "request.compiler_revision")
    if request.get("target") not in dependency.targets:
        raise ValueError("worker target is not bound by the admitted Compiler")
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
    workload = load_workload(Path(str(request["workload_path"])).resolve(strict=True))
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
    if isinstance(manifest, (TensorLaunchManifest, MetalTensorLaunchManifest)):
        manifest.check_workload(workload, case_id)
    baseline = None
    timed_assay_available = True
    evaluation = request.get('evaluation_protocol')
    if evaluation is not None:
        evaluation = _object(evaluation, 'request.evaluation_protocol')
        if sha256(_canonical_json_bytes(evaluation)).hexdigest() != request['evaluation_protocol_sha256']:
            raise ValueError('worker evaluation policy identity differs')
        # Whether this run is timed is the Study's statement, not this worker's guess and
        # not a property of the target read here. A Study for a target with no named
        # timer carries a measurement-coverage limitation instead of a paired assay.
        coverage = evaluation.get('measurement_coverage')
        if isinstance(coverage, Mapping) and coverage.get('timed_assay') == 'unavailable':
            timed_assay_available = False
        if paired_protocol(evaluation) is not None:
            # The assay the Study declares belongs to one platform row, and so does the
            # candidate's target; a pair timed under another row's assay is refused here.
            if platform_for_paired_kind(evaluation['paired_timing']['kind']) is not platform_for(candidate.target):
                raise ValueError('paired assay backend differs from sealed manifest')
            partner = _object(request.get('baseline'), 'request.baseline')
            paths = _object(partner.get('artifact_paths'), 'request.baseline.artifact_paths')
            baseline = candidate_from_identity({k: v for k, v in partner.items() if k != 'artifact_paths'},
                {role: _input_path(request_root, path, f'baseline.{role}').read_bytes()
                 for role, path in paths.items()})
            manifests = validate_pair_candidates(candidate, baseline, workload, case_id)
            if not isinstance(manifest, MetalTensorLaunchManifest) and 'validation_case_ids' in evaluation:
                cases = validation_case_ids(evaluation)
                if (cases != workload.case_ids or workload.document['validation'].get('all_cases_required') is not True
                        or case_id != workload.document['validation'].get('primary_case')):
                    raise ValueError('CUDA evaluation case projection differs from Workload validation')
                for validation_case in cases:
                    for bound_manifest in manifests.values():
                        bound_manifest.check_validation_case(workload, validation_case)
        elif 'baseline' in request:
            raise ValueError('worker baseline has no paired policy')
    elif 'baseline' in request:
        raise ValueError('worker baseline has no evaluation authority')
    return _Authority(
        request,
        request_root,
        executor,
        workload,
        manifest,
        candidate,
        payloads,
        case_id,
        baseline,
        timed_assay_available=timed_assay_available,
        allocation_mode=allocation_mode(candidate.target),
    )


def _execution_platform(authority: _Authority) -> CodeObject:
    """The declared object that selects how this candidate is executed.

    A Target declares what its toolchain produces, so the execution path follows that
    declaration rather than the type of the launch manifest. No platform is reached by
    falling through another's branch: before this, every candidate that was not Metal's
    reached CUDA's device admission, including one built for a target CUDA never names.

    The manifest and the declaration must agree. Either alone would be a second owner of
    the same fact, and a mismatched pair is a sealed candidate nobody can launch.
    """
    row = platform_for(authority.candidate.target)
    metal_manifest = isinstance(authority.manifest, MetalTensorLaunchManifest)
    if metal_manifest != (row.code_object is CodeObject.METAL_BINARY_ARCHIVE):
        raise ValueError(
            "launch manifest and declared execution platform differ: "
            f"{authority.candidate.target!r} declares {row.code_object.value!r}"
        )
    return row.code_object


def _route_calls_per_cohort(authority: _Authority) -> int:
    """The cohort's route-call count the candidate's platform row declares."""
    row = platform_for(authority.candidate.target)
    if row.route_calls_per_cohort is None:
        raise ValueError(
            f"{row.code_object.value!r} declares no route-call count for a tensor-tile "
            "cohort; its cohort is shaped by its own observer")
    return row.route_calls_per_cohort


def _observe_cuda(authority: _Authority):
    """Admit the CUDA device through the allocator the Study's row declares (D6).

    A local-broker admission supports a correctness check; the paired CUPTI receipt still
    requires the cluster lease and `paired.admit_device_identity` refuses a local job.
    """
    observers = {"exclusive": observe_exclusive_cuda, "local_serialized": observe_local_cuda}
    observe = observers.get(authority.allocation_mode)
    if observe is None:
        raise ValueError(
            f"allocation mode {authority.allocation_mode!r} names no CUDA device admission")
    return observe(authority.candidate.target)


def _fresh_tile_cohort(loaded, strict_cupti, workload, inputs, expected, *,
                       samples_per_cohort, route_calls_per_cohort):
    """The single retained CUPTI path for both historical and paired tensor assays."""
    validation_inputs = getattr(loaded, 'validation_inputs', inputs)
    if validation_inputs is not inputs and not _same_tensor_inputs(inputs, validation_inputs):
        raise ValueError('retained validation inputs differ from the Workload case')
    arguments = loaded.fresh_argument_sets(route_calls_per_cohort)
    used = 0
    def launch_fresh():
        nonlocal used
        if used >= len(arguments):
            raise RuntimeError('CUPTI invocation budget exceeded; output reuse is forbidden')
        loaded.launch(arguments[used])
        used += 1
    samples = [float(value) for value in strict_cupti(launch_fresh,
        dry_run_iters=11, repeat_iters=samples_per_cohort, cold_l2_cache=True, use_cuda_graph=False)]
    if len(samples) != samples_per_cohort:
        raise ValueError('worker CUPTI sample count differs')
    if used != len(arguments):
        raise RuntimeError('retained CUPTI helper invocation count differs')
    check = {'checked_launches': used, 'passed': True, 'output_mismatches': 0,
             'max_abs_error': 0.0, 'inputs_unchanged': True}
    # A loaded tensor candidate retained a value-identical native CPU array at
    # admission. Generic callables keep their original input representation.
    for values in arguments:
        observed, after = loaded.snapshot(values)
        correct, observation = compare_tile_outputs(workload, validation_inputs, expected, observed, after)
        check['passed'] = check['passed'] and correct
        check['output_mismatches'] += observation['output_mismatches']
        check['max_abs_error'] = max(check['max_abs_error'], observation['max_abs_error'])
        check['inputs_unchanged'] = check['inputs_unchanged'] and observation['inputs_unchanged']
    return samples, check


def _evaluate_paired_tile(authority, result, benchmark_for, admission):
    """Execute both sealed participants in one allocation, in the frozen order.

    `benchmark_for(role, manifest)` returns the assay that times one arm, because which
    assay that is belongs to the backend and not to this function -- the same move
    `_evaluate_tile_candidate` already made. It is a factory rather than one instance
    because an arm's assay may be bound to the kernel it times: CUPTI is not, and returns
    the same object for both, while the HIP assay names the dispatch it attributes and so
    is one per role. Handing a single instance to both arms would have attributed the
    baseline's dispatches to the candidate's kernel name, and the assay would have refused
    -- correctly, and one layer too late to say why.
    """
    evaluation = authority.request['evaluation_protocol']
    protocol = paired_protocol(evaluation)
    all_cases = 'validation_case_ids' in evaluation
    cases = validation_case_ids(evaluation) if all_cases else (authority.case_id,)
    if all_cases and (cases != authority.workload.case_ids
            or authority.workload.document['validation'].get('all_cases_required') is not True
            or authority.case_id != authority.workload.document['validation'].get('primary_case')):
        raise ValueError('CUDA evaluation case projection differs from Workload validation')
    candidates = {'candidate': authority.candidate, 'baseline': authority.baseline}
    manifests = validate_pair_candidates(authority.candidate, authority.baseline,
                                        authority.workload, authority.case_id)
    for case_id in cases:
        if all_cases:
            for manifest in manifests.values():
                manifest.check_validation_case(authority.workload, case_id)
    input_cases = {case_id: materialize_case(authority.workload, case_id) for case_id in cases}
    inputs = input_cases[authority.case_id]
    expected = reference_outputs(authority.workload, authority.case_id, inputs)
    loaded = {}
    checks = {role: {'preflight': None, 'postflight': None, 'timed_output_checks': []}
              for role in protocol.arms}
    measurements = []
    counters = result['counters']
    passed = True
    metrics = {'output_mismatches': 0, 'max_abs_error': 0.0, 'inputs_unchanged': True}
    correctness_calls = 0
    def accumulate(check):
        nonlocal passed
        passed = passed and check['passed']
        values = check.get('metrics', check)
        metrics['output_mismatches'] += values['output_mismatches']
        metrics['max_abs_error'] = max(metrics['max_abs_error'], values['max_abs_error'])
        metrics['inputs_unchanged'] = metrics['inputs_unchanged'] and values['inputs_unchanged']
    def correctness(role, phase):
        nonlocal correctness_calls
        launches = []
        combined = {'output_mismatches': 0, 'max_abs_error': 0.0, 'inputs_unchanged': True}
        for case_id in cases:
            correctness_protocol = EvaluationProtocol('workload-tensor-worker-correctness',
                authority.request['purpose'], authority.workload.canonical_sha256, case_id, 'none')
            evaluate = evaluate_tile_validation_case if all_cases else evaluate_tile_workload
            receipt = evaluate(candidates[role], authority.workload,
                               correctness_protocol, loaded[(role, case_id)])
            values = dict(receipt.correctness)
            launches.append({'input_case_id': case_id, 'passed': receipt.correctness_passed, 'metrics': values})
            combined['output_mismatches'] += values['output_mismatches']
            combined['max_abs_error'] = max(combined['max_abs_error'], values['max_abs_error'])
            combined['inputs_unchanged'] &= values['inputs_unchanged']
            correctness_calls += 1
        check = {'passed': all(row['passed'] for row in launches), 'metrics': combined}
        if all_cases:
            check['launches'] = launches
        checks[role][phase] = check
        accumulate(check)
    try:
        for role in protocol.arms:
            for case_id in cases:
                loaded[(role, case_id)] = LoadedTorchTensorCandidate(candidates[role], manifests[role], input_cases[case_id], admission)
                counters['module_loads'] += 1
            correctness(role, 'preflight')
            counters['preflight_calls'] += len(cases)
        if passed:
            assays = {role: benchmark_for(role, manifests[role]) for role in protocol.arms}
            for index, order in enumerate(protocol.pair_order):
                row = {'pair_index': index, 'order': list(order), 'arms': {}}
                for position, role in enumerate(order):
                    samples, check = _fresh_tile_cohort(loaded[(role, authority.case_id)], assays[role],
                        authority.workload, inputs, expected,
                        samples_per_cohort=protocol.samples_per_cohort,
                        route_calls_per_cohort=protocol.route_calls_per_cohort)
                    counters['timing_samples'] += len(samples)
                    checks[role]['timed_output_checks'].append(check)
                    accumulate(check)
                    row['arms'][role] = {'position': position,
                        'candidate_record_sha256': candidates[role].canonical_sha256,
                        'samples_ms': samples, 'summary': summarize_cohort(samples),
                        'route_calls': check['checked_launches'], 'output_check': check}
                    # This path produces every paired comparison's evidence, and it was
                    # dropping what the assay saw beside each arm's samples. An arm that
                    # timed part of its candidate is only visible here.
                    seen = getattr(assays[role], 'non_target_dispatches', None)
                    if seen is not None:
                        row['arms'][role]['non_target_dispatches'] = seen
                measurements.append(row)
            for role in protocol.arms:
                correctness(role, 'postflight')
        identities = {role: candidate_identity(item) for role, item in candidates.items()}
        # The assay the Study declared, not the one this producer was written against.
        # It read PAIRED_KIND, which was true while CUPTI was the only source a tensor
        # pair could be timed by; the Metal producer below already reads the declaration,
        # and the receipt validator compares the two, so a third source turned a
        # hardcoded name into 'paired raw kind differs from the declared assay'.
        raw = {'kind': evaluation['paired_timing']['kind'],
            'evaluation_protocol': authority.request['evaluation_protocol'],
            'participants': identities, 'workload_sha256': authority.workload.canonical_sha256,
            'case_id': authority.case_id, 'purpose': authority.request['purpose'],
            'job_id': admission.broker_job_id, 'gpu_uuid': admission.gpu_uuid,
            'measurements': measurements}
        if not measurements:
            raw['not_measured'] = 'correctness_rejected'
        timing = paired_summary(raw) if measurements else None
        _write_new(authority.request_root / 'timing-samples.json', raw)
        _write_new(authority.request_root / 'correctness-output.json', {
            'passed': passed, 'metrics': metrics, 'participants': checks})
        _write_new(authority.request_root / 'launch-receipt.json', {
            'job_id': admission.broker_job_id, 'gpu_uuid': admission.gpu_uuid,
            'candidate_sha256': authority.candidate.candidate_sha256, 'participants': identities,
            'correctness_launches': correctness_calls, 'fallback_calls': 0,
            'resources': {role: loaded[(role, authority.case_id)].loaded.resources for role in protocol.arms}})
        result['receipt'] = {'correctness_passed': passed, 'correctness': metrics,
            'kernel_calls': 1, 'fallback_calls': 0, 'timing': timing,
            'artifacts': {'correctness_output': 'correctness-output.json',
                          'launch_receipt': 'launch-receipt.json', 'timing_samples': 'timing-samples.json'}}
    finally:
        counters['kernel_calls'] = sum(item.loaded.launch_calls for item in loaded.values())
        pending_error = sys.exc_info()[1]
        cleanup_error = None
        for item in loaded.values():
            try:
                item.close()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
        if cleanup_error is not None:
            if pending_error is not None:
                raise pending_error from cleanup_error
            raise cleanup_error


def _evaluate_untimed_validation_cases(authority, result, admission):
    """Keep the full Workload distribution contract when a platform has no timer."""
    from dataclasses import asdict
    cases = validation_case_ids(authority.request["evaluation_protocol"])
    if (cases != authority.workload.case_ids
            or authority.workload.document["validation"].get("all_cases_required") is not True):
        raise ValueError("untimed validation cases differ from the Workload")
    rows = []
    metrics = {"output_mismatches": 0, "max_abs_error": 0.0, "inputs_unchanged": True}
    counters = result["counters"]
    for case_id in cases:
        authority.manifest.check_validation_case(authority.workload, case_id)
        inputs = materialize_case(authority.workload, case_id)
        loaded = LoadedTorchTensorCandidate(authority.candidate, authority.manifest, inputs, admission)
        counters["module_loads"] += 1
        try:
            protocol = EvaluationProtocol("workload-tensor-worker-correctness", authority.request["purpose"],
                authority.workload.canonical_sha256, case_id, "none")
            receipt = evaluate_tile_validation_case(authority.candidate, authority.workload, protocol, loaded)
            counters["preflight_calls"] += 1
            values = dict(receipt.correctness)
            metrics["output_mismatches"] += values["output_mismatches"]
            metrics["max_abs_error"] = max(metrics["max_abs_error"], values["max_abs_error"])
            metrics["inputs_unchanged"] &= values["inputs_unchanged"]
            rows.append({"input_case_id": case_id, "passed": receipt.correctness_passed,
                         "metrics": values, "resources": loaded.loaded.resources})
        finally:
            counters["kernel_calls"] += loaded.loaded.launch_calls
            loaded.close()
    passed = all(row["passed"] for row in rows)
    _write_new(authority.request_root / "correctness-output.json", {
        "passed": passed, "metrics": metrics, "validation_cases": rows,
        "correctness_launches": len(rows),
    })
    _write_new(authority.request_root / "launch-receipt.json", {
        "job_id": admission.broker_job_id, "gpu_uuid": admission.gpu_uuid,
        "device_admission": asdict(admission),
        "candidate_sha256": authority.candidate.candidate_sha256,
        "correctness_launches": len(rows), "fallback_calls": 0,
        "allocation_mode": job_mode(admission.broker_job_id), "external_gpu_activity": "not_excluded",
    })
    # The common receipt retains a timing artifact even when it contains JSON null.
    # Absence of measurement is explicit; omitting the role breaks broker admission.
    _write_new(authority.request_root / "timing-samples.json", None)
    result["receipt"] = {"correctness_passed": passed, "correctness": metrics,
        "kernel_calls": 1, "fallback_calls": 0, "timing": None,
        "artifacts": {"correctness_output": "correctness-output.json",
                      "launch_receipt": "launch-receipt.json", "timing_samples": "timing-samples.json"}}


def _evaluate_tile_candidate(authority, result, benchmark, admission, collect_timing,
                             *, route_calls_per_cohort, profile_source=None):
    """Use the common oracle and one loaded module across correctness and timing.

    `benchmark` is the timing source itself, not the host it came from: a callable taking
    the function to run, the untimed and timed counts, and whether the device is reset,
    and returning one millisecond value per timed call. Each platform builds its own --
    `StrictCuptiBenchmark` over the Executor's CUPTI helper, `HipDispatchBenchmark` over
    roctracer -- because the cohort shape is shared and the source is not. Wrapping the
    argument in CUPTI's strict adapter here made that the only source this path could use.

    `route_calls_per_cohort` belongs to the source for the same reason: CUPTI spends six
    calls on its own calibration callbacks and the HIP benchmark spends none, so the
    Study's count and this one have to be the same number, and it comes from the caller
    that knows which source is running.

    `profile_source` is the attribution source, passed for the same reason and never
    inferred: a platform with one supplies it, a platform without one passes None and its
    receipt carries no profile rather than an empty one. It takes the single-dispatch
    launch and the kernel name, and returns the raw activity its own profiler saw.
    """
    if collect_timing and benchmark is None:
        raise ValueError("a timed tile evaluation requires its timing source")
    if (not collect_timing and profile_source is None and authority.request["purpose"] != "attribution"
            and "validation_case_ids" in authority.request["evaluation_protocol"]):
        return _evaluate_untimed_validation_cases(authority, result, admission)
    inputs = materialize_case(authority.workload, authority.case_id)
    loaded = LoadedTorchTensorCandidate(authority.candidate, authority.manifest, inputs, admission)
    counters = result['counters']
    counters['module_loads'] = 1
    # Attribution's child supplies correctness; the parent adds the profiler assay.
    purpose = 'confirmatory' if authority.request['purpose'] == 'attribution' else authority.request['purpose']
    protocol = EvaluationProtocol('workload-tensor-worker-correctness', purpose,
        authority.workload.canonical_sha256, authority.case_id, 'none')
    cohorts = []
    non_target = []
    timed_checks = []
    try:
        preflight = evaluate_tile_workload(authority.candidate, authority.workload, protocol, loaded)
        counters['preflight_calls'] = 1
        passed = preflight.correctness_passed
        metrics = dict(preflight.correctness)
        timing = None
        correctness_calls = 1
        if collect_timing and passed:
            expected = reference_outputs(authority.workload, authority.case_id, inputs)
            for _ in range(5):
                samples, check = _fresh_tile_cohort(loaded, benchmark, authority.workload,
                    inputs, expected, samples_per_cohort=25,
                    route_calls_per_cohort=route_calls_per_cohort)
                cohorts.append(samples)
                # Per cohort, because the assay overwrites this on every call. Reading it
                # once after the loop reported the fifth cohort and dropped four, beside a
                # `cohort_count: 5` in the same record -- a number that reads as "this run
                # saw none" when four fifths of the run was not looked at.
                seen = getattr(benchmark, 'non_target_dispatches', None)
                if seen is not None:
                    non_target.append(seen)
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
            # The bound is the Study's, not this function's. It was the literal 0.05
            # here while the Study declared its own, so a Study that widened or tightened
            # the bound was judged against a number it never named -- one fact with two
            # owners, and the literal winning. The assay owns the threshold; this reports
            # against it and says which one it used.
            assay = paired_protocol(authority.request.get('evaluation_protocol') or {})
            maximum_cv = assay.maximum_cv if assay is not None else 0.05
            timing = {
                'measurement_quality_passed': all(
                    summarize_cohort(s)['cv'] <= maximum_cv for s in cohorts),
                'maximum_cv': maximum_cv,
                'observed_maximum_cv': max(summarize_cohort(s)['cv'] for s in cohorts),
                'pooled_median_ms': statistics.median(v for s in cohorts for v in s),
                'cohort_count': 5, 'samples_per_cohort': 25,
            }
            # An assay that can tell a dispatch it did not name from one it did says so
            # here. A count it keeps to itself is not a report: this is the field a reader
            # checks to know a cohort timed one kernel and not part of one. CUPTI's assay
            # does not distinguish them and declares nothing.
            if non_target:
                timing['non_target_dispatches_per_cohort'] = list(non_target)
                timing['non_target_dispatches'] = sum(non_target)
        if profile_source is not None and not passed:
            # Raised before anything is written: `_write_new` opens "xb", so a retry into
            # the same request root would surface FileExistsError instead of this. The
            # Metal sibling refuses at the same point for the same reason.
            raise ValueError(
                "instrumented dispatch requires the candidate to pass the external oracle")
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
        if profile_source is not None:
            # One separate instrumented dispatch, after correctness and outside every
            # cohort. It is attribution, not a sample: no device-state reset precedes it
            # and the record says so, so nobody compares it to a cohort median.
            from open_cake_ir.evaluation.hip_observations import (
                HIP_PROFILE_KIND, hip_profile_summary)
            instrumented = loaded.fresh_argument_sets(1)[0]
            raw = profile_source(lambda: loaded.launch(instrumented),
                                 authority.manifest.kernel_name)
            profile_path = authority.request_root / 'profile.json'
            _write_new(profile_path, {
                'kind': HIP_PROFILE_KIND,
                'candidate_sha256': authority.candidate.candidate_sha256,
                'case_id': authority.case_id,
                'kernel_name': authority.manifest.kernel_name,
                'job_id': admission.broker_job_id,
                'gpu_uuid': admission.gpu_uuid,
                'allocation_mode': 'local_serialized',
                'external_gpu_activity': 'not_excluded',
                'separate_instrumented_launch': True,
                'evaluation_protocol': authority.request['evaluation_protocol'],
                'raw': raw, 'summary': hip_profile_summary(raw)})
            artifacts['profile'] = profile_path.name
        if collect_timing:
            timing_path = authority.request_root / 'timing-samples.json'
            _write_new(timing_path, {'cohorts_ms': cohorts} if cohorts else {'not_measured': 'correctness_rejected'})
            artifacts['timing_samples'] = timing_path.name
        elif profile_source is None and authority.request["purpose"] != "attribution":
            timing_path = authority.request_root / 'timing-samples.json'
            _write_new(timing_path, None)
            artifacts['timing_samples'] = timing_path.name
        # The common receipt describes the final correctness launch; counters and
        # the raw launch artifact retain the separate preflight and timing work.
        result['receipt'] = {'correctness_passed': passed, 'correctness': metrics,
            'kernel_calls': 1, 'fallback_calls': 0, 'timing': timing, 'artifacts': artifacts}
    finally:
        counters['kernel_calls'] = loaded.loaded.launch_calls
        counters['timing_samples'] = sum(len(s) for s in cohorts)
        loaded.close()


def _evaluate_metal_candidate(authority, result):
    """One real Metal assay through the existing broker worker and common receipts."""
    import re
    from open_cake_ir.evaluation.metal_runtime import observe
    from open_cake_ir.evaluation.metal_observations import (
        METAL_TIMER, METAL_CACHE, METAL_PROFILE_KIND, amortized_dispatch_ms, metal_profile_summary)

    evaluation = authority.request['evaluation_protocol']
    protocol = paired_protocol(evaluation)
    if protocol is None or evaluation['paired_timing']['kind'] not in METAL_KINDS:
        raise ValueError('Metal worker requires its explicit fixed-baseline paired assay')
    cases = validation_case_ids(evaluation)
    if (cases != authority.workload.case_ids or authority.workload.document['validation'].get('all_cases_required') is not True
            or authority.case_id != authority.workload.document['validation'].get('primary_case')):
        raise ValueError('Metal evaluation case projection differs from Workload validation')
    row = PLATFORMS[CodeObject.METAL_BINARY_ARCHIVE]
    job_id = os.environ.get('METAL_JOB_ID', '')
    if os.environ.get('GPUQ_JOB_ID'):
        from open_cake_ir.evaluation.gpuq import observe_allocation
        job_id = observe_allocation(authority.candidate.target)['job_id']
    elif (re.fullmatch(rf'{row.local_job_prefix}-[0-9a-f]{{12}}', job_id) is None
            or job_id == f'{row.local_job_prefix}-000000000000'):
        raise ValueError('Metal worker requires a real broker job allocation')
    from open_cake_ir.evaluation.local_broker import observe_local_metal_job
    if not os.environ.get('GPUQ_JOB_ID') and observe_local_metal_job() != job_id:
        raise ValueError('Metal broker lock identity differs')
    admission = authority.executor.admit_host()
    if not isinstance(admission, Mapping) or admission.get('kind') != row.host_kind:
        raise ValueError('Metal worker requires an admitted Metal Executor')
    result['job_id'] = job_id
    result['mode'] = job_mode(job_id)
    result['admitted'] = True
    profile = authority.request['purpose'] == 'attribution'
    candidates = {'candidate': authority.candidate}
    manifests = {'candidate': authority.manifest}
    if not profile:
        if authority.baseline is None:
            raise ValueError('Metal paired assay requires its sealed baseline')
        candidates['baseline'] = authority.baseline
        manifests = validate_pair_candidates(authority.candidate, authority.baseline, authority.workload, authority.case_id)
    from open_cake_ir.tasks.workloads import materialize_case as task_materialize, reference_outputs as task_reference
    input_cases = {}
    for case_id in cases:
        inputs = task_materialize(authority.workload, case_id)
        input_cases[case_id] = {'inputs': inputs, 'expected': task_reference(authority.workload, case_id, inputs)}
    plan = []
    def append(role, phase, case_id, *, timed=False, instrumented=False, pair_index=None, position=None,
               dispatches=1):
        plan.append({'index': len(plan), 'role': role, 'phase': phase, 'input_case_id': case_id,
                     'timed': timed, 'profile': instrumented, 'pair_index': pair_index, 'position': position,
                     'dispatches': dispatches})
    for role in candidates:
        for case_id in cases:
            append(role, 'preflight', case_id)
    if profile:
        append('candidate', 'profile', authority.case_id, instrumented=True)
    else:
        for pair_index, order in enumerate(protocol.pair_order):
            for position, role in enumerate(order):
                for call in range(protocol.route_calls_per_cohort):
                    # Warmups share the sample's command shape so the timed buffers
                    # observe an already warmed pipeline at the same dispatch count.
                    append(role, 'cohort', authority.case_id,
                           timed=call >= protocol.route_calls_per_cohort - protocol.samples_per_cohort,
                           pair_index=pair_index, position=position,
                           dispatches=protocol.dispatches_per_sample)
        for role in candidates:
            for case_id in cases:
                append(role, 'postflight', case_id)
    observation = observe(workload=authority.workload, candidates=candidates, manifests=manifests,
        input_cases=input_cases, launch_plan=plan, observer_executable=Path(admission['observer_executable']),
        expected_host=dict(admission['host']), directory=authority.request_root / 'metal-observation')
    launches = observation['launches']
    counters = result['counters']
    counters.update(module_loads=len(candidates), preflight_calls=sum(row['phase'] == 'preflight' for row in launches),
                    kernel_calls=len(launches), timing_samples=sum(row['timed'] for row in launches))
    def aggregate(rows, *, timed=False):
        metrics = {'output_mismatches': 0, 'max_abs_error': 0.0, 'inputs_unchanged': True}
        passed = True
        for row in rows:
            passed &= row['passed']
            metrics['output_mismatches'] += row['metrics']['output_mismatches']
            metrics['max_abs_error'] = max(metrics['max_abs_error'], row['metrics']['max_abs_error'])
            metrics['inputs_unchanged'] &= row['metrics']['inputs_unchanged']
        result = {'passed': passed, 'launches': [
            {key: row[key] for key in ('input_case_id', 'passed', 'metrics', 'command_buffer')} for row in rows]}
        result.update(metrics if timed else {'metrics': metrics})
        if timed:
            result['checked_launches'] = len(rows)
        return result
    overall = aggregate(launches)
    host = observation['host']
    if profile:
        profiled = [row for row in launches if row['phase'] == 'profile']
        if len(profiled) != 1 or not overall['passed']:
            raise ValueError('instrumented Metal launch and all validation inputs must pass the external oracle')
        row = profiled[0]
        raw_profile = row['profile_raw']
        profile_document = {'kind': METAL_PROFILE_KIND, 'candidate_sha256': authority.candidate.candidate_sha256,
            'case_id': authority.case_id, 'kernel_name': authority.candidate.entry_point, 'job_id': job_id, 'host': host,
            'separate_instrumented_launch': True, 'archive_miss_policy': 'failOnBinaryArchiveMiss',
            'evaluation_protocol': evaluation, 'allocation_mode': 'local_serialized', 'external_gpu_activity': 'not_excluded',
            'raw': raw_profile, 'summary': metal_profile_summary(raw_profile)}
        _write_new(authority.request_root / 'profile.json', profile_document)
        _write_new(authority.request_root / 'correctness-output.json', overall)
        _write_new(authority.request_root / 'launch-receipt.json', {'candidate_sha256': authority.candidate.candidate_sha256,
            'job_id': job_id, 'host': host, 'allocation_mode': 'local_serialized',
            'external_gpu_activity': 'not_excluded', 'instrumented_command': row['command_buffer'],
            'correctness_launches': len(launches), 'fallback_calls': 0})
        result['receipt'] = {'correctness_passed': True, 'correctness': overall['metrics'], 'kernel_calls': 1,
            'fallback_calls': 0, 'timing': None, 'artifacts': {'correctness_output': 'correctness-output.json',
            'launch_receipt': 'launch-receipt.json', 'profile': 'profile.json'}}
        return
    checks = {}
    for role in candidates:
        checks[role] = {'preflight': aggregate([row for row in launches if row['role'] == role and row['phase'] == 'preflight']),
            'postflight': None, 'timed_output_checks': []}
        post = [row for row in launches if row['role'] == role and row['phase'] == 'postflight']
        if post:
            checks[role]['postflight'] = aggregate(post)
    measurements = []
    if any(row['phase'] == 'cohort' for row in launches):
        for pair_index, order in enumerate(protocol.pair_order):
            measurement = {'pair_index': pair_index, 'order': list(order), 'arms': {}}
            for position, role in enumerate(order):
                rows = [row for row in launches if row['phase'] == 'cohort' and row['pair_index'] == pair_index and row['role'] == role]
                check = aggregate(rows, timed=True)
                checks[role]['timed_output_checks'].append(check)
                samples = [amortized_dispatch_ms(row['command_buffer']) for row in rows if row['timed']]
                measurement['arms'][role] = {'position': position,
                    'candidate_record_sha256': candidates[role].canonical_sha256,
                    'samples_ms': samples, 'summary': summarize_cohort(samples), 'route_calls': len(rows),
                    'output_check': check, 'command_buffers': [row['command_buffer'] for row in rows]}
            measurements.append(measurement)
    identities = {role: candidate_identity(candidate) for role, candidate in candidates.items()}
    raw = {'kind': evaluation['paired_timing']['kind'], 'evaluation_protocol': evaluation, 'participants': identities,
        'workload_sha256': authority.workload.canonical_sha256, 'case_id': authority.case_id,
        'purpose': authority.request['purpose'], 'job_id': job_id, 'device_registry_id': host['device_registry_id'],
        'host': host, 'allocation_mode': 'local_serialized', 'external_gpu_activity': 'not_excluded',
        'timer': METAL_TIMER, 'cache_policy': METAL_CACHE, 'measurements': measurements}
    if not measurements:
        raw['not_measured'] = 'correctness_rejected'
    timing = paired_summary(raw) if measurements else None
    _write_new(authority.request_root / 'timing-samples.json', raw)
    _write_new(authority.request_root / 'correctness-output.json', {'passed': overall['passed'], 'metrics': overall['metrics'], 'participants': checks})
    _write_new(authority.request_root / 'launch-receipt.json', {'candidate_sha256': authority.candidate.candidate_sha256,
        'participants': identities, 'job_id': job_id, 'host': host,
        'allocation_mode': 'local_serialized', 'external_gpu_activity': 'not_excluded',
        'correctness_launches': sum(row['phase'] in {'preflight', 'postflight'} for row in launches), 'fallback_calls': 0})
    result['receipt'] = {'correctness_passed': overall['passed'], 'correctness': overall['metrics'], 'kernel_calls': 1,
        'fallback_calls': 0, 'timing': timing, 'artifacts': {'correctness_output': 'correctness-output.json',
        'launch_receipt': 'launch-receipt.json', 'timing_samples': 'timing-samples.json'}}


def _evaluate_hip_candidate(authority, result, *, collect_timing, admission=None):
    """Evaluate one sealed AMDGCN candidate on its admitted DCU.

    Correctness, the oracle, the cohort shape and the receipts are the same ones every
    tensor-tile evaluation uses. Timing is this target's own: `HipDispatchBenchmark` reads
    roctracer's per-dispatch device time through the profiler the admitted torch carries,
    which is what made this a named source rather than a coverage limitation.

    Whether timing runs at all is still the Study's statement, not this function's. A
    target whose Study carries a `measurement_coverage` limitation arrives here with
    `collect_timing` false and gets correctness alone, and the receipt says `timing` is
    absent rather than implying none was possible.
    """
    from open_cake_ir.evaluation.hip_benchmark import HipDispatchBenchmark
    from open_cake_ir.evaluation.hip_observations import collect_hip_dispatch_activity
    from open_cake_ir.evaluation.triton_hip import observe_local_hip

    if admission is None:
        try:
            admission = observe_local_hip(authority.candidate.target)
        except (ValueError, RuntimeError):
            result["error"] = "gpu_admission_differs"
            return
    result["job_id"] = admission.broker_job_id
    result["mode"] = job_mode(admission.broker_job_id)
    result["admitted"] = True
    if authority.request["purpose"] == "attribution":
        # Attribution is correctness plus one instrumented dispatch. It is neither timed
        # nor paired: a cohort here would be a second latency taken under different device
        # state than the assay's, reported beside it and comparable to nothing. Reaching
        # the paired branch instead refuses inside the tile evaluation, because an
        # attribution purpose is not a correctness protocol.
        _evaluate_tile_candidate(
            authority, result, None, admission, False,
            route_calls_per_cohort=_route_calls_per_cohort(authority),
            profile_source=collect_hip_dispatch_activity)
        return
    if authority.baseline is not None and collect_timing:
        # A Study with a paired policy sends both participants, and the assay is one per
        # arm because it attributes by kernel name.
        _evaluate_paired_tile(
            authority, result,
            lambda role, manifest: HipDispatchBenchmark(manifest.kernel_name), admission)
        return
    benchmark = (HipDispatchBenchmark(authority.manifest.kernel_name)
                 if collect_timing else None)
    _evaluate_tile_candidate(authority, result, benchmark, admission, collect_timing,
                             route_calls_per_cohort=_route_calls_per_cohort(authority))


def _evaluate_metax_candidate(authority, result, *, collect_timing, admission=None):
    from open_cake_ir.evaluation.triton_metax import observe_local_metax

    if collect_timing or authority.request["purpose"] == "attribution":
        raise ValueError("MACA timing and profiler coverage are unavailable")
    host = authority.executor.admit_host()
    if admission is None:
        admission = observe_local_metax(authority.candidate.target, runtime_library=host["runtime_library"])
    elif admission.runtime_library != host["runtime_library"]:
        raise ValueError("MACA device admission refers to another runtime library")
    result.update(job_id=admission.broker_job_id, mode=job_mode(admission.broker_job_id), admitted=True)
    _evaluate_tile_candidate(authority, result, None, admission, False, route_calls_per_cohort=None)


def _evaluate_candidate(
    authority: _Authority,
    result: dict[str, object],
    *,
    collect_timing: bool,
    admission: CudaDeviceAdmission | None = None,
) -> None:
    platform = _execution_platform(authority)
    if platform is not CodeObject.CUBIN:
        # This is the cubin row's own evaluate, reached again by the profile child. Named,
        # not fallen through, and never stepped down onto another platform's path: every
        # other row's evaluate is reached through its own `_PLATFORMS` entry.
        raise ValueError(
            f"{platform.value!r} does not launch through this path: the hsaco row launches "
            "its own, and a Metal binary archive is observed rather than launched"
        )
    helper = authority.executor.admit_host()
    if admission is None:
        try:
            admission = _observe_cuda(authority)
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
    if authority.baseline is not None and collect_timing:
        # CUPTI times whatever the callable dispatches and is not bound to a kernel
        # name, so both arms share one instance.
        strict_cupti = StrictCuptiBenchmark(helper)
        _evaluate_paired_tile(authority, result,
                              lambda role, manifest: strict_cupti, admission)
        return
    if isinstance(authority.manifest, TensorLaunchManifest):
        _evaluate_tile_candidate(
            authority, result,
            StrictCuptiBenchmark(helper) if collect_timing else None,
            admission, collect_timing,
            route_calls_per_cohort=_route_calls_per_cohort(authority))
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
        admission = _observe_cuda(authority)
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
            "output_owner": {"uid": os.geteuid(), "gid": os.getegid()},
        },
    )
    child_result_path = authority.request_root / "profile-child-result.json"
    from open_cake_ir.evaluation.source_bootstrap import module_command
    command = [
        str(profiler["path"]),
        "--forward-signals",
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
        *module_command(sys.executable, "open_cake_ir.tasks.evaluate"),
        "--profile-child",
        "--profile-admission",
        str(admission_path),
        "--request",
        str(request_path),
        "--output",
        str(child_result_path),
    ]
    from open_cake_ir.lab.ncu_process import NcuProcessCancelled, run_ncu
    try:
        completed = run_ncu(
            command,
            cwd=ROOT,
            environment=sanitized_environment(),
            timeout_seconds=900,
            maximum_output_bytes=16 * 1024 * 1024,
        )
    except (SupervisedProcessTimeout, SupervisedProcessOutputLimit, NcuProcessCancelled) as error:
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


@dataclass(frozen=True)
class _ExecutionPlatform:
    """The evaluate callable this worker registers on one execution platform row.

    The registry (`evaluation.platforms`) states how a declared code object is attributed
    and whether its profile comes from a separate supervised child; it cannot hold the
    callable, because the callable imports task code the Evaluation layer may not (ADR
    0055). So the row is held here beside its worker function, and the facts are read
    off the row rather than restated. A code object nothing implements is refused by
    name instead of reaching whichever branch it happens to fall into -- that was not
    hypothetical: a candidate built for an AMDGCN target asking for a profile reached
    CUDA's exclusive device admission and was refused by nvidia-smi.
    """

    evaluate: Callable[[_Authority, dict], None] | None
    platform: ExecutionPlatform | None = None

    @property
    def attribution(self) -> str | None:
        return None if self.platform is None else self.platform.attribution

    @property
    def profiled_child(self) -> bool:
        return self.platform is not None and self.platform.profiled_child


_PLATFORMS = {
    CodeObject.CUBIN: _ExecutionPlatform(
        # Whether a run is timed is the Study's statement, not this table's: a Study for
        # a target with no named timer carries a measurement-coverage limitation instead
        # of a paired assay, and this was an unconditional True.
        evaluate=lambda authority, result: _evaluate_candidate(
            authority, result, collect_timing=authority.timed_assay_available),
        platform=PLATFORMS[CodeObject.CUBIN],
    ),
    CodeObject.METAL_BINARY_ARCHIVE: _ExecutionPlatform(
        evaluate=_evaluate_metal_candidate,
        platform=PLATFORMS[CodeObject.METAL_BINARY_ARCHIVE],
    ),
    # The AMDGCN half admits a device, loads a candidate and launches it, verified on a
    # DCU (F-2026-09-15-003). Whether it is timed is the Study's statement, as above.
    CodeObject.HSACO: _ExecutionPlatform(
        evaluate=lambda authority, result: _evaluate_hip_candidate(
            authority, result, collect_timing=authority.timed_assay_available),
        platform=PLATFORMS[CodeObject.HSACO],
    ),
    CodeObject.MCFATBIN: _ExecutionPlatform(
        evaluate=lambda authority, result: _evaluate_metax_candidate(
            authority, result, collect_timing=authority.timed_assay_available),
        platform=PLATFORMS[CodeObject.MCFATBIN],
    ),
}


def _platform(authority: _Authority) -> _ExecutionPlatform:
    """The row for this candidate's declared object, or a refusal naming the object."""
    name = _execution_platform(authority)
    platform = _PLATFORMS.get(name)
    if platform is None or platform.evaluate is None:
        raise ValueError(
            f"no execution platform implements {name.value!r}: this worker launches and "
            "times a cubin, launches an hsaco, and observes a Metal binary archive"
        )
    return platform


def main() -> int:
    global _PROFILE_OUTPUT_OWNER, _BROKER_ALLOCATION
    _BROKER_ALLOCATION = None
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--profile-admission", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    request_path = args.request.resolve(strict=True)
    # The allocator that admitted this process names its job in its own variable: the
    # local broker in METAL_JOB_ID under the family's prefix, the cluster allocator in
    # GPUQ_JOB_ID. Before either, the worker carries the cluster placeholder.
    result = _base_result(os.environ.get("METAL_JOB_ID", os.environ.get(
        "GPUQ_JOB_ID", f"{PLATFORMS[CodeObject.CUBIN].exclusive_job_prefix}-000000000000")))
    try:
        authority = _load_authority(request_path)
        if os.environ.get("GPUQ_BACKEND"):
            from open_cake_ir.evaluation.gpuq import observe_allocation
            _BROKER_ALLOCATION = observe_allocation(authority.candidate.target)
        purpose = str(authority.request["purpose"])
        if args.profile_child:
            if purpose != "attribution" or args.profile_admission is None:
                raise ValueError("profile child requires attribution purpose")
            if not _platform(authority).profiled_child:
                raise ValueError(
                    f"{_execution_platform(authority).value!r} has no profiled child launch"
                )
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
                "output_owner",
            } or admission_document.get("schema_version") != 1:
                raise ValueError("profile admission fields differ")
            from open_cake_ir.lab.ncu_process import profile_output_owner
            _PROFILE_OUTPUT_OWNER = profile_output_owner(admission_document["output_owner"])
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
            platform = _platform(authority)
            if platform.attribution is None:
                raise ValueError(
                    f"{_execution_platform(authority).value!r} has no attribution source; a "
                    "profile is not taken on another platform's behalf"
                )
            if platform.attribution == "inside_evaluate":
                if args.profile_admission is not None:
                    # Two platforms take their profile inside evaluate now, so this can no
                    # longer be phrased as Metal's rule: a DCU caller was refused in
                    # Metal's words for a mistake of its own.
                    raise ValueError(
                        f"{_execution_platform(authority).value!r} takes its profile inside "
                        "evaluate; that admission is supplied by its own Executor")
                platform.evaluate(authority, result)
            elif platform.attribution == "separate":
                if args.profile_admission is not None:
                    raise ValueError("profile admission is internal-only")
                _profile_candidate(authority, request_path, result)
            else:
                # Nsight was the fall-through here too. Three sites read this profile and
                # two were closed; this is the third, and leaving it meant any future
                # attribution value routed a candidate to CUDA's profiler.
                raise ValueError(
                    f"attribution source {platform.attribution!r} is not implemented; a "
                    "profile is taken inside evaluate or by this platform's own separate "
                    "profiler, and never on another platform's behalf")
        else:
            _platform(authority).evaluate(authority, result)
    except Exception as error:
        result["error"] = "evaluator_failed"
        result["failure_class"] = type(error).__name__
        result["receipt"] = None
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
    _write_new(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
