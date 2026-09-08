#!/usr/bin/env python3
"""Evaluate one sealed artifact on its admitted exact backend.

The existing worker retains CUDA assays and observes Metal binary archives through
the same common Evaluation receipt and broker boundary.
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from open_cake_ir.tasks.workloads import load_workload

from open_cake_ir.evaluation import CudaDeviceAdmission, LaunchableCandidate, LoadedCudaCandidate, WorkloadContract, build_ncu_attribution_profile, NCU_ATTRIBUTION_METRICS, summarize_cohort
from open_cake_ir.tasks.flash_kmeans.cuda_manifest import CudaLaunchManifest
from open_cake_ir.tasks.flash_kmeans.cuda import CudaTensorContract
from open_cake_ir.evaluation.benchmark import StrictCuptiBenchmark
from open_cake_ir.tasks.flash_kmeans.workload import assignment_raw_sha256, classify_flash_kmeans_output, flash_kmeans_oracle, generate_flash_kmeans_case
from open_cake_ir.lab.executor import ExecutorRevision
from open_cake_ir.evaluation.admission import observe_exclusive_cuda
from open_cake_ir.evaluation.core import EvaluationProtocol, LoadedTorchTensorCandidate, TensorLaunchManifest, compare_tile_outputs
from open_cake_ir.tasks.tiles.evaluation import evaluate_tile_workload
from open_cake_ir.tasks.launch import parse_launch_manifest
from open_cake_ir.evaluation.metal_manifest import MetalTensorLaunchManifest
from open_cake_ir.tasks.tiles.workload import materialize_case, reference_outputs
from open_cake_ir.lab.process import SupervisedProcessOutputLimit, SupervisedProcessTimeout, run_supervised, sanitized_environment
from open_cake_ir.evaluation.paired import (
    PAIRED_KIND, PAIRED_METAL_KIND, paired_protocol, paired_summary, candidate_identity, validation_case_ids,
    candidate_from_identity, validate_pair_candidates,
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
    manifest: CudaLaunchManifest | TensorLaunchManifest | MetalTensorLaunchManifest
    candidate: LaunchableCandidate
    payloads: Mapping[str, bytes]
    case_id: str
    baseline: LaunchableCandidate | None = None


def _load_authority(request_path: Path) -> _Authority:
    request_root = request_path.parent
    request = _object(json.loads(request_path.read_text(encoding="utf-8")), "request")
    executor = ExecutorRevision.load_reference(ROOT, request.get("executor_revision"), "request.executor_revision")
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
    evaluation = request.get('evaluation_protocol')
    if evaluation is not None:
        evaluation = _object(evaluation, 'request.evaluation_protocol')
        if sha256(_canonical_json_bytes(evaluation)).hexdigest() != request['evaluation_protocol_sha256']:
            raise ValueError('worker evaluation policy identity differs')
        if paired_protocol(evaluation) is not None:
            if isinstance(manifest, MetalTensorLaunchManifest) != (evaluation['paired_timing']['kind'] == PAIRED_METAL_KIND):
                raise ValueError('paired assay backend differs from sealed manifest')
            partner = _object(request.get('baseline'), 'request.baseline')
            paths = _object(partner.get('artifact_paths'), 'request.baseline.artifact_paths')
            baseline = candidate_from_identity({k: v for k, v in partner.items() if k != 'artifact_paths'},
                {role: _input_path(request_root, path, f'baseline.{role}').read_bytes()
                 for role, path in paths.items()})
            validate_pair_candidates(candidate, baseline, workload, case_id)
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
    )


def _fresh_tile_cohort(loaded, strict_cupti, workload, inputs, expected, *,
                       samples_per_cohort, route_calls_per_cohort):
    """The single retained CUPTI path for both historical and paired tensor assays."""
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
    for values in arguments:
        observed, after = loaded.snapshot(values)
        correct, observation = compare_tile_outputs(workload, inputs, expected, observed, after)
        check['passed'] = check['passed'] and correct
        check['output_mismatches'] += observation['output_mismatches']
        check['max_abs_error'] = max(check['max_abs_error'], observation['max_abs_error'])
        check['inputs_unchanged'] = check['inputs_unchanged'] and observation['inputs_unchanged']
    return samples, check


def _evaluate_paired_tile(authority, result, helper, admission):
    """Execute both sealed participants in one allocation, in the frozen order."""
    protocol = paired_protocol(authority.request['evaluation_protocol'])
    candidates = {'candidate': authority.candidate, 'baseline': authority.baseline}
    manifests = validate_pair_candidates(authority.candidate, authority.baseline,
                                        authority.workload, authority.case_id)
    inputs = materialize_case(authority.workload, authority.case_id)
    expected = reference_outputs(authority.workload, authority.case_id, inputs)
    loaded = {}
    checks = {role: {'preflight': None, 'postflight': None, 'timed_output_checks': []}
              for role in protocol.arms}
    measurements = []
    counters = result['counters']
    correctness_protocol = EvaluationProtocol('workload-tensor-worker-correctness',
        authority.request['purpose'], authority.workload.canonical_sha256, authority.case_id, 'none')
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
        receipt = evaluate_tile_workload(candidates[role], authority.workload,
                                        correctness_protocol, loaded[role])
        check = {'passed': receipt.correctness_passed, 'metrics': dict(receipt.correctness)}
        checks[role][phase] = check
        correctness_calls += 1
        accumulate(check)
    try:
        for role in protocol.arms:
            loaded[role] = LoadedTorchTensorCandidate(candidates[role], manifests[role], inputs, admission)
            counters['module_loads'] += 1
            correctness(role, 'preflight')
            counters['preflight_calls'] += 1
        if passed:
            strict_cupti = StrictCuptiBenchmark(helper)
            for index, order in enumerate(protocol.pair_order):
                row = {'pair_index': index, 'order': list(order), 'arms': {}}
                for position, role in enumerate(order):
                    samples, check = _fresh_tile_cohort(loaded[role], strict_cupti,
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
                measurements.append(row)
            for role in protocol.arms:
                correctness(role, 'postflight')
        identities = {role: candidate_identity(item) for role, item in candidates.items()}
        raw = {'kind': PAIRED_KIND, 'evaluation_protocol': authority.request['evaluation_protocol'],
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
            'resources': {role: item.loaded.resources for role, item in loaded.items()}})
        result['receipt'] = {'correctness_passed': passed, 'correctness': metrics,
            'kernel_calls': 1, 'fallback_calls': 0, 'timing': timing,
            'artifacts': {'correctness_output': 'correctness-output.json',
                          'launch_receipt': 'launch-receipt.json', 'timing_samples': 'timing-samples.json'}}
    finally:
        counters['kernel_calls'] = sum(item.loaded.launch_calls for item in loaded.values())
        for item in loaded.values():
            item.close()


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
                samples, check = _fresh_tile_cohort(loaded, strict_cupti, authority.workload,
                    inputs, expected, samples_per_cohort=25, route_calls_per_cohort=42)
                cohorts.append(samples)
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


def _evaluate_metal_candidate(authority, result):
    """One real Metal assay through the existing broker worker and common receipts."""
    import re
    from open_cake_ir.evaluation.metal_runtime import observe
    from open_cake_ir.evaluation.metal_observations import (
        METAL_TIMER, METAL_CACHE, METAL_PROFILE_KIND, command_buffer_ms, metal_profile_summary)

    evaluation = authority.request['evaluation_protocol']
    protocol = paired_protocol(evaluation)
    if protocol is None or evaluation['paired_timing']['kind'] != PAIRED_METAL_KIND:
        raise ValueError('Metal worker requires its explicit fixed-baseline paired assay')
    cases = validation_case_ids(evaluation)
    if (cases != authority.workload.case_ids or authority.workload.document['validation'].get('all_cases_required') is not True
            or authority.case_id != authority.workload.document['validation'].get('primary_case')):
        raise ValueError('Metal evaluation case projection differs from Workload validation')
    job_id = os.environ.get('METAL_JOB_ID', '')
    if re.fullmatch(r'metal-[0-9a-f]{12}', job_id) is None or job_id == 'metal-000000000000':
        raise ValueError('Metal worker requires a real broker job allocation')
    admission = authority.executor.admit_host()
    if not isinstance(admission, Mapping) or admission.get('kind') != 'metal':
        raise ValueError('Metal worker requires an admitted Metal Executor')
    result['job_id'] = job_id
    result['mode'] = 'local_serialized'
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
    def append(role, phase, case_id, *, timed=False, instrumented=False, pair_index=None, position=None):
        plan.append({'index': len(plan), 'role': role, 'phase': phase, 'input_case_id': case_id,
                     'timed': timed, 'profile': instrumented, 'pair_index': pair_index, 'position': position})
    for role in candidates:
        for case_id in cases:
            append(role, 'preflight', case_id)
    if profile:
        append('candidate', 'profile', authority.case_id, instrumented=True)
    else:
        for pair_index, order in enumerate(protocol.pair_order):
            for position, role in enumerate(order):
                for call in range(protocol.route_calls_per_cohort):
                    append(role, 'cohort', authority.case_id,
                           timed=call >= protocol.route_calls_per_cohort - protocol.samples_per_cohort,
                           pair_index=pair_index, position=position)
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
                samples = [command_buffer_ms(row['command_buffer']) for row in rows if row['timed']]
                measurement['arms'][role] = {'position': position,
                    'candidate_record_sha256': candidates[role].canonical_sha256,
                    'samples_ms': samples, 'summary': summarize_cohort(samples), 'route_calls': len(rows),
                    'output_check': check, 'command_buffers': [row['command_buffer'] for row in rows]}
            measurements.append(measurement)
    identities = {role: candidate_identity(candidate) for role, candidate in candidates.items()}
    raw = {'kind': PAIRED_METAL_KIND, 'evaluation_protocol': evaluation, 'participants': identities,
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


def _evaluate_candidate(
    authority: _Authority,
    result: dict[str, object],
    *,
    collect_timing: bool,
    admission: CudaDeviceAdmission | None = None,
) -> None:
    if isinstance(authority.manifest, MetalTensorLaunchManifest):
        _evaluate_metal_candidate(authority, result)
        return
    helper = authority.executor.admit_host()
    if admission is None:
        try:
            admission = observe_exclusive_cuda(authority.candidate.target)
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
        _evaluate_paired_tile(authority, result, helper, admission)
        return
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
        admission = observe_exclusive_cuda(authority.candidate.target)
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
        elif isinstance(authority.manifest, MetalTensorLaunchManifest):
            if args.profile_admission is not None:
                raise ValueError("Metal profile admission is provided by its Executor")
            _evaluate_metal_candidate(authority, result)
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
