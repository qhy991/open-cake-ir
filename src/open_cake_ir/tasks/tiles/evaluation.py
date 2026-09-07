from __future__ import annotations
import json
from hashlib import sha256
from typing import Mapping
from open_cake_ir.evaluation.core import EvaluationProtocol,EvaluationReceipt,LaunchableCandidate,TensorLaunchManifest,compare_tile_outputs,_canonical_json_bytes
from open_cake_ir.evaluation.workload import WorkloadContract

def evaluate_tile_workload(candidate: LaunchableCandidate, workload: WorkloadContract,
                           protocol: EvaluationProtocol, launcher) -> EvaluationReceipt:
    """Common correctness assay: Workload data/oracle, sealed launch, every output."""
    from .workload import materialize_case, reference_outputs
    if protocol.workload_sha256 != workload.canonical_sha256 or protocol.timing != 'none' or protocol.purpose == 'attribution':
        raise ValueError('tile correctness Evaluation protocol differs')
    manifest = TensorLaunchManifest.from_dict(json.loads(candidate.artifact_payloads['launch_manifest']))
    manifest.check_workload(workload, protocol.case_id)
    if (candidate.launch_spec_sha256 != manifest.canonical_sha256
        or candidate.target != manifest.target or candidate.entry_point != manifest.kernel_name):
        raise ValueError('sealed tensor manifest differs')
    inputs = materialize_case(workload, protocol.case_id)
    before = {name: list(values) for name, values in inputs.items()}
    expected = reference_outputs(workload, protocol.case_id, before)
    observed, after, launch = launcher.launch_tensors(candidate, manifest, inputs)
    passed, metrics = compare_tile_outputs(workload, before, expected, observed, after)
    if (not isinstance(launch, Mapping) or launch.get('candidate_sha256') != candidate.candidate_sha256
        or launch.get('kernel_calls') != 1 or launch.get('fallback_calls') != 0):
        raise ValueError('tile launch receipt differs')
    launch_bytes = _canonical_json_bytes(launch)
    return EvaluationReceipt(candidate.candidate_sha256, workload.canonical_sha256,
        protocol.canonical_sha256, protocol.purpose, protocol.case_id, passed, metrics,
        1, 0, sha256(launch_bytes).hexdigest(), None, artifact_payloads={
            'correctness_output': _canonical_json_bytes({'passed': passed, 'metrics': metrics}),
            'launch_receipt': launch_bytes, 'timing_samples': b'null'})
