from __future__ import annotations
import json
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Mapping
from open_cake_ir.evaluation.core import EvaluationProtocol,EvaluationReceipt,LaunchableCandidate,TensorLaunchManifest,compare_tile_outputs,_canonical_json_bytes,_is_torch_tensor
from open_cake_ir.evaluation.workload import WorkloadContract


@dataclass(frozen=True, init=False)
class PreparedTensorCase:
    """CPU-only task inputs and original oracle, retained for one worker process.

    Construction always calls the registered task. This is not a serialized
    caller-supplied reference or a new cache authority.
    """
    workload_sha256: str
    case_id: str
    inputs: Mapping
    expected: Mapping

    def __init__(self, workload: WorkloadContract, case_id: str):
        from open_cake_ir.tasks.workloads import materialize_evaluation_case
        inputs, expected = materialize_evaluation_case(workload, case_id)
        object.__setattr__(self, 'workload_sha256', workload.canonical_sha256)
        object.__setattr__(self, 'case_id', case_id)
        object.__setattr__(self, 'inputs', MappingProxyType({k: v if _is_torch_tensor(v) else tuple(v)
                                                         for k, v in inputs.items()}))
        object.__setattr__(self, 'expected', MappingProxyType({k: tuple(v.reshape(-1).tolist()) if _is_torch_tensor(v) else tuple(v)
                                                           for k, v in expected.items()}))

    def check(self, workload: WorkloadContract, case_id: str):
        if self.workload_sha256 != workload.canonical_sha256 or self.case_id != case_id:
            raise ValueError('prepared tensor case differs from the Workload or input case')

def evaluate_tile_workload(candidate: LaunchableCandidate, workload: WorkloadContract,
                           protocol: EvaluationProtocol, launcher, *, prepared=None) -> EvaluationReceipt:
    """Common correctness assay: Workload data/oracle, sealed launch, every output."""
    return _evaluate_tile(candidate, workload, protocol, launcher, validation_case=False, prepared=prepared)


def evaluate_tile_validation_case(candidate: LaunchableCandidate, workload: WorkloadContract,
                                  protocol: EvaluationProtocol, launcher, *, prepared=None) -> EvaluationReceipt:
    """Check a required same-ABI distribution under the unchanged primary artifact."""
    return _evaluate_tile(candidate, workload, protocol, launcher, validation_case=True, prepared=prepared)


def _evaluate_tile(candidate, workload, protocol, launcher, *, validation_case, prepared):
    if protocol.workload_sha256 != workload.canonical_sha256 or protocol.timing != 'none' or protocol.purpose == 'attribution':
        raise ValueError('tile correctness Evaluation protocol differs')
    from open_cake_ir.tasks.launch import parse_launch_manifest
    manifest = parse_launch_manifest(json.loads(candidate.artifact_payloads['launch_manifest']))
    manifest.check_complete_domain()
    if validation_case:
        manifest.check_validation_case(workload, protocol.case_id)
    else:
        manifest.check_workload(workload, protocol.case_id)
    if (candidate.launch_spec_sha256 != manifest.canonical_sha256
        or candidate.target != manifest.target or candidate.entry_point != manifest.kernel_name):
        raise ValueError('sealed tensor manifest differs')
    if prepared is None:
        prepared = PreparedTensorCase(workload, protocol.case_id)
    if not isinstance(prepared, PreparedTensorCase):
        raise ValueError('tensor preparation must come from the task-owned CPU constructor')
    prepared.check(workload, protocol.case_id)
    inputs = {name: values if _is_torch_tensor(values) else list(values) for name, values in prepared.inputs.items()}
    before = {name: values if _is_torch_tensor(values) else list(values) for name, values in inputs.items()}
    expected = prepared.expected
    observed, after, launch = launcher.launch_tensors(candidate, manifest, inputs)
    passed, metrics = compare_tile_outputs(workload, before, expected, observed, after)
    if (not isinstance(launch, Mapping) or launch.get('candidate_sha256') != candidate.candidate_sha256
        or launch.get('kernel_calls') != manifest.kernels_per_call or launch.get('fallback_calls') != 0):
        raise ValueError('tile launch receipt differs')
    launch_bytes = _canonical_json_bytes(launch)
    return EvaluationReceipt(candidate.candidate_sha256, workload.canonical_sha256,
        protocol.canonical_sha256, protocol.purpose, protocol.case_id, passed, metrics,
        manifest.kernels_per_call, 0, sha256(launch_bytes).hexdigest(), None, artifact_payloads={
            'correctness_output': _canonical_json_bytes({'passed': passed, 'metrics': metrics}),
            'launch_receipt': launch_bytes, 'timing_samples': b'null'})
