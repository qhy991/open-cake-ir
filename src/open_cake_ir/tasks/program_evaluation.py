"""Original tensor Workload oracle to common untimed Evaluation receipts.

Preparation is CPU-only and precedes allocation. This is correctness qualification;
it does not claim a Program timer, profiler or optimization Run endpoint.
"""
import base64
from dataclasses import asdict, dataclass
from hashlib import sha256
import os
from types import MappingProxyType

from open_cake_ir.evaluation.core import EvaluationReceipt, compare_tile_output_values
from open_cake_ir.evaluation.loaders import LifecycleError
from open_cake_ir.evaluation.torch_tensor_inputs import LoadedTorchTensorInputs, check_cpu_tensor_inputs
from open_cake_ir.lab.faults import RunProtocolFault
from open_cake_ir.serialization import canonical_json_bytes
from .launch import parse_launch_manifest
from .workloads import materialize_tensors, reference_tensors


@dataclass(frozen=True, init=False)
class PreparedProgramCase:
    workload_sha256: str
    case_id: str
    inputs: object
    expected: object

    def __init__(self, workload, case_id):
        if os.environ.get('METAL_BROKER_LOCK_FD') or os.environ.get('GPUQ_JOB_ID'):
            raise ValueError('tensor input/oracle preparation must precede GPU allocation')
        inputs = materialize_tensors(workload, case_id)
        expected = reference_tensors(workload, case_id, inputs)
        object.__setattr__(self, 'workload_sha256', workload.canonical_sha256)
        object.__setattr__(self, 'case_id', case_id)
        object.__setattr__(self, 'inputs', MappingProxyType(dict(inputs)))
        object.__setattr__(self, 'expected', MappingProxyType(dict(expected)))


def _output_record(tensors):
    import torch
    return {name: {'shape': list(value.shape), 'dtype': str(value.dtype),
                   'bytes_base64': base64.b64encode(value.view(torch.uint8).numpy().tobytes()).decode('ascii')}
            for name, value in tensors.items()}


def evaluate_program_case(candidate, workload, protocol, admission, *, prepared, observe=None):
    """Check one complete same-ABI case under its original oracle and tolerance."""
    import json
    if (protocol.workload_sha256 != workload.canonical_sha256 or protocol.timing != 'none'
            or protocol.purpose == 'attribution' or not isinstance(prepared, PreparedProgramCase)
            or prepared.workload_sha256 != workload.canonical_sha256 or prepared.case_id != protocol.case_id):
        raise ValueError('tensor Program correctness protocol or CPU preparation differs')
    manifest = parse_launch_manifest(json.loads(candidate.artifact_payloads['launch_manifest']))
    manifest.check_complete_domain()
    if protocol.case_id == manifest.case_id:
        manifest.check_workload(workload, protocol.case_id)
    else:
        manifest.check_validation_case(workload, protocol.case_id)
    check_cpu_tensor_inputs(manifest, prepared.inputs)
    loaded = LoadedTorchTensorInputs(candidate, manifest, prepared.inputs, admission)
    primary = None
    retained = {}
    before = loaded.loaded.launch_calls
    diagnostic = {'candidate_sha256': candidate.candidate_sha256,
                  'manifest_sha256': manifest.canonical_sha256,
                  'expected_kernel_calls': manifest.kernels_per_call,
                  'device_admission': asdict(admission)}
    try:
        if observe is None:
            loaded.launch()
        else:
            # The observation source owns instrumentation, while this assay still
            # owns exact call count, original output checks and native teardown.
            # Retain acquired activity before snapshots or later validation fail.
            retained['program_activity'] = canonical_json_bytes(observe(loaded.launch))
        observed, input_checks = loaded.snapshot()
        # Freeze completed observations before any count/comparison/teardown check
        # can fail. A rejected execution retains evidence, never a passing receipt.
        observation = {'input_checks': input_checks, 'observed_tensors': _output_record(observed),
                       'expected_tensors': _output_record(prepared.expected)}
        if 'program_activity' in retained:
            observation['native_activity'] = json.loads(retained['program_activity'])
        retained['program_observation'] = canonical_json_bytes(observation)
        count = loaded.loaded.launch_calls - before
        if count != manifest.kernels_per_call:
            raise ValueError('native Program did not execute its exact stage count')
        expected_values = {name: value.reshape(-1).tolist() for name, value in prepared.expected.items()}
        observed_values = {name: value.reshape(-1).tolist() for name, value in observed.items()}
        correct, metrics = compare_tile_output_values(workload, expected_values, observed_values)
        metrics = {**metrics, 'inputs_unchanged': all(input_checks.values())}
        passed = correct and metrics['inputs_unchanged']
        correctness = {'passed': passed, 'metrics': metrics, **observation}
        resources = loaded.loaded.resources
    except BaseException as error:
        for role, payload in getattr(error, 'artifact_payloads', {}).items():
            if isinstance(role, str) and role.isidentifier() and isinstance(payload, bytes):
                retained[role] = payload
        primary = error
    finally:
        try:
            loaded.close()
        except BaseException as cleanup:
            primary = LifecycleError(primary, cleanup) if primary is not None else cleanup
    if not loaded.loaded.closed and primary is None:
        primary = ValueError('native tensor modules remain open after teardown')
    diagnostic.update(kernel_calls=loaded.loaded.launch_calls - before,
                      module_unloaded=loaded.loaded.closed, resources=loaded.loaded.resources)
    if primary is not None:
        retained['program_launch'] = canonical_json_bytes({**diagnostic, 'failure_class': type(primary).__name__,
                                                          'error': str(primary)})
        raise RunProtocolFault('harness_fault', str(primary), artifact_payloads=retained) from primary
    launch = {'candidate_sha256': candidate.candidate_sha256, 'kernel_calls': count, 'fallback_calls': 0,
              'manifest_sha256': manifest.canonical_sha256, 'device_admission': asdict(admission),
              'module_unloaded': loaded.loaded.closed, 'resources': resources}
    launch_bytes = canonical_json_bytes(launch)
    try:
        return EvaluationReceipt(candidate.candidate_sha256, workload.canonical_sha256, protocol.canonical_sha256,
            protocol.purpose, protocol.case_id, passed, metrics, count, 0, sha256(launch_bytes).hexdigest(), None,
            artifact_payloads={'correctness_output': canonical_json_bytes(correctness),
                               'launch_receipt': launch_bytes, 'timing_samples': b'null'})
    except Exception as error:
        retained['program_launch'] = launch_bytes
        raise RunProtocolFault('harness_fault', str(error), artifact_payloads=retained) from error
