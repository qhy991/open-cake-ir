"""Observed MACA launch resource failures; these never constitute an Evaluation receipt."""
from __future__ import annotations

import re

from open_cake_ir.serialization import canonical_json_bytes


class MetaxLaunchResourceError(RuntimeError):
    """Status 32 from target dispatch, with the loaded resource facts retained."""

    def retain_launch(self, candidate, manifest, resources, *, completed_target_calls):
        self.launch_resource = {
            'kind': 'maca_launch_resource_failure_v1',
            'operation': 'mcModuleLaunchKernel', 'status': 32,
            'status_name': 'mcErrorMemoryValueTooLarge',
            'target': candidate.target, 'candidate_sha256': candidate.candidate_sha256,
            'kernel_name': manifest.kernel_name, 'grid': list(manifest.grid),
            'block': list(manifest.block), 'resources': dict(resources),
            'completed_target_calls': completed_target_calls,
            'coverage': 'failed_dispatch_no_correctness_or_timing_receipt',
        }
        self.artifact_payloads = {'launch_resource': canonical_json_bytes(self.launch_resource)}


def is_local_launch_resource_failure(result, diagnostic, stdout, stderr):
    """Permit only a zero-dispatch search failure with matching SDK limit evidence.

    The limit belongs to this worker's observed host configuration. It is not a
    Target capability and is never substituted when the SDK omits its account.
    Unknown errors, teardown failures and failures after successful launches stop.
    """
    if not isinstance(result, dict) or not isinstance(diagnostic, dict):
        return False
    counters = result.get('counters')
    resources = diagnostic.get('resources')
    if (not isinstance(counters, dict) or not isinstance(resources, dict)
            or any(type(value) is not int or value < 0 for value in counters.values())
            or result.get('failure_class') != 'MetaxLaunchResourceError'
            or result.get('schema_version') != 2
            or set(result.get('failure_artifacts', {})) != {'launch_resource'}
            or result.get('error') != 'evaluator_failed' or result.get('receipt') is not None
            or result.get('admitted') is not True or result.get('mode') != 'local_serialized'
            or result.get('counters', {}).get('kernel_calls') != 0
            or result.get('counters', {}).get('timing_samples') != 0
            or result.get('counters', {}).get('compiler_invocations') != 0
            or result.get('counters', {}).get('fallback_calls') != 0
            or result.get('counters', {}).get('preflight_calls') != 0
            or diagnostic.get('kind') != 'maca_launch_resource_failure_v1'
            or diagnostic.get('operation') != 'mcModuleLaunchKernel'
            or diagnostic.get('status') != 32
            or diagnostic.get('status_name') != 'mcErrorMemoryValueTooLarge'
            or diagnostic.get('completed_target_calls') != 0
            or type(diagnostic.get('completed_target_calls')) is not int
            or diagnostic.get('purpose') != 'search' or diagnostic.get('arm') != 'candidate'
            or diagnostic.get('phase') != 'preflight'
            or diagnostic.get('teardown_completed') is not True
            or diagnostic.get('job_id') != result.get('job_id')
            or diagnostic.get('coverage') != 'failed_dispatch_no_correctness_or_timing_receipt'):
        return False
    # SDK stdout is independent of the Python exception and reports the host limit.
    limits = re.findall(r'The system is set to :(\d+) KB/Thread,kernel request:(\d+) KB/thread', stdout)
    if len(limits) != 1 or 'Error in allocating private memory' not in stdout:
        return False
    limit, requested = map(int, limits[0])
    local_bytes = resources.get('local_bytes')
    # The SDK prints whole KB while the function metadata reports bytes. Both
    # adjacent integer renderings preserve the observed limit comparison.
    # This does not infer a device limit or admit an unrelated allocation error.
    if (limit <= 0 or requested <= limit or type(local_bytes) is not int
            or local_bytes <= limit * 1024
            or requested not in {local_bytes // 1024, (local_bytes + 1023) // 1024}):
        return False
    expected = 'MetaxLaunchResourceError: MACA mcModuleLaunchKernel failed with status 32 (mcErrorMemoryValueTooLarge)'
    lines = [line for line in stderr.splitlines() if line and not line.startswith('[maca-run] accepted job ')]
    return lines == [expected]
