"""Explicit whole-Program MCPTI span sampling; ordinary timing admission stays closed.

Each sample starts at its first sealed stage and ends at its last, including stage
gaps. A completed same-stream reset precedes each invocation; host work before the
first kernel or after the last is excluded. This source needs device qualification
before any paired policy or optimization environment may select it.
"""
from collections.abc import Mapping

from open_cake_ir.compiler.target import CodeObject, declared_target
from .metax_benchmark import _McptiBenchmark, _identity, kernel_records, RESET
from .metax_program_profile import program_launch_manifests, validate_program_dispatches

PROGRAM_TIMER = 'mcpti_first_stage_start_last_stage_end_ns'
SYNCHRONIZATION = 'device_synchronize_after_reset_and_after_each_program'
# Captured MACA 3.5.3 mcpti_runtime_cbid.h, not a CUDA callback identifier.
_DEVICE_SYNCHRONIZE = 18


def program_samples(native, *, repeats):
    """Reconstruct complete cold Program invocations, preserving every stage gap."""
    from .paired import candidate_from_identity
    if not isinstance(native, Mapping) or type(repeats) is not int or repeats <= 0:
        raise ValueError('MACA Program sample request differs')
    manifest, manifests = program_launch_manifests(native)
    candidate = candidate_from_identity(native.get('candidate'))
    target = declared_target(manifest.target)
    if (target.code_object is not CodeObject.MCFATBIN or target.l2_cache_bytes is None
            or native.get('timer') != PROGRAM_TIMER or native.get('cache_policy') != RESET
            or native.get('synchronization') != SYNCHRONIZATION
            or native.get('l2_cache_bytes') != target.l2_cache_bytes
            or native.get('reset_bytes') != 4 * target.l2_cache_bytes
            or candidate.target != manifest.target or candidate.entry_point != manifest.kernel_name
            or candidate.launch_spec_sha256 != manifest.canonical_sha256
            or any(child['candidate_sha256'] != candidate.candidate_sha256
                   for child in native['stage_candidates'].values())):
        raise ValueError('MACA Program candidate, interval, reset or synchronization differs')
    reset_records = kernel_records(native.get('reset_activity'))
    reset_apis = [row for row in native['reset_activity']['records'] if row['kind'] == 5 and row['cbid'] in (56, 60)]
    if (len(reset_records) != 1 or reset_records[0] != native.get('reset_record')
            or len(reset_apis) != 1 or reset_apis[0]['cbid'] != 56
            or reset_records[0]['name'] in {item.kernel_name for item in manifests.values()}):
        raise ValueError('MACA Program reset calibration differs')
    records = kernel_records(native.get('activity'))
    width = len(manifests) + 1
    if len(records) != repeats * width or records[0]['start_ns'] < reset_records[0]['end_ns']:
        raise ValueError('MACA Program sample/reset count or calibration order differs')
    apis = {row['correlation']: row for row in native['activity']['records'] if row['kind'] == 5}
    syncs = sorted((row for row in apis.values() if row['cbid'] == _DEVICE_SYNCHRONIZE),
                   key=lambda row: row['start_ns'])
    # Two per invocation plus the collector's final completed-device drain.
    if len(syncs) != 2 * repeats + 1:
        raise ValueError('MACA Program sample synchronization coverage differs')
    samples = []
    previous = None
    stream = None
    for index in range(repeats):
        reset, *stages = records[index * width:(index + 1) * width]
        validate_program_dispatches(stages, apis, manifests)
        current = (reset['device'], reset['context'], reset['stream'])
        before, after = syncs[2 * index:2 * index + 2]
        if (_identity(reset) != _identity(reset_records[0]) or apis[reset['correlation']]['cbid'] != 56
                or current != (stages[0]['device'], stages[0]['context'], stages[0]['stream'])
                or stream is not None and current != stream
                or previous is not None and reset['start_ns'] < previous
                or reset['end_ns'] > stages[0]['start_ns']
                or reset['end_ns'] > before['end_ns']
                or apis[reset['correlation']]['end_ns'] > before['start_ns']
                or before['end_ns'] > apis[stages[0]['correlation']]['start_ns']
                or apis[stages[-1]['correlation']]['end_ns'] > after['start_ns']
                or stages[-1]['end_ns'] > after['end_ns']):
            raise ValueError('MACA Program reset, stream or synchronized sample boundary differs')
        samples.append((stages[-1]['end_ns'] - stages[0]['start_ns']) / 1e6)
        previous, stream = after['end_ns'], current
    if syncs[-1]['start_ns'] < previous:
        raise ValueError('MACA Program final drain precedes the last sample')
    return samples


class McptiProgramBenchmark(_McptiBenchmark):
    timer = PROGRAM_TIMER

    def __init__(self, candidate, *, activity_library, l2_cache_bytes):
        from .program import program_components
        from .paired import candidate_identity
        manifest, children, manifests = program_components(candidate)
        if declared_target(manifest.target).code_object is not CodeObject.MCFATBIN:
            raise ValueError('MCPTI Program sampling requires mcfatbin')
        self._facts = {'candidate': candidate_identity(candidate), 'manifest': manifest.as_dict(),
            'stage_manifests': {name: spec.as_dict() for name, spec in manifests.items()},
            'stage_candidates': {name: candidate_identity(child) for name, child in children.items()},
            'synchronization': SYNCHRONIZATION}
        self._kernel_names = tuple(spec.kernel_name for spec in manifests.values())
        super().__init__(manifest, activity_library=activity_library, l2_cache_bytes=l2_cache_bytes)

    @property
    def kernel_names(self):
        return self._kernel_names

    def _capture_fields(self):
        return self._facts

    def _launch_sample(self, function):
        import torch
        torch.cuda.synchronize()
        function()
        torch.cuda.synchronize()

    def _samples(self, repeats, reset):
        return program_samples(self.last_activity, repeats=repeats)

    def __call__(self, function, *, dry_run_iters, repeat_iters, cold_l2_cache, use_cuda_graph):
        if cold_l2_cache is not True:
            raise ValueError('MACA Program sampling requires its declared cold reset')
        return super().__call__(function, dry_run_iters=dry_run_iters, repeat_iters=repeat_iters,
                                cold_l2_cache=cold_l2_cache, use_cuda_graph=use_cuda_graph)
