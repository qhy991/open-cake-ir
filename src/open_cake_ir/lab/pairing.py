"""Bindings for same-backend known-baseline optimization on the existing Lab path."""
from __future__ import annotations

import json
from typing import Mapping

from open_cake_ir.compiler.toolchain import project_triton_kernel


def comparison_arm(arms: Mapping[str, object]) -> str:
    if set(arms) == {'open_cake', 'direct_cuda'}:
        return 'direct_cuda'
    if set(arms) == {'open_cake', 'native_triton'}:
        return 'native_triton'
    raise ValueError('matched_search requires Open Cake and one explicitly declared comparison environment')


def bind_baseline(schedule: Mapping[str, object], workload, case_id: str) -> dict:
    """Bind an already shaped baseline; the Workload ABI is the only tensor owner."""
    document = json.loads(json.dumps(schedule))
    abi = workload.tensor_abi(case_id)
    buffers = [buffer for buffer in document['buffers'] if buffer['space'] == 'global']
    if [(b['name'], tuple(b['shape']), b['dtype'], b['mode']) for b in buffers] != [
            (arg.name, arg.shape, arg.dtype, arg.mode) for arg in abi]:
        raise ValueError('baseline Schedule must already match the selected Workload ABI; use baseline preparation for another shape')
    if document.get('lowering', {}).get('backend') != 'triton':
        raise ValueError('paired baseline requires the explicit Triton lowering route')
    document['metadata']['workload_contract_sha256'] = workload.canonical_sha256
    return document


def native_baseline(lowering) -> dict:
    """Explicitly project the same Compiler kernel, retaining its launch decisions."""
    requirements = lowering.toolchain_requirements
    return {'kernel_source': project_triton_kernel(lowering.source.encode(), requirements).decode(),
            'compile_constants': dict(requirements['compile_constants']),
            'compile_options': dict(requirements['compile_options']), 'grid': list(requirements['grid'])}


def triton_optimization_analysis_plan() -> dict:
    return {
        'experimental_unit': 'run', 'target_population': 'prescheduled_runs_under_exact_campaign_lock',
        'primary_endpoint': ['qualified_by_budget', 'best_confirmed_latency_ms_if_qualified'],
        'contrast': 'two_part_open_cake_vs_native_triton',
        'estimand': 'terminal-budget qualification-rate difference and conditional confirmed performance from the same known baseline',
        'treatment': {
            'open_cake': 'Schedule_or_restricted_Python_IR_to_frozen_Triton',
            'native_triton': 'kernel_only_Triton_same_frozen_backend',
            'reference_access': 'known_baseline_optimization',
            'baseline': 'same_Compiler_lowering_kernel_and_compile_launch_metadata',
            'held_fixed': ['workload', 'case', 'oracle', 'backend', 'toolchain', 'launch_abi', 'provider', 'scaffold', 'budget', 'evaluation'],
        },
        'missingness': {'candidate_failure': 'observed_outcome', 'external_fault': 'missing', 'replacement': 'forbidden'},
        'pooling': 'forbidden_without_successor_analysis_plan',
        'availability': 'all_prescheduled_runs_observed_and_each_arm_has_qualified_run',
        'summary_statistics': {
            'qualification': 'arm_rate', 'qualification_contrast': 'open_cake_rate_minus_native_triton_rate',
            'conditional_latency': 'arm_median_ms', 'contrast': 'native_triton_median_divided_by_open_cake_median',
            'uncertainty': 'per_arm_observed_range_ms',
        },
        'direction': 'lower_latency_is_better',
        'confirmation': 'fresh_fixed_candidate_correctness_then_paired_cupti',
        'threshold_view': 'descriptive_first_fresh_confirmation_from_retained_events_no_search_latency_substitution',
    }
