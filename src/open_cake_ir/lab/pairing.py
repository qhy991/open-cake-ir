"""Bindings for same-backend known-baseline optimization on the existing Lab path."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

from open_cake_ir.compiler.toolchain import project_triton_kernel


@dataclass(frozen=True)
class NativeBackend:
    """One owner for each same-backend treatment's public spelling and adapters."""

    backend: str
    arm: str
    label: str
    media_type: str
    submit_tool: str
    document: str
    baseline_file: str
    analysis_version: str
    hidden_null_pointer_parameters: int
    candidate_schema: str | None = None

    @property
    def authoring_file(self):
        return self.document.lower().removesuffix(".md").replace("_", "-") + "-authoring.md"

    def isolated_compiler(self, config):
        if self.backend == "triton":
            from .triton_build import IsolatedTritonCompiler
            return IsolatedTritonCompiler(**config)
        from .cute_build import IsolatedCuTeCompiler
        return IsolatedCuTeCompiler(**config)

    def builder(self, *, workload, case_id, isolated_compiler):
        if self.backend == "triton":
            from .environments import TritonToolchainBuilder
            cls = TritonToolchainBuilder
        else:
            from .cute_build import CuTeToolchainBuilder
            cls = CuTeToolchainBuilder
        return cls(workload=workload, case_id=case_id, isolated_compiler=isolated_compiler)

    def environment(self, builder, **kwargs):
        from .environments import NativeCuTeEnvironment, NativeTritonEnvironment
        cls = NativeTritonEnvironment if self.backend == "triton" else NativeCuTeEnvironment
        return cls(builder, **kwargs)


_NATIVE_BACKENDS = (
    NativeBackend("triton", "native_triton", "Triton", "application/vnd.open-cake.triton+json",
                  "submit_triton_kernel", "PAIRED_TRITON.md", "candidate-baseline.triton.json",
                  "triton_optimization_v1", 2),
    NativeBackend("cutlass_cute_dsl", "native_cute_dsl", "CuTeDSL", "application/vnd.open-cake.cute+json",
                  "submit_cute_kernel", "PAIRED_CUTE.md", "candidate-baseline.cute.json",
                  "cute_optimization_v1", 0, "contracts/providers/native-cute-candidate-v1.schema.json"),
)


def backend_policy(backend: str) -> NativeBackend:
    for policy in _NATIVE_BACKENDS:
        if policy.backend == backend:
            return policy
    raise ValueError("unsupported paired lowering backend")


def native_backend(comparison: str | None) -> NativeBackend | None:
    if comparison in {None, "direct_cuda"}:
        return None
    for policy in _NATIVE_BACKENDS:
        if policy.arm == comparison:
            return policy
    raise ValueError("unsupported comparison environment")


def native_source(source: bytes, requirements: Mapping[str, object]) -> bytes:
    """Project/admit the exact kernel under the frozen backend's source contract."""
    policy = backend_policy(requirements.get("compiler"))
    if policy.backend == "triton":
        return project_triton_kernel(source, requirements)
    from open_cake_ir.compiler.cute_toolchain import validate_cute_kernel
    validate_cute_kernel(source, requirements)
    return source


def native_block(requirements: Mapping[str, object]) -> list[int]:
    policy = backend_policy(requirements.get("compiler"))
    return ([requirements["compile_options"]["num_warps"] * 32, 1, 1]
            if policy.backend == "triton" else list(requirements["block"]))


def comparison_arm(arms: Mapping[str, object]) -> str | None:
    if set(arms) == {"open_cake"}:
        return None
    if set(arms) == {'open_cake', 'direct_cuda'}:
        return 'direct_cuda'
    for policy in _NATIVE_BACKENDS:
        if set(arms) == {'open_cake', policy.arm}:
            return policy.arm
    raise ValueError('matched_search requires Open Cake and one explicitly declared comparison environment')


def bind_baseline(schedule: Mapping[str, object], workload, case_id: str, *, backend: str | None = None) -> dict:
    """Bind an already shaped baseline; the Workload ABI is the only tensor owner."""
    document = json.loads(json.dumps(schedule))
    if document.get('target') != workload.document['semantics'].get('target'):
        raise ValueError('baseline Schedule target differs from the Workload target')
    abi = workload.tensor_abi(case_id)
    buffers = [buffer for buffer in document['buffers'] if buffer['space'] == 'global']
    if [(b['name'], tuple(b['shape']), b['dtype'], b['mode']) for b in buffers] != [
            (arg.name, arg.shape, arg.dtype, arg.mode) for arg in abi]:
        raise ValueError('baseline Schedule must already match the selected Workload ABI; use baseline preparation for another shape')
    backend = backend or document.get('lowering', {}).get('backend')
    if backend not in {'triton', 'metal', 'cutlass_cute_dsl'} or document.get('lowering', {}).get('backend') != backend:
        raise ValueError('baseline differs from the explicitly requested lowering backend')
    document['metadata']['workload_contract_sha256'] = workload.canonical_sha256
    return document


def native_baseline(lowering) -> dict:
    """Explicitly project the same Compiler kernel, retaining its launch decisions."""
    requirements = lowering.toolchain_requirements
    policy = backend_policy(requirements.get("compiler"))
    source = native_source(lowering.source.encode(), requirements).decode()
    if policy.backend == "triton":
        return {'kernel_source': source, 'compile_constants': dict(requirements['compile_constants']),
                'compile_options': dict(requirements['compile_options']), 'grid': list(requirements['grid'])}
    return {'kernel_source': source, 'grid': list(requirements['grid']),
            'block': list(requirements['block']),
            'dynamic_shared_memory_bytes': requirements['dynamic_shared_memory_bytes']}


def native_optimization_analysis_plan(comparison: str) -> dict:
    policy = native_backend(comparison)
    if policy is None:
        raise ValueError("native optimization analysis requires a native comparison")
    return {
        'experimental_unit': 'run', 'target_population': 'prescheduled_runs_under_exact_campaign_lock',
        'primary_endpoint': ['qualified_by_budget', 'best_confirmed_latency_ms_if_qualified'],
        'contrast': f'two_part_open_cake_vs_{policy.arm}',
        'estimand': 'terminal-budget qualification-rate difference and conditional confirmed performance from the same known baseline',
        'treatment': {
            'open_cake': f'Schedule_or_restricted_Python_IR_to_frozen_{policy.label}',
            policy.arm: f'kernel_only_{policy.label}_same_frozen_backend',
            'reference_access': 'known_baseline_optimization',
            'baseline': 'same_Compiler_lowering_kernel_and_compile_launch_metadata',
            'held_fixed': ['workload', 'case', 'oracle', 'backend', 'toolchain', 'launch_abi', 'provider', 'scaffold', 'budget', 'evaluation'],
        },
        'missingness': {'candidate_failure': 'observed_outcome', 'external_fault': 'missing', 'replacement': 'forbidden'},
        'pooling': 'forbidden_without_successor_analysis_plan',
        'availability': 'all_prescheduled_runs_observed_and_each_arm_has_qualified_run',
        'summary_statistics': {
            'qualification': 'arm_rate', 'qualification_contrast': f'open_cake_rate_minus_{policy.arm}_rate',
            'conditional_latency': 'arm_median_ms', 'contrast': f'{policy.arm}_median_divided_by_open_cake_median',
            'uncertainty': 'per_arm_observed_range_ms',
        },
        'direction': 'lower_latency_is_better',
        'confirmation': 'fresh_fixed_candidate_correctness_then_paired_cupti',
        'threshold_view': 'descriptive_first_fresh_confirmation_from_retained_events_no_search_latency_substitution',
    }


def triton_optimization_analysis_plan() -> dict:
    """Preserve the historical public Triton analysis contract."""
    return native_optimization_analysis_plan("native_triton")


def matched_run_arms(arms: Mapping[str, object], claim_scope: str) -> list[str]:
    """The single authoring environment is confined to non-comparative optimization."""
    comparison = comparison_arm(arms)
    if comparison is None:
        if claim_scope != "artifact_optimization_only":
            raise ValueError("one Authoring Environment requires artifact_optimization_only")
        return ["open_cake"]
    names = [comparison, "open_cake"]
    return sorted(names if claim_scope in {"artifact_optimization_only", "system_qualification_only"} else names * 3)
