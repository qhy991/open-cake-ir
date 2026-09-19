"""Bindings for same-backend known-baseline optimization on the existing Lab path."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Protocol

from .native_triton_adapter import TritonNativeAdapter
from .native_cute_adapter import CuTeNativeAdapter


class NativeAdapter(Protocol):
    """Backend-owned build factories and projection of its native candidate format."""

    def isolated_compiler(self, config: Mapping[str, object]) -> object: ...
    def builder(self, *, workload: object, case_id: str, isolated_compiler: object) -> object: ...
    def environment(self, builder: object, **kwargs: object) -> object: ...
    def source(self, source: bytes, requirements: Mapping[str, object]) -> bytes: ...
    def block(self, requirements: Mapping[str, object], *, warp_size: int) -> list[int]: ...
    def baseline(self, source: str, requirements: Mapping[str, object]) -> dict: ...


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
    adapter: NativeAdapter
    candidate_schema: str | None = None

    @property
    def authoring_file(self):
        return self.document.lower().removesuffix(".md").replace("_", "-") + "-authoring.md"

    def isolated_compiler(self, config):
        return self.adapter.isolated_compiler(config)

    def builder(self, *, workload, case_id, isolated_compiler):
        return self.adapter.builder(workload=workload, case_id=case_id,
                                    isolated_compiler=isolated_compiler)

    def environment(self, builder, **kwargs):
        return self.adapter.environment(builder, **kwargs)


_NATIVE_BACKENDS = (
    NativeBackend("triton", "native_triton", "Triton", "application/vnd.open-cake.triton+json",
                  "submit_triton_kernel", "PAIRED_TRITON.md", "candidate-baseline.triton.json",
                  "triton_optimization_v1", 2, TritonNativeAdapter()),
    NativeBackend("cutlass_cute_dsl", "native_cute_dsl", "CuTeDSL", "application/vnd.open-cake.cute+json",
                  "submit_cute_kernel", "PAIRED_CUTE.md", "candidate-baseline.cute.json",
                  "cute_optimization_v1", 0, CuTeNativeAdapter(), "contracts/providers/native-cute-candidate-v1.schema.json"),
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
    return policy.adapter.source(source, requirements)


def native_block(requirements: Mapping[str, object], *, warp_size: int) -> list[int]:
    """Threads per block a Triton route's `num_warps` commits to, at this target's width.

    `num_warps` counts role slots, not threads, so the width is the Target's to declare
    and this function's to be told. It used to read a literal 32, which is a fourth copy
    of a fact `Target.warp_size` already owns and the only one that could not be corrected
    by fixing the Target document. Measured on gfx938, whose document declares 64: a
    baseline that launched at 64 threads was compared against an expected 32 and the
    fixed-baseline admission gate refused it -- a correct baseline rejected in a message
    about the Compiler kernel, which is the shape a borrowed constant always takes.
    """
    if not isinstance(warp_size, int) or isinstance(warp_size, bool) or warp_size <= 0:
        raise ValueError("a block width needs the target's declared role-slot width")
    policy = backend_policy(requirements.get("compiler"))
    return policy.adapter.block(requirements, warp_size=warp_size)


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
    if backend is None:
        backend = backend_policy(document.get('lowering', {}).get('backend')).backend
    if backend not in {'triton', 'metal', 'cutlass_cute_dsl'} or document.get('lowering', {}).get('backend') != backend:
        raise ValueError('baseline differs from the explicitly requested lowering backend')
    document['metadata']['workload_contract_sha256'] = workload.canonical_sha256
    return document


def native_baseline(lowering) -> dict:
    """Explicitly project the same Compiler kernel, retaining its launch decisions."""
    requirements = lowering.toolchain_requirements
    policy = backend_policy(requirements.get("compiler"))
    source = native_source(lowering.source.encode(), requirements).decode()
    return policy.adapter.baseline(source, requirements)


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
