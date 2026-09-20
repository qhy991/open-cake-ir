"""The one place the Evaluation and Lab layers consult per declared code object.

A Target declares `code_object`; that declaration selects how a candidate is built,
admitted, launched, timed, attributed and profiled. Before this module those facts were
re-spelled in seven independent string vocabularies -- build-role sets in artifacts.py,
paired-policy kinds in paired.py, EvaluationProtocol.timing, job-id prefixes in
attempts.py and local_broker.py, a host-kind ladder in lab/executor.py, a toolchain-kind
ladder in lab/runtime_config.py, and a private three-row table in tasks/evaluate.py --
and CUDA was the fall-through in several of them. A new platform was therefore a dozen
shared edits. Here it is one row, and an object no row implements is refused by name.

The rows carry facts, not vendor code: drivers, benchmarks, observation readers and host
validators stay in their own modules and are named from a row or held equal to the row
set by a contract test (the Lab cannot be imported from here; ADR 0055 keeps task code
out of this layer). `evaluate` callables are registered from the task side.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from open_cake_ir.compiler.target import CodeObject, Target, TargetParseError, declared_target


@dataclass(frozen=True)
class ExecutionPlatform:
    code_object: CodeObject
    # The `kind` a host capture under runtime/hosts/ declares. None is the explicitly
    # named pre-kind CUDA form that runtime/hosts/sm_103a.json still carries; it is a row,
    # not a fall-through, and retires when that host is recaptured.
    host_kind: str | None
    # Roles a build of this object produces beyond the arm-owned source role.
    build_roles: frozenset[str]
    # Every role a sealed candidate for this object may carry.
    allowed_artifact_roles: frozenset[str]
    # The `abi` a sealed Workload-tensor launch manifest for this object declares. Two
    # spellings exist and each is parsed exactly as it always was; the row says which one
    # a candidate for this object seals, so nothing decides that from the manifest's type.
    launch_abi: str
    # Which measurement source a timed assay on this object names, and the paired
    # policy kinds that name it. None means no timer is declared: a Study for the target
    # carries a measurement-coverage limitation instead of a paired assay.
    measurement_source: str | None
    paired_kinds: frozenset[str]
    # The EvaluationProtocol.timing value a paired assay on this object declares.
    protocol_timing: str | None
    # How many times a paired cohort calls the route, when the source declares it here
    # (Metal's count lives beside the observer snapshot it must not drift from).
    route_calls_per_cohort: int | None
    # Job-id prefixes: the allocator that issued the job, not the API the candidate uses.
    # `exclusive` is the cluster allocator (gpuq, exclusive mode); `local` is the local
    # broker (local_serialized mode). D6: a CUDA device outside the cluster is reached
    # through the local broker like Metal and HIP, under its own prefix.
    exclusive_job_prefix: str | None
    local_job_prefix: str | None
    # Where a profile comes from: a separate supervised entry (CUDA's NCU child) or the
    # platform's own activity taken inside evaluate (Metal's observer, HIP's roctracer).
    attribution: str
    profiled_child: bool


_ROWS = (
    ExecutionPlatform(
        code_object=CodeObject.CUBIN,
        host_kind=None,
        build_roles=frozenset({"compiler_expanded_source", "ptx", "cubin", "launch_manifest"}),
        allowed_artifact_roles=frozenset({
            "authored_source", "lowered_source", "compiler_expanded_source", "ttir", "ttgir",
            "llir", "ptx", "cubin", "sass", "toolchain_resource_report", "launch_manifest", "kernel_bundle", "stage_compilation",
        }),
        launch_abi="workload_tensors_v1",
        measurement_source="cupti",
        paired_kinds=frozenset({"fixed_baseline_paired_cupti_v1"}),
        protocol_timing="paired_cupti",
        route_calls_per_cohort=6 + 11 + 25,
        exclusive_job_prefix="gpuq",
        local_job_prefix="cuda",
        attribution="separate",
        profiled_child=True,
    ),
    ExecutionPlatform(
        code_object=CodeObject.METAL_BINARY_ARCHIVE,
        host_kind="metal",
        build_roles=frozenset({"metal_binary_archive", "metal_build_report", "launch_manifest"}),
        allowed_artifact_roles=frozenset({
            "authored_source", "lowered_source", "metal_binary_archive", "metal_build_report",
            "launch_manifest",
        }),
        launch_abi="metal_workload_tensors_v1",
        measurement_source="metal",
        paired_kinds=frozenset({"fixed_baseline_paired_metal_v1", "fixed_baseline_paired_metal_v2"}),
        protocol_timing="paired_metal",
        route_calls_per_cohort=None,
        exclusive_job_prefix=None,
        local_job_prefix="metal",
        attribution="inside_evaluate",
        profiled_child=False,
    ),
    ExecutionPlatform(
        code_object=CodeObject.HSACO,
        host_kind="hip",
        build_roles=frozenset({"compiler_expanded_source", "amdgcn", "hsaco", "launch_manifest"}),
        allowed_artifact_roles=frozenset({
            "authored_source", "lowered_source", "compiler_expanded_source", "ttir", "ttgir",
            "llir", "amdgcn", "hsaco", "toolchain_resource_report", "launch_manifest", "kernel_bundle",
        }),
        launch_abi="workload_tensors_v1",
        measurement_source="hip_dispatch",
        paired_kinds=frozenset({"fixed_baseline_paired_hip_dispatch_v1"}),
        protocol_timing="paired_hip",
        route_calls_per_cohort=11 + 25,
        exclusive_job_prefix=None,
        local_job_prefix="hip",
        attribution="inside_evaluate",
        profiled_child=False,
    ),
    ExecutionPlatform(
        code_object=CodeObject.MCFATBIN,
        host_kind="maca",
        build_roles=frozenset({"compiler_expanded_source", "ttir", "ttgir", "mcfatbin", "launch_manifest"}),
        allowed_artifact_roles=frozenset({
            "authored_source", "lowered_source", "compiler_expanded_source", "ttir", "ttgir",
            "mcfatbin", "launch_manifest", "kernel_bundle",
        }),
        launch_abi="workload_tensors_v1",
        measurement_source=None,
        paired_kinds=frozenset(),
        protocol_timing=None,
        route_calls_per_cohort=None,
        exclusive_job_prefix=None,
        local_job_prefix="maca",
        attribution="unavailable",
        profiled_child=False,
    ),
)

PLATFORMS: Mapping[CodeObject, ExecutionPlatform] = MappingProxyType(
    {row.code_object: row for row in _ROWS}
)


def platform_for(target: object) -> ExecutionPlatform:
    """The row for one exact target id, a Target, or a code object; refused by name."""
    if isinstance(target, ExecutionPlatform):
        return target
    if isinstance(target, CodeObject):
        code_object = target
    elif isinstance(target, Target):
        code_object = target.code_object
    else:
        try:
            code_object = (CodeObject(target) if isinstance(target, str) and target in {c.value for c in CodeObject}
                           else declared_target(target).code_object)
        except (TargetParseError, ValueError, TypeError) as error:
            raise ValueError(
                f"{target!r} names no declared target and no code object; each execution "
                "platform is one row keyed by the object a Target declares"
            ) from error
    try:
        return PLATFORMS[code_object]
    except KeyError as error:
        raise ValueError(
            f"no execution platform implements the {code_object.value!r} code object"
        ) from error


def platform_for_host_kind(kind: str | None) -> ExecutionPlatform:
    """The row whose host captures declare `kind`; None is the pre-kind CUDA form."""
    for row in _ROWS:
        if row.host_kind == kind:
            return row
    raise ValueError(f"Executor host kind {kind!r} is not admitted")


def platform_for_paired_kind(kind: object) -> ExecutionPlatform:
    """The row whose declared paired assay names `kind`; a kind no row declares is refused."""
    for row in _ROWS:
        if kind in row.paired_kinds:
            return row
    raise ValueError(f"paired assay {kind!r} is declared by no execution platform")
