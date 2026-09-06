"""NCU-aligned static profile envelopes over one Schedule and lowering.

The report is intentionally not a synthetic profiler.  It maps facts already owned by
the IR, Target, work model, and deterministic lowering onto the names an NCU user asks
about.  Exact values and bounds stay numeric; throughput and stall counters remain
unknown or qualitative risk until a target-specific calibration covers them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TYPE_CHECKING

from .compiled_resources import CompiledResources

if TYPE_CHECKING:
    from .core import Lowering

from .analysis import (
    logical_register_pressure_per_thread,
    residency_upper_bound,
    top_k_merge_structure,
    top_k_selection_structure,
)
from .ir import (
    AccessIndexKind,
    MemorySpace,
    OperationKind,
    Schedule,
    TopKParameters,
)
from .target import Target
from .work import WorkBound, work_bound

_ESTIMATE_KINDS = {
    "exact",
    "lower_bound",
    "upper_bound",
    "uncalibrated_risk",
    "unknown",
}
_RISKS = {"low", "medium", "high"}


@dataclass(frozen=True)
class MetricEstimate:
    """One NCU-oriented value with its epistemic status carried beside it."""

    metric: str
    estimate_kind: str
    value: int | float | str | None
    unit: str
    coverage: str
    reasons: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.metric
            or self.estimate_kind not in _ESTIMATE_KINDS
            or not self.unit
            or not self.coverage
            or (
                self.estimate_kind == "uncalibrated_risk"
                and self.value not in _RISKS
            )
            or (
                self.estimate_kind == "unknown"
                and self.value is not None
            )
        ):
            raise ValueError("profile metric estimate differs")

    def as_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "estimate_kind": self.estimate_kind,
            "value": self.value,
            "unit": self.unit,
            "coverage": self.coverage,
            "reasons": list(self.reasons),
            "missing": list(self.missing),
        }


@dataclass(frozen=True)
class ProfileEnvelope:
    """Static/compiled feature report aligned with the retained NCU metric set."""

    schedule_id: str
    target_id: str
    work: Mapping[str, object] | None
    residency: Mapping[str, object] | None
    lowering: Mapping[str, object]
    ncu_metrics: tuple[MetricEstimate, ...]
    abstentions: tuple[str, ...]
    compiled_resources: CompiledResources | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": "ncu_aligned_profile_envelope",
            "schedule_id": self.schedule_id,
            "target_id": self.target_id,
            "work": None if self.work is None else dict(self.work),
            "residency": None if self.residency is None else dict(self.residency),
            "lowering": dict(self.lowering),
            "ncu_metrics": [metric.as_dict() for metric in self.ncu_metrics],
            "abstentions": list(self.abstentions),
            "compiled_resources": (
                None if self.compiled_resources is None else self.compiled_resources.as_dict()
            ),
        }


def _work_document(bound: WorkBound | None) -> Mapping[str, object] | None:
    if bound is None:
        return None
    return {
        "program_tiles": bound.program_tiles,
        "flops": bound.flops,
        "flops_estimate_kind": "exact" if bound.flops_exact else "lower_bound",
        "mma_flops": bound.mma_flops,
        "mma_fraction": bound.mma_fraction,
        "arithmetic_contracts": list(bound.arithmetic_contracts),
        "contended_contract": bound.contended_contract,
        "compulsory_read_bytes": bound.compulsory_read_bytes,
        "compulsory_written_bytes": bound.compulsory_written_bytes,
        "scheduled_transfer_payload": {
            "scope": "explicit IR LOAD/STORE values before access masks; not physical memory traffic",
            "read_bytes_upper_bound": bound.scheduled_read_bytes_upper_bound,
            "written_bytes_upper_bound": bound.scheduled_written_bytes_upper_bound,
            "operations": [
                {"operation": item.operation,
                 "read_bytes_upper_bound": item.read_bytes_upper_bound,
                 "written_bytes_upper_bound": item.written_bytes_upper_bound,
                 "missing": list(item.missing)}
                for item in bound.scheduled_transfers
            ],
        },
        "compulsory_bytes_estimate_kind": (
            "exact" if bound.compulsory_bytes_exact else "upper_bound"
        ),
        "arithmetic_intensity": bound.arithmetic_intensity,
        "uncounted_arithmetic": list(bound.uncounted_arithmetic),
        "operation_repetitions": [
            {
                "operation": row.operation,
                "estimate_kind": row.estimate_kind,
                "whole_grid": row.whole_grid,
                "missing": list(row.missing),
            }
            for row in bound.operation_repetitions
        ],
        "partially_addressed": list(bound.partially_addressed),
    }


def _residency_document(schedule: Schedule, target: Target,
                        compiled_resources: CompiledResources | None = None) -> Mapping[str, object] | None:
    envelope = residency_upper_bound(schedule, target, compiled_resources=compiled_resources)
    if envelope is None:
        return None
    return {
        "ctas_per_sm_upper_bound": envelope.ctas_per_multiprocessor,
        "binding_resource": envelope.binding.resource if envelope.binding else None,
        "coverage": "compiled allocation" if compiled_resources else "Schedule declarations",
        "logical_register_pressure_per_thread": logical_register_pressure_per_thread(
            schedule, target
        ),
        "bounds": [
            {
                "resource": bound.resource,
                "per_cta": bound.per_cta,
                "per_sm": bound.per_multiprocessor,
                "ctas_per_sm": bound.ctas,
                "unit": bound.unit,
            }
            for bound in envelope.bounds
        ],
    }


def _bound_ctas(schedule: Schedule, target: Target, resource: str,
                compiled_resources: CompiledResources | None = None) -> int | None:
    envelope = residency_upper_bound(schedule, target, compiled_resources=compiled_resources)
    if envelope is None:
        return None
    match = next((bound for bound in envelope.bounds if bound.resource == resource), None)
    return None if match is None else match.ctas


def _explicit_allocation_bytes(schedule: Schedule, space: MemorySpace) -> int:
    return sum(
        allocation.size_bytes
        for allocation in schedule.allocations
        if allocation.space is space
    )


def _top_k_features(schedule: Schedule) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for operation in schedule.operations:
        if operation.kind is not OperationKind.TOP_K or not operation.reads:
            continue
        parameters = operation.parameters
        source = schedule.buffer(operation.reads[0])
        if not isinstance(parameters, TopKParameters) or source is None:
            continue
        structure = top_k_merge_structure(schedule, operation)
        selection = (
            top_k_selection_structure(
                parameters.k,
                structure.merge_width,
                parameters.source_tiles_per_merge,
            )
            if structure is not None and parameters.across_loop
            else None
        )
        extent = source.shape[0] if len(source.shape) == 1 else None
        rows.append(
            {
                "operation": operation.op_id,
                "k": parameters.k,
                "source_extent": extent,
                "across_loop": parameters.across_loop,
                "source_tiles_per_merge": (
                    structure.source_tiles_per_merge if structure is not None else None
                ),
                "source_elements_per_merge": (
                    structure.source_elements_per_merge if structure is not None else None
                ),
                "merge_width": structure.merge_width if structure is not None else None,
                "selection_algorithm": (
                    selection.algorithm if selection is not None else None
                ),
                "selection_comparison_model": (
                    selection.comparison_model if selection is not None else None
                ),
                "selection_comparison_estimate_kind": (
                    "toolchain_frontend_model" if selection is not None else None
                ),
                "selection_comparison_model_source": (
                    "Triton v3.7.1 standard.py" if selection is not None else None
                ),
                "baseline_comparison_lane_work": (
                    selection.baseline_comparison_lane_work
                    if selection is not None
                    else None
                ),
                "selected_comparison_lane_work": (
                    selection.selected_comparison_lane_work
                    if selection is not None
                    else None
                ),
                "comparison_lane_work_reduction_fraction": (
                    selection.comparison_lane_reduction_fraction
                    if selection is not None
                    else None
                ),
                "selection_declared_shared_memory_delta_bytes": 0,
                "selection_compiled_shared_memory_delta_bytes": None,
                "selection_compiled_resource_missing": [
                    "matched compiled allocation for selected algorithm and control"
                ],
                "loop_carried_state_elements": (
                    2 * parameters.k if parameters.across_loop else 0
                ),
                "loop_carried_state_bytes": (
                    structure.loop_carried_state_bytes if structure is not None else None
                ),
                "pending_source_key_elements": (
                    structure.pending_source_key_elements if structure is not None else None
                ),
                "pending_source_state_bytes": (
                    structure.pending_source_state_bytes if structure is not None else None
                ),
                "structural_count_estimate_kind": (
                    structure.count_estimate_kind if structure is not None else "unknown"
                ),
                "whole_grid_source_tile_update_count": (
                    structure.whole_grid_source_tile_update_count
                    if structure is not None
                    else None
                ),
                "whole_grid_full_group_merge_count": (
                    structure.whole_grid_full_group_merge_count
                    if structure is not None
                    else None
                ),
                "whole_grid_tail_flush_merge_count": (
                    structure.whole_grid_tail_flush_merge_count
                    if structure is not None
                    else None
                ),
                "whole_grid_merge_update_count": (
                    structure.whole_grid_merge_update_count
                    if structure is not None
                    else None
                ),
                "structural_count_missing": (
                    list(structure.count_missing)
                    if structure is not None
                    else ["top_k merge structure is unavailable"]
                ),
            }
        )
    return rows


def _runtime_indexed_buffers(schedule: Schedule) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                access.buffer
                for access in schedule.access_maps
                if any(index.source is AccessIndexKind.BUFFER for index in access.indices)
            }
        )
    )


def _synchronization_risk(schedule: Schedule, top_k: list[dict[str, object]]) -> tuple[str, ...]:
    reasons: list[str] = []
    if schedule.barriers:
        reasons.append(
            f"{len(schedule.barriers)} explicit barrier declaration(s)"
        )
    for row in top_k:
        if row["across_loop"]:
            if row["source_tiles_per_merge"] == 1:
                reasons.append(
                    f"loop-carried top_k k={row['k']} merge_width={row['merge_width']}"
                )
            else:
                reasons.append(
                    f"loop-carried top_k k={row['k']} "
                    f"source_tiles_per_merge={row['source_tiles_per_merge']} "
                    f"merge_width={row['merge_width']} "
                    f"pending_source_state_bytes={row['pending_source_state_bytes']}"
                )
        else:
            reasons.append(f"resident top_k k={row['k']}")
    if any(operation.kind is OperationKind.SCAN for operation in schedule.operations):
        reasons.append("scan lowering owns a parallel synchronization decomposition")
    if any(
        operation.kind is OperationKind.ONLINE_SOFTMAX
        for operation in schedule.operations
    ):
        reasons.append("online_softmax carries a reduction state across tiles")
    return tuple(reasons)


def _risk(reasons: tuple[str, ...], *, high: bool = False) -> str:
    if not reasons:
        return "low"
    return "high" if high or len(reasons) > 1 else "medium"


def _loop_body_operations(schedule: Schedule) -> frozenset[str]:
    operation_ids = {operation.op_id for operation in schedule.operations}
    return frozenset(
        entry
        for loop in schedule.tile_loops
        for entry in loop.body
        if entry in operation_ids
    )


def _lowering_document(
    schedule: Schedule, lowered_source: str | None
) -> Mapping[str, object]:
    top_k = _top_k_features(schedule)
    source = lowered_source or ""
    return {
        "generated_source_bytes": (
            len(source.encode("utf-8")) if lowered_source is not None else None
        ),
        "top_k": top_k,
        "explicit_barrier_count": len(schedule.barriers),
        "tile_loop_count": len(schedule.tile_loops),
        "global_load_operations": sum(
            operation.kind is OperationKind.LOAD for operation in schedule.operations
        ),
        "global_store_operations": sum(
            operation.kind is OperationKind.STORE for operation in schedule.operations
        ),
        "runtime_indexed_buffers": list(_runtime_indexed_buffers(schedule)),
        "triton_dot_count": source.count("tl.dot("),
        "triton_top_k_count": source.count("tl.topk("),
        "triton_sort_count": source.count("tl.sort("),
        "triton_bitonic_merge_count": source.count("tl.bitonic_merge("),
        "triton_range_count": source.count("tl.range("),
    }


def profile_envelope(
    schedule: Schedule,
    target: Target,
    *,
    lowered_source: str | None = None,
    lowering: "Lowering | None" = None,
    compiled_resources: CompiledResources | None = None,
) -> ProfileEnvelope:
    """Derive one NCU-aligned report without inventing measured percentages."""

    if lowering is not None:
        if lowered_source is not None:
            raise ValueError("provide one lowering authority, not two source representations")
        if lowering.schedule_id != schedule.schedule_id or lowering.target != target.target_id:
            raise ValueError("profile lowering context differs from Schedule or Target")
        lowered_source = lowering.source
    if target.compute_capability is None:
        if compiled_resources is not None:
            raise ValueError("CUDA compiled-resource feedback does not describe a Metal kernel")
        return ProfileEnvelope(
            schedule.schedule_id, target.target_id, _work_document(work_bound(schedule)),
            None, _lowering_document(schedule, lowered_source), (),
            ("This target has no calibrated performance model or occupancy facts; "
             "NVIDIA NCU metrics and CUDA compiled-resource feedback do not apply.",),
            None,
        )
    if compiled_resources is not None:
        if lowering is None:
            raise ValueError("compiled resource feedback requires its complete Lowering context")
        requirements = lowering.toolchain_requirements
        options = requirements.get("compile_options", {})
        compiled_resources.require_context(
            source=lowering.source, target=target.target_id,
            entry_point=str(requirements.get("kernel_entry_point", "")),
            threads_per_cta=int(options.get("num_warps", 0)) * target.warp_size,
        )
    work = work_bound(schedule)
    residency = residency_upper_bound(schedule, target, compiled_resources=compiled_resources)
    register_pressure = logical_register_pressure_per_thread(schedule, target)
    shared_bytes = (
        compiled_resources.shared_bytes if compiled_resources
        else _explicit_allocation_bytes(schedule, MemorySpace.SHARED)
    )
    shared_ctas = _bound_ctas(schedule, target, "shared_memory", compiled_resources)
    warp_ctas = _bound_ctas(schedule, target, "threads", compiled_resources)
    register_ctas = _bound_ctas(schedule, target, "registers", compiled_resources)
    top_k = _top_k_features(schedule)
    synchronization = _synchronization_risk(schedule, top_k)
    runtime_indexed = _runtime_indexed_buffers(schedule)
    contended_contract = work.contended_contract if work is not None else None
    matched_arithmetic_peak = (
        target.peak.for_contract(contended_contract)
        if target.peak is not None and contended_contract is not None
        else None
    )
    throughput_missing = []
    if matched_arithmetic_peak is None:
        throughput_missing.append(
            (
                f"matching arithmetic peak for instruction contract "
                f"{contended_contract!r}"
                if contended_contract is not None
                else "matching arithmetic peak"
            )
        )
    throughput_missing.append("measured or calibrated duration")
    throughput_reasons = []
    if contended_contract is not None:
        throughput_reasons.append(f"instruction contract={contended_contract}")
    throughput_reasons.append(
        f"counted FLOPs={work.flops}" if work is not None else "work domain unavailable"
    )
    occupancy_ctas = residency.ctas_per_multiprocessor if residency else None
    facts = target.occupancy
    active_warps_upper = None
    if facts is not None and occupancy_ctas is not None:
        maximum_warps = facts.maximum_threads_per_multiprocessor // target.warp_size
        active_warps_upper = min(
            100.0,
            100.0
            * occupancy_ctas
            * schedule.total_warp_extent
            / maximum_warps,
        )

    backend_intrinsics = []
    if lowered_source is not None and "tl.dot(" in lowered_source:
        backend_intrinsics.append("tl.dot may allocate implicit shared/register storage")
    if lowered_source is not None and "tl.topk(" in lowered_source:
        backend_intrinsics.append("tl.topk owns an internal bitonic synchronization/storage plan")
    if lowered_source is not None and (
        "tl.sort(" in lowered_source or "tl.bitonic_merge(" in lowered_source
    ):
        backend_intrinsics.append(
            "tl.sort and tl.bitonic_merge own internal synchronization/storage plans"
        )
        backend_intrinsics.append(
            "half-selection compiled resource delta requires matched toolchain artifacts"
        )
    scoreboard_reasons = list(
        f"runtime-indexed global buffer {name}" for name in runtime_indexed
    )
    if occupancy_ctas is not None and occupancy_ctas <= 2:
        scoreboard_reasons.append(
            f"static residency upper bound is {occupancy_ctas} CTA/SM"
        )
    loop_body = _loop_body_operations(schedule)
    if any(
        operation.kind is OperationKind.LOAD and operation.op_id in loop_body
        for operation in schedule.operations
    ):
        scoreboard_reasons.append("global loads execute inside a tiled program")

    metrics = (
        MetricEstimate(
            "launch__registers_per_thread",
            "exact" if compiled_resources else "unknown",
            compiled_resources.registers_per_thread if compiled_resources else None,
            "register/thread",
            "compiled backend allocation",
            reasons=(
                (f"logical register pressure proxy={register_pressure}",)
                if register_pressure
                else ()
            ),
            missing=() if compiled_resources else ("compiled-kernel register allocation",),
        ),
        MetricEstimate(
            "launch__occupancy_limit_registers",
            "upper_bound" if register_ctas is not None else "unknown",
            register_ctas,
            "CTA/SM",
            "compiled backend allocation",
            reasons=(
                (f"logical register pressure proxy={register_pressure}",)
                if register_pressure
                else ()
            ),
            missing=("register allocation granularity",) if compiled_resources else ("ptxas register allocation",),
        ),
        MetricEstimate(
            "launch__occupancy_limit_shared_mem",
            "upper_bound" if shared_ctas is not None else "unknown",
            shared_ctas,
            "CTA/SM",
            "compiled static plus launch dynamic allocation" if compiled_resources else "explicit Schedule shared-memory allocations",
            reasons=((f"{shared_bytes} shared bytes/CTA",) if shared_bytes else ()),
            missing=("allocation granularity and driver reservation",) if compiled_resources else ("backend implicit shared memory",),
        ),
        MetricEstimate(
            "launch__occupancy_limit_blocks",
            "unknown",
            None,
            "CTA/SM",
            "Target structural facts",
            missing=("maximum resident blocks per SM",),
        ),
        MetricEstimate(
            "launch__occupancy_limit_warps",
            "upper_bound" if warp_ctas is not None else "unknown",
            warp_ctas,
            "CTA/SM",
            "Target thread capacity divided by CTA threads",
        ),
        MetricEstimate(
            "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
            "upper_bound" if active_warps_upper is not None else "unknown",
            active_warps_upper,
            "%",
            "static residency and CTA warp extent",
            missing=("scheduler eligibility", "tail effects", "backend resource allocation"),
        ),
        MetricEstimate(
            "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
            "uncalibrated_risk",
            _risk(
                synchronization,
                high=any(row["across_loop"] and int(row["k"]) >= 128 for row in top_k),
            ),
            "risk",
            "IR synchronization and stateful-reduction structure",
            reasons=synchronization,
            missing=("B200 NCU calibration",),
        ),
        MetricEstimate(
            "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
            "uncalibrated_risk",
            _risk(tuple(scoreboard_reasons), high=bool(runtime_indexed)),
            "risk",
            "global dependency and latency-hiding proxies",
            reasons=tuple(scoreboard_reasons),
            missing=("SASS load-use distance", "cache behavior", "B200 NCU calibration"),
        ),
        MetricEstimate(
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "unknown",
            None,
            "%",
            "declared work without a measured duration/rate",
            reasons=tuple(throughput_reasons),
            missing=tuple(throughput_missing),
        ),
        MetricEstimate(
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "unknown",
            None,
            "%",
            "compulsory traffic without a measured duration/rate",
            reasons=(
                f"compulsory bytes={work.compulsory_bytes}"
                if work is not None
                else "traffic domain unavailable",
            ),
            missing=("B200 bandwidth peak", "measured or calibrated duration"),
        ),
        MetricEstimate(
            "lts__throughput.avg.pct_of_peak_sustained_elapsed",
            "unknown",
            None,
            "%",
            "IR access structure has no cache transaction model",
            missing=("working-set residency", "reuse realization", "B200 NCU calibration"),
        ),
    )

    abstentions = [
        "throughput percentages require a measured or calibrated duration and target rate",
        "L2 behavior requires a cache/transaction calibration",
        "stall percentages remain qualitative until B200 NCU coverage exists",
    ]
    if compiled_resources is None:
        abstentions.insert(0, "physical register allocation requires a compiled artifact")
    else:
        abstentions.extend((
            "residency remains an upper bound without allocation granularity, carveout, barriers and implicit tensor memory",
            "stack and local allocation are static bytes, not dynamic spill traffic",
        ))
    if backend_intrinsics:
        abstentions.extend(backend_intrinsics)
    return ProfileEnvelope(
        schedule.schedule_id,
        target.target_id,
        _work_document(work),
        _residency_document(schedule, target, compiled_resources),
        _lowering_document(schedule, lowered_source),
        metrics,
        tuple(abstentions),
        compiled_resources,
    )
