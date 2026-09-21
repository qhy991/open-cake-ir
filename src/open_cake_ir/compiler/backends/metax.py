"""The bounded MACA capability checks used by the shared Triton emitter."""

from ..diagnostics import Finding
from ..ir import DType, OperationKind, Schedule
from ..target import Target
from .common import refusal


_BUFFER_DTYPES = frozenset({DType.FP32, DType.FP16, DType.BF16, DType.INT32, DType.FP8_E4M3})


def preflight(schedule: Schedule, target: Target) -> tuple[Finding, ...]:
    findings = []
    for index, buffer in enumerate(schedule.buffers):
        if buffer.dtype not in _BUFFER_DTYPES:
            findings.append(refusal(
                "MACA_DTYPE_UNQUALIFIED", f"buffers[{index}].dtype",
                "this MACA route admits FP32, FP16, BF16, INT32 and E4M3FN buffers; "
                "other tensor representations require their own device qualification",
            ))
    for index, operation in enumerate(schedule.operations):
        if operation.kind is OperationKind.TOP_K:
            source = schedule.buffer(operation.reads[0]) if operation.reads else None
            if (source is None or source.dtype is not DType.FP32
                    or operation.parameters.across_loop):
                findings.append(refusal(
                    "MACA_TOP_K_UNQUALIFIED", f"operations[{index}]",
                    "the MACA top_k route is qualified for a resident FP32 tile; "
                    "integer scores and loop-carried selection require their own device qualification",
                ))
        operands = [schedule.buffer(name) for name in (*operation.reads, *operation.writes)]
        if not any(buffer is not None and buffer.dtype is DType.FP8_E4M3 for buffer in operands):
            continue
        path = f"operations[{index}]"
        if operation.kind is OperationKind.CAST:
            source = schedule.buffer(operation.reads[0]) if operation.reads else None
            # The complete E4M3FN decoding domain is measured. The inverse
            # diagnostic covers finite points only, so it cannot admit arbitrary
            # FP32 values or a different destination format here.
            if (source is None or source.dtype is not DType.FP8_E4M3
                    or operation.parameters.to not in (DType.FP16, DType.FP32)):
                findings.append(refusal(
                    "MACA_FP8_CAST_UNQUALIFIED", path,
                    "the MACA FP8 cast route admits E4M3FN to FP16 or FP32 only; "
                    "other conversion directions have no complete device contract",
                ))
            elif source.is_scalar:
                findings.append(refusal(
                    "MACA_FP8_SCALAR_CAST_UNSUPPORTED", path,
                    "the captured MACA compiler asserts on scalar FP8 conversion; "
                    "this route requires a non-scalar FP8 tile before casting",
                ))
        elif operation.kind not in (OperationKind.LOAD, OperationKind.STORE):
            findings.append(refusal(
                "MACA_FP8_OPERATION_UNQUALIFIED", path,
                "the MACA E4M3FN route admits load, store and decoded FP16/FP32 arithmetic; "
                "arithmetic directly on FP8 values requires separate qualification",
            ))
    if schedule.residency is not None and schedule.residency.registers_per_thread is not None:
        findings.append(refusal(
            "MACA_REGISTER_BUDGET_UNQUALIFIED", "residency.registers_per_thread",
            "the MACA maxnreg option has no verified enforcement contract on this route",
        ))
    if any(loop.range_options.warp_specialize for loop in schedule.tile_loops):
        findings.append(refusal(
            "MACA_WARP_SPECIALIZATION_UNSUPPORTED", "tile_loops",
            "the admitted MACA Triton version has no qualified warp-specialization route",
        ))
    # The captured MACA Triton 3.1 range accepts num_stages only. These options
    # would otherwise reach its JIT as unknown keywords; dropping them would
    # silently change the authored Schedule's performance commitments.
    for index, loop in enumerate(schedule.tile_loops):
        for name, default in (("loop_unroll_factor", 1), ("flatten", False),
                              ("disallow_acc_multi_buffer", False), ("disable_licm", False)):
            if getattr(loop.range_options, name) != default:
                findings.append(refusal(
                    "MACA_LOOP_OPTION_UNSUPPORTED", f"tile_loops[{index}].range_options.{name}",
                    f"the admitted MACA Triton range does not support {name}; "
                    "the backend does not discard a declared loop option",
                ))
    return tuple(findings)
