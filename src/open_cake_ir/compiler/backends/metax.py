"""The bounded MACA capability checks used by the shared Triton emitter."""

from ..diagnostics import Finding
from ..ir import DType, Schedule
from ..target import Target
from .common import refusal


_BUFFER_DTYPES = frozenset({DType.FP32, DType.FP16, DType.BF16, DType.INT32})


def preflight(schedule: Schedule, target: Target) -> tuple[Finding, ...]:
    findings = []
    for index, buffer in enumerate(schedule.buffers):
        if buffer.dtype not in _BUFFER_DTYPES:
            findings.append(refusal(
                "MACA_DTYPE_UNQUALIFIED", f"buffers[{index}].dtype",
                "this MACA route admits FP32, FP16, BF16 and INT32 buffers; "
                "other tensor representations require their own device qualification",
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
