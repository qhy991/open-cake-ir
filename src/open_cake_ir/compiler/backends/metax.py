"""The bounded MACA capability checks used by the shared Triton emitter."""

from ..diagnostics import Finding
from ..ir import DType, Schedule
from ..target import Target
from .common import refusal


def preflight(schedule: Schedule, target: Target) -> tuple[Finding, ...]:
    findings = []
    for index, buffer in enumerate(schedule.buffers):
        if buffer.dtype is not DType.FP32:
            findings.append(refusal(
                "MACA_DTYPE_UNQUALIFIED", f"buffers[{index}].dtype",
                "this MACA route is bounded to FP32 buffers; mixed-precision and integer "
                "tensor ABIs require their own device qualification",
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
    return tuple(findings)
