"""Shared emitted results, typed refusals and host-wrapper dtype spellings.

Backend-native syntax and capability predicates stay with their implementations.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..diagnostics import Finding, FindingCategory
from ..ir import DType, OperationKind, Schedule


class EmitError(ValueError):
    """The Schedule does not determine the source to emit."""




@dataclass(frozen=True)
class Emission:
    source: str
    entry_point: str
    constants: dict[str, object]
    """Every value derived from the Schedule, so a test can compare them against the
    constants a hand-written artifact would have carried."""

    toolchain: dict[str, object] | None = None
    """What the backend needs to compile this source, when the source alone does not
    imply it. Triton compiles a kernel function against an explicit signature and
    constexpr set; CuTe-DSL compiles the module."""


def require(condition: object, message: str) -> None:
    """Refuse rather than invent a decision the Schedule declined to make."""

    if not condition:
        raise EmitError(message)


def refusal(code: str, path: str, message: str) -> Finding:
    """One backend refusal blocks lowering while preserving IR acceptance."""
    return Finding(code, path, message, FindingCategory.HARDWARE_CONFORMANCE,
                   blocks_acceptance=False)


TORCH_DTYPES = {
    DType.BF16: "torch.bfloat16",
    DType.FP16: "torch.float16",
    DType.FP32: "torch.float32",
    DType.FP8_E4M3: "torch.float8_e4m3fn",
    DType.INT32: "torch.int32",
}


def vocabulary_findings(
    schedule: Schedule, dtypes: frozenset[DType], operation_kinds: frozenset[OperationKind]
) -> tuple[Finding, ...]:
    """Check the representations each concrete backend actually implements."""
    backend = schedule.lowering.backend.value
    findings = []
    for index, buffer in enumerate(schedule.buffers):
        if buffer.dtype not in dtypes:
            findings.append(refusal(
                "BACKEND_DTYPE_UNEMITTABLE", f"buffers[{index}].dtype",
                f"backend {backend!r} cannot name dtype {buffer.dtype.value!r}",
            ))
    for index, operation in enumerate(schedule.operations):
        if operation.kind not in operation_kinds:
            findings.append(refusal(
                "BACKEND_OPERATION_UNEMITTABLE", f"operations[{index}].kind",
                f"backend {backend!r} has no body for operation kind {operation.kind.value!r}",
            ))
    return tuple(findings)
