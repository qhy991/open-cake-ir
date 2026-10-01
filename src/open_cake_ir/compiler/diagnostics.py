"""Canonical typed Compiler diagnostics and blocking dispositions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FindingCategory(str, Enum):
    SCHEDULE_SEMANTICS = "schedule_semantics"
    HARDWARE_CONFORMANCE = "hardware_conformance"
    DATA_CONSISTENCY = "data_consistency"
    PROGRAM_SAFETY = "program_safety"


class FindingSeverity(str, Enum):
    """The three dispositions the paper's harness returns.

    ``BLOCKING`` rejects the candidate with a localized reason, ``REPORT`` describes a
    likely limit, and ``HINT`` suggests a non-blocking improvement. Only ``BLOCKING``
    stops a Schedule from reaching the toolchain.
    """

    BLOCKING = "blocking"
    REPORT = "report"
    HINT = "hint"


# A Schedule with only these blocking Findings is valid Cake IR whose selected
# emission route has no implementation for a declared operation, dtype, access or
# exact target. Keep ownership here with Compiler diagnostics; Lab imports the set
# to route evidence, rather than maintaining another backend capability table.
BACKEND_LOWERING_GAP_CODES = frozenset({
    'BACKEND_TARGET_UNSUPPORTED',
    'BACKEND_DTYPE_UNEMITTABLE',
    'BACKEND_OPERATION_UNEMITTABLE',
    'BACKEND_ACCESS_INDEX_UNSUPPORTED',
    'BACKEND_ARITHMETIC_UNSUPPORTED',
    'BACKEND_CAST_UNSUPPORTED',
    'BACKEND_REDUCTION_UNSUPPORTED',
    'TRITON_BARRIER_UNSUPPORTED',
    'TRITON_ARANGE_RANGE_UNSUPPORTED',
    'TRITON_WARP_SPECIALIZED_ARGMIN_UNSUPPORTED',
    'TRITON_MMA_INSTRUCTION_UNSUPPORTED',
    'TRITON_MMA_K_RANGES_UNSUPPORTED',
    'TRITON_LOOP_STOP_UNSUPPORTED',
    'TRITON_ELEMENTWISE_UNSUPPORTED',
    'TOP_K_SOURCE_UNLOWERABLE',
    'TOP_K_INT32_ACROSS_LOOP_UNLOWERABLE',
    'TOP_K_K_UNLOWERABLE',
    'INDEX_EXPAND_SOURCE_UNLOWERABLE',
    'INDEX_EXPAND_EXTENT_UNLOWERABLE',
    'VALID_EXTENT_ACCESS_UNLOWERABLE',
    'METAL_MMA_K_RANGES_UNSUPPORTED',
    'METAL_INSTRUCTION_UNSUPPORTED',
    'METAL_ELEMENTWISE_UNSUPPORTED',
    'METAL_REDUCTION_UNSUPPORTED',
    'CUTE_MMA_K_RANGES_UNSUPPORTED',
    'CUTE_MMA_INSTRUCTION_UNSUPPORTED',
    'CUTE_ACCESS_SUBRANGE_UNSUPPORTED',
    'CUTE_STATE_UNSUPPORTED',
    'NATIVE_OPERATION_UNSUPPORTED',
    'NATIVE_ARITHMETIC_UNSUPPORTED',
    'NATIVE_BARRIER_UNSUPPORTED',
    'NATIVE_DTYPE_UNSUPPORTED',
})

# These lowering-only backend diagnostics name a declaration the author can change
# without extending Cake IR or implementing a new emitter body.
AUTHOR_FIXABLE_LOWERING_CODES = frozenset({
    'BACKEND_IDENTIFIER_COLLISION',
    'BACKEND_IDENTIFIER_UNSAFE',
    'BACKEND_SOURCE_ID_UNSAFE',
    'BACKEND_MMA_INSTRUCTION_REQUIRED',
    'METAL_ENTRY_POINT_UNSUPPORTED',
    'NATIVE_NAME_UNSUPPORTED',
    'NATIVE_REFERENCE_UNKNOWN',
})


@dataclass(frozen=True)
class Finding:
    """One typed Compiler diagnostic, retained unchanged through Assessment.

    Severity owns whether lowering is blocked. Blocking findings normally also reject
    the Schedule; a backend capability refusal may explicitly leave acceptance intact.
    Reports and hints are advisory and cannot authorize acceptance or performance.

    The constructor keeps the verifier's typed category/severity signature. Callers of
    the former core-only boolean constructor must name the category and express lowering
    disposition through severity; no category is inferred from a diagnostic code.
    Both blocking dispositions remain dataclass fields for existing ``asdict`` consumers,
    but ``blocks_lowering`` is derived and cannot be independently supplied.
    """

    code: str
    path: str
    message: str
    category: FindingCategory
    severity: FindingSeverity = FindingSeverity.BLOCKING
    blocks_acceptance: bool | None = field(default=None, kw_only=True)
    blocks_lowering: bool = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.category, FindingCategory) or not isinstance(
            self.severity, FindingSeverity
        ):
            raise ValueError("Finding category and severity must be typed")
        object.__setattr__(self, "blocks_lowering", self.severity is FindingSeverity.BLOCKING)
        if self.blocks_acceptance is None:
            object.__setattr__(self, "blocks_acceptance", self.blocks_lowering)
        elif not isinstance(self.blocks_acceptance, bool):
            raise ValueError("Finding blocks_acceptance must be a boolean")
        if self.blocks_acceptance and not self.blocks_lowering:
            raise ValueError("non-blocking Findings cannot block acceptance")

    def to_dict(self) -> dict[str, object]:
        """Project the public diagnostic without losing either blocking disposition."""

        return {
            "code": self.code,
            "path": self.path,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "blocks_acceptance": self.blocks_acceptance,
            "blocks_lowering": self.blocks_lowering,
        }

    def __str__(self) -> str:
        return f"[{self.severity.value}] {self.code} at {self.path}: {self.message}"
