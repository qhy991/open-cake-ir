"""The fixed current backend inventory; concrete implementations own their rules."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Protocol

from ..diagnostics import Finding
from ..ir import DType, LoweringBackend, OperationKind, Schedule
from ..target import CodeObject, Target
from .common import Emission
from . import cutedsl, metal, native_cuda, triton


class BackendModule(Protocol):
    SUPPORTED_DTYPES: frozenset[DType]
    SUPPORTED_OPERATION_KINDS: frozenset[OperationKind]
    # The code objects this backend's emission compiles to. The Compiler holds a
    # Target's declared code object against it before preflight, so a backend admits
    # targets by what they run and never by a table of their ids.
    CODE_OBJECTS: frozenset[CodeObject]

    def requirements(self, schedule: Schedule) -> tuple[Finding, ...]: ...
    def preflight(self, schedule: Schedule, target: Target) -> tuple[Finding, ...]: ...
    def emit(self, schedule: Schedule, target: Target, *, entry_point: str | None = None) -> Emission: ...


@dataclass(frozen=True)
class Backend:
    module: BackendModule
    source_language: str
    compiler: str
    # Optional constraints on raw containers, checked after IR construction and before
    # canonicalization. None explicitly declares no additional input restriction.
    validate_input: Callable[[Mapping[str, object]], None] | None = None


BACKENDS = MappingProxyType({
    LoweringBackend.NATIVE_CUDA: Backend(native_cuda, "cuda_cpp", "nvcc"),
    LoweringBackend.METAL: Backend(metal, "metal", "MTLDevice.makeLibrary"),
    LoweringBackend.TRITON: Backend(triton, "python", "triton", validate_input=triton.validate_input),
    LoweringBackend.CUTLASS_CUTE_DSL: Backend(cutedsl, "python", "cutlass_cute_dsl"),
})
