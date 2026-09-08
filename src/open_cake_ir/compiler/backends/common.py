"""Shared emitted results, typed refusals and host-wrapper dtype spellings.

Backend-native syntax and capability predicates stay with their implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
import keyword

from ..diagnostics import Finding, FindingCategory
from ..ir import BufferMode, DType, MemorySpace, OperationKind, Schedule


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


_PYTHON_IMPORT_NAMES = frozenset({"tl", "torch", "triton", "cutlass", "cute", "__debug__"})
_GENERATED_PREFIXES = ("N_", "D_", "BLOCK_", "NUM_WARPS", "_work")


def safe_python_identifier(name: str, *, register_route: bool = False) -> bool:
    """One lexical/namespace rule for Python-emitting backend symbols."""
    return (isinstance(name, str) and name.isascii() and name.isidentifier()
            and not keyword.iskeyword(name) and name not in _PYTHON_IMPORT_NAMES
            and not name.startswith(_GENERATED_PREFIXES)
            and (not register_route or ("__" not in name and name not in {"warp", "open_cake_cute_launch"})))


def python_name_findings(schedule: Schedule, *, register_route: bool = False) -> tuple[Finding, ...]:
    """Keep IR labels expressive while refusing unsafe Python source commitments."""
    findings = []
    code = "CUTE_REGISTER_IDENTIFIER" if register_route else "BACKEND_IDENTIFIER_UNSAFE"
    symbols = [("lowering.entry_point", schedule.lowering.entry_point)]
    symbols.extend((f"buffers[{i}].name", value.name) for i, value in enumerate(schedule.buffers))
    for field in ("roles", "allocations", "pipelines", "barriers", "tile_loops"):
        symbols.extend((f"{field}[{i}].name", value.name) for i, value in enumerate(getattr(schedule, field)))
    symbols.extend((f"tile_loops[{i}].iterator", value.iterator) for i, value in enumerate(schedule.tile_loops))
    if schedule.program_map is not None:
        symbols.extend((f"program_map.axes[{i}].name", value.name) for i, value in enumerate(schedule.program_map.axes))
    symbols.extend((f"operations[{i}].id", value.op_id) for i, value in enumerate(schedule.operations))
    for path, name in symbols:
        if not safe_python_identifier(name, register_route=register_route):
            findings.append(refusal(code, path,
                "Python lowering requires ASCII identifiers outside keywords, imports and generated namespaces"))
    if not register_route and (any(ord(char) < 32 or ord(char) == 127 for char in schedule.schedule_id)
            or schedule.schedule_id.splitlines() != [schedule.schedule_id]):
        findings.append(refusal("BACKEND_SOURCE_ID_UNSAFE", "schedule_id",
            "generated source comments require a Schedule id without control characters"))
    globals_ = [(i, value) for i, value in enumerate(schedule.buffers) if value.space is MemorySpace.GLOBAL]
    for i, value in globals_:
        if value.name in {"out", "tensor", "shape", "dtype"} and value.mode is BufferMode.INPUT:
            findings.append(refusal(code, f"buffers[{i}].name",
                f"input name {value.name!r} conflicts with Python wrapper arguments or validation locals"))
        if value.name == schedule.lowering.entry_point:
            findings.append(refusal(code, f"buffers[{i}].name",
                "a global pointer cannot shadow the generated entry point"))
    local_names = {}
    for path, name in symbols:
        if not (path.startswith("buffers[") or path.startswith("program_map.axes[")
                or path.startswith("tile_loops[") and path.endswith(".iterator")):
            continue
        previous = local_names.setdefault(name, path)
        if previous != path:
            findings.append(refusal("BACKEND_IDENTIFIER_COLLISION", path,
                f"{name!r} shares a generated local namespace with {previous}"))
    # Constexpr names normalize buffer and coordinate spellings to uppercase.
    # Distinct declarations must not collapse onto the same generated symbol.
    normalized = {}
    for path, name in symbols:
        if not (path.startswith("buffers[") or path.startswith("program_map.axes[") or
                path.startswith("tile_loops[") and path.endswith(".name")):
            continue
        group = "buffer" if path.startswith("buffers[") else "coordinate"
        key = (group, name.upper())
        previous = normalized.setdefault(key, (path, name))
        if previous[0] != path:
            findings.append(refusal("BACKEND_IDENTIFIER_COLLISION", path,
                f"{name!r} and {previous[1]!r} share generated uppercase symbols"))
    return tuple(findings)
