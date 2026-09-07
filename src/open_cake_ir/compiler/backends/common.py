"""Vocabulary shared by every backend emitter.

Small on purpose. The backends differ because their targets differ -- one places
operands explicitly and the other leaves that to the compiler -- so what they share is
the shape of a result and the way they refuse, not a framework. Extracting more would
build one for two instances.
"""

from __future__ import annotations

from dataclasses import dataclass


class EmitError(ValueError):
    """The Schedule does not determine the source to emit."""


@dataclass(frozen=True)
class BackendPrecondition:
    """One backend-owned reason a well-formed Schedule cannot be emitted."""

    code: str
    path: str
    message: str


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
