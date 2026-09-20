"""Lowered static Programs, independent of storage allocation and execution."""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from .ir.program import Program
if TYPE_CHECKING:
    from .core import Lowering

@dataclass(frozen=True)
class LoweredProgram:
    program: Program
    compiler_revision_id: str
    lowerings: tuple[Lowering, ...]
