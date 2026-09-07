"""Common typed Schedule verification, composed in four fixed contract classes.

Structural admissibility belongs to IR construction. Each rule retains its code,
category, severity and localized meaning; backend dispatch remains with Compiler.
No registry or diagnostic deduplication can replace ownership of a rule.
"""

from __future__ import annotations

from ..diagnostics import Finding, FindingCategory, FindingSeverity
from ..ir import Schedule
from ..target import Target
from . import data_consistency, hardware_conformance, program_safety, schedule_semantics
from ._collector import _Collector
from .hardware_conformance import resolve_grid
from .schedule_semantics import NameConflict, name_conflicts

__all__ = [
    "Finding", "FindingCategory", "FindingSeverity", "NameConflict",
    "name_conflicts", "resolve_grid", "verify",
]


def verify(schedule: Schedule, target: Target) -> tuple[Finding, ...]:
    """Return localized findings for a structurally admissible typed Schedule."""

    out = _Collector()
    schedule_semantics.verify(schedule, out)
    hardware_conformance.verify(schedule, target, out)
    data_consistency.verify(schedule, out)
    program_safety.verify(schedule, out)
    return out.result()
