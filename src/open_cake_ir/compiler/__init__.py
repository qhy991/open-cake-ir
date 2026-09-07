"""Public Compiler Interface.

The Compiler owns four things, in the order a Schedule meets them:

* `ir` -- the canonical typed Schedule. Structural admissibility has one owner.
* `target` -- the exact hardware contract a Schedule is verified against.
* `verifier` -- target-derived hard gates, grouped by the four contract classes the
  paper's harness reports, each finding naming a path and a violated contract.
* `emit_cutedsl` -- lowering that derives what the artifact used to hardcode.

`Compiler.assess` composes the first three; `Compiler.lower` dispatches accepted Schedules
to deterministic target emitters.
"""

from .core import (
    Assessment,
    Compiler,
    CompilerError,
    CorpusCaseReport,
    CorpusGateReport,
    Lowering,
)
from .emit_cutedsl import EmitError, Emission, emit
from .compiled_resources import CompiledResources
from .empirical_cost import EmpiricalCostModel
from .ir import LoweringBackend, LoweringRoute, Schedule, ScheduleParseError
from .profile_model import MetricEstimate, ProfileEnvelope, profile_envelope
from .target import Target, TargetParseError
from .verifier import Finding, FindingCategory, FindingSeverity, verify

__all__ = [
    "Assessment",
    "Compiler",
    "CompilerError",
    "CompiledResources",
    "CorpusCaseReport",
    "CorpusGateReport",
    "EmitError",
    "EmpiricalCostModel",
    "Emission",
    "Finding",
    "FindingCategory",
    "FindingSeverity",
    "Lowering",
    "LoweringBackend",
    "LoweringRoute",
    "MetricEstimate",
    "ProfileEnvelope",
    "Schedule",
    "ScheduleParseError",
    "Target",
    "TargetParseError",
    "emit",
    "profile_envelope",
    "verify",
]
