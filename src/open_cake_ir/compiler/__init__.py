"""Public Compiler Interface.

The Compiler owns four things, in the order a Schedule meets them:

* `ir` -- the canonical typed Schedule. Structural admissibility has one owner.
* `target` -- the exact hardware contract a Schedule is verified against.
* `verifier` -- target-derived hard gates, grouped by the four contract classes the
  paper's harness reports, each finding naming a path and a violated contract.
* `backends` -- native CUDA, Triton, CuTe-DSL and Metal emission with shared result/error types.

`Compiler.assess` composes the first three; `Compiler.lower` dispatches accepted Schedules
to deterministic target emitters. Direct emission uses an explicit backend, for example
`from open_cake_ir.compiler.backends.triton import emit`; there is no generic emitter.
"""

from .core import (
    Assessment,
    Compiler,
    Lowering,
)
from .passes import FusionResult, SpecializationResult
from .corpus import CorpusCaseReport, CorpusGateReport
from .errors import CompilerError, LoweringRefusedError
from .backends.common import EmitError, Emission
from .performance.compiled_resources import CompiledResources
from .performance.empirical_cost import EmpiricalCostModel
from .ir import LoweringBackend, LoweringRoute, Schedule, ScheduleParseError
from .performance.profile import MetricEstimate, ProfileEnvelope, profile_envelope
from .target import Target, TargetParseError
from .diagnostics import Finding, FindingCategory, FindingSeverity
from .verifier import verify

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
    "FusionResult",
    "SpecializationResult",
    "Lowering",
    "LoweringRefusedError",
    "LoweringBackend",
    "LoweringRoute",
    "MetricEstimate",
    "ProfileEnvelope",
    "Schedule",
    "ScheduleParseError",
    "Target",
    "TargetParseError",
    "profile_envelope",
    "verify",
]
