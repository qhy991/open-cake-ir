"""Send each diagnosis to the thing that has to change.

The paper's fourth stage routes a result to the candidate, the verifier, the cost model or
the IR vocabulary according to the diagnosis. Without it every failure reads the same way
-- as the candidate's fault -- and the outer loop that turns recurring failures into new
rules has nothing to accumulate.

Each destination is inferred from a signal the loop already produces, not from a new one:

* **candidate** -- a gate refused it, or the set repeated itself. The Schedule is wrong,
  or two of them are one program under two names, and its author can see why either way.
* **verifier** -- every gate passed and the toolchain refused Compiler-produced source. Something was true of
  this Schedule that the pre-compile gates do not model, which is a missing rule rather
  than a bad candidate.
* **backend_lowering** -- Cake IR admits the Schedule, but the selected backend has
  no implementation for its declared instruction, dtype, access or exact target.
  This is a Compiler evolution candidate, not a reason to corrupt the Schedule.
* **backend_triage** -- Cake IR admits the Schedule but a lowering-only Finding has
  no reviewed owner yet, or several owners are mixed. Preserve it for a maintenance
  agent instead of blaming the author by default.
* **ir_vocabulary** -- a typed Compiler Finding explicitly establishes that a physical
  decision is not expressible in Cake IR. Generic emitter errors do not prove that.
* **cost_model** -- the order was wrong. This one is not inferred from a rejection, because
  it needs two measurements to compare: the Lab raises it when a Turn searches more than
  one launchable candidate and the fastest is not the one the ranking put first, by more
  than the Study says counts. A Turn that searches one candidate cannot reach the claim
  and does not make it, and an inversion inside the noise is not one either -- the loss
  surface is a plateau, so most inversions are inside the noise and routing them all
  would bury the cost model in reports of its own measurement error.

A route is a claim about what should change, so a wrong one is worse than none: routing a
missing verifier rule to the candidate tells an author to fix a Schedule that was correct
by the rules it was given.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from open_cake_ir.compiler.diagnostics import (AUTHOR_FIXABLE_LOWERING_CODES,
                                               BACKEND_LOWERING_GAP_CODES)

CANDIDATE = "candidate"
VERIFIER = "verifier"
IR_VOCABULARY = "ir_vocabulary"
BACKEND_LOWERING = "backend_lowering"
BACKEND_TRIAGE = "backend_triage"
COST_MODEL = "cost_model"

DESTINATIONS = (CANDIDATE, VERIFIER, IR_VOCABULARY, BACKEND_LOWERING,
                BACKEND_TRIAGE, COST_MODEL)

# Every backend EmitError currently receives this phrase, including implementation
# defects. Its cause is not typed, so it needs owner triage rather than an IR claim.
_UNDER_DETERMINED = "does not determine its source"

@dataclass(frozen=True)
class Route:
    """Where one diagnosis belongs, and the evidence that put it there."""

    destination: str | None
    reason: str

    def __post_init__(self) -> None:
        if self.destination is not None and self.destination not in DESTINATIONS or not self.reason:
            raise ValueError("routing destination or reason differs")


def route_rejection(feedback: Mapping[str, object], *, arm: str = "open_cake") -> Route:
    """Classify Environment feedback using its assigned arm, never authored metadata.

    Compile feedback is a CandidateCompileRejected, not an infrastructure failure.
    The default describes Compiler-only assessment callers; Lab execution and replay
    always supply the frozen arm.
    """

    if arm not in {"open_cake", "direct_cuda", "native_triton", "native_cute_dsl"}:
        raise ValueError("diagnosis Authoring Environment arm differs")
    stage = feedback.get("stage")
    if stage == 'budget':
        return Route(None,'the Run exhausted its compilation quota; no implementation defect is inferred')

    if stage == "compile" and arm != "open_cake":
        return Route(CANDIDATE, "the toolchain refused source authored by this arm")

    if stage == "compile":
        # The gates admitted it and the toolchain did not. Whatever was wrong is outside
        # what the verifier models, so the rule is missing rather than the Schedule bad.
        return Route(
            VERIFIER,
            "every gate passed and the toolchain still refused, so a contract this "
            "Schedule violated is not yet modelled",
        )

    error = feedback.get("error")
    if feedback.get("code") == "LOWERING_UNDETERMINED" or (isinstance(error, str) and _UNDER_DETERMINED in error):
        return Route(
            BACKEND_TRIAGE,
            "emission refused after admission; the generic EmitError does not prove "
            "whether Cake IR, backend lowering, or a missing preflight rule owns it",
        )

    findings = feedback.get("findings")
    if isinstance(findings, list):
        blocking = [item for item in findings if isinstance(item, Mapping)
                    and (item.get('blocks_acceptance') or item.get('blocks_lowering'))]
        candidate_codes = sorted(str(item.get('code')) for item in blocking
                                 if item.get('blocks_acceptance'))
        if candidate_codes:
            return Route(CANDIDATE, f"a gate refused it: {', '.join(candidate_codes)}")
        author_codes = sorted(str(item.get('code')) for item in blocking
                              if item.get('code') in AUTHOR_FIXABLE_LOWERING_CODES)
        if author_codes and len(author_codes) == len(blocking):
            return Route(CANDIDATE,
                         f"the selected backend requires an author declaration change: {', '.join(author_codes)}")
        backend_codes = sorted(str(item.get('code')) for item in blocking
                               if item.get('blocks_lowering')
                               and item.get('code') in BACKEND_LOWERING_GAP_CODES)
        if backend_codes and len(backend_codes) == len(blocking):
            return Route(
                BACKEND_LOWERING,
                f"Cake IR accepts this Schedule, but the selected backend cannot lower: "
                f"{', '.join(backend_codes)}",
            )
        if blocking:
            return Route(
                BACKEND_TRIAGE,
                "Cake IR accepted this Schedule; a maintenance agent must assign "
                "the lowering-only Finding owner before changing it: "
                + ', '.join(sorted(str(item.get('code')) for item in blocking)),
            )

    return Route(
        CANDIDATE,
        f"the Environment refused it at the {stage!r} stage",
    )
