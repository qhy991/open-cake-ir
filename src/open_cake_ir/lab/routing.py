"""Send each diagnosis to the thing that has to change.

The paper's fourth stage routes a result to the candidate, the verifier, the cost model or
the IR vocabulary according to the diagnosis. Without it every failure reads the same way
-- as the candidate's fault -- and the outer loop that turns recurring failures into new
rules has nothing to accumulate.

Each destination is inferred from a signal the loop already produces, not from a new one:

* **candidate** -- a gate refused it. The Schedule is wrong and its author can see why.
* **verifier** -- every gate passed and the toolchain still refused. Something was true of
  this Schedule that the pre-compile gates do not model, which is a missing rule rather
  than a bad candidate.
* **ir_vocabulary** -- the Schedule is well-formed and this backend cannot lower it. Some
  of that is knowable before emission and arrives as a finding with a code; the rest is
  discovered while emitting and arrives as a refusal to determine the source. The author
  declared everything the IR can express and it was not enough either way.
* **cost_model** -- the order was wrong. This one is not inferred from a rejection, because
  it needs two measurements to compare: the Lab raises it when a Turn searches more than
  one launchable candidate and the fastest is not the one the ranking put first. A Turn
  that searches one candidate cannot reach the claim and does not make it.

A route is a claim about what should change, so a wrong one is worse than none: routing a
missing verifier rule to the candidate tells an author to fix a Schedule that was correct
by the rules it was given.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

CANDIDATE = "candidate"
VERIFIER = "verifier"
IR_VOCABULARY = "ir_vocabulary"
COST_MODEL = "cost_model"

DESTINATIONS = (CANDIDATE, VERIFIER, IR_VOCABULARY, COST_MODEL)

# The Compiler says this when a profile emits and the Schedule leaves a decision open.
# It is the one rejection that is about the vocabulary rather than about the Schedule.
_UNDER_DETERMINED = "does not determine its source"

# Findings that say the same thing early enough to carry a code. Matching a code beats
# matching a message: a message is prose the Compiler is free to reword.
_VOCABULARY_CODES = frozenset({"PROFILE_OPERATION_UNEMITTABLE"})


@dataclass(frozen=True)
class Route:
    """Where one diagnosis belongs, and the evidence that put it there."""

    destination: str
    reason: str

    def __post_init__(self) -> None:
        if self.destination not in DESTINATIONS or not self.reason:
            raise ValueError("routing destination or reason differs")


def route_rejection(feedback: Mapping[str, object]) -> Route:
    """Classify one rejected candidate from the feedback the Environment returned."""

    stage = feedback.get("stage")

    if stage == "compile":
        # The gates admitted it and the toolchain did not. Whatever was wrong is outside
        # what the verifier models, so the rule is missing rather than the Schedule bad.
        return Route(
            VERIFIER,
            "every gate passed and the toolchain still refused, so a contract this "
            "Schedule violated is not yet modelled",
        )

    error = feedback.get("error")
    if isinstance(error, str) and _UNDER_DETERMINED in error:
        return Route(
            IR_VOCABULARY,
            "lowering refused because the Schedule leaves a decision the vocabulary "
            "cannot express",
        )

    findings = feedback.get("findings")
    if isinstance(findings, list):
        blocking = [
            item.get("code")
            for item in findings
            if isinstance(item, Mapping)
            and (item.get("blocks_acceptance") or item.get("blocks_lowering"))
        ]
        vocabulary = sorted(
            str(code) for code in blocking if code in _VOCABULARY_CODES
        )
        if vocabulary:
            # Checked before the candidate route: these findings block a Schedule that
            # is not wrong, so telling its author to fix it would be telling them to
            # work around a gap in the compiler.
            return Route(
                IR_VOCABULARY,
                f"the Schedule is well-formed and no backend body exists for it: "
                f"{', '.join(vocabulary)}",
            )
        if blocking:
            return Route(
                CANDIDATE,
                f"a gate refused it: {', '.join(sorted(str(code) for code in blocking))}",
            )

    return Route(
        CANDIDATE,
        f"the Environment refused it at the {stage!r} stage",
    )
