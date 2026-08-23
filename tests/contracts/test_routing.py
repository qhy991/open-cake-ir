"""Contract tests for diagnosis routing.

A route is a claim about what has to change. Sending a missing verifier rule to the
candidate tells an author to fix a Schedule that was correct by the rules it was given,
so a wrong route is worse than no route.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.lab.routing import (  # noqa: E402
    CANDIDATE,
    COST_MODEL,
    DESTINATIONS,
    IR_VOCABULARY,
    VERIFIER,
    Route,
    route_rejection,
)


class RoutingContractTests(unittest.TestCase):
    def test_a_gate_refusal_is_the_candidates_to_fix(self) -> None:
        decision = route_rejection(
            {
                "stage": "assessment",
                "findings": [
                    {"code": "RESIDENCY_IMPOSSIBLE", "blocking": True},
                    {"code": "RESIDENCY_BOUND", "blocking": False},
                ],
            }
        )
        self.assertEqual(decision.destination, CANDIDATE)
        self.assertIn("RESIDENCY_IMPOSSIBLE", decision.reason)
        # A report is not a reason to reject, so it must not appear as one.
        self.assertNotIn("RESIDENCY_BOUND", decision.reason)

    def test_passing_every_gate_and_failing_to_compile_is_the_verifiers(self) -> None:
        """The one route that matters most, and the easiest to get backwards.

        The gates admitted this Schedule and the toolchain refused it, so something was
        true of it that the pre-compile model does not cover. Blaming the candidate here
        is how a missing rule stays missing.
        """

        decision = route_rejection({"stage": "compile", "diagnostic": "ptxas exit 255"})
        self.assertEqual(decision.destination, VERIFIER)

    def test_an_inexpressible_schedule_is_the_vocabularys(self) -> None:
        decision = route_rejection(
            {
                "stage": "assessment",
                "error": "Schedule does not determine its source: expected one mma",
            }
        )
        self.assertEqual(decision.destination, IR_VOCABULARY)

    def test_a_route_must_name_a_real_destination_and_a_reason(self) -> None:
        for bad in ({"destination": "elsewhere", "reason": "x"}, {"destination": CANDIDATE, "reason": ""}):
            with self.subTest(**bad):
                with self.assertRaises(ValueError):
                    Route(**bad)

    def test_the_cost_model_route_exists_but_is_not_inferred_yet(self) -> None:
        """Recorded rather than guessed.

        Deciding the order was wrong needs more than one candidate per Turn to reach a
        measurement, and today one does. Naming the destination while refusing to infer it
        keeps the gap visible instead of letting a plausible guess fill it.
        """

        self.assertIn(COST_MODEL, DESTINATIONS)
        for feedback in (
            {"stage": "assessment", "findings": []},
            {"stage": "compile"},
            {"stage": "manifest"},
        ):
            self.assertNotEqual(route_rejection(feedback).destination, COST_MODEL)


if __name__ == "__main__":
    unittest.main()
