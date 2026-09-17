"""The measurement-coverage limitation, and the gate that admits it.

`evaluation_policy` has written this shape since the DCU was admitted -- a Study for a
target whose backend names no timing source carries a limitation instead of a paired
assay. Nothing downstream accepted it until now, and nothing tested it either way.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from open_cake_ir.lab._policies import untimed  # noqa: E402
from open_cake_ir.tasks.devices import BACKENDS, timing_source  # noqa: E402
from open_cake_ir.tasks.normalization import study as study_module  # noqa: E402
from open_cake_ir.tasks.normalization.study import arm_feedback, evaluation_policy  # noqa: E402


TIMED_FEEDBACK = ["findings", "correctness", "qualified_timing", "profile"]
UNTIMED_FEEDBACK = ["findings", "correctness"]


class UntimedPredicateTests(unittest.TestCase):
    def test_only_an_explicit_unavailable_coverage_is_untimed(self):
        self.assertTrue(untimed({"measurement_coverage": {"timed_assay": "unavailable"}}))
        self.assertFalse(untimed({}))
        self.assertFalse(untimed({"measurement_coverage": {"timed_assay": "available"}}))
        self.assertFalse(untimed({"measurement_coverage": None}))

    def test_a_malformed_declaration_is_refused_rather_than_dereferenced(self):
        """It reads an external Study document, so a crash here is a gate that isn't one."""

        with self.assertRaisesRegex(ValueError, "measurement_coverage"):
            untimed({"measurement_coverage": "unavailable"})
        with self.assertRaisesRegex(ValueError, "measurement_coverage"):
            untimed({"measurement_coverage": ["unavailable"]})
        with self.assertRaisesRegex(ValueError, "must be an object"):
            untimed("evaluation_protocol")


class CoverageMatchesTheRegistryTests(unittest.TestCase):
    """The limitation states what a device cannot do; the registry owns that fact."""

    def test_every_backend_without_a_timing_source_produces_an_untimed_policy(self):
        for backend, device in BACKENDS.items():
            with self.subTest(backend=backend):
                expected = timing_source(backend) is None
                self.assertEqual(expected, device["timing_source"] is None)

    def test_the_two_feedback_shapes_follow_the_policy_and_nothing_else(self):
        self.assertEqual(arm_feedback({"measurement_coverage": {"timed_assay": "unavailable"}}),
                         UNTIMED_FEEDBACK)
        self.assertEqual(arm_feedback({}), TIMED_FEEDBACK)


class _Workload:
    """The smallest thing `evaluation_policy` reads, so the shapes can be compared."""

    def __init__(self, target: str) -> None:
        self.target = target
        self.document = {"validation": {"primary_case": "primary"}}
        self.case_ids = ("primary",)


class PolicyShapeTests(unittest.TestCase):
    def test_a_backend_with_no_timing_source_carries_the_limitation_and_no_paired_assay(self):
        """The condition is constructed, not borrowed from whichever row is untimed.

        This named gfx1151 while gfx1151 declared no source. It gained one the day
        something measured on it, and the test then failed for a reason that had nothing
        to do with what it checks -- a registry row is not this test's to depend on.
        """

        with mock.patch.object(study_module, "timing_source", return_value=None):
            policy = evaluation_policy(_Workload("gfx1151"))
        self.assertTrue(untimed(policy))
        self.assertNotIn("paired_timing", policy)
        self.assertEqual(policy["search_evaluation"], "correctness_only")
        self.assertEqual(policy["attribution_evaluation"], "correctness_only")
        self.assertEqual(arm_feedback(policy), UNTIMED_FEEDBACK)

    def test_every_registered_backend_currently_names_a_source(self):
        """Not a rule, a record: today no registered backend is untimed.

        If one is added the untimed path gains a live user, and whoever adds it should see
        that here rather than discover it from a Study that quietly loses its timing.
        """

        untimed_backends = sorted(name for name in BACKENDS if timing_source(name) is None)
        self.assertEqual(untimed_backends, [])

    def test_a_timed_target_is_untouched_by_the_untimed_branch(self):
        """The regression pin: the shape that was already shipping must not move."""

        policy = evaluation_policy(_Workload("sm_103a"))
        self.assertFalse(untimed(policy))
        self.assertNotIn("measurement_coverage", policy)
        self.assertEqual(policy["search_evaluation"], "correctness_then_paired_cupti")
        self.assertEqual(policy["attribution_evaluation"],
                         "correctness_then_profile_each_search_survivor")
        self.assertEqual(policy["paired_timing"]["kind"], "fixed_baseline_paired_cupti_v1")
        self.assertEqual(arm_feedback(policy), TIMED_FEEDBACK)


if __name__ == "__main__":
    unittest.main()
