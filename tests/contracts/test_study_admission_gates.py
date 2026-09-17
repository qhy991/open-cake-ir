"""The two task-layer gates that check a Study against the registry that owns the fact.

These exist because `lab.preflight` admits a Study's own declaration and cannot consult
the device registry -- `tests/contracts/test_task_boundaries.py` forbids `lab` importing
the task layer. Both gates were added without tests, and an independent review proved the
gap by mutation: replacing them with no-ops left a 1993-test suite byte-identical. These
tests fail if either gate is neutered, which is the property that was missing.
"""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from open_cake_ir.tasks import runtime  # noqa: E402
from open_cake_ir.tasks.devices import BACKENDS, allocation, timing_source  # noqa: E402


UNTIMED = {"measurement_coverage": {"timed_assay": "unavailable"}}


def _study(target, *, evaluation=None, mode=None):
    execution = {"target": target}
    if mode is not None:
        execution["gpu"] = {"name": "device", "count": 1, "mode": mode}
    return SimpleNamespace(document={
        # Not portfolio: `TaskLab.preflight` routes that shape elsewhere, before either
        # gate -- which is itself worth knowing and is why the reachability test below
        # pins the matched path.
        "kind": "matched_search",
        "execution": execution,
        **({"evaluation_protocol": evaluation} if evaluation is not None else {}),
    })


def _timed_backend():
    return next(name for name in BACKENDS if timing_source(name) is not None)


def _target_of(backend):
    return BACKENDS[backend]["target"]


class MeasurementCoverageGateTests(unittest.TestCase):
    """A Study may not claim a limitation the device it names does not have."""

    def test_a_timed_target_cannot_declare_that_no_timed_assay_exists(self):
        backend = _timed_backend()
        with self.assertRaises(ValueError) as raised:
            runtime._admit_measurement_coverage(
                _study(_target_of(backend), evaluation=UNTIMED))
        message = str(raised.exception)
        self.assertIn(backend, message)
        self.assertIn(repr(timing_source(backend)), message)

    def test_an_untimed_target_may_declare_it(self):
        """Constructed, not borrowed: no registered backend is untimed today."""

        backend = _timed_backend()
        # The gate imports `timing_source` inside the function, so patching the module
        # attribute is what reaches it; without the patch this same call raises.
        with self.assertRaises(ValueError):
            runtime._admit_measurement_coverage(
                _study(_target_of(backend), evaluation=UNTIMED))
        with mock.patch("open_cake_ir.tasks.devices.timing_source", return_value=None):
            runtime._admit_measurement_coverage(
                _study(_target_of(backend), evaluation=UNTIMED))

    def test_a_target_no_backend_admits_is_refused_by_name(self):
        with self.assertRaisesRegex(ValueError, "no registered backend admits"):
            runtime._admit_measurement_coverage(_study("sm_999z", evaluation=UNTIMED))

    def test_a_study_making_no_claim_is_not_examined(self):
        """The gate reads a claim; it does not invent one for a Study that made none."""

        runtime._admit_measurement_coverage(_study("sm_999z"))
        runtime._admit_measurement_coverage(_study("sm_999z", evaluation={}))


class ExecutionModeGateTests(unittest.TestCase):
    """How a device is reached is the registry's fact, not the lowering route's."""

    def test_each_backend_is_admitted_only_at_the_mode_it_declares(self):
        for backend, device in BACKENDS.items():
            expected = ("local_serialized" if allocation(backend) == "local_broker"
                        else "exclusive")
            wrong = "exclusive" if expected == "local_serialized" else "local_serialized"
            with self.subTest(backend=backend):
                runtime._admit_execution_mode(_study(device["target"], mode=expected))
                with self.assertRaises(ValueError) as raised:
                    runtime._admit_execution_mode(_study(device["target"], mode=wrong))
                self.assertIn(backend, str(raised.exception))

    def test_a_locally_allocated_triton_target_is_not_judged_by_its_route(self):
        """The regression this gate exists for: gfx1151 lowers through Triton and is
        reached by the local broker, and the old rule read the mode off the route."""

        runtime._admit_execution_mode(_study("gfx1151", mode="local_serialized"))
        with self.assertRaises(ValueError):
            runtime._admit_execution_mode(_study("gfx1151", mode="exclusive"))

    def test_a_target_no_backend_admits_is_refused_by_name(self):
        with self.assertRaisesRegex(ValueError, "no registered backend"):
            runtime._admit_execution_mode(_study("sm_999z", mode="exclusive"))

    def test_a_study_declaring_no_mode_is_not_examined(self):
        runtime._admit_execution_mode(_study("gfx1151"))


class GatesAreReachedTests(unittest.TestCase):
    """Both gates run on the path a launch actually takes.

    Testing the functions alone would pass even if `TaskLab.preflight` stopped calling
    them, which is the shape of the gap these tests were written to close.
    """

    def test_task_lab_preflight_calls_both_before_delegating(self):
        called = []
        study = _study("gfx1151", evaluation=UNTIMED, mode="local_serialized")
        with mock.patch.object(runtime.StudyContract, "load", return_value=study), \
             mock.patch.object(runtime, "_admit_measurement_coverage",
                               side_effect=lambda s: called.append("coverage")), \
             mock.patch.object(runtime, "_admit_execution_mode",
                               side_effect=lambda s: called.append("mode")), \
             mock.patch.object(runtime.Lab, "preflight",
                               side_effect=lambda *a, **k: called.append("delegated")):
            runtime.TaskLab(ROOT).preflight(Path("unused.json"))
        self.assertEqual(called, ["coverage", "mode", "delegated"])


if __name__ == "__main__":
    unittest.main()
