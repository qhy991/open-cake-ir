"""Which contracts realize a barrier is declared once, by the vocabulary that owns it.

Two shared rules each kept their own copy of `{"mbarrier", "barrier.sync"}`: the
hardware-conformance rule intersected the Target's declared set with that literal, and
the program-safety diagnostic named the same pair in its message. `BarrierMechanism`
already is that set -- a Schedule's `mechanism` parses through it and the authoring
Schema projects it -- so a Target declaring its own barrier contract under another name
would have been told it admits none, while the enum that owns the fact said otherwise.

Today both spellings agree, which is exactly when a duplicate is worth removing: it is
found when they stop agreeing.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler.ir import BarrierMechanism, Schedule
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import verify

ROOT = Path(__file__).resolve().parents[2]


def _barrier_schedule() -> Schedule:
    """The first declared case that actually carries barriers, found by scanning.

    A hardcoded name that stopped matching would make these pass by checking nothing.
    """
    for path in sorted((ROOT / "corpus/schedules").glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("barriers") and document.get("target") == "sm_100a":
            try:
                return Schedule.from_dict(document)
            except Exception:
                continue
    raise AssertionError("no declared Corpus case carries barriers on sm_100a")


def _codes(schedule, target) -> set[str]:
    return {finding.code for finding in verify(schedule, target)}


class BarrierMechanismOwnership(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schedule = _barrier_schedule()
        cls.target = Target.load(ROOT / "compiler/targets/sm_100a.json")

    def _with_contracts(self, *names: str) -> Target:
        return dataclasses.replace(
            self.target, synchronization_contracts=frozenset(names))

    def test_the_declared_target_admits_its_barriers(self) -> None:
        self.assertNotIn("TARGET_SYNCHRONIZATION_UNSUPPORTED",
                         _codes(self.schedule, self.target))

    def test_every_mechanism_the_vocabulary_names_is_admitted_on_its_own(self) -> None:
        """Adding a member to the enum reaches this rule; it is not a second list."""
        for mechanism in BarrierMechanism:
            with self.subTest(mechanism=mechanism.value):
                target = self._with_contracts(mechanism.value)
                self.assertNotIn("TARGET_SYNCHRONIZATION_UNSUPPORTED",
                                 _codes(self.schedule, target))

    def test_a_contract_that_realizes_no_barrier_does_not_admit_one(self) -> None:
        """`triton_program_order` is declared by sm_100a and is not a mechanism.

        The permissive reading -- any declared synchronization contract admits barriers
        -- would accept a Schedule whose handshakes nothing can realize.
        """
        self.assertIn("triton_program_order", self.target.synchronization_contracts)
        self.assertNotIn("triton_program_order",
                         {mechanism.value for mechanism in BarrierMechanism})
        target = self._with_contracts("triton_program_order")
        self.assertIn("TARGET_SYNCHRONIZATION_UNSUPPORTED", _codes(self.schedule, target))

    def test_a_target_admitting_no_synchronization_contract_is_refused(self) -> None:
        self.assertIn("TARGET_SYNCHRONIZATION_UNSUPPORTED",
                      _codes(self.schedule, self._with_contracts()))

    def test_neither_rule_keeps_its_own_copy_of_the_pair(self) -> None:
        for relative in ("src/open_cake_ir/compiler/verifier/hardware_conformance.py",
                         "src/open_cake_ir/compiler/verifier/program_safety.py"):
            source = (ROOT / relative).read_text(encoding="utf-8")
            with self.subTest(module=relative):
                self.assertNotIn('{"mbarrier", "barrier.sync"}', source)


if __name__ == "__main__":
    unittest.main()
