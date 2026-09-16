"""An instruction contract whose dtypes are not modelled is reported, not skipped.

`_CONTRACT_DTYPES` names what each admitted contract reads and accumulates in, and the
operand check ran only when the contract had a row. A contract without one produced no
dtype Finding at all -- indistinguishable, to a reader, from a check that passed. That is
the generous reading absence always gets, and the fail-open direction is the dangerous
one.

Measured when this was written: every contract the Corpus uses on an mma operation has a
row, so nothing here changes an existing verdict. What these hold is that a future
contract admitted by a Target, but unmodelled here, says so.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler.diagnostics import FindingSeverity
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import verify
from open_cake_ir.compiler.verifier.hardware_conformance import _CONTRACT_DTYPES

ROOT = Path(__file__).resolve().parents[2]
UNMODELLED = "vendor.mma.synthetic.f32"


def _mma_case() -> tuple[dict, str]:
    """A declared case carrying an mma contract, found by scanning rather than named."""
    for path in sorted((ROOT / "corpus/schedules").glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("target") != "sm_100a":
            continue
        for operation in document.get("operations", []):
            instruction = (operation.get("parameters") or {}).get("instruction")
            if operation.get("kind") == "mma" and isinstance(instruction, dict) \
                    and instruction.get("contract") in _CONTRACT_DTYPES:
                return document, instruction["contract"]
    raise AssertionError("no declared Corpus case carries a modelled mma contract")


class ContractDtypeCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document, cls.contract = _mma_case()
        cls.target = Target.load(ROOT / "compiler/targets/sm_100a.json")

    def _findings(self, document: dict, target: Target):
        return verify(Schedule.from_dict(document), target)

    def test_every_contract_the_corpus_uses_is_modelled(self) -> None:
        """The report exists for a gap that does not currently occur."""
        codes = {f.code for f in self._findings(self.document, self.target)}
        self.assertNotIn("MMA_CONTRACT_DTYPES_UNMODELED", codes)

    def test_an_admitted_but_unmodelled_contract_says_it_was_not_checked(self) -> None:
        document = json.loads(json.dumps(self.document))
        for operation in document["operations"]:
            instruction = (operation.get("parameters") or {}).get("instruction")
            if operation.get("kind") == "mma" and isinstance(instruction, dict):
                instruction["contract"] = UNMODELLED
        target = dataclasses.replace(
            self.target,
            instruction_contracts=self.target.instruction_contracts | {UNMODELLED})
        findings = self._findings(document, target)
        unmodelled = [f for f in findings if f.code == "MMA_CONTRACT_DTYPES_UNMODELED"]
        self.assertTrue(unmodelled, sorted({f.code for f in findings}))
        self.assertIs(unmodelled[0].severity, FindingSeverity.REPORT)
        self.assertFalse(unmodelled[0].blocks_lowering)
        self.assertIn(UNMODELLED, unmodelled[0].message)
        self.assertIn("not checked", unmodelled[0].message)

    def test_a_modelled_contract_still_refuses_a_disagreeing_operand(self) -> None:
        """The rule the report sits beside is unchanged."""
        self.assertIn(self.contract, _CONTRACT_DTYPES)
        operands, _ = _CONTRACT_DTYPES[self.contract]
        self.assertTrue(operands)


if __name__ == "__main__":
    unittest.main()
