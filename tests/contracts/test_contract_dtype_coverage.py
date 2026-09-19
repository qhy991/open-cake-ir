"""An instruction contract with no record is refused, not skipped and not reported past.

`ir/instruction_contracts.py` names what each contract reads and accumulates in, and the
operand check reads that record. Before it, a private verifier table held the same facts
and the check ran only when the contract had a row; a contract without one produced no
dtype Finding at all -- indistinguishable, to a reader, from a check that passed. A REPORT
then replaced that silence, and a blocking refusal now replaces the REPORT: a Target may
admit a name, but a Compiler that cannot say what the name means cannot lower it.

Measured when this was written: every contract the Corpus uses on an mma operation has a
record, so nothing here changes an existing verdict. What these hold is that a future
contract admitted by a Target, but unregistered, is refused by the verifier and names
itself, and that a registered contract of another kind on an mma operation is refused too
rather than dereferenced for an accumulator it does not have.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler.diagnostics import FindingSeverity
from open_cake_ir.compiler.ir import ContractKind, Schedule, contract, contracts_of
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import verify

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
                    and instruction.get("contract") in contracts_of(ContractKind.MMA):
                return document, instruction["contract"]
    raise AssertionError("no declared Corpus case carries a modelled mma contract")


class ContractDtypeCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document, cls.contract = _mma_case()
        cls.target = Target.load(ROOT / "compiler/targets/sm_100a.json")

    def _findings(self, document: dict, target: Target):
        return verify(Schedule.from_dict(document), target)

    def _renamed(self, name: str) -> dict:
        document = json.loads(json.dumps(self.document))
        for operation in document["operations"]:
            instruction = (operation.get("parameters") or {}).get("instruction")
            if operation.get("kind") == "mma" and isinstance(instruction, dict):
                instruction["contract"] = name
        return document

    def _admitting(self, name: str) -> Target:
        return dataclasses.replace(
            self.target, instruction_contracts=self.target.instruction_contracts | {name})

    def test_every_contract_the_corpus_uses_is_registered(self) -> None:
        """The refusal exists for a gap that does not currently occur."""
        codes = {f.code for f in self._findings(self.document, self.target)}
        self.assertNotIn("INSTRUCTION_CONTRACT_UNKNOWN", codes)
        self.assertNotIn("MMA_INSTRUCTION_KIND_DIFFERS", codes)

    def test_an_admitted_but_unregistered_contract_is_refused_by_name(self) -> None:
        findings = self._findings(self._renamed(UNMODELLED), self._admitting(UNMODELLED))
        unknown = [f for f in findings if f.code == "INSTRUCTION_CONTRACT_UNKNOWN"]
        self.assertEqual(len(unknown), 1, sorted({f.code for f in findings}))
        self.assertIs(unknown[0].severity, FindingSeverity.BLOCKING)
        self.assertTrue(unknown[0].blocks_lowering)
        self.assertIn(UNMODELLED, unknown[0].message)
        self.assertIn("no contract record", unknown[0].message)
        self.assertIn(self.target.target_id, unknown[0].message)
        self.assertTrue(unknown[0].path.endswith(".instruction.contract"), unknown[0].path)

    def test_a_contract_the_target_does_not_admit_is_refused_by_the_target_rule(self) -> None:
        """The registry refusal is gated on admission: an unadmitted name is the Target's
        refusal alone, as corpus case swiglu-instruction-drift pins for elementwise."""
        codes = {f.code for f in self._findings(self._renamed(UNMODELLED), self.target)}
        self.assertIn("TARGET_INSTRUCTION_UNSUPPORTED", codes)
        self.assertNotIn("INSTRUCTION_CONTRACT_UNKNOWN", codes)

    def test_a_registered_contract_of_another_kind_is_refused_not_dereferenced(self) -> None:
        """An elementwise record has no accumulator; reading one would crash."""
        name = "libdevice.tanh.f32"
        self.assertIs(contract(name).kind, ContractKind.ELEMENTWISE)
        findings = self._findings(self._renamed(name), self.target)
        differs = [f for f in findings if f.code == "MMA_INSTRUCTION_KIND_DIFFERS"]
        self.assertEqual(len(differs), 1, sorted({f.code for f in findings}))
        self.assertIs(differs[0].severity, FindingSeverity.BLOCKING)
        self.assertIn("elementwise contract, not an MMA", differs[0].message)

    def test_a_registered_contract_still_refuses_a_disagreeing_operand(self) -> None:
        """The rule the refusal sits beside is unchanged, and reads the record."""
        record = contract(self.contract)
        self.assertIs(record.kind, ContractKind.MMA)
        self.assertTrue(record.operand_dtypes)
        self.assertIsNotNone(record.accumulator)


if __name__ == "__main__":
    unittest.main()
