from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler  # noqa: E402
from open_cake_ir.tasks.qsa.program import ProgramContract
from tests.contracts._historical_qsa_program import replay_program_v2


class ProgramContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.lock.json")
        cls.path = ROOT / "contracts/programs/qsa-prefill-t32768-v2.json"

    def test_qsa_program_has_one_public_abi_and_five_ordered_schedules(self) -> None:
        observed = replay_program_v2()
        self.assertEqual(observed["public_outputs"], ["output"])
        self.assertEqual(observed["program_id"], "qsa-prefill-t32768-cake-port-v2")
        self.assertEqual(observed["nodes"], ["pool", "layernorm", "score_topk", "expand", "attention"])
        self.assertEqual(observed["last_dependencies"], ["expand"])
        self.assertEqual(observed["reference_visibility"],
                         "compiler_validation_only_not_clean_start_authority")

    def test_current_compiler_does_not_rebind_the_frozen_program_v2(self) -> None:
        with self.assertRaisesRegex(ValueError, "program node 'score_topk' lowering differs"):
            ProgramContract.load(ROOT, self.path, self.compiler)

    def test_schedule_identity_and_dataflow_drift_fail_closed(self) -> None:
        refusals = replay_program_v2()["mutation_refusals"]
        self.assertEqual(len(refusals), 3)
        self.assertIn("program node 'pool' Schedule differs", refusals[0])
        self.assertIn("program node 'layernorm' dataflow differs", refusals[1])
        self.assertEqual(refusals[2], "program node 'attention' tensor 'output' differs")


if __name__ == "__main__":
    unittest.main()
