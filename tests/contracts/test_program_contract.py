from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler  # noqa: E402
from open_cake_ir.tasks.qsa.program import ProgramContract


class ProgramContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.lock.json")
        cls.path = ROOT / "contracts/programs/qsa-prefill-t32768-v2.json"

    def test_qsa_program_has_one_public_abi_and_five_ordered_schedules(self) -> None:
        program = ProgramContract.load(ROOT, self.path, self.compiler)

        self.assertEqual(program.public_outputs, ("output",))
        self.assertEqual(program.program_id, "qsa-prefill-t32768-cake-port-v2")
        self.assertEqual(
            tuple(node.node_id for node in program.nodes),
            ("pool", "layernorm", "score_topk", "expand", "attention"),
        )
        self.assertEqual(program.nodes[-1].depends_on, ("expand",))
        self.assertEqual(
            program.reference_visibility,
            "compiler_validation_only_not_clean_start_authority",
        )

    def test_schedule_identity_and_dataflow_drift_fail_closed(self) -> None:
        original = json.loads(self.path.read_text(encoding="utf-8"))
        mutations = (
            lambda document: document["nodes"][0].update(schedule_sha256="0" * 64),
            lambda document: document["nodes"][1].update(depends_on=["ghost"]),
            lambda document: document["nodes"][-1]["views"].update(output="identity"),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as directory:
                document = json.loads(json.dumps(original))
                mutate(document)
                path = Path(directory) / "program.json"
                path.write_text(json.dumps(document), encoding="utf-8")
                with self.assertRaises(ValueError):
                    ProgramContract.load(ROOT, path, self.compiler)


if __name__ == "__main__":
    unittest.main()
