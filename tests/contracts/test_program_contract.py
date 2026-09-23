from __future__ import annotations

import sys
import json
from dataclasses import replace
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler, Program  # noqa: E402
from open_cake_ir.compiler.ir import ProgramStage  # noqa: E402
from open_cake_ir.serialization import canonical_json_bytes  # noqa: E402
from open_cake_ir.tasks.qsa.program import ProgramContract
from tests.contracts._historical_qsa_program import replay_program_v2


class ProgramContractTest(unittest.TestCase):
    def test_program_stage_reuses_its_validated_typed_schedule(self) -> None:
        document = json.loads((ROOT / 'corpus/schedules/fma-b8-smoke.json').read_text())
        program = Program.from_schedule(document)
        stage = program.stages[0]
        self.assertIs(stage.schedule, stage.schedule)
        self.assertEqual(stage.schedule.schedule_id, document['schedule_id'])
        self.assertEqual(json.loads(stage.schedule_bytes), document)
        changed = json.loads(stage.schedule_bytes)
        changed['schedule_id'] = 'different-id'
        replaced = replace(stage, schedule_bytes=canonical_json_bytes(changed))
        self.assertEqual(replaced.schedule.schedule_id, 'different-id')
        with self.assertRaisesRegex(TypeError, 'immutable bytes'):
            ProgramStage(stage.name, bytearray(stage.schedule_bytes), stage.bindings)

    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
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
        with self.assertRaisesRegex(ValueError, "program node 'pool' Schedule differs"):
            ProgramContract.load(ROOT, self.path, self.compiler)

    def test_current_program_successor_preserves_the_frozen_workload_and_composition(self) -> None:
        path = ROOT / "contracts/programs/qsa-prefill-t32768-v4.json"
        program = ProgramContract.load(ROOT, path, self.compiler)
        self.assertEqual(program.implementation.outputs, program.public_outputs)
        self.assertEqual([stage.name for stage in program.implementation.stages],
                         [node.node_id for node in program.nodes])
        self.assertTrue(program.implementation.stages[0].bindings['index_k'].singleton_view)
        self.assertFalse(program.implementation.stages[0].bindings['pooled'].singleton_view)
        previous = json.loads(self.path.read_text())
        successor = json.loads(path.read_text())
        self.assertEqual(program.program_id, successor["program_id"])
        self.assertNotEqual(successor["program_id"], previous["program_id"])
        successor["program_id"] = previous["program_id"]
        for old_node, new_node in zip(previous["nodes"], successor["nodes"]):
            for field in ("schedule_sha256", "lowering_source_sha256"):
                self.assertNotEqual(new_node[field], old_node[field])
                new_node[field] = old_node[field]
        self.assertEqual(successor, previous)

    def test_schedule_identity_and_dataflow_drift_fail_closed(self) -> None:
        refusals = replay_program_v2()["mutation_refusals"]
        self.assertEqual(len(refusals), 3)
        self.assertIn("program node 'pool' Schedule differs", refusals[0])
        self.assertIn("program node 'layernorm' dataflow differs", refusals[1])
        self.assertEqual(refusals[2], "program node 'attention' tensor 'output' differs")


if __name__ == "__main__":
    unittest.main()
