"""C550 successor keeps the frozen gather contract and specializes every case ABI."""
from pathlib import Path
import json
import subprocess
import unittest

from examples.paired_triton.prepare import baseline_schedule
from open_cake_ir.compiler import Compiler
from open_cake_ir.tasks.workloads import load_workload
from open_cake_ir.tasks.tiles.workload import materialize_case, reference_outputs

ROOT = Path(__file__).resolve().parents[2]


class IndexedGatherMetax(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = load_workload(ROOT / 'contracts/workloads/indexed-gather-bf16-v2.json')
        cls.new = load_workload(ROOT / 'contracts/workloads/indexed-gather-bf16-v3.json')
        cls.compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')

    def test_successor_changes_only_identity_target_and_provenance(self):
        for field in ('cases', 'tensors', 'oracle'):
            self.assertEqual(self.new.document[field], self.old.document[field])
        self.assertEqual({k:v for k,v in self.new.document['validation'].items() if k != 'qualification'},
                         {k:v for k,v in self.old.document['validation'].items() if k != 'qualification'})
        old_semantics = self.old.document['semantics']
        new_semantics = self.new.document['semantics']
        self.assertEqual({k:v for k,v in new_semantics.items() if k != 'target'},
                         {k:v for k,v in old_semantics.items() if k != 'target'})
        self.assertEqual(self.new.target, 'xcore1002')
        self.assertEqual(self.new.document['revision'], '3')
        parent = next(row for row in self.new.document['provenance']
                      if row['kind'] == 'target_adaptation_parent')
        retained = subprocess.run(
            ['git', 'show', f"{parent['source_commit']}:{parent['path']}"],
            cwd=ROOT, check=True, capture_output=True,
        ).stdout
        self.assertEqual(json.loads(retained), self.old.document)

    def test_every_case_has_its_own_complete_lowerable_schedule_and_same_oracle(self):
        self.assertEqual(self.new.case_ids, self.old.case_ids)
        for case in self.new.case_ids:
            old_inputs = materialize_case(self.old, case)
            new_inputs = materialize_case(self.new, case)
            self.assertEqual(new_inputs, old_inputs)
            self.assertEqual(reference_outputs(self.new, case, new_inputs),
                             reference_outputs(self.old, case, old_inputs))
            schedule = baseline_schedule(self.new, case, project_root=ROOT)
            globals_ = [b for b in schedule['buffers'] if b['space'] == 'global']
            self.assertEqual([(b['name'], tuple(b['shape']), b['dtype'], b['mode']) for b in globals_],
                [(a.name, a.shape, a.dtype, a.mode) for a in self.new.tensor_abi(case)])
            assessment = self.compiler.assess(schedule)
            self.assertTrue(assessment.lowering_eligible, assessment.findings)
            lowering = self.compiler.lower(assessment)
            self.assertEqual(lowering.target, 'xcore1002')
            self.assertEqual(lowering.toolchain_requirements['code_object'], 'mcfatbin')

    def test_old_documents_and_case_specific_shapes_remain_distinct(self):
        self.assertEqual(self.old.document,
            json.loads((ROOT / 'contracts/workloads/indexed-gather-bf16-v2.json').read_text()))
        self.assertEqual(len({self.new.tensor_abi(case) for case in self.new.case_ids}), 4)
