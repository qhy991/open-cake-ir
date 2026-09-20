"""Exact-target successors retain the original composed task's entire mathematics."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from open_cake_ir.compiler import Compiler
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.solx_fib import attention, moe
from open_cake_ir.tasks.workloads import load_workload

ROOT = Path(__file__).resolve().parents[2]


def contracts():
    for owner in (attention, moe):
        for task in owner.TASKS:
            for variant in owner.VARIANTS:
                yield owner, task, variant


class ComposedTargetSuccessors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')

    def test_default_b300_contracts_are_identical_to_the_retained_documents(self):
        for owner, task, variant in contracts():
            with self.subTest(task=task, variant=variant):
                original = owner.workload_document(task, variant=variant)
                retained = json.loads((ROOT / 'contracts/workloads' / f"{original['workload_id']}.json").read_text())
                self.assertEqual(original, retained)
                owner.validate_contract(original)

    def test_c550_successors_change_only_identity_provenance_and_target(self):
        for owner, task, variant in contracts():
            with self.subTest(task=task, variant=variant):
                original = owner.workload_document(task, variant=variant)
                successor = owner.workload_document(task, variant=variant, backend='triton-metax')
                self.assertEqual(successor['revision'], '2')
                self.assertEqual(successor['semantics']['target'], 'xcore1002')
                self.assertNotEqual(successor['workload_id'], original['workload_id'])
                self.assertEqual(successor['provenance'][:-1], original['provenance'])
                self.assertEqual(successor['provenance'][-1]['workload_id'], original['workload_id'])
                mathematical = deepcopy(successor)
                for key in ('workload_id', 'revision', 'provenance'):
                    mathematical[key] = original[key]
                mathematical['semantics']['target'] = original['semantics']['target']
                self.assertEqual(mathematical, original)
                owner.validate_contract(successor)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'workload.json'
                    path.write_text(json.dumps(successor))
                    workload = load_workload(path)
                program = owner.launch_plan(workload)
                lowered = self.compiler.lower_program(program)
                self.assertEqual(program.target, 'xcore1002')
                self.assertTrue(all(item.target == 'xcore1002' for item in lowered.lowerings))
                self.assertEqual(tuple(program.inputs), tuple(a.name for a in workload.tensor_abi('primary') if a.mode == 'input'))

    def test_mutating_a_frozen_target_or_successor_semantics_is_not_a_migration(self):
        for owner in (attention, moe):
            task = next(iter(owner.TASKS))
            old = owner.workload_document(task)
            old['semantics']['target'] = 'xcore1002'
            with self.assertRaises(ValueError):
                owner.validate_contract(old)
            successor = owner.workload_document(task, backend='triton-metax')
            for section, field, value in (('validation', 'atol', 1.0), ('oracle', 'callable', 'other.oracle'),
                                          ('semantics', 'input_effects', 'may_mutate')):
                bad = deepcopy(successor)
                bad[section][field] = value
                with self.subTest(owner=owner.__name__, field=field), self.assertRaises(ValueError):
                    owner.validate_contract(bad)
            bad = deepcopy(successor)
            bad['cases'][0]['seed'] += 1
            with self.assertRaises(ValueError):
                owner.validate_contract(bad)

    def test_registry_and_target_operations_own_admission(self):
        for backend in ('not-registered', 'metal-m1-pro', 'triton-gfx1151'):
            for owner in (attention, moe):
                with self.subTest(backend=backend, owner=owner.__name__), self.assertRaises(ValueError):
                    owner.workload_document(next(iter(owner.TASKS)), backend=backend)
        # Another already-declared Triton Target uses the same successor mechanism.
        document = attention.workload_document(next(iter(attention.TASKS)), backend='triton-b200')
        self.assertEqual(document['revision'], '2')
        self.assertEqual(document['semantics']['target'], 'sm_100a')
