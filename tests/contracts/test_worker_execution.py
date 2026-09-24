"""Versioned Program worker control is typed and cannot replay as ordered stages."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Program
from open_cake_ir.compiler.ir import HandoffScope, RankPlacement, WorkerExecution
from open_cake_ir.evaluation.program import ProgramLaunchManifest


ROOT = Path(__file__).resolve().parents[2]
SCHEDULE = ROOT / "corpus/schedules/fma-b8-smoke.json"


def document() -> dict:
    schedule = json.loads(SCHEDULE.read_text())
    shape = [8, 128]
    tensors = {name: {"shape": shape, "dtype": "fp32"}
               for name in ("a", "b", "c", "middle0", "middle1", "out")}
    tensors.update({name: {"shape": [1], "dtype": "int32"}
                    for name in ("first_ctas", "chunks", "steal_budget")})
    stages = [
        {"name": name, "schedule": schedule,
         "bindings": {"a": source, "b": "b", "c": "c", "y": destination}}
        for name, source, destination in (
            ("dispatch", "a", "middle0"),
            ("compute", "middle0", "middle1"),
            ("combine", "middle1", "out"),
        )
    ]
    return {
        "schema_version": 2, "program_id": "fma-worker-topology", "target": "sm_100a",
        "tensors": tensors,
        "inputs": ["a", "b", "c", "first_ctas", "chunks", "steal_budget"],
        "outputs": ["out"], "stages": stages,
        "execution": {
            "kind": "cooperative_workers",
            "lowering": {"backend": "native_cuda", "entry_point": "cake_worker_fma"},
            "controls": {"first_class_ctas": "first_ctas", "chunk_count": "chunks",
                         "steal_budget": "steal_budget"},
            "workers": [
                {"name": "communication", "phases": ["dispatch", "combine"]},
                {"name": "computation", "phases": ["compute", "combine"]},
            ],
            "queues": [
                {"stage": "dispatch", "workers": ["communication"]},
                {"stage": "compute", "workers": ["communication", "computation"]},
                {"stage": "combine", "workers": ["communication", "computation"]},
            ],
            "handoffs": [
                {"payload": "middle0", "producer": "dispatch", "consumer": "compute",
                 "order": "release_acquire", "scope": "device"},
                {"payload": "middle1", "producer": "compute", "consumer": "combine",
                 "order": "release_acquire", "scope": "device"},
            ],
            "steal": {"borrower": "communication", "stage": "compute",
                      "after": "dispatch", "before": "combine"},
        },
    }


def rank_document() -> dict:
    value = document()
    value['schema_version'] = 3
    value['execution']['workers'][1]['phases'] = ['compute']
    value['execution']['queues'][2]['workers'] = ['communication']
    for handoff in value['execution']['handoffs']:
        handoff['scope'] = 'system'
    value['execution']['placement'] = {
        'world_size': 2, 'state_rank': 0,
        'worker_ranks': {'communication': 0, 'computation': 1},
        'tensor_ranks': {name: (1 if name == 'middle0' else 0)
                         for name in value['tensors']},
    }
    return value


class WorkerExecutionContract(unittest.TestCase):
    def test_rank_placement_roundtrips_and_never_uses_ordered_fallback(self) -> None:
        program = Program.from_dict(rank_document())
        self.assertIsInstance(program.execution.placement, RankPlacement)
        self.assertEqual(dict(program.execution.placement.stage_ranks),
                         {'dispatch': 0, 'compute': 1, 'combine': 0})
        self.assertEqual(program.document['schema_version'], 3)
        self.assertEqual(Program.from_dict(program.document), program)
        with self.assertRaisesRegex(ValueError, 'worker'):
            Compiler.load(ROOT).lower_program(program)
        manifest = {'schema_version': 1, 'abi': ProgramLaunchManifest.abi,
                    'workload_sha256': '0' * 64, 'case_id': 'primary',
                    'program': program.document, 'lowered_sources': {}}
        with self.assertRaisesRegex(ValueError, 'cannot use ordered Program execution'):
            ProgramLaunchManifest.from_dict(manifest)

    def test_rank_placement_rejects_wrong_scope_and_incomplete_ownership(self) -> None:
        mutations = [
            (lambda d: d['execution']['handoffs'][0].__setitem__('scope', 'device'),
             'scope differs from rank placement'),
            (lambda d: d['execution']['placement']['tensor_ranks'].pop('middle0'),
             'own every Program tensor'),
            (lambda d: d['execution']['placement']['worker_ranks'].__setitem__(
                'computation', 0), 'both valid ranks'),
            (lambda d: d['execution']['placement']['tensor_ranks'].__setitem__(
                'out', 1), 'producing rank'),
            (lambda d: d['execution']['placement']['tensor_ranks'].__setitem__(
                'chunks', 1), 'controls must reside'),
            (lambda d: d['execution']['placement'].__setitem__('world_size', 3),
             'exactly two ranks'),
        ]
        for mutate, reason in mutations:
            value = rank_document()
            mutate(value)
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                Program.from_dict(value)
        value = rank_document()
        value['execution']['workers'][1]['phases'].append('combine')
        value['execution']['queues'][2]['workers'].append('computation')
        with self.assertRaisesRegex(ValueError, 'one regular worker rank'):
            Program.from_dict(value)

    def test_complete_topology_roundtrips_without_changing_version_one(self) -> None:
        program = Program.from_dict(document())
        self.assertIsNotNone(program.execution)
        self.assertIsInstance(program.execution, WorkerExecution)
        self.assertEqual([stage.name for stage in program.stages],
                         ["dispatch", "compute", "combine"])
        self.assertEqual(program.execution.steal.stage, "compute")
        self.assertEqual(Program.from_dict(program.document), program)
        old = document()
        old["schema_version"] = 1
        old.pop("execution")
        for name in ("first_ctas", "chunks", "steal_budget"):
            old["tensors"].pop(name)
            old["inputs"].remove(name)
        original = Program.from_dict(old)
        self.assertIsNone(original.execution)
        self.assertEqual(original.document["schema_version"], 1)

    def test_controls_and_handoffs_are_not_optional_or_opaque(self) -> None:
        mutations = [
            (lambda d: d["tensors"]["chunks"].__setitem__("dtype", "fp32"), "control 'chunk_count'"),
            (lambda d: d["execution"]["controls"].__setitem__("chunk_count", "a"), "control 'chunk_count'"),
            (lambda d: d["execution"]["handoffs"].pop(), "one handoff"),
            (lambda d: d["execution"]["handoffs"][0].__setitem__("order", "relaxed"), "handoff 'middle0'"),
            (lambda d: d["execution"]["handoffs"][0].__setitem__("scope", "cluster"), "requires device or system scope"),
            (lambda d: d["execution"]["handoffs"][0].__setitem__("consumer", "combine"), "handoff 'middle0'"),
            (lambda d: d["execution"]["handoffs"].__setitem__(
                1, dict(d["execution"]["handoffs"][0])), "unique and complete"),
            (lambda d: d["execution"]["handoffs"].reverse(), "producer stage order"),
        ]
        for mutate, reason in mutations:
            changed = document()
            mutate(changed)
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                Program.from_dict(changed)

    def test_system_scope_handoff_is_preserved_without_claiming_a_lowering(self) -> None:
        changed = document()
        changed["execution"]["handoffs"][0]["scope"] = "system"
        program = Program.from_dict(changed)
        self.assertIs(program.execution.handoffs[0].scope, HandoffScope.SYSTEM)
        self.assertIs(program.execution.handoffs[1].scope, HandoffScope.DEVICE)
        self.assertEqual(Program.from_dict(program.document), program)
        with self.assertRaisesRegex(ValueError, "dedicated native lowering"):
            Compiler.load(ROOT).lower_program(program)

    def test_queue_owners_phase_order_and_steal_window_are_checked(self) -> None:
        mutations = [
            (lambda d: d["execution"]["queues"].pop(), "one queue"),
            (lambda d: d["execution"]["queues"][1].__setitem__("workers", ["computation"]), "workers differ"),
            (lambda d: d["execution"]["queues"].reverse(), "Program stage and CTA class order"),
            (lambda d: d["execution"]["queues"][1].__setitem__(
                "workers", ["computation", "communication"]), "Program stage and CTA class order"),
            (lambda d: d["execution"]["workers"][0].__setitem__("phases", ["combine", "dispatch"]), "Program dataflow order"),
            (lambda d: d["execution"]["steal"].__setitem__("after", "combine"), "between two ordered"),
            (lambda d: d["execution"]["steal"].__setitem__("borrower", "computation"), "first CTA class"),
        ]
        for mutate, reason in mutations:
            changed = document()
            mutate(changed)
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                Program.from_dict(changed)

    def test_v2_never_replays_as_ordered_or_unqualified_lowering(self) -> None:
        program = Program.from_dict(document())
        compiler = Compiler.load(ROOT)
        with self.assertRaisesRegex(ValueError, "dedicated native lowering"):
            compiler.lower_program(program)
        rewritten = compiler.rewrite_program(program, "specialize_triton_warps", {
            "stage": "dispatch", "num_warps": 4,
            "schedule_id": "other", "entry_point": "other",
        })
        self.assertEqual(rewritten.reason, "execution_kind")
        manifest = {"schema_version": 1, "abi": ProgramLaunchManifest.abi,
                    "workload_sha256": "0" * 64, "case_id": "primary",
                    "program": program.document, "lowered_sources": {}}
        with self.assertRaisesRegex(ValueError, "cannot use ordered Program execution"):
            ProgramLaunchManifest.from_dict(manifest)


if __name__ == "__main__":
    unittest.main()
