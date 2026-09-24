"""A system instruction does not by itself admit a peer-owned pointer."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import Target
from open_cake_ir.evaluation.peer_atomic import (
    PeerAtomicBinding, PeerCapabilities, bind_peer_atomic_state,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'corpus/schedules/atomic-reservation-b8-smoke.json'
CONTRACT = 'ptx.atom.relaxed.sys.global.add.s32'


def schedule_document():
    document = json.loads(SOURCE.read_text())
    document['target'] = 'sm_103a'
    document['lowering']['backend'] = 'native_cuda'
    next(row for row in document['operations'] if row['kind'] == 'atomic_rmw')[
        'parameters']['scope'] = 'system'
    return document


class PeerAtomicBindingContract(unittest.TestCase):
    def setUp(self):
        self.schedule = Schedule.from_dict(schedule_document())
        target = Target.load(ROOT / 'compiler/targets/sm_103a.json')
        self.target = replace(target,
            instruction_contracts=target.instruction_contracts | {CONTRACT})
        self.buffers = {buffer.name: index + 1000
                        for index, buffer in enumerate(self.schedule.buffers)
                        if buffer.space.value == 'global'}
        self.calls = []

    def bind(self, **changes):
        arguments = dict(
            execution_device=0,
            pointer_owner=lambda pointer: 1 if pointer == self.buffers['counts'] else 0,
            probe_peer=lambda source, owner: self.observe(source, owner),
            enable_peer=lambda source, owner: self.enable(source, owner),
        )
        arguments.update(changes)
        return bind_peer_atomic_state(self.schedule, self.target, self.buffers,
                                      **arguments)

    def observe(self, source, owner):
        self.calls.append(('probe', source, owner))
        return PeerCapabilities(True, True)

    def enable(self, source, owner):
        self.calls.append(('enable', source, owner))
        return True

    def test_exact_pair_is_observed_before_enabling(self):
        bound = self.bind()
        self.assertEqual(bound, PeerAtomicBinding('sm_103a', 'counts', 0, 1))
        self.assertEqual(self.calls, [('probe', 0, 1), ('enable', 0, 1)])

    def test_missing_contract_scope_or_remote_owner_refuses(self):
        missing = replace(self.target,
            instruction_contracts=self.target.instruction_contracts - {CONTRACT})
        with self.assertRaisesRegex(ValueError, 'exact native CUDA system contract'):
            bind_peer_atomic_state(self.schedule, missing, self.buffers,
                execution_device=0, pointer_owner=lambda _: 1,
                probe_peer=self.observe, enable_peer=self.enable)
        local_document = schedule_document()
        next(row for row in local_document['operations'] if row['kind'] == 'atomic_rmw')[
            'parameters']['scope'] = 'device'
        local = Schedule.from_dict(local_document)
        with self.assertRaisesRegex(ValueError, 'one system-scope state target'):
            bind_peer_atomic_state(local, self.target,
                self.buffers, execution_device=0, pointer_owner=lambda _: 1,
                probe_peer=self.observe, enable_peer=self.enable)
        with self.assertRaisesRegex(ValueError, 'belong to another observed GPU'):
            self.bind(pointer_owner=lambda _: 0)
        self.assertEqual(self.calls, [])

    def test_pair_capability_and_mapping_are_not_inherited(self):
        for fact in (PeerCapabilities(False, True), PeerCapabilities(True, False)):
            with self.subTest(fact=fact):
                self.calls.clear()
                with self.assertRaisesRegex(ValueError, 'exact directed GPU pair'):
                    self.bind(probe_peer=lambda source, owner: fact)
                self.assertEqual(self.calls, [])
        with self.assertRaisesRegex(ValueError, 'was not enabled'):
            self.bind(enable_peer=lambda source, owner: False)
        self.assertEqual(self.calls, [('probe', 0, 1)])
        with self.assertRaisesRegex(ValueError, 'cover every global'):
            bind_peer_atomic_state(self.schedule, self.target, {'counts': 1000},
                execution_device=0, pointer_owner=lambda _: 1,
                probe_peer=self.observe, enable_peer=self.enable)


if __name__ == '__main__':
    unittest.main()
