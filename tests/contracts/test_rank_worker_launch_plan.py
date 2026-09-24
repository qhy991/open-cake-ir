"""Two-rank worker storage and launch are bound without ordered fallback."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import unittest

from open_cake_ir.compiler import Program
from open_cake_ir.compiler.program import LoweredWorkerProgram
from open_cake_ir.evaluation.launch_plan import prepare_rank_worker_program
from tests.contracts.test_worker_execution import rank_document


@dataclass
class Storage:
    address: int
    nbytes: int
    device: int
    status: int = 0


def lowered_worker():
    program = Program.from_dict(rank_document())
    placement = program.execution.placement
    return LoweredWorkerProgram(program, 'fixture', 'two-rank generated source', {
        'target': program.target,
        'entry_point': program.execution.lowering.entry_point,
        'world_size': 2, 'state_rank': 0,
        'tensor_ranks': dict(placement.tensor_ranks),
        'state_bytes': 32, 'state_status_offset_bytes': 28,
        'state_reset': 'zero_before_both_launches_on_rank0_stream',
        'argument_order': list(program.tensors),
        'grid_per_rank': [[148, 1, 1], [148, 1, 1]],
        'cooperative_grid': True, 'peer_pair_runtime_check': True,
        'host_abi': {'create_rank': 'create', 'launch_two': 'launch',
                     'destroy_rank': 'destroy'},
    })


class RankWorkerLaunchPlanTests(unittest.TestCase):
    def setUp(self):
        self.lowered = lowered_worker()
        self.inputs = {name: Storage(10000 + index * 100000,
                                     self.lowered.program.tensors[name].nbytes,
                                     self.lowered.program.execution.placement.tensor_ranks[name])
                       for index, name in enumerate(self.lowered.program.inputs)}
        self.address = 1000000
        self.contexts = ((0, 'stream0'), (1, 'stream1'))
        self.calls = []

    def allocate(self, name, spec, rank):
        self.address += 100000
        return Storage(self.address, spec.nbytes, rank)

    def state(self, size, rank):
        self.address += 100000
        return Storage(self.address, size, rank)

    @staticmethod
    def span(tensor):
        return tensor.device, tensor.address, tensor.address + tensor.nbytes

    def load(self, lowered, buffers, state):
        self.assertIs(lowered, self.lowered)
        self.assertEqual(tuple(buffers), tuple(lowered.program.tensors))
        self.assertEqual(state.device, 0)
        self.calls.append('load')
        def launch(contexts):
            self.assertEqual(contexts, self.contexts)
            self.calls.append('two-kernel-launch')
            state.status = 0
        return launch

    def read_status(self, state, offset, contexts):
        self.assertEqual(offset, 28)
        self.assertEqual(contexts, self.contexts)
        self.calls.append('synchronized-status')
        return state.status

    def prepare(self, **changes):
        callbacks = dict(allocate=self.allocate, allocate_state=self.state,
            load_worker=self.load,
            check_tensor=lambda tensor, spec: self.assertEqual(tensor.nbytes, spec.nbytes),
            storage_span=self.span,
            execution_context=lambda rank: self.contexts[rank],
            read_status=self.read_status)
        callbacks.update(changes)
        return prepare_rank_worker_program(self.lowered, self.inputs, **callbacks)

    def test_one_combined_call_uses_declared_rank_owners(self):
        prepared = self.prepare()
        self.assertEqual(prepared.buffers['middle0'].device, 1)
        self.assertEqual(prepared.buffers['middle1'].device, 0)
        self.assertEqual(prepared.state.device, 0)
        for _ in range(2):
            self.assertIs(prepared.run()['out'], prepared.buffers['out'])
        self.assertEqual(prepared.launch_calls, 2)
        self.assertEqual(self.calls, ['load', 'two-kernel-launch', 'synchronized-status',
                                      'two-kernel-launch', 'synchronized-status'])

    def test_wrong_rank_extent_and_alias_refuse_before_load(self):
        self.inputs['a'].device = 1
        with self.assertRaisesRegex(ValueError, 'storage owner'):
            self.prepare()
        self.assertEqual(self.calls, [])
        self.inputs['a'].device = 0
        with self.assertRaisesRegex(ValueError, 'storage owner'):
            self.prepare(allocate_state=lambda size, rank: Storage(2000000, size, 1))
        with self.assertRaisesRegex(ValueError, 'extent differs'):
            self.prepare(allocate_state=lambda size, rank: Storage(2000000, size - 4, 0))
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            self.prepare(allocate_state=lambda size, rank:
                         Storage(self.inputs['a'].address, size, 0))

    def test_metadata_status_and_stream_fail_closed(self):
        original = self.lowered
        for field, value in (
            ('world_size', 1), ('state_rank', 1), ('argument_order', ['a']),
            ('peer_pair_runtime_check', False), ('cooperative_grid', False),
            ('state_status_offset_bytes', 32),
        ):
            with self.subTest(field=field):
                self.lowered = replace(original, toolchain_requirements={
                    **original.toolchain_requirements, field: value})
                with self.assertRaisesRegex(ValueError, 'placement, state or host ABI'):
                    self.prepare()
        self.lowered = original
        prepared = self.prepare(read_status=lambda state, offset, contexts: -1)
        with self.assertRaisesRegex(ValueError, 'returned status -1'):
            prepared.run()
        prepared = self.prepare()
        self.contexts = ((0, 'changed'), (1, 'stream1'))
        with self.assertRaisesRegex(ValueError, 'device/stream changed'):
            prepared.run()


if __name__ == '__main__':
    unittest.main()
