"""Exact integer controls for the emitted CAKE starter; no device claims."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler, Target
from open_cake_ir.compiler.backends.triton import emit
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.program_frontend import parse_program
from open_cake_ir.evaluation.workload import TensorABI
from tests.contracts.test_triton_loop_scopes import _execute

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('stable_routing', Path(__file__).with_name('stable_routing.py'))
routing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(routing)


def execute(program, keys):
    """Run actual emitted stages with independent bounded CPU memory effects."""
    memory = {'topk_idx': keys[:]}
    target = Target.load(ROOT / 'compiler/targets/xcore1002.json')
    traces = []
    for stage in program.stages:
        schedule = Schedule.from_dict(json.loads(stage.schedule_bytes))
        values = {}
        for buffer in schedule.buffers:
            if buffer.space.value != 'global': continue
            name = stage.bindings[buffer.name].tensor
            if buffer.mode.value == 'output':
                memory[name] = [-999] * math.prod(buffer.shape)
            values[buffer.name] = memory[name]
        trace = _execute(emit(schedule, target), values)
        traces.append(trace)
    return {name: memory[name] for name in program.outputs}, memory, traces


class StableRouting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)

    def assert_semantics(self, batch, sequence, keys, *, tile=4):
        with patch.object(routing, 'SCAN_ROWS', tile):
            program = routing.program_for(batch, sequence)
        observed, memory, traces = execute(program, keys)
        expected_order = sorted(range(len(keys)), key=lambda index: (keys[index], index))
        expected_offsets = [sum(value < boundary for value in keys) for boundary in range(257)]
        self.assertEqual(observed['sorted_token_indices'], expected_order)
        self.assertEqual(observed['expert_offsets'], expected_offsets)
        self.assertEqual(sorted(memory['ranks']), list(range(len(keys))))
        self.assertEqual(memory['topk_idx'], keys)
        self.assertTrue(all(set(trace.stores.values()) == {1} for trace in traces))
        self.assertNotEqual(expected_order, [index // 8 for index in expected_order])
        return program, observed

    def test_stable_ties_across_batch_row_and_slot_with_masked_tails(self):
        keys = [((index * 7) % 11) if index % 9 else 255 for index in range(80)]
        self.assert_semantics(2, 5, keys)

    def test_empty_experts_and_boundary_offsets_use_the_actual_baseline_tile(self):
        self.assert_semantics(1, 1, [0, 255, 0, 4, 4, 255, 1, 0], tile=128)
        self.assert_semantics(1, 1, [0] * 8)
        self.assert_semantics(1, 1, [255] * 8)

    def test_wrong_tie_order_is_a_permutation_but_fails_the_original_stability_rule(self):
        keys = [3] * 16
        with patch.object(routing, 'SCAN_ROWS', 4):
            source = routing.source_for(1, 2)
        bad = source.replace('lm.compare(slot_indices, wanted_slot, op="lt", id="earlier_slot")',
                             'lm.compare(slot_indices, wanted_slot, op="gt", id="earlier_slot")')
        self.assertNotEqual(source, bad)
        observed, _, _ = execute(parse_program(bad).program, keys)
        self.assertEqual(sorted(observed['sorted_token_indices']), list(range(16)))
        self.assertNotEqual(observed['sorted_token_indices'], list(range(16)))

    def test_original_abi_unique_ownership_and_full_shape_lower(self):
        for batch, sequence in ((8, 256), (64, 128), (1, 2080)):
            program = routing.program_for(batch, sequence)
            lowered = self.compiler.lower_program(program)
            self.assertEqual(program.inputs, ('topk_idx',))
            self.assertEqual(program.outputs, ('sorted_token_indices', 'expert_offsets'))
            self.assertEqual(program.tensors['topk_idx'].shape, (batch, sequence, 8))
            self.assertEqual(program.tensors['ranks'].shape, (batch, sequence, 8))
            self.assertEqual([item.toolchain_requirements['grid'] for item in lowered.lowerings],
                             [[batch, sequence, 8], [batch * sequence * 8, 1, 1], [257, 1, 1]])
            for stage in program.stages:
                raw = json.loads(stage.schedule_bytes)
                self.assertFalse({'scan', 'atomic_rmw', 'top_k'} & {op['kind'] for op in raw['operations']})
                stores = [op for op in raw['operations'] if op['kind'] == 'store']
                self.assertEqual(len(stores), 1)
                access = next(item for item in raw['access_maps'] if item['operation'] == stores[0]['id'])
                self.assertTrue(all(index['source'] == 'program' for index in access['indices']))

    def test_external_workload_abi_cannot_be_flattened_or_return_token_ids(self):
        abi = [TensorABI('topk_idx', (2, 3, 8), 'int32', 'input'),
               TensorABI('sorted_token_indices', (48,), 'int32', 'output'),
               TensorABI('expert_offsets', (257,), 'int32', 'output')]
        workload = SimpleNamespace(target='xcore1002', tensor_abi=lambda case: abi)
        self.assertEqual(routing.source_for_workload(workload, 'primary'), routing.source_for(2, 3))
        for index, changed in ((0, replace(abi[0], shape=(48,))),
                               (1, replace(abi[1], shape=(6,))),
                               (2, replace(abi[2], dtype='fp32'))):
            wrong = list(abi); wrong[index] = changed
            with self.assertRaises(ValueError):
                routing.source_for_workload(SimpleNamespace(target='xcore1002', tensor_abi=lambda case: wrong), 'primary')


if __name__ == '__main__':
    unittest.main()
