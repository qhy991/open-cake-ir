"""Emitted-source semantic controls, with no device or reduction-order claim."""
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import math
import operator
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler, Target
from open_cake_ir.compiler.backends.triton import emit
from open_cake_ir.compiler.program_frontend import parse_program
from open_cake_ir.evaluation.workload import TensorABI
from tests.contracts.test_epilogue_fusion import rounded
from tests.contracts.test_triton_loop_scopes import _TL, _Tile, _execute

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('expert_execution', Path(__file__).with_name('expert_execution.py'))
expert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(expert)


def execute(program, inputs):
    memory = deepcopy(inputs)
    target = Target.load(ROOT / 'compiler/targets/xcore1002.json')
    traces = []
    binary = _Tile.binary

    def rounded_binary(self, other, function):
        tile = binary(self, other, function)
        return _Tile(tile.shape, [rounded(value, 'fp32') if isinstance(value, float) else value
                                 for value in tile.values])

    def cast(tile, dtype):
        return _Tile(tile.shape, [int(value) if dtype in (_TL.int32, 'int64') else
                                 rounded(value, 'bf16' if dtype == 'bf16' else 'fp32')
                                 for value in tile.values])

    with ExitStack() as stack:
        stack.enter_context(patch.object(_Tile, 'binary', rounded_binary))
        stack.enter_context(patch.object(_Tile, 'to', cast))
        stack.enter_context(patch.object(_Tile, '__ge__', lambda a, b: a.binary(b, operator.ge), create=True))
        stack.enter_context(patch.object(_TL, 'int64', 'int64', create=True))
        stack.enter_context(patch.object(_TL, 'bfloat16', 'bf16', create=True))
        for stage in program.stages:
            values = {}
            for buffer in stage.schedule.buffers:
                if buffer.space.value != 'global':
                    continue
                name = stage.bindings[buffer.name].tensor
                if buffer.mode.value == 'output':
                    memory[name] = [float('nan')] * math.prod(buffer.shape)
                values[buffer.name] = memory[name]
            traces.append(_execute(emit(stage.schedule, target), values))
    return memory, traces


def scalar_reference(inputs, tokens, hidden, intermediate, experts, slots, *, round_intermediate=False):
    result = []
    for token in range(tokens):
        rows = []
        for slot in range(slots):
            index = token * slots + slot
            eid = inputs['topk_indices'][index]
            gate, up = [], []
            for column in range(intermediate):
                base = (eid * intermediate + column) * hidden
                gate.append(sum(inputs['hidden_states'][token * hidden + k] * inputs['gate_proj_weights'][base + k]
                                for k in range(hidden)))
                up.append(sum(inputs['hidden_states'][token * hidden + k] * inputs['up_proj_weights'][base + k]
                              for k in range(hidden)))
            value = [rounded(g / (1 + math.exp(-g)) * u, 'bf16' if round_intermediate else 'fp32')
                     for g, u in zip(gate, up)]
            rows.append([rounded(sum(value[k] * inputs['down_proj_weights'][(eid * hidden + column) * intermediate + k]
                                      for k in range(intermediate)) * inputs['topk_weights'][index], 'fp32')
                         for column in range(hidden)])
        order = sorted(range(slots), key=lambda slot: (inputs['topk_indices'][token * slots + slot], slot))
        for column in range(hidden):
            acc = 0.0
            for slot in order:
                acc = rounded(acc + rows[slot][column], 'fp32')
            result.append(rounded(acc, 'bf16'))
    return result


class ExpertExecution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)

    def test_duplicate_routes_weights_and_masked_projection_tails(self):
        tokens, hidden, intermediate, experts, slots = 2, 5, 5, 3, 4
        rng = random.Random(39)
        def values(count):
            return [rounded(rng.uniform(-0.8, 0.8), 'bf16') for _ in range(count)]
        inputs = {'hidden_states': values(tokens * hidden), 'topk_indices': [2, 0, 2, 1, 1, 1, 0, 2],
                  'topk_weights': [0.125, 0.25, 0.5, 0.125, 0.25, 0.5, 0.125, 0.125],
                  'gate_proj_weights': values(experts * intermediate * hidden),
                  'up_proj_weights': values(experts * intermediate * hidden),
                  'down_proj_weights': values(experts * hidden * intermediate)}
        program = parse_program(expert._source(tokens, hidden, intermediate, experts, slots,
                                               tile=4, reduction_tile=4)).program
        observed, traces = execute(program, inputs)
        expected = scalar_reference(inputs, tokens, hidden, intermediate, experts, slots)
        self.assertEqual(observed['output'], expected)
        self.assertTrue(all(observed[name] == value for name, value in inputs.items()))
        self.assertTrue(all(set(trace.stores.values()) == {1} for trace in traces))
        rounded_too_soon = scalar_reference(inputs, tokens, hidden, intermediate, experts, slots,
                                           round_intermediate=True)
        self.assertNotEqual(expected, rounded_too_soon)
        dropped = deepcopy(inputs)
        dropped['topk_weights'][2] = 0.0
        self.assertNotEqual(expected, scalar_reference(dropped, tokens, hidden, intermediate, experts, slots))

    def test_stable_expert_order_has_a_float32_counterexample(self):
        program = parse_program(expert._source(1, 4, 4, 4, 4, tile=4, reduction_tile=4)).program
        final = program.stages[-1]
        single = SimpleNamespace(stages=(final,))
        # Expert order contributes 2**24, 1, -2**24, 0. Slot order contributes
        # 2**24, -2**24, 1, 0 and therefore produces a different FP32 sum.
        inputs = {'topk_indices': [0, 2, 1, 3],
                  'contributions': [float(2**24)] * 4 + [-float(2**24)] * 4 + [1.0] * 4 + [0.0] * 4}
        result, _ = execute(single, inputs)
        self.assertEqual(result['output'], [0.0] * 4)

    def test_public_abi_and_full_size_program_are_explicit(self):
        program = expert.program_for(2447)
        lowered = self.compiler.lower_program(program)
        self.assertEqual(len(lowered.lowerings), 3)
        self.assertEqual(program.tensors['topk_indices'].dtype.value, 'int64')
        self.assertEqual(program.tensors['intermediate_values'].dtype.value, 'fp32')
        self.assertEqual(program.tensors['contributions'].dtype.value, 'fp32')
        abi = tuple(TensorABI(name, program.tensors[name].shape, program.tensors[name].dtype.value, mode)
                    for mode, names in (('input', program.inputs), ('output', program.outputs)) for name in names)
        workload = SimpleNamespace(target=expert.TARGET, tensor_abi=lambda case: abi)
        self.assertEqual(expert.source_for_workload(workload, 'primary'), expert.source_for(2447))
        with self.assertRaises(ValueError):
            expert.source_for_workload(SimpleNamespace(target='gfx938', tensor_abi=lambda case: abi), 'primary')
        with self.assertRaises(ValueError):
            expert.source_for_workload(SimpleNamespace(target=expert.TARGET, tensor_abi=lambda case: abi[::-1]), 'primary')


if __name__ == '__main__':
    unittest.main()
