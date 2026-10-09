"""Independent small numerical and memory controls for emitted GQA backward."""
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from open_cake_ir.compiler import Compiler, Target
from open_cake_ir.compiler.backends.triton import emit
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.program_frontend import parse_program
from open_cake_ir.evaluation.workload import TensorABI
from tests.contracts.test_epilogue_fusion import rounded
from tests.contracts.test_triton_loop_scopes import _execute, _TL, _Tile

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('gqa_backward', Path(__file__).with_name('gqa_backward.py'))
gqa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gqa)


def bf16(values):
    array = np.asarray(values, dtype=np.float32)
    return np.array([rounded(float(value), 'bf16') for value in array.flat], dtype=np.float32).reshape(array.shape)


def reference(inputs, *, heads, kv_heads):
    """Independent FP32 array contractions and explicit original rounding seams."""
    go = inputs['grad_attn_output']
    weights, dropped = inputs['attn_weights'], inputs['attn_weights_dropped']
    values, mask = inputs['value_states'], inputs['dropout_mask']
    batch, queries, _, width = go.shape
    keys = values.shape[2]
    groups = heads // kv_heads
    dw = np.empty_like(weights)
    dv = np.empty_like(values)
    for b in range(batch):
        for h in range(heads):
            dw[b, h] = go[b, :, h] @ values[b, h // groups].T
        for kv in range(kv_heads):
            head_results = [dropped[b, h].T @ go[b, :, h]
                            for h in range(kv * groups, (kv + 1) * groups)]
            dv[b, kv] = np.sum(np.stack(head_results), axis=0, dtype=np.float32)
    grad = np.asarray(dw * mask / np.float32(.9), dtype=np.float32)
    row = np.sum(grad * weights, axis=-1, keepdims=True, dtype=np.float32)
    scores = weights * (grad - row)
    return {'grad_attn_scores': bf16(scores), 'grad_value_states': bf16(dv)}


def execute(program, inputs):
    """Run emitted stages with typed FP32 arithmetic and bounded CPU memory."""
    memory = {name: value.reshape(-1).tolist() for name, value in inputs.items()}
    target = Target.load(ROOT / 'compiler/targets/xcore1002.json')
    traces = []
    def cast(tile, dtype):
        return _Tile(tile.shape, [int(value) if dtype is _TL.int32 else
            rounded(value, 'fp32' if dtype is _TL.float32 else dtype) for value in tile.values])
    def binary(function):
        def method(left, right):
            return left.binary(right, lambda x, y: rounded(function(x, y), 'fp32')
                if isinstance(x, float) or isinstance(y, float) else function(x, y))
        return method
    def sum_values(values):
        if all(type(value) in (int, bool) for value in values): return sum(values)
        total = 0.
        for value in values: total = rounded(total + value, 'fp32')
        return total
    with ExitStack() as stack:
        stack.enter_context(patch.object(_Tile, 'to', cast))
        stack.enter_context(patch.object(_TL, 'bfloat16', 'bf16', create=True))
        for name, fn in [('__add__', lambda x,y: x+y), ('__radd__', lambda x,y: y+x),
                         ('__sub__', lambda x,y: x-y), ('__mul__', lambda x,y: x*y),
                         ('__rmul__', lambda x,y: y*x), ('__truediv__', lambda x,y: x/y)]:
            stack.enter_context(patch.object(_Tile, name, binary(fn), create=True))
        for name, fn in [('__floordiv__', lambda x,y: x//y), ('__mod__', lambda x,y: x%y),
                         ('__ge__', lambda x,y: x>=y)]:
            stack.enter_context(patch.object(_Tile, name, lambda a,b,fn=fn: a.binary(b,fn), create=True))
        stack.enter_context(patch.object(_TL, 'sum', staticmethod(lambda tile, axis: _TL.reduce(tile, axis, sum_values))))
        for stage in program.stages:
            schedule = Schedule.from_dict(json.loads(stage.schedule_bytes))
            values = {}
            for buffer in schedule.buffers:
                if buffer.space.value != 'global': continue
                name = stage.bindings[buffer.name].tensor
                if buffer.mode.value == 'output': memory[name] = [float('nan')] * int(np.prod(buffer.shape))
                values[buffer.name] = memory[name]
            traces.append(_execute(emit(schedule, target), values))
    for name, values in inputs.items():
        if memory[name] != values.reshape(-1).tolist(): raise AssertionError('input mutated')
    return {name: np.asarray(memory[name], dtype=np.float32).reshape(program.tensors[name].shape)
            for name in program.outputs}, traces


class GQABackward(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.compiler = Compiler.load(ROOT)

    def fixture(self):
        small = dict(QUERY_HEADS=4, KV_HEADS=2, HEAD_DIM=4, KEY_TILE=2, QUERY_TILE=2)
        with patch.multiple(gqa, **small): source = gqa.source_for(1, 3, 3)
        go = bf16(((np.arange(48).reshape(1,3,4,4) % 7) - 3) / 8.)
        values = bf16(((np.arange(24).reshape(1,2,3,4) * 3 % 11) - 5) / 16.)
        weights = bf16(np.tile([.25,.25,.5], (1,4,3,1)))
        mask = (np.arange(36).reshape(1,4,3,3) % 3) != 1
        dropped = bf16(weights * mask / np.float32(.9))
        return source, {'grad_attn_output':go,'attn_weights':weights,'attn_weights_dropped':dropped,
                        'value_states':values,'dropout_mask':mask}

    def test_complete_emitted_program_matches_independent_grouped_backward(self):
        source, inputs = self.fixture()
        program = parse_program(source).program
        observed, traces = execute(program, inputs)
        expected = reference(inputs, heads=4, kv_heads=2)
        for name in expected: np.testing.assert_array_equal(observed[name], expected[name])
        self.assertTrue(all(set(trace.stores.values()) == {1} for trace in traces))

    def test_wrong_dropout_or_group_mapping_changes_the_original_gradient(self):
        source, inputs = self.fixture()
        expected = reference(inputs, heads=4, kv_heads=2)
        no_scale = source.replace('masked / 0.9', 'masked / 1.0')
        wrong_group = source.replace('head_base = kv_index * 2', 'head_base = kv_index * 1')
        wrong_value = source.replace('kv_head = head_index // 2', 'kv_head = head_index % 2')
        for changed, output in [(no_scale,'grad_attn_scores'),(wrong_group,'grad_value_states'),
                                (wrong_value,'grad_attn_scores')]:
            self.assertNotEqual(changed, source)
            observed, _ = execute(parse_program(changed).program, inputs)
            self.assertFalse(np.array_equal(observed[output], expected[output]))

    def test_original_shapes_rounding_and_small_scratch_lower_without_new_capability(self):
        for batch, query, keys in ((32,691,773),(1,4096,4096),(64,128,128)):
            program = gqa.program_for(batch,query,keys)
            lowered = self.compiler.lower_program(program)
            self.assertEqual([item.toolchain_requirements['grid'] for item in lowered.lowerings],
                             [[batch,80,query],[batch,80,query],[batch,8,keys]])
            internal = set(program.tensors) - set(program.inputs) - set(program.outputs)
            self.assertEqual(internal, {'row_sum'})
            self.assertEqual(program.tensors['row_sum'].nbytes, batch*80*query*4)
            self.assertEqual(program.tensors['dropout_mask'].dtype.value, 'bool')
            narrowed = [op['id'] for stage in program.stages for op in json.loads(stage.schedule_bytes)['operations']
                        if op['kind']=='cast' and op['parameters']['to']=='bf16']
            self.assertEqual(narrowed, ['round_score_gradient','round_value_gradient'])
            self.assertEqual(len(program.stages[2].schedule.tile_loops),10)

    def test_original_scalar_and_tensor_order_are_not_optional(self):
        program = gqa.program_for(1,128,128)
        abi = [TensorABI(name,program.tensors[name].shape,program.tensors[name].dtype.value,mode)
               for mode,names in [('input',program.inputs),('output',program.outputs)] for name in names]
        semantics={'fixed_scalar_inputs': {'attention_dropout': {'dtype':'float32','value':.1}}}
        workload=SimpleNamespace(target='xcore1002',tensor_abi=lambda case:abi,document={'semantics':semantics})
        self.assertEqual(gqa.source_for_workload(workload,'primary'),gqa.source_for(1,128,128))
        semantics['fixed_scalar_inputs']={}
        with self.assertRaises(ValueError):gqa.source_for_workload(workload,'primary')
        for value in (0., .2, True):
            with self.assertRaises(ValueError):gqa.source_for(1,128,128,attention_dropout=value)


if __name__ == '__main__': unittest.main()
