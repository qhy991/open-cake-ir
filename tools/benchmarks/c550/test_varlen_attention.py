"""Bounded emitted-source controls; no native or original-device qualification."""
from contextlib import ExitStack
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler, Target
from open_cake_ir.compiler.backends.triton import emit
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.program_frontend import parse_program
from open_cake_ir.evaluation.workload import TensorABI
from tests.contracts.test_triton_loop_scopes import _TL, _Tile, _execute

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('varlen_attention', Path(__file__).with_name('varlen_attention.py'))
attention = importlib.util.module_from_spec(spec)
spec.loader.exec_module(attention)


def fp32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def bf16(value):
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    if bits & 0x7f800000 != 0x7f800000:
        bits += 0x7fff + ((bits >> 16) & 1)
    return struct.unpack('<f', struct.pack('<I', bits & 0xffff0000))[0]


def fixtures(tokens, boundaries, dimensions):
    embed, heads, depth = dimensions
    quantized = lambda count, multiplier, modulus, scale: [bf16(((i * multiplier) % modulus - modulus // 2) * scale) for i in range(count)]
    return {
        'hidden_states': quantized(tokens * embed, 5, 13, .125),
        'cu_seqlens': list(boundaries),
        'cos': quantized(tokens * embed, 3, 11, .125),
        'sin': quantized(tokens * embed, 7, 9, .125),
        'qkv_weight': quantized(3 * embed * embed, 7, 9, .125),
        'qkv_bias': quantized(3 * embed, 3, 5, .0625),
        'proj_weight': quantized(embed * embed, 11, 7, .125),
        'proj_bias': [bf16(.125)] * embed,
    }


def linear(values, weights, bias, rows, inner, columns):
    return [bf16(fp32(sum(fp32(values[r*inner+k] * weights[c*inner+k]) for k in range(inner))) + bias[c])
            for r in range(rows) for c in range(columns)]


def oracle(inputs, tokens, dimensions):
    embed, heads, depth = dimensions
    if inputs['cu_seqlens'][-1] == 0:
        return [0.] * (tokens * embed)
    packed = linear(inputs['hidden_states'], inputs['qkv_weight'], inputs['qkv_bias'], tokens, embed, 3*embed)
    projections = [[packed[t*3*embed+p*embed+j] for t in range(tokens) for j in range(embed)] for p in range(3)]
    for part in (0, 1):
        before = projections[part][:]
        for t in range(tokens):
            for h in range(heads):
                for d in range(depth):
                    index = t*embed+h*depth+d
                    rotated = before[t*embed+h*depth+(d+depth//2 if d<depth//2 else d-depth//2)]
                    rotated *= -1 if d<depth//2 else 1
                    projections[part][index] = bf16(bf16(before[index]*inputs['cos'][index])
                                                          + bf16(rotated*inputs['sin'][index]))
    q, k, v = projections
    result = [0.] * (tokens * embed)
    start = 0
    for stop in inputs['cu_seqlens']:
        if stop <= start:
            continue
        for t in range(start, stop):
            for h in range(heads):
                scores = [bf16(bf16(fp32(sum(fp32(q[t*embed+h*depth+d]*k[j*embed+h*depth+d])
                                          for d in range(depth)))) * fp32(depth ** -.5)) for j in range(start,stop)]
                maximum = max(scores)
                exps = [fp32(math.exp(fp32(value-maximum))) for value in scores]
                total = fp32(sum(exps))
                probs = [bf16(fp32(value/total)) for value in exps]
                for d in range(depth):
                    result[t*embed+h*depth+d] = bf16(fp32(sum(fp32(probs[j-start]*v[j*embed+h*depth+d])
                                                                              for j in range(start,stop))))
        start = stop
    return linear(result, inputs['proj_weight'], inputs['proj_bias'], tokens, embed, embed)


def execute(program, inputs):
    memory = {name: values[:] for name, values in inputs.items()}
    target = Target.load(ROOT/'compiler/targets/xcore1002.json')
    bf, i64 = object(), object()
    original_binary = _Tile.binary
    def binary(tile, other, function):
        result = original_binary(tile, other, function)
        return _Tile(result.shape, [fp32(x) if isinstance(x, float) else x for x in result.values])
    def cast(tile, dtype):
        fn = bf16 if dtype is bf else (fp32 if dtype is _TL.float32 else int)
        return _Tile(tile.shape, [fn(x) for x in tile.values])
    def reduce(tile, axis):
        result = _TL.reduce(tile, axis, sum)
        return _Tile(result.shape, [fp32(x) if isinstance(x, float) else x for x in result.values])
    traces = []
    with ExitStack() as stack:
        stack.enter_context(patch.object(_TL, 'bfloat16', bf, create=True))
        stack.enter_context(patch.object(_TL, 'int64', i64, create=True))
        stack.enter_context(patch.object(_Tile, 'binary', binary))
        stack.enter_context(patch.object(_Tile, 'to', cast))
        stack.enter_context(patch.object(_Tile, '__le__', lambda tile,value:tile.binary(value,lambda a,b:a<=b), create=True))
        stack.enter_context(patch.object(_Tile, '__ge__', lambda tile,value:tile.binary(value,lambda a,b:a>=b), create=True))
        stack.enter_context(patch.object(_Tile, '__gt__', lambda tile,value:tile.binary(value,lambda a,b:a>b), create=True))
        stack.enter_context(patch.object(_TL, 'sum', staticmethod(reduce)))
        stack.enter_context(patch.object(_TL, 'exp', staticmethod(lambda tile:_Tile(tile.shape,[fp32(math.exp(x)) for x in tile.values]))))
        for stage in program.stages:
            schedule = Schedule.from_dict(json.loads(stage.schedule_bytes))
            arguments = {}
            for buffer in schedule.buffers:
                if buffer.space.value != 'global':
                    continue
                public = stage.bindings[buffer.name].tensor
                if buffer.mode.value == 'output':
                    memory[public] = [-999] * math.prod(buffer.shape)
                arguments[buffer.name] = memory[public]
            traces.append(_execute(emit(schedule,target), arguments))
    return memory, traces


class VarlenVisionAttention(unittest.TestCase):
    def check(self, tokens, boundaries, dimensions):
        inputs = fixtures(tokens, boundaries, dimensions)
        program = attention.program_for(tokens, len(boundaries), dimensions=dimensions)
        observed, traces = execute(program, inputs)
        expected = oracle(inputs, tokens, dimensions)
        self.assertEqual(observed['output'], expected)
        for name, values in inputs.items():
            self.assertEqual(observed[name], values)
        self.assertTrue(all(set(trace.stores.values()) == {1} for trace in traces))
        return observed

    def test_ragged_boundaries_empty_segments_and_bidirectional_attention(self):
        observed = self.check(5, [0,2,2,5], (8,2,4))
        self.assertEqual(observed['segments'], [1,1,3,3,3])
        self.assertTrue(math.isfinite(observed['scores'][1]))
        self.assertEqual(observed['scores'][2], -math.inf)
        self.check(3, [1,2,3,3,3], (8,2,4))

    def test_all_zero_lengths_return_zero_even_with_projection_bias(self):
        self.check(3, [0,0], (8,2,4))

    def test_projection_and_head_loop_boundaries(self):
        self.check(1, [1], (144,18,8))
        self.check(3, [1,3], (32,2,16))

    def test_padded_key_and_value_loop_tail(self):
        self.check(129, [64,129], (2,1,2))

    def test_cross_sequence_mask_counterexample(self):
        inputs = fixtures(5, [2,5], (8,2,4))
        source = attention.source_for(5,2,dimensions=(8,2,4))
        changed = source.replace('masked = lm.select(same_segment, scaled_fp32, "negative_infinity", id="mask_segments")',
                                 'masked = scaled_fp32 + 0.0')
        self.assertNotEqual(changed,source)
        observed,_ = execute(parse_program(changed).program,inputs)
        self.assertNotEqual(observed['output'],oracle(inputs,5,(8,2,4)))

    def test_rope_rounds_products_before_their_sum(self):
        program = attention.program_for(1,1,dimensions=(4,1,4))
        rotary = next(stage for stage in program.stages if stage.name=='rotary')
        data = {'q0':[1.0078125,1.0078125,1.015625,1.015625],
                'k0':[1.0078125]*4,'cos':[.9921875]*4,'sin':[1.0078125]*4}
        # Only the emitted rotary stage is needed for this numerical discriminator.
        observed,_ = execute(SimpleNamespace(stages=(rotary,)),data)
        expected = bf16(bf16(1.0078125*.9921875)+bf16(-1.015625*1.0078125))
        fused = bf16(1.0078125*.9921875-1.015625*1.0078125)
        self.assertNotEqual(expected,fused)
        self.assertEqual(observed['q'][0],expected)

    def test_public_binder_keeps_original_int64_endpoints_and_tensor_order(self):
        t,s,e,h,d=5,4,*attention.DIMENSIONS
        abi=[TensorABI('hidden_states',(t,e),'bf16','input'), TensorABI('cu_seqlens',(s,),'int64','input'),
             TensorABI('cos',(t,h,d),'bf16','input'), TensorABI('sin',(t,h,d),'bf16','input'),
             TensorABI('qkv_weight',(3*e,e),'bf16','input'), TensorABI('qkv_bias',(3*e,),'bf16','input'),
             TensorABI('proj_weight',(e,e),'bf16','input'), TensorABI('proj_bias',(e,),'bf16','input'),
             TensorABI('output',(t,e),'bf16','output')]
        workload=SimpleNamespace(target=attention.TARGET,tensor_abi=lambda case:abi)
        self.assertEqual(attention.source_for_workload(workload),attention.source_for(t,s))
        for index,change in ((1,{'dtype':'int32'}),(2,{'shape':(t,e)}),
                             (4,{'shape':(e,3*e)}),(8,{'dtype':'fp32'})):
            altered=list(abi);altered[index]=replace(abi[index],**change)
            with self.assertRaisesRegex(ValueError,'original tensor shape, dtype or order'):
                attention.source_for_workload(SimpleNamespace(target=attention.TARGET,tensor_abi=lambda case:altered))


if __name__=='__main__':
    unittest.main()
