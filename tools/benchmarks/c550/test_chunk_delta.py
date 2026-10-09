"""Small emitted arithmetic checks; original device tolerances remain separate."""
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import json
import math
import operator
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from open_cake_ir.compiler import Compiler, Target
from open_cake_ir.compiler.backends.triton import emit
from open_cake_ir.compiler.program_frontend import parse_program
from open_cake_ir.evaluation.workload import TensorABI
from tests.contracts.test_epilogue_fusion import rounded
from tests.contracts.test_triton_loop_scopes import _execute, _TL, _Tile

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('chunk_delta', Path(__file__).with_name('chunk_delta.py'))
delta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(delta)


def bf16(values):
    array = np.asarray(values, dtype=np.float32)
    return np.array([rounded(float(value), 'bf16') for value in array.flat], dtype=np.float32).reshape(array.shape)


def normalized_bf16(x):
    square = bf16(x * x)
    total = bf16(np.sum(square, axis=-1, keepdims=True, dtype=np.float32))
    epsilon = bf16(total + np.float32(1e-6))
    inverse = bf16(1. / np.sqrt(epsilon))
    return bf16(x * inverse)


def reference(inputs, scale, chunk):
    """Independent array implementation of the original chunk equations."""
    q = normalized_bf16(inputs['query']) * np.float32(scale)
    k = normalized_bf16(inputs['key'])
    v, beta, g = (inputs[name].astype(np.float32) for name in ('value','beta','g'))
    batch, hk, sequence, width = k.shape
    hv = v.shape[1]
    pad = (-sequence) % chunk
    q = np.pad(q, ((0,0),(0,0),(0,pad),(0,0)))
    k = np.pad(k, ((0,0),(0,0),(0,pad),(0,0)))
    v = np.pad(v, ((0,0),(0,0),(0,pad),(0,0)))
    g = np.pad(g, ((0,0),(0,0),(0,pad)))
    beta = np.pad(beta, ((0,0),(0,0),(0,pad)))
    padded = sequence + pad
    out = np.zeros((batch,hv,padded,width), dtype=np.float32)
    for b in range(batch):
        for h in range(hv):
            state = np.zeros((width,width), dtype=np.float32)
            for start in range(0,padded,chunk):
                qs, ks, vs = q[b,h%hk,start:start+chunk], k[b,h%hk,start:start+chunk], v[b,h,start:start+chunk]
                gate, update = g[b,h,start:start+chunk], beta[b,h,start:start+chunk]
                prefix = np.cumsum(gate, dtype=np.float32)
                decay = np.exp(prefix[:,None]-prefix[None,:]).astype(np.float32)
                kb = ks * update[:,None]
                vb = vs * update[:,None]
                triangular = np.where(np.arange(chunk)[:,None] > np.arange(chunk)[None,:],
                                      -(kb @ ks.T) * decay, 0.).astype(np.float32)
                for i in range(1,chunk):
                    row, sub = triangular[i,:i].copy(), triangular[:i,:i].copy()
                    triangular[i,:i] = row + np.sum(row[:,None] * sub, axis=0, dtype=np.float32)
                transform = triangular + np.eye(chunk, dtype=np.float32)
                transformed_v = transform @ vb
                transformed_k = transform @ (kb * np.exp(gate)[:,None])
                new_v = transformed_v - transformed_k @ state
                intra = np.where(np.arange(chunk)[:,None] >= np.arange(chunk)[None,:],
                                 (qs @ ks.T) * decay, 0.).astype(np.float32)
                inter = (qs * np.exp(gate)[:,None]) @ state
                out[b,h,start:start+chunk] = inter + intra @ new_v
                state = state * np.exp(gate[-1]) + (ks * np.exp(gate[-1]-gate)[:,None]).T @ new_v
    return bf16(out[:,:,:sequence].transpose(0,2,1,3))


def execute(program, inputs):
    """Actual emitted loops/memory with explicit FP32 arithmetic CPU doubles."""
    memory = {name: values.reshape(-1).tolist() for name, values in inputs.items()}
    target = Target.load(ROOT/'compiler/targets/xcore1002.json')
    original_binary = _Tile.binary
    def fp32(value): return rounded(float(value),'fp32')
    def binary(self, other, function):
        result = original_binary(self,other,function)
        return _Tile(result.shape,[fp32(value) if isinstance(value,float) else value for value in result.values])
    def cast(tile,dtype):
        return _Tile(tile.shape,[int(value) if dtype is _TL.int32 else
                                rounded(value,'fp32' if dtype is _TL.float32 else dtype) for value in tile.values])
    def sum_values(values):
        values=list(values)
        if all(type(value) in (int,bool) for value in values):return sum(values)
        result=0.
        for value in values:result=fp32(result+value)
        return result
    def dot(a,b,**kwargs):
        m,k=a.shape;kb,n=b.shape
        if k!=kb:raise ValueError('CPU dot dimensions differ')
        return _Tile((m,n),[sum_values(fp32(a.at((i,t))*b.at((t,j))) for t in range(k))
                            for i in range(m) for j in range(n)])
    traces=[]
    with ExitStack() as stack:
        stack.enter_context(patch.object(_Tile,'binary',binary))
        stack.enter_context(patch.object(_Tile,'to',cast))
        for name,fn in [('__floordiv__',operator.floordiv),('__mod__',operator.mod),
                        ('__ge__',operator.ge),('__le__',operator.le),('__gt__',operator.gt)]:
            stack.enter_context(patch.object(_Tile,name,lambda a,b,fn=fn:a.binary(b,fn),create=True))
        stack.enter_context(patch.object(_TL,'bfloat16','bf16',create=True))
        stack.enter_context(patch.object(_TL,'sum',staticmethod(lambda value,axis:_TL.reduce(value,axis,sum_values))))
        stack.enter_context(patch.object(_TL,'dot',staticmethod(dot)))
        stack.enter_context(patch.object(_TL,'exp',staticmethod(lambda tile:_Tile(tile.shape,[fp32(math.exp(value)) for value in tile.values]))))
        stack.enter_context(patch.object(_TL,'rsqrt',staticmethod(lambda tile:_Tile(tile.shape,[fp32(1/math.sqrt(value)) for value in tile.values])),create=True))
        for stage in program.stages:
            values={}
            for buffer in stage.schedule.buffers:
                if buffer.space.value!='global':continue
                name=stage.bindings[buffer.name].tensor
                if buffer.mode.value=='output':memory[name]=[float('nan')]*math.prod(buffer.shape)
                values[buffer.name]=memory[name]
            traces.append(_execute(emit(stage.schedule,target),values))
    for name,values in inputs.items():
        if memory[name]!=values.reshape(-1).tolist():raise AssertionError('Input bytes changed in the CPU model')
    output=np.asarray(memory['output'],dtype=np.float32).reshape(program.tensors['output'].shape)
    return output,memory,traces


class ChunkDelta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.compiler=Compiler.load(ROOT)

    def fixture(self, sequence=9):
        rng=np.random.default_rng(71)
        inputs={'query':bf16(rng.uniform(-.8,.8,(1,2,sequence,4))),
                'key':bf16(rng.uniform(-.8,.8,(1,2,sequence,4))),
                'value':bf16(rng.uniform(-.5,.5,(1,4,sequence,4))),
                'g':bf16(rng.uniform(-.2,.15,(1,4,sequence))),
                'beta':bf16(rng.uniform(.1,.3,(1,4,sequence)))}
        source=delta._source(1,sequence,.5,heads_k=2,heads_v=4,width=4,chunk=4,column_tile=2)
        return source,inputs

    def test_emitted_chunk_equations_cover_tail_boundary_and_zero_initial_state(self):
        for sequence in (3,4,5,9):
            source,inputs=self.fixture(sequence)
            program=parse_program(source).program
            observed,_,traces=execute(program,inputs)
            expected=reference(inputs,.5,4)
            np.testing.assert_allclose(observed,expected,rtol=.01,atol=1e-4)
            self.assertTrue(all(set(trace.stores.values())=={1} for trace in traces))
            self.assertEqual(program.outputs,('output',))
            self.assertNotIn(f'state_{(sequence+3)//4}',program.tensors)
        source,inputs=self.fixture(9)
        inputs['beta']=bf16(inputs['beta']-np.float32(.2))
        observed,_,_=execute(parse_program(source).program,inputs)
        np.testing.assert_allclose(observed,reference(inputs,.5,4),rtol=.01,atol=1e-4)

    def test_causal_group_and_recurrent_dependency_counterexamples(self):
        source,inputs=self.fixture(9)
        expected=reference(inputs,.5,4)
        mutations=[source.replace('key_head = head_index % 2','key_head = head_index // 2'),
                   source.replace('lm.compare(lanes, row, op="le", id="causal_domain")',
                                  'lm.compare(lanes, row, op="lt", id="causal_domain")'),
                   source.replace('corrected = u - correction','corrected = u + correction'),
                   source.replace('        decay = lm.exp(raw_g, id="raw_gate_exp")',
                                  '        decay = lm.exp(row_prefix, id="raw_gate_exp")')]
        for changed in mutations:
            self.assertNotEqual(changed,source)
            actual,_,_=execute(parse_program(changed).program,inputs)
            self.assertFalse(np.allclose(actual,expected,rtol=.01,atol=1e-4))

    def test_normalization_keeps_bf16_boundaries_before_fp32(self):
        source,inputs=self.fixture(3)
        program=parse_program(source).program
        _,memory,_=execute(program,inputs)
        expected=normalized_bf16(inputs['key'])
        observed=np.asarray(memory['k_normalized'],dtype=np.float32).reshape(expected.shape)
        np.testing.assert_array_equal(observed,expected)
        fp32=inputs['key']/np.sqrt(np.sum(inputs['key']**2,axis=-1,keepdims=True)+1e-6)
        self.assertFalse(np.array_equal(observed,fp32))
        first=json.loads(program.stages[0].schedule_bytes)
        rounds=[op for op in first['operations'] if op['kind']=='cast' and op['parameters']['to']=='bf16']
        self.assertEqual(len(rounds),10)

    def test_original_shape_has_explicit_states_and_checked_scalar(self):
        program=delta.program_for(1,131)
        lowered=self.compiler.lower_program(program)
        self.assertEqual(len(lowered.lowerings),12)
        self.assertEqual(program.tensors['output'].shape,(1,131,16,128))
        self.assertEqual(program.tensors['state_1'].shape,(1,16,128,128))
        self.assertEqual(program.tensors['state_2'].shape,(1,16,128,128))
        abi=[TensorABI(name,program.tensors[name].shape,program.tensors[name].dtype.value,mode)
             for mode,names in [('input',program.inputs),('output',program.outputs)] for name in names]
        scale=1/math.sqrt(128)
        semantics={'fixed_scalar_inputs':{'scale':{'dtype':'float32','value':scale}}}
        workload=SimpleNamespace(target='xcore1002',tensor_abi=lambda case:abi,document={'semantics':semantics})
        self.assertEqual(delta.source_for_workload(workload,'primary'),delta.source_for(1,131))
        semantics['fixed_scalar_inputs']={}
        with self.assertRaises(ValueError):delta.source_for_workload(workload,'primary')


if __name__=='__main__':unittest.main()
