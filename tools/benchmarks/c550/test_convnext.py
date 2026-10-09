"""Small emitted-source CPU controls. Device reduction order remains unqualified."""
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler, Target, frontend
from open_cake_ir.compiler.backends.triton import emit
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.program_frontend import parse_program
from open_cake_ir.evaluation.workload import TensorABI
from tests.contracts.test_triton_loop_scopes import _TL, _Tile, _execute

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('convnext', Path(__file__).with_name('convnext.py'))
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


def fp32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def fixture(b,c,h,w):
    shapes = {'x':(b,c,h,w), 'dwconv_weight':(c,1,7,7), 'dwconv_bias':(c,),
        'layernorm_weight':(c,), 'layernorm_bias':(c,), 'pwconv1_weight':(4*c,c),
        'pwconv1_bias':(4*c,), 'grn_weight':(1,1,1,4*c), 'grn_bias':(1,1,1,4*c),
        'pwconv2_weight':(c,4*c), 'pwconv2_bias':(c,)}
    return {name:[fp32(((i*7+3*j)%19-9)*(.125 if name=='x' else .03125))
                  for i in range(math.prod(shape))] for j,(name,shape) in enumerate(shapes.items())}


def reference(data,b,c,h,w,eps,ln_eps):
    """Scalar high-level computation with independent index loops and math.erf."""
    p=h*w;f=4*c
    dw=[]
    for batch in range(b):
        for y in range(h):
            for x in range(w):
                for channel in range(c):
                    total=0.
                    for i in range(7):
                        for j in range(7):
                            yi,xi=y+i-3,x+j-3
                            if 0<=yi<h and 0<=xi<w:
                                total += data['x'][((batch*c+channel)*h+yi)*w+xi]*data['dwconv_weight'][(channel*7+i)*7+j]
                    dw.append(total+data['dwconv_bias'][channel])
    normalized=[]
    for row in range(b*p):
        values=dw[row*c:(row+1)*c];mean=sum(values)/c
        inverse=1/math.sqrt(sum((v-mean)**2 for v in values)/c+ln_eps)
        normalized.extend((v-mean)*inverse*data['layernorm_weight'][j]+data['layernorm_bias'][j] for j,v in enumerate(values))
    activated=[]
    for row in range(b*p):
        for feature in range(f):
            z=sum(normalized[row*c+k]*data['pwconv1_weight'][feature*c+k] for k in range(c))+data['pwconv1_bias'][feature]
            activated.append(z*.5*(1+math.erf(z/math.sqrt(2))))
    norm=[]
    for batch in range(b):
        spatial=[math.sqrt(sum(activated[(batch*p+pixel)*f+k]**2 for pixel in range(p))) for k in range(f)]
        mean=sum(spatial)/f
        norm.extend(value/(mean+eps) for value in spatial)
    out=[0.]*(b*c*h*w)
    for batch in range(b):
        for pixel in range(p):
            grn=[data['grn_weight'][k]*(activated[(batch*p+pixel)*f+k]*norm[batch*f+k])+data['grn_bias'][k]+activated[(batch*p+pixel)*f+k] for k in range(f)]
            for channel in range(c):
                index=(batch*c+channel)*p+pixel
                out[index]=sum(grn[k]*data['pwconv2_weight'][channel*f+k] for k in range(f))+data['pwconv2_bias'][channel]+data['x'][index]
    return out


def execute(program, inputs):
    memory=deepcopy(inputs);traces=[]
    target=Target.load(ROOT/'compiler/targets/xcore1002.json')
    binary=_Tile.binary;dot=_TL.dot
    def rounded_binary(tile,other,fn):
        result=binary(tile,other,fn)
        return _Tile(result.shape,[fp32(v) if isinstance(v,float) else v for v in result.values])
    def sum_axis(tile,axis):
        result=_TL.reduce(tile,axis,sum)
        return _Tile(result.shape,[fp32(v) for v in result.values])
    def dot_rounded(a,b,**kwargs):
        result=dot(a,b,**kwargs)
        return _Tile(result.shape,[fp32(v) for v in result.values])
    with ExitStack() as stack:
        stack.enter_context(patch.object(_Tile,'binary',rounded_binary))
        stack.enter_context(patch.object(_TL,'sum',staticmethod(sum_axis)))
        stack.enter_context(patch.object(_TL,'dot',staticmethod(dot_rounded)))
        stack.enter_context(patch.object(_TL,'rsqrt',staticmethod(lambda tile:_Tile(tile.shape,[fp32(1/math.sqrt(v)) if v else math.inf for v in tile.values])),create=True))
        stack.enter_context(patch.object(_TL,'exp',staticmethod(lambda tile:_Tile(tile.shape,[fp32(math.exp(v)) for v in tile.values]))))
        stack.enter_context(patch.object(_Tile,'__floordiv__',lambda tile,x:tile.binary(x,lambda a,b:a//b),create=True))
        stack.enter_context(patch.object(_Tile,'__mod__',lambda tile,x:tile.binary(x,lambda a,b:a%b),create=True))
        stack.enter_context(patch.object(_Tile,'__ge__',lambda tile,x:tile.binary(x,lambda a,b:a>=b),create=True))
        stack.enter_context(patch.object(_Tile,'to',lambda tile,dtype:_Tile(tile.shape,[fp32(v) if dtype is _TL.float32 else int(v) for v in tile.values])))
        for stage in program.stages:
            schedule=Schedule.from_dict(json.loads(stage.schedule_bytes));args={}
            for buf in schedule.buffers:
                if buf.space.value!='global':continue
                public=stage.bindings[buf.name].tensor
                if buf.mode.value=='output':memory[public]=[math.nan]*math.prod(buf.shape)
                args[buf.name]=memory[public]
            traces.append(_execute(emit(schedule,target),args))
    return memory,traces


def single(source, inputs, outputs):
    name=next(line.split('(')[0][4:] for line in source.splitlines() if line.startswith('def '))
    names=tuple(inputs)+tuple(outputs)
    program=parse_program('from open_cake_ir.compiler import frontend as cake\n'+source+
        f'\ncake.program(program_id="cpu_control", inputs={tuple(inputs)!r}, outputs={tuple(outputs)!r}, stages=(cake.stage(name="{name}",schedule={name}, bindings={dict(zip(names,names))!r}),))\n').program
    return execute(program, inputs)


class ConvNextContracts(unittest.TestCase):
    def test_complete_program_padding_axes_and_residual_against_scalar_erf(self):
        dims=(2,33,2,3);inputs=fixture(*dims)
        program=candidate.program_for(*dims,eps=.2,layer_norm_eps=.03)
        observed,traces=execute(program,inputs)
        expected=reference(inputs,*dims,.2,.03)
        self.assertLess(max(abs(a-b) for a,b in zip(observed['output'],expected)),2e-5)
        self.assertTrue(all(set(trace.stores.values())=={1} for trace in traces))
        for name,values in inputs.items():self.assertEqual(observed[name],values)
        self.assertNotEqual(observed['output'],[a-b for a,b in zip(expected,inputs['x'])])

    def test_depthwise_zero_padding_does_not_wrap_between_rows_or_channels(self):
        dims=(2,3,2,5);inputs=fixture(*dims)
        selected={name:inputs[name] for name in ('x','dwconv_weight','dwconv_bias')}
        observed,_=single(candidate._depthwise(*dims),selected,('convolved',))
        b,c,h,w=dims
        for batch in range(b):
            for y in range(h):
                for x in range(w):
                    for ch in range(c):
                        expect=inputs['dwconv_bias'][ch]
                        for i in range(7):
                            for j in range(7):
                                if 0<=y+i-3<h and 0<=x+j-3<w:
                                    expect+=inputs['x'][((batch*c+ch)*h+y+i-3)*w+x+j-3]*inputs['dwconv_weight'][(ch*7+i)*7+j]
                        self.assertAlmostEqual(observed['convolved'][((batch*h+y)*w+x)*c+ch],expect,places=6)

    def test_layernorm_centered_variance_masks_tail_and_uses_original_epsilon(self):
        inputs={'convolved':[1000.,1000.125,1000.25], 'layernorm_weight':[1.,2.,3.], 'layernorm_bias':[.1,.2,.3]}
        observed,_=single(candidate._layernorm(1,3,1,.02),inputs,('normalized',))
        mean=sum(inputs['convolved'])/3
        inverse=1/math.sqrt(sum((x-mean)**2 for x in inputs['convolved'])/3+.02)
        for j,value in enumerate(observed['normalized']):self.assertAlmostEqual(value,(inputs['convolved'][j]-mean)*inverse*(j+1)+inputs['layernorm_bias'][j],places=5)

    def test_spatial_norm_uses_each_batch_and_tail_without_nan_on_zero(self):
        for p in (7,129):
            values=[0.]*(p*8)+[float((i%7)-3) for i in range(p*8)]
            observed,_=single(candidate._spatial_norm(2,2,p),{'activated':values},('global_norm',))
            expected=[math.sqrt(sum(values[(batch*p+j)*8+k]**2 for j in range(p))) for batch in range(2) for k in range(8)]
            for a,b in zip(observed['global_norm'],expected):self.assertAlmostEqual(a,b,places=4)

    def test_gelu_approximation_measured_error_against_erf(self):
        values=[fp32(i/128) for i in range(-2048,2049)]
        n=len(values)
        source=candidate._header('gelu_test',[candidate._tensor('values',(n,)),candidate._tensor('output',(n,),True)])+'''    element = lm.program(output, axis=0, dimension=0, tile=128)
    with compute:
        value = lm.load(values[element], id="value")
'''+candidate._gelu_lines('value','result')+'''        lm.store(output[element], result, id="output")
'''
        observed,_=single(source,{'values':values},('output',))
        expected=[x*.5*(1+math.erf(x/math.sqrt(2))) for x in values]
        error=max(abs(a-b) for a,b in zip(observed['output'],expected))
        self.assertLess(error,2e-6)
        self.assertLess(observed['output'][2048-5*128],0.)

    def test_original_abi_and_scalar_provenance_are_checked(self):
        program=candidate.program_for(1,96,7,7,eps=1e-6,layer_norm_eps=1e-6)
        abi=[TensorABI(name,program.tensors[name].shape,'fp32',mode) for mode,names in [('input',program.inputs),('output',program.outputs)] for name in names]
        scalars={name:{'dtype':'float32','value':value,'binding':'literal_input'} for name,value in [('eps',.2),('layer_norm_eps',.03)]}
        work=SimpleNamespace(target='xcore1002',tensor_abi=lambda case:abi,document={'semantics':{'fixed_scalar_inputs':scalars}})
        self.assertEqual(candidate.source_for_workload(work),candidate.source_for(1,96,7,7,eps=.2,layer_norm_eps=.03))
        with self.assertRaises(ValueError):
            candidate.source_for_workload(SimpleNamespace(target='xcore1002',tensor_abi=lambda case:list(reversed(abi)),document=work.document))
        for bad in ({'eps':scalars['eps']},{**scalars,'eps':{'dtype':'float32','value':1e-6,'binding':'guessed'}}):
            with self.assertRaises(ValueError):candidate.source_for_workload(SimpleNamespace(target='xcore1002',tensor_abi=lambda case:abi,document={'semantics':{'fixed_scalar_inputs':bad}}))


if __name__=='__main__':unittest.main()
