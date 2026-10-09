"""Bounded CPU controls for the complete Decoder ABI and explicit rounding seams."""
from contextlib import ExitStack
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.program_frontend import parse_program
from open_cake_ir.evaluation.workload import TensorABI
from tools.benchmarks.c550 import decoder_backward as decoder
from tools.benchmarks.c550 import _decoder_stages as stages
from tools.benchmarks.c550.test_varlen_attention import bf16, fp32, execute as execute_program
from tests.contracts.test_triton_loop_scopes import _TL, _Tile

ROOT=Path(__file__).resolve().parents[3]


def execute(program,inputs):
    with ExitStack() as stack:
        stack.enter_context(patch.object(_Tile,'__floordiv__',lambda tile,x:tile.binary(x,lambda a,b:a//b),create=True))
        stack.enter_context(patch.object(_Tile,'__mod__',lambda tile,x:tile.binary(x,lambda a,b:a%b),create=True))
        stack.enter_context(patch.object(_TL,'rsqrt',staticmethod(lambda tile:_Tile(tile.shape,[fp32(1/math.sqrt(v)) if v else math.inf for v in tile.values])),create=True))
        return execute_program(program,inputs)


def single(source,inputs,outputs):
    name=next(line.split('(')[0][4:] for line in source.splitlines() if line.startswith('def '))
    names=tuple(inputs)+tuple(outputs)
    text='from open_cake_ir.compiler import frontend as cake\n'+source+f'\ncake.program(program_id="decoder_cpu_control", inputs={tuple(inputs)!r}, outputs={tuple(outputs)!r}, stages=(cake.stage(name="{name}",schedule={name},bindings={dict(zip(names,names))!r}),))\n'
    return execute(parse_program(text).program,inputs)


def quantized(count,seed,scale=.03125):
    return [bf16(((i*7+seed)%19-9)*scale) for i in range(count)]


def linear(x,w,b,s,k,n):
    return [bf16(fp32(sum(x[row*k+j]*w[j*n+column] for j in range(k))))
            for row in range(b*s) for column in range(n)]


class DecoderBackwardContracts(unittest.TestCase):
    def test_two_linear_branches_round_before_their_sum(self):
        b,s,k,n=2,33,64,32
        x=quantized(b*s*k,3,.25);w=quantized(k*n,5,.125)
        y=quantized(b*s*k,7,.0625);v=quantized(k*n,11,.25)
        source=stages.linear_paths('two_linear',b,s,n,[('x','w',k),('y','v',k)],'out')
        memory,traces=single(source,{'x':x,'w':w,'y':y,'v':v},('out',))
        expected=[bf16(a+b) for a,b in zip(linear(x,w,b,s,k,n),linear(y,v,b,s,k,n))]
        self.assertEqual(memory['out'],expected)
        self.assertTrue(all(set(trace.stores.values())=={1} for trace in traces))

    def test_swiglu_uses_silu_of_up_and_two_up_product_rounds(self):
        b,s,h,i=1,33,64,32
        go=[bf16(1.25)]*(b*s*h)
        weight=[bf16(.03125)]*(h*i)
        gate=[bf16(.0517578125)]*(b*s*i)
        up=[bf16(-1.375)]*(b*s*i)
        silu=[bf16(-.27734375)]*(b*s*i)
        memory,_=single(stages.swiglu_backward(b,s,h,i,{}),
            {'grad_output':go,'down_weight':weight,'gate':gate,'up':up,'silu_up':silu},('grad_gate','grad_up'))
        grad=bf16(sum(go[j]*weight[j*i] for j in range(h)))
        sigmoid=fp32(1/fp32(fp32(math.exp(fp32(-up[0])))+1.))
        derivative=bf16(fp32(sigmoid*fp32(fp32(up[0]*fp32(1.-sigmoid))+1.)))
        expected=bf16(bf16(grad*gate[0])*derivative)
        self.assertEqual(memory['grad_gate'],[bf16(grad*silu[0])]*(b*s*i))
        self.assertEqual(memory['grad_up'],[expected]*(b*s*i))
        changed=stages.swiglu_backward(b,s,h,i,{}).replace('lm.cast(first_bf16, to="fp32", id="widen_first_product")',
                                                        'lm.cast(first_product, to="fp32", id="widen_first_product")')
        mutant,_=single(changed,{'grad_output':go,'down_weight':weight,'gate':gate,'up':up,'silu_up':silu},('grad_gate','grad_up'))
        self.assertTrue(any(a != b for a,b in zip(mutant['grad_up'],memory['grad_up'])), 'the first BF16 product cast must change this control')

    def test_rms_input_reads_supplied_variance_and_rounds_before_residual_add(self):
        b,s,h=1,2,3
        inputs={'g':[bf16(x) for x in [.13,-.27,.31,.19,.07,-.41]],
            'x':[bf16(x) for x in [.3,-.2,.1,.8,.4,-.9]],
            'variance':[.73,.29],'weight':[bf16(1.1),bf16(.9),bf16(1.3)],
            'residual_gradient':[bf16(x) for x in [.5,-.25,.125,.5,-.25,.125]]}
        result,_=single(stages.rms_input('rms',b,s,h,.017,'g','x','variance','weight','residual_gradient','out'),inputs,('out',))
        expected=[]
        for row in range(s):
            inv=fp32(1/math.sqrt(fp32(inputs['variance'][row]+fp32(.017))))
            normalized=[fp32(inputs['g'][row*h+j]*inputs['weight'][j]) for j in range(h)]
            total=fp32(sum(fp32(normalized[j]*inputs['x'][row*h+j]) for j in range(h)))
            gv=fp32(fp32(total*-.5)*fp32(fp32(inv*inv)*inv))
            for j in range(h):
                base=fp32(normalized[j]*inv)
                correction=fp32(fp32(inputs['x'][row*h+j]*fp32(2/h))*gv)
                hidden=bf16(fp32(base+correction))
                expected.append(bf16(inputs['residual_gradient'][row*h+j]+hidden))
        self.assertEqual(result['out'],expected)

    def test_fp32_norm_weight_gradient_includes_both_batches_and_sequence_tail(self):
        b,s,h=2,33,3
        grad=[bf16((batch+1)*(j%5-2)*.25) for batch in range(b) for row in range(s) for j in range(h)]
        normalized=[fp32((batch+2)*(row%7-3)*.0625) for batch in range(b) for row in range(s) for j in range(h)]
        result,_=single(stages.rms_weight('norm_weight',b,s,h,'grad','normalized','out'),
                        {'grad':grad,'normalized':normalized},('out',))
        expected=[fp32(sum(fp32(grad[(batch*s+row)*h+j]*normalized[(batch*s+row)*h+j])
                     for batch in range(b) for row in range(s))) for j in range(h)]
        self.assertEqual(result['out'],expected)

    def test_inverse_rope_uses_rotated_gradient_negative_sine_and_rounded_products(self):
        b,s,heads,d=2,3,2,64;width=heads*d
        grad=quantized(b*s*width,2,.125);cosine=quantized(b*s*d,7,.125);sine=quantized(b*s*d,11,.125)
        result,_=single(stages.inverse_rope('rope',b,s,heads,d,'grad','out'),
                        {'grad':grad,'cos':cosine,'sin':sine},('out',))
        expected=[]
        for row in range(b*s):
            for head in range(heads):
                for feature in range(d):
                    index=row*width+head*d+feature
                    rotated=grad[row*width+head*d+((feature+d//2)%d)]*(-1 if feature<d//2 else 1)
                    expected.append(bf16(bf16(grad[index]*cosine[row*d+feature])+bf16(rotated*-sine[row*d+feature])))
        self.assertEqual(result['out'],expected)

    def test_softmax_uses_given_weights_without_an_extra_causal_mask(self):
        b,s,heads,d=1,33,2,64
        g=quantized(b*heads*s*s,9,.25)
        w=[bf16((j%7+1)/64) for j in range(b*heads*s*s)]
        result,_=single(stages.softmax_backward(b,s,heads,d),
            {'grad_attn_weights':g,'attn_weights':w},('grad_attn_logits',))
        expected=[]
        for row in range(b*heads*s):
            start=row*s;total=fp32(sum(fp32(g[start+j]*w[start+j]) for j in range(s)))
            expected.extend(bf16(fp32(fp32(w[start+j]*fp32(g[start+j]-total))/8.)) for j in range(s))
        self.assertEqual(result['grad_attn_logits'],expected)
        self.assertNotEqual(result['grad_attn_logits'][1],0.)

    def test_original_abi_views_dtypes_scalar_and_all_outputs_are_checked(self):
        b,s=2,128;d=decoder.Dimensions();original,physical,outputs=decoder.tensor_shapes(b,s,d)
        fp32_names={'variance1','variance2','hidden_states_normalized1','hidden_states_normalized2',
                     'grad_input_ln_weight','grad_post_attn_ln_weight'}
        abi=[TensorABI(n,physical[n],'fp32' if n in fp32_names else 'bf16','input') for n in decoder.INPUT_NAMES]
        abi += [TensorABI(n,outputs[n],'fp32' if n in fp32_names else 'bf16','output') for n in decoder.OUTPUT_NAMES]
        semantics={'input_view_contract':'zero_copy_dense_axis_permutation_before_native_submission',
            'input_views':{n:([0,2,1,3] if n in decoder.VIEW_INPUTS else list(range(len(original[n])))) for n in decoder.INPUT_NAMES},
            'original_tensor_shapes':{n:list(shape) for n,shape in {**original,**outputs}.items()},
            'fixed_scalar_inputs':{'eps':{'dtype':'float32','value':1e-6,'binding':'original_factory_literal'}}}
        work=SimpleNamespace(target='xcore1002',tensor_abi=lambda case:abi,document={'semantics':semantics})
        program=parse_program(decoder.source_for_workload(work)).program
        self.assertEqual(program.inputs,decoder.INPUT_NAMES)
        self.assertEqual(program.outputs,decoder.OUTPUT_NAMES)
        self.assertEqual(len(program.stages),22)
        for name in decoder.OUTPUT_NAMES:
            self.assertEqual(program.tensors[name].dtype.value,'fp32' if 'ln_weight' in name else 'bf16')
        for mutation in ('views','dtype','missing_output','eps'):
            changed=json.loads(json.dumps(semantics));new_abi=list(abi)
            if mutation=='views':changed['input_views']['query_states_rotated']=[0,1,2,3]
            elif mutation=='dtype':new_abi[-2]=replace(new_abi[-2],dtype='fp32')
            elif mutation=='missing_output':new_abi.pop()
            else:changed['fixed_scalar_inputs']['eps']['binding']='guessed'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                decoder.source_for_workload(SimpleNamespace(target='xcore1002',tensor_abi=lambda case:new_abi,document={'semantics':changed}))


if __name__=='__main__':unittest.main()
