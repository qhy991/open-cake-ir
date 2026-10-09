"""Complete original Decoder backward expressed as static CAKE stages.

The original ABI is checked at the public Workload boundary. This module never
constructs inputs, executes an oracle or performs framework/device arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from tools.benchmarks.c550 import _decoder_stages as stages

TASK = 'L2/056_language_model_decoder_prenorm_attention_ffn_residual_backward'
TARGET = 'xcore1002'
VIEW_INPUTS = ('query_states','key_states','value_states','query_states_rotated','key_states_rotated')
INPUT_NAMES = ('grad_output','residual','attn_input',*VIEW_INPUTS,'key_states_repeated',
    'value_states_repeated','cos','sin','attn_weights','attn_output','residual2',
    'ffn_input','gate','up','silu_up','swiglu_output','input_ln_weight','q_weight',
    'k_weight','v_weight','o_weight','post_attn_ln_weight','gate_weight','up_weight',
    'down_weight','variance1','variance2','hidden_states_normalized1','hidden_states_normalized2')
OUTPUT_NAMES = ('grad_input','grad_input_ln_weight','grad_q_weight','grad_k_weight',
    'grad_v_weight','grad_o_weight','grad_post_attn_ln_weight','grad_gate_weight',
    'grad_up_weight','grad_down_weight')


@dataclass(frozen=True)
class Dimensions:
    hidden: int = 5120
    heads: int = 32
    kv_heads: int = 8
    depth: int = 160
    intermediate: int = 14336

    def checked(self):
        if (any(type(v) is not int or v <= 0 for v in (self.hidden,self.heads,self.kv_heads,self.depth,self.intermediate))
                or self.heads % self.kv_heads or self.depth < 64 or self.depth % 32
                or self.hidden < 64 or self.intermediate < 64):
            raise ValueError('Decoder dimensions need complete head groups and tiled contraction extents')
        return self


def tensor_shapes(b,s,d):
    h,q,k,v,i = d.hidden,d.heads*d.depth,d.kv_heads*d.depth,d.depth,d.intermediate
    original={name:(b,s,h) for name in ('grad_output','residual','attn_input','residual2','ffn_input')}
    original.update({name:(b,d.heads if name in ('query_states','query_states_rotated') else d.kv_heads,s,v) for name in VIEW_INPUTS})
    original.update({name:(b,d.heads,s,v) for name in ('key_states_repeated','value_states_repeated')})
    original.update({name:(b,s,v) for name in ('cos','sin')})
    original.update(attn_weights=(b,d.heads,s,s),attn_output=(b,s,q))
    original.update({name:(b,s,i) for name in ('gate','up','silu_up','swiglu_output')})
    original.update(input_ln_weight=(h,),post_attn_ln_weight=(h,),q_weight=(q,h),
        k_weight=(k,h),v_weight=(k,h),o_weight=(h,q),gate_weight=(i,h),up_weight=(i,h),down_weight=(h,i))
    original.update(variance1=(b,s,1),variance2=(b,s,1),hidden_states_normalized1=(b,s,h),hidden_states_normalized2=(b,s,h))
    physical={name:((shape[0],shape[2],shape[1],shape[3]) if name in VIEW_INPUTS else shape)
              for name,shape in original.items()}
    outputs={'grad_input':(b,s,h),'grad_input_ln_weight':(h,),'grad_q_weight':(q,h),
        'grad_k_weight':(k,h),'grad_v_weight':(k,h),'grad_o_weight':(h,q),
        'grad_post_attn_ln_weight':(h,),'grad_gate_weight':(i,h),'grad_up_weight':(i,h),'grad_down_weight':(h,i)}
    return original,physical,outputs


def source_for(batch,sequence,*,eps,dimensions=Dimensions()):
    """Small dimensions are a CPU control domain; original binding fixes Dimensions()."""
    d=dimensions.checked()
    if (type(batch) is not int or batch <= 0 or type(sequence) is not int or sequence <= 32
            or type(eps) not in (int,float) or not math.isfinite(eps) or eps <= 0):
        raise ValueError('Decoder needs positive batch, sequence above 32 and a finite positive epsilon')
    b,s,h,q,k,i = batch,sequence,d.hidden,d.heads*d.depth,d.kv_heads*d.depth,d.intermediate
    _,physical,_=tensor_shapes(b,s,d)
    unused={name:physical[name] for name in ('query_states','key_states','value_states','key_states_rotated')}
    records=[]
    def add(name,source):records.append((name,source))
    add('swiglu_backward',stages.swiglu_backward(b,s,h,i,unused))
    add('down_weight_gradient',stages.weight_gradient('down_weight_gradient',b,s,h,i,'grad_output','swiglu_output','grad_down_weight'))
    add('ffn_input_gradient',stages.linear_paths('ffn_input_gradient',b,s,h,
        [('grad_gate','gate_weight',i),('grad_up','up_weight',i)],'grad_ffn_input'))
    for name,gradient in [('gate','grad_gate'),('up','grad_up')]:
        add(name+'_weight_gradient',stages.weight_gradient(name+'_weight_gradient',b,s,i,h,gradient,'ffn_input','grad_'+name+'_weight'))
    add('post_norm_weight_gradient',stages.rms_weight('post_norm_weight_gradient',b,s,h,'grad_ffn_input','hidden_states_normalized2','grad_post_attn_ln_weight'))
    add('post_norm_input_gradient',stages.rms_input('post_norm_input_gradient',b,s,h,float(eps),'grad_ffn_input','residual2','variance2','post_attn_ln_weight','grad_output','grad_hidden_states_attn'))
    add('attention_output_gradient',stages.attention_output_gradient(b,s,h,d.heads,d.depth))
    add('output_weight_gradient',stages.weight_gradient('output_weight_gradient',b,s,h,q,'grad_hidden_states_attn','attn_output','grad_o_weight'))
    add('attention_weight_gradient',stages.attention_weight_gradient(b,s,d.heads,d.depth))
    add('softmax_backward',stages.softmax_backward(b,s,d.heads,d.depth))
    add('rotated_query_gradient',stages.query_gradient(b,s,d.heads,d.depth))
    add('grouped_key_gradient',stages.grouped_gradient('grouped_key_gradient',b,s,d.heads,d.kv_heads,d.depth,key_gradient=True))
    add('grouped_value_gradient',stages.grouped_gradient('grouped_value_gradient',b,s,d.heads,d.kv_heads,d.depth,key_gradient=False))
    add('query_rope_gradient',stages.inverse_rope('query_rope_gradient',b,s,d.heads,d.depth,'grad_query_rotated','grad_query'))
    add('key_rope_gradient',stages.inverse_rope('key_rope_gradient',b,s,d.kv_heads,d.depth,'grad_key_rotated','grad_key'))
    add('qkv_input_gradient',stages.linear_paths('qkv_input_gradient',b,s,h,
        [('grad_query','q_weight',q),('grad_key','k_weight',k),('grad_value','v_weight',k)],'grad_attn_input'))
    for name,gradient,width in [('q','grad_query',q),('k','grad_key',k),('v','grad_value',k)]:
        add(name+'_weight_gradient',stages.weight_gradient(name+'_weight_gradient',b,s,width,h,gradient,'attn_input','grad_'+name+'_weight'))
    add('input_norm_weight_gradient',stages.rms_weight('input_norm_weight_gradient',b,s,h,'grad_attn_input','hidden_states_normalized1','grad_input_ln_weight'))
    add('input_norm_input_gradient',stages.rms_input('input_norm_input_gradient',b,s,h,float(eps),'grad_attn_input','residual','variance1','input_ln_weight','grad_hidden_states_attn','grad_input'))
    # Derive bindings from the same Schedule declaration that owns each local ABI.
    from open_cake_ir.compiler.frontend import parse
    tail=f'\ncake.program(program_id="decoder_backward_b{b}_s{s}", inputs={INPUT_NAMES!r}, outputs={OUTPUT_NAMES!r}, stages=(\n'
    for name,source in records:
        document=parse('from open_cake_ir.compiler import frontend as cake\n'+source).document
        names=[buffer['name'] for buffer in document['buffers'] if buffer['space']=='global']
        tail+=f'    cake.stage(name="{name}", schedule={name}, bindings={dict(zip(names,names))!r}),\n'
    return 'from open_cake_ir.compiler import frontend as cake\n\n'+'\n'.join(source for _,source in records)+tail+'))\n'


def program_for(batch,sequence,*,eps,dimensions=Dimensions()):
    from open_cake_ir.compiler.program_frontend import parse_program
    return parse_program(source_for(batch,sequence,eps=eps,dimensions=dimensions)).program


def source_for_workload(workload,case_id='primary'):
    abi=workload.tensor_abi(case_id)
    if workload.target != TARGET or len(abi)!=43 or abi[0].name!='grad_output' or len(abi[0].shape)!=3:
        raise ValueError('Decoder requires the complete exact-target original tensor ABI')
    b,s,_=abi[0].shape
    original,physical,outputs=tensor_shapes(b,s,Dimensions())
    fp32_names={'variance1','variance2','hidden_states_normalized1','hidden_states_normalized2',
                 'grad_input_ln_weight','grad_post_attn_ln_weight'}
    expected=[(name,physical[name],'fp32' if name in fp32_names else 'bf16','input') for name in INPUT_NAMES]
    expected += [(name,outputs[name],'fp32' if name in fp32_names else 'bf16','output') for name in OUTPUT_NAMES]
    if [(a.name,tuple(a.shape),a.dtype,a.mode) for a in abi] != expected:
        raise ValueError('Decoder original ordered shapes or storage dtypes differ')
    semantics=workload.document['semantics']
    views={name:([0,2,1,3] if name in VIEW_INPUTS else list(range(len(original[name])))) for name in INPUT_NAMES}
    if (semantics.get('input_view_contract')!='zero_copy_dense_axis_permutation_before_native_submission'
            or semantics.get('input_views')!=views
            or semantics.get('original_tensor_shapes')!={name:list(shape) for name,shape in {**original,**outputs}.items()}):
        raise ValueError('Decoder requires the complete original dense zero-copy input views')
    scalars=semantics.get('fixed_scalar_inputs')
    if (not isinstance(scalars,dict) or set(scalars)!={'eps'} or not isinstance(scalars['eps'],dict)
            or set(scalars['eps'])!={'dtype','value','binding'} or scalars['eps']['dtype']!='float32'
            or scalars['eps']['binding'] not in {'literal_input','original_factory_literal'}):
        raise ValueError('Decoder requires its original epsilon binding')
    return source_for(b,s,eps=scalars['eps']['value'])
