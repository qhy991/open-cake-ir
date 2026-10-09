"""Independent small scalar reference for tests, never a candidate fallback."""
import math

from tools.benchmarks.c550.test_varlen_attention import bf16,fp32


def fixture(b,s,d):
    from tools.benchmarks.c550.decoder_backward import tensor_shapes,INPUT_NAMES
    _,shapes,_=tensor_shapes(b,s,d)
    fp32_inputs={'variance1','variance2','hidden_states_normalized1','hidden_states_normalized2'}
    data={}
    for seed,name in enumerate(INPUT_NAMES):
        count=math.prod(shapes[name])
        scale=.125 if name=='grad_output' else .03125
        cast=fp32 if name in fp32_inputs else bf16
        data[name]=[cast(((j*7+seed*3)%19-9)*scale) for j in range(count)]
    for name in ('variance1','variance2'):
        data[name]=[fp32(.3+(j%5)*.03) for j in range(b*s)]
    for name in ('input_ln_weight','post_attn_ln_weight'):
        data[name]=[bf16(1+(j%5-2)*.0625) for j in range(d.hidden)]
    data['attn_weights']=[bf16((j%7+1)/64) for j in range(b*d.heads*s*s)]
    for name in ('query_states','key_states','value_states','key_states_rotated'):
        data[name]=[float('nan')]*len(data[name])
    return data


def linear(values,weight,rows,k,n):
    return [bf16(fp32(sum(values[row*k+j]*weight[j*n+column] for j in range(k))))
            for row in range(rows) for column in range(n)]


def weight_gradient(left,right,rows,m,n):
    return [bf16(fp32(sum(left[row*m+i]*right[row*n+j] for row in range(rows))))
            for i in range(m) for j in range(n)]


def add_bf16(left,right):
    return [bf16(a+b) for a,b in zip(left,right)]


def normalization(gradient,residual,weight,variance,normalized,residual_gradient,rows,h,eps):
    weight_grad=[fp32(sum(fp32(gradient[row*h+j]*normalized[row*h+j]) for row in range(rows))) for j in range(h)]
    output=[]
    for row in range(rows):
        inverse=fp32(1/math.sqrt(fp32(variance[row]+fp32(eps))))
        gn=[fp32(gradient[row*h+j]*weight[j]) for j in range(h)]
        dot=fp32(sum(fp32(gn[j]*residual[row*h+j]) for j in range(h)))
        gv=fp32(fp32(dot*-.5)*fp32(fp32(inverse*inverse)*inverse))
        for j in range(h):
            base=fp32(gn[j]*inverse)
            correction=fp32(fp32(residual[row*h+j]*fp32(2/h))*gv)
            hidden=bf16(fp32(base+correction))
            output.append(bf16(residual_gradient[row*h+j]+hidden))
    return output,weight_grad


def rope(gradient,cosine,sine,b,s,heads,depth):
    width=heads*depth;result=[]
    for row in range(b*s):
        for head in range(heads):
            for feature in range(depth):
                index=row*width+head*depth+feature
                rotated=gradient[row*width+head*depth+(feature+depth//2)%depth]*(-1 if feature<depth//2 else 1)
                result.append(bf16(bf16(gradient[index]*cosine[row*depth+feature])+bf16(rotated*-sine[row*depth+feature])))
    return result


def evaluate(data,b,s,d,eps):
    """The full original expression, with scalar indexing and independent loop order."""
    h,heads,kv,depth,i=d.hidden,d.heads,d.kv_heads,d.depth,d.intermediate
    q,k,rows=heads*depth,kv*depth,b*s
    go=data['grad_output'];out={}
    grad_swiglu=linear(go,data['down_weight'],rows,h,i)
    out['grad_down_weight']=weight_gradient(go,data['swiglu_output'],rows,h,i)
    grad_gate=[bf16(a*z) for a,z in zip(grad_swiglu,data['silu_up'])]
    grad_up=[]
    for a,gate,up in zip(grad_swiglu,data['gate'],data['up']):
        sigmoid=fp32(1/fp32(fp32(math.exp(fp32(-up)))+1))
        derivative=bf16(fp32(sigmoid*fp32(fp32(up*fp32(1-sigmoid))+1)))
        grad_up.append(bf16(bf16(a*gate)*derivative))
    grad_ffn=add_bf16(linear(grad_gate,data['gate_weight'],rows,i,h),linear(grad_up,data['up_weight'],rows,i,h))
    out['grad_gate_weight']=weight_gradient(grad_gate,data['ffn_input'],rows,i,h)
    out['grad_up_weight']=weight_gradient(grad_up,data['ffn_input'],rows,i,h)
    attn_grad,out['grad_post_attn_ln_weight']=normalization(grad_ffn,data['residual2'],data['post_attn_ln_weight'],data['variance2'],data['hidden_states_normalized2'],go,rows,h,eps)
    grad_attention=linear(attn_grad,data['o_weight'],rows,h,q)
    out['grad_o_weight']=weight_gradient(attn_grad,data['attn_output'],rows,h,q)
    score_grad=[]
    for batch in range(b):
        for head in range(heads):
            for query in range(s):
                for key in range(s):
                    score_grad.append(bf16(fp32(sum(grad_attention[(batch*s+query)*q+head*depth+t]*data['value_states_repeated'][((batch*heads+head)*s+key)*depth+t] for t in range(depth)))))
    logits_grad=[]
    for row in range(b*heads*s):
        start=row*s;weights=data['attn_weights'][start:start+s];grad=score_grad[start:start+s]
        dot=fp32(sum(fp32(a*z) for a,z in zip(weights,grad)))
        logits_grad.extend(bf16(fp32(fp32(a*fp32(z-dot))/fp32(math.sqrt(depth)))) for a,z in zip(weights,grad))
    query_grad=[0.]*(b*s*q)
    for batch in range(b):
        for query in range(s):
            for head in range(heads):
                for t in range(depth):
                    query_grad[(batch*s+query)*q+head*depth+t]=bf16(fp32(sum(logits_grad[((batch*heads+head)*s+query)*s+key]*data['key_states_repeated'][((batch*heads+head)*s+key)*depth+t] for key in range(s))))
    key_grad=[0.]*(b*s*k);value_grad=[0.]*(b*s*k)
    for batch in range(b):
        for key in range(s):
            for kv_head in range(kv):
                for t in range(depth):
                    parts_k=[];parts_v=[]
                    for group in range(heads//kv):
                        head=kv_head*(heads//kv)+group
                        parts_k.append(bf16(fp32(sum(logits_grad[((batch*heads+head)*s+query)*s+key]*data['query_states_rotated'][((batch*s+query)*heads+head)*depth+t] for query in range(s)))))
                        parts_v.append(bf16(fp32(sum(data['attn_weights'][((batch*heads+head)*s+query)*s+key]*grad_attention[(batch*s+query)*q+head*depth+t] for query in range(s)))))
                    index=(batch*s+key)*k+kv_head*depth+t
                    key_grad[index]=bf16(fp32(sum(parts_k)))
                    value_grad[index]=bf16(fp32(sum(parts_v)))
    query_grad=rope(query_grad,data['cos'],data['sin'],b,s,heads,depth)
    key_grad=rope(key_grad,data['cos'],data['sin'],b,s,kv,depth)
    q_path=linear(query_grad,data['q_weight'],rows,q,h)
    k_path=linear(key_grad,data['k_weight'],rows,k,h)
    v_path=linear(value_grad,data['v_weight'],rows,k,h)
    grad_attn_input=add_bf16(add_bf16(q_path,k_path),v_path)
    out['grad_q_weight']=weight_gradient(query_grad,data['attn_input'],rows,q,h)
    out['grad_k_weight']=weight_gradient(key_grad,data['attn_input'],rows,k,h)
    out['grad_v_weight']=weight_gradient(value_grad,data['attn_input'],rows,k,h)
    out['grad_input'],out['grad_input_ln_weight']=normalization(grad_attn_input,data['residual'],data['input_ln_weight'],data['variance1'],data['hidden_states_normalized1'],attn_grad,rows,h,eps)
    return out
