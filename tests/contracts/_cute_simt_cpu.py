"""Execute emitted CuTe SIMT source with 32 cooperating CPU lanes.

This models explicit source operations and collective participation, not SDK codegen,
GPU numerical approximations, register allocation or performance.
"""
import ast
from concurrent.futures import ThreadPoolExecutor
import itertools
import math
import struct
import threading
from types import SimpleNamespace


class F32(float):
    def __new__(cls,value=0.0):
        try:value=struct.unpack('<f',struct.pack('<f',float(value)))[0]
        except OverflowError:value=math.copysign(float('inf'),value)
        return super().__new__(cls,value)
    def __add__(self,x):return F32(float(self)+float(x))
    __radd__=__add__
    def __sub__(self,x):return F32(float(self)-float(x))
    def __rsub__(self,x):return F32(float(x)-float(self))
    def __mul__(self,x):return F32(float(self)*float(x))
    __rmul__=__mul__
    def __truediv__(self,x):
        if x==0:return F32(float('nan') if self==0 else math.copysign(float('inf'),self*x))
        return F32(float(self)/float(x))
    def __rtruediv__(self,x):return F32(x).__truediv__(self)


def fmax(a,b):
    if math.isnan(a):return F32(b)
    if math.isnan(b):return F32(a)
    if a==b==0:return F32(0.0 if math.copysign(1,a)>0 or math.copysign(1,b)>0 else -0.0)
    return F32(max(a,b))


class Rmem:
    def __init__(self,shape):self.values=[F32(0)]*math.prod(shape)
    def fill(self,value):self.values[:]=[F32(value)]*len(self.values)
    def __getitem__(self,i):return self.values[i]
    def __setitem__(self,i,v):self.values[i]=F32(v)


class Pointer:
    def __init__(self,values,writes=None,offset=0):self.values,self.writes,self.offset=values,writes,offset
    def __add__(self,offset):return Pointer(self.values,self.writes,self.offset+offset)
    def __getitem__(self,index):
        index+=self.offset
        assert 0<=index<len(self.values),'out-of-bounds source load'
        return F32(self.values[index])
    def __setitem__(self,index,value):
        index+=self.offset
        assert 0<=index<len(self.values),'out-of-bounds source store'
        assert self.writes is not None,'input mutated'
        self.writes[index]+=1
        self.values[index]=F32(value)


def execute(emission, inputs, output_sizes):
    local=threading.local();barrier=threading.Barrier(32,timeout=10);shared=[F32(0)]*32
    def exchange(value,lane):
        shared[local.lane]=value;barrier.wait()
        out=shared[lane];barrier.wait()
        return out
    def reduction(value,op):
        for distance in (16,8,4,2,1):value=op(value,exchange(value,local.lane^distance))
        return value
    def exp(value):
        try:return F32(math.exp(value))
        except OverflowError:return F32(float('inf'))
    arch=SimpleNamespace(thread_idx=lambda:(local.lane,0,0),block_idx=lambda:local.program,
        shuffle_sync=exchange,warp_reduction_sum=lambda x:reduction(x,lambda a,b:a+b),
        warp_reduction_max=lambda x:reduction(x,fmax),fmax=fmax)
    cute=SimpleNamespace(Pointer=Pointer,arch=arch,make_layout=lambda shape:shape,
        make_rmem_tensor=lambda shape,dtype:Rmem(shape),make_tensor=lambda ptr,layout:ptr,
        math=SimpleNamespace(exp=lambda x,fastmath=False:exp(x),
            exp2=lambda x,fastmath=False:F32(2**x),rsqrt=lambda x,fastmath=False:F32(1/math.sqrt(x)),
            tanh=lambda x,fastmath=False:F32(math.tanh(x))))
    cutlass=SimpleNamespace(Float32=F32,range_constexpr=range)
    tree=ast.parse(emission.source);kernel=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
    kernel.decorator_list=[]
    for arg in kernel.args.args:arg.annotation=None
    env={'cute':cute,'cutlass':cutlass}
    exec(compile(ast.Module(body=[kernel],type_ignores=[]),'<CuTe CPU source model>','exec'),env)
    outputs={k:[float('nan')]*n for k,n in output_sizes.items()};writes={k:[0]*n for k,n in output_sizes.items()}
    pointers={k:Pointer(list(v)) for k,v in inputs.items()}
    pointers.update({k:Pointer(v,writes[k]) for k,v in outputs.items()})
    def run(lane,program):
        local.lane,local.program=lane,program
        try:env[kernel.name](**pointers)
        except BaseException:barrier.abort();raise
    with ThreadPoolExecutor(max_workers=32) as pool:
        for program in itertools.product(*(range(n) for n in emission.toolchain['grid'])):
            futures=[pool.submit(run,lane,program) for lane in range(32)]
            for future in futures:future.result()
    assert all(n==1 for values in writes.values() for n in values),'output coverage/ownership differs'
    return outputs
