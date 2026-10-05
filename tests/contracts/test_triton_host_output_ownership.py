"""Fresh outputs keep their ABI without querying newly allocated metadata twice."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from open_cake_ir.compiler.backends.triton import emit
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import Target

ROOT = Path(__file__).resolve().parents[2]


class Tensor:
    def __init__(self, shape, dtype, device='cuda:0', contiguous=True, pointer=1):
        self._shape, self._dtype, self._device = shape, dtype, device
        self.contiguous, self.pointer = contiguous, pointer
        self.reads = []

    @property
    def shape(self):
        self.reads.append('shape')
        return self._shape

    @property
    def dtype(self):
        self.reads.append('dtype')
        return self._dtype

    @property
    def device(self):
        self.reads.append('device')
        return self._device

    @property
    def is_cuda(self):
        return self._device.startswith('cuda:')

    def is_contiguous(self):
        self.reads.append('contiguous')
        return self.contiguous

    def data_ptr(self):
        return self.pointer


class HostOutputOwnership(unittest.TestCase):
    def entry(self, multi=False, state=False):
        name = 'atomic-reservation-b8-smoke' if state else 'gemm-bias-b1-smoke'
        d = json.loads((ROOT/'corpus/schedules'/(name+'.json')).read_text())
        if multi:
            output = next(b for b in d['buffers'] if b['name'] == d['outputs'][0])
            other = deepcopy(output);other['name']='second_output';d['buffers'].append(other)
            store = next(o for o in d['operations'] if o['kind']=='store')
            other_store=deepcopy(store);other_store.update(id='second_store',writes=['second_output'])
            d['operations'].append(other_store)
            access=deepcopy(next(a for a in d['access_maps'] if a['operation']==store['id']))
            access.update(operation='second_store',buffer='second_output');d['access_maps'].append(access)
            d['outputs'].append('second_output')
        typed=Schedule.from_dict(d);target=Target.load(ROOT/'compiler/targets'/(typed.target+'.json'))
        source=emit(typed,target).source;tree=ast.parse(source)
        entry=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==d['lowering']['entry_point'])
        allocated=[];launches=[]
        def empty(shape,*,dtype,device):
            t=Tensor(shape,dtype,device,pointer=100+len(allocated));allocated.append(t);return t
        class Kernel:
            def __getitem__(self,grid):
                def run(*args,**options):launches.append([t.data_ptr() for t in args])
                return run
        namespace={'torch':SimpleNamespace(empty=empty,**{n:n for n in ['float32','float16','bfloat16','int32','float8_e4m3fn']})}
        for n in tree.body:
            if isinstance(n,ast.FunctionDef) and n is not entry:namespace[n.name]=Kernel()
        exec(compile(ast.Module(body=[entry],type_ignores=[]),'<generated host>','exec'),namespace)
        inputs=[b for b in d['buffers'] if b['space']=='global' and b['mode'] in ['input','state']]
        outputs=[next(b for b in d['buffers'] if b['name']==n) for n in d['outputs']]
        spelling={'fp32':'float32','bf16':'bfloat16','fp16':'float16','int32':'int32'}
        args=[Tensor(tuple(b['shape']),spelling[b['dtype']],pointer=i+1) for i,b in enumerate(inputs)]
        return namespace[entry.name],args,outputs,allocated,launches,spelling

    def test_fresh_single_multi_and_state_outputs_preserve_allocation_and_current_inputs(self):
        for multi,state in [(False,False),(True,False),(False,True)]:
            with self.subTest(multi=multi,state=state):
                fn,args,outputs,allocated,launches,_=self.entry(multi,state)
                first=fn(*args);args[0].pointer=999;second=fn(*args)
                self.assertIsNot(first,second)
                self.assertEqual(len(allocated),2*len(outputs))
                self.assertTrue(all(t.reads==[] for t in allocated))
                self.assertEqual(launches[-1][0],999)

    def test_caller_outputs_are_checked_on_every_call(self):
        for multi,state in [(False,False),(True,False),(False,True)]:
            with self.subTest(multi=multi,state=state):
                fn,args,outputs,allocated,launches,spelling=self.entry(multi,state)
                values=[Tensor(tuple(b['shape']),spelling[b['dtype']],pointer=80+i) for i,b in enumerate(outputs)]
                out=values if multi else values[0]
                fn(*args,out=out);self.assertEqual(len(allocated),0)
                for field,bad in [('_shape',(1,)),('_dtype','wrong'),('_device','cuda:1'),('contiguous',False)]:
                    target=values[-1];old=getattr(target,field);setattr(target,field,bad)
                    with self.assertRaises((ValueError,TypeError)):fn(*args,out=out)
                    setattr(target,field,old)
                self.assertEqual(len(launches),1)
                self.assertTrue(all('contiguous' in t.reads for t in values))

    def test_multi_output_count_and_mutable_input_metadata_are_rejected_before_launch(self):
        fn,args,_,_,launches,_=self.entry(multi=True)
        with self.assertRaises(ValueError):fn(*args,out=[])
        args[0]._shape=(1,)
        with self.assertRaises(ValueError):fn(*args)
        self.assertEqual(launches,[])
