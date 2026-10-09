"""OCML FMA keeps the existing typed ternary operation and exact Target admission."""
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.compiler.backends.triton import emit
from open_cake_ir.compiler.backends.common import EmitError
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import declared_target
from open_cake_ir.compiler.performance.work import work_bound
from open_cake_ir.compiler.toolchain import project_triton_kernel, validate_triton_kernel

ROOT = Path(__file__).resolve().parents[2]


def document(nested=False):
    name = 'fma-chain-b8-smoke' if nested else 'fma-b8-smoke'
    d = json.loads((ROOT/'corpus/schedules'/f'{name}.json').read_text())
    d['target'] = 'gfx938'
    d.pop('residency', None)
    for op in d['operations']:
        if op['kind'] == 'elementwise' and op['parameters']['op'] == 'fma':
            op['parameters']['instruction']['contract'] = 'ocml.fma.f32'
    return d


class OcmlFma(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)

    def test_ordinary_and_nested_lowering_preserve_producer_operations(self):
        for nested,count,flops in [(False,1,2048),(True,2,5120)]:
            with self.subTest(nested=nested):
                d=document(nested);a=self.compiler.assess(d)
                self.assertTrue(a.lowering_eligible,a.findings)
                l=self.compiler.lower(a)
                self.assertIn('from triton.language.extra import libdevice',l.source)
                self.assertEqual(l.source.count('libdevice.fma('),count)
                self.assertNotIn('fma.rn.f32 $0',l.source)
                self.assertIn(b'libdevice.fma(',project_triton_kernel(l.source.encode(),l.toolchain_requirements))
                if nested:
                    self.assertIn('product = a_tile * b_tile',l.source)
                    self.assertIn('libdevice.fma(inner, c_tile, product)',l.source)
                work=work_bound(Schedule.from_dict(d))
                self.assertEqual(work.flops,flops)
                self.assertEqual(work.compulsory_bytes,4*8*128*4)

    def test_frontend_uses_existing_explicit_fma_spelling(self):
        source='''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="ocml-fma", target="gfx938", backend="triton", entry_point="fma_value")
def candidate(lm,a:cake.Tensor((17,),"fp32"),b:cake.Tensor((17,),"fp32"),c:cake.Tensor((17,),"fp32"),out:cake.Tensor((17,),"fp32",mode="output")):
    role=lm.role(execution_groups=[0])
    n=lm.program(a,axis=0,dimension=0,tile=16)
    with role:
        x=lm.load(a[n],id="a")
        y=lm.load(b[n],id="b")
        z=lm.load(c[n],id="c")
        value=lm.fma(x,y,z,instruction={"contract":"ocml.fma.f32"},id="fma")
        lm.store(out[n],value,id="out")
'''
        a=self.compiler.assess(frontend.parse(source).document)
        self.assertTrue(a.lowering_eligible,a.findings)
        self.assertIn('libdevice.fma(x, y, z)',self.compiler.lower(a).source)

    def test_dtype_arity_shape_and_space_remain_checked(self):
        cases=[]
        for dtype in ['fp16','bf16','int32']:
            d=document();next(b for b in d['buffers'] if b['name']=='a_tile')['dtype']=dtype
            cases.append((d,'ELEMENTWISE_INSTRUCTION_DTYPE_DIFFERS'))
        d=document();next(o for o in d['operations'] if o['id']=='fma')['reads'].pop()
        cases.append((d,'ELEMENTWISE_ARITY'))
        d=document();next(b for b in d['buffers'] if b['name']=='c_tile')['shape']=[64]
        cases.append((d,'ELEMENTWISE_SHAPE_MISMATCH'))
        d=document();next(o for o in d['operations'] if o['id']=='fma')['reads'][0]='a'
        cases.append((d,'ELEMENTWISE_FMA_SPACE'))
        for d,code in cases:
            with self.subTest(code=code):
                a=self.compiler.assess(d)
                self.assertFalse(a.lowering_eligible)
                self.assertIn(code,{f.code for f in a.findings})

    def test_other_targets_and_foreign_instruction_names_remain_refused(self):
        for target in ['gfx1151','sm_100a','xcore1002']:
            d=document();d['target']=target
            a=self.compiler.assess(d)
            self.assertIn('TARGET_INSTRUCTION_UNSUPPORTED',{f.code for f in a.findings})
            with self.assertRaises(EmitError):emit(Schedule.from_dict(d),declared_target(target))
        d=document();next(o for o in d['operations'] if o['id']=='fma')['parameters']['instruction']['contract']='ptx.fma.rn.f32'
        self.assertIn('TARGET_INSTRUCTION_UNSUPPORTED',{f.code for f in self.compiler.assess(d).findings})

    def test_native_call_boundary_is_direct_ternary_and_hsaco_only(self):
        source='''import triton
import triton.language as tl
from triton.language.extra import libdevice
@triton.jit
def kernel(a, b):
    x = tl.load(a)
    y = libdevice.fma(x, x, x)
    tl.store(b, y)
'''
        req={'kernel_entry_point':'kernel','signature':{'a':'*fp32','b':'*fp32'},
             'compile_constants':{},'code_object':'hsaco'}
        validate_triton_kernel(source.encode(),req)
        for route in ['cubin','mcfatbin',None]:
            with self.subTest(route=route),self.assertRaises(ValueError):
                validate_triton_kernel(source.encode(),{**req,'code_object':route})
        for expr in ['libdevice.fma(x,x)','libdevice.fma(x,x,x,x)',
                     'libdevice.fma(x,x,c=x)','libdevice.fma(*x)',
                     'libdevice.fma_rn(x,x,x)','libdevice.add(x,x)']:
            with self.subTest(expr=expr),self.assertRaises(ValueError):
                validate_triton_kernel(source.replace('libdevice.fma(x, x, x)',expr).encode(),req)
        alias=source.replace('    y = libdevice.fma(x, x, x)',
                             '    fn = libdevice.fma\n    y = fn(x, x, x)')
        with self.assertRaises(ValueError):validate_triton_kernel(alias.encode(),req)
