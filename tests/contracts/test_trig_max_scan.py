"""Observed RoPE and segment-start expression gaps; CPU admission is not device proof."""
import ast
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np
from jsonschema import Draft202012Validator
from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.compiler.ir import Schedule, ScheduleParseError
from open_cake_ir.compiler.backends.triton import emit, preflight, MAX_SCAN_COMBINE_SOURCE
from open_cake_ir.compiler.schema import schedule_schema
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import verify
from open_cake_ir.compiler.performance.work import work_bound
from open_cake_ir.compiler.toolchain import project_triton_kernel, validate_triton_kernel
from tests.contracts.test_resident_integer_scan import SOURCE as SCAN_SOURCE

ROOT = Path(__file__).resolve().parents[2]
TRIG_SOURCE = '''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="ocml-trig", target="gfx938", backend="triton", entry_point="run")
def candidate(lm, x: cake.Tensor((2, 8), "fp32"), y: cake.Tensor((2, 8), "fp32", mode="output")):
    row = lm.program(x, axis=0, dimension=0, tile=1)
    compute = lm.role(execution_groups=[0])
    with compute:
        values = lm.load(x[row, :])
        result = lm.sin(values, instruction={"contract": "ocml.sin.f32"})
        lm.store(y[row, :], result, coalesced=False)
'''

class TrigContract(unittest.TestCase):
    def test_explicit_contract_schema_and_fp32_typing(self):
        for op in ('sin', 'cos'):
            source = TRIG_SOURCE.replace('sin', op)
            doc = frontend.parse(source).document
            self.assertEqual(list(Draft202012Validator(schedule_schema()).iter_errors(doc)), [])
            s = Schedule.from_dict(doc)
            target = Target.load(ROOT/'compiler/targets/gfx938.json')
            # A synthetic admission isolates shared semantics. The real Target changes
            # only after its separate device qualification.
            admitted = replace(target, instruction_contracts=target.instruction_contracts | {f'ocml.{op}.f32'})
            self.assertFalse([f for f in verify(s, admitted) if f.blocks_lowering])
            self.assertFalse([f for f in preflight(s, admitted) if f.blocks_lowering])
            lowered = emit(s, admitted)
            self.assertIn(f'result = libdevice.{op}(values)', lowered.source)
            projected = project_triton_kernel(lowered.source.encode(), lowered.toolchain)
            validate_triton_kernel(projected, lowered.toolchain)
            self.assertIn('TARGET_INSTRUCTION_UNSUPPORTED', {f.code for f in verify(s,target)})
            work = work_bound(s)
            self.assertFalse(work.flops_exact)
            self.assertIn('result',work.uncounted_arithmetic)
            d=deepcopy(doc)
            next(o for o in d['operations'] if o['id']=='result')['parameters'].pop('instruction')
            with self.assertRaisesRegex(ScheduleParseError,'instruction is required'):
                Schedule.from_dict(d)
            self.assertTrue(list(Draft202012Validator(schedule_schema()).iter_errors(d)))
            for dtype in ('fp16','bf16','int32'):
                with self.assertRaises(Exception) as context:
                    frontend.parse(source.replace('"fp32"',f'"{dtype}"'))
                self.assertIn('no implicit dtype promotion',str(context.exception))
            d=deepcopy(doc)
            next(o for o in d['operations'] if o['id']=='result')['parameters']['instruction']['contract']='ocml.tanh.f32'
            self.assertIn('ELEMENTWISE_INSTRUCTION_KIND_DIFFERS',{f.code for f in verify(Schedule.from_dict(d),admitted)})

    def test_direct_unary_jail_calls_are_hsaco_only_without_escape(self):
        source='import triton\nimport triton.language as tl\nfrom triton.language.extra import libdevice\n@triton.jit\ndef kernel(a,b):\n    x=tl.load(a)\n    y=libdevice.sin(x)\n    tl.store(b,y)\n'
        req={'kernel_entry_point':'kernel','signature':{'a':'*fp32','b':'*fp32'},'compile_constants':{},'code_object':'hsaco'}
        validate_triton_kernel(source.encode(),req)
        for expr in ('libdevice.sin', 'libdevice.sin(*x)', 'libdevice.sin(x,x)', 'libdevice.sin(x=x)', 'libdevice.sin.to(x)'):
            with self.subTest(expr=expr),self.assertRaises(ValueError):
                validate_triton_kernel(source.replace('libdevice.sin(x)',expr).encode(),req)
        for obj in ('cubin','mcfatbin',None):
            with self.assertRaises(ValueError):
                validate_triton_kernel(source.encode(),{**req,'code_object':obj})

class MaxScan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler=Compiler.load(ROOT,ROOT/'compiler/revision.json')

    def lower(self,source):
        doc=frontend.parse(source).document
        a=self.compiler.assess(doc)
        self.assertTrue(a.lowering_eligible,a.findings)
        return self.compiler.lower(a)

    def test_signed_exact_prefix_and_suffix_and_segment_start_counterexample(self):
        values=np.array([-2147483648,-20,-100,16777217,16777216,2147483647,0,-1],dtype=np.int32)
        for direction in ('forward','reverse'):
            lowered=self.lower(SCAN_SOURCE.replace('op="sum"','op="max"').replace('direction="forward"',f'direction="{direction}"'))
            projected=project_triton_kernel(lowered.source.encode(),lowered.toolchain_requirements)
            self.assertIn(MAX_SCAN_COMBINE_SOURCE.strip().encode(),projected)
            line=next(x.strip() for x in lowered.source.splitlines() if x.strip().startswith('prefix ='))
            class Value:
                def to(self,dtype): return values.astype(dtype)
            class TL:
                int32=np.int32
                @staticmethod
                def associative_scan(value,axis,combine_fn,reverse):
                    if reverse:value=np.flip(value,axis)
                    out=np.maximum.accumulate(value,axis=axis)
                    return np.flip(out,axis) if reverse else out
            ns={'tl':TL,'values':Value(),'_cake_max_combine':np.maximum}
            exec(line,ns)
            order=range(len(values)) if direction=='forward' else reversed(range(len(values)))
            expected=[0]*len(values); high=-2147483648
            for i in order:
                high=max(high,int(values[i]));expected[i]=high
            self.assertEqual(ns['prefix'].tolist(),expected)
        starts=np.array([0,0,2,0,4,0,0,7],dtype=np.int32)
        self.assertEqual(np.maximum.accumulate(starts).tolist(),[0,0,2,2,4,4,4,7])
        self.assertNotEqual(np.cumsum(starts).tolist(),np.maximum.accumulate(starts).tolist())

    def test_float_max_refused_and_scalar_identity_and_no_floating_work(self):
        source=SCAN_SOURCE.replace('op="sum"','op="max"')
        for dtype in ('fp32','bf16','fp16'):
            doc=frontend.parse(source.replace('"int32"',f'"{dtype}"')).document
            self.assertIn('SCAN_DTYPE_MISMATCH',{f.code for f in self.compiler.assess(doc).findings})
        scalar=source.replace('(2, 8)','(2,)').replace('x[row, :]','x[row]').replace('y[row, :]','y[row]')
        lowered=self.lower(scalar)
        self.assertNotIn('associative_scan',lowered.source)
        project_triton_kernel(lowered.source.encode(),lowered.toolchain_requirements)
        work=work_bound(Schedule.from_dict(frontend.parse(source).document))
        self.assertEqual(work.flops,0); self.assertTrue(work.flops_exact)

    def test_helper_and_callback_admission_cannot_be_modified_or_aliased(self):
        lowered=self.lower(SCAN_SOURCE.replace('op="sum"','op="max"'))
        source=project_triton_kernel(lowered.source.encode(),lowered.toolchain_requirements)
        for old,new in ((b'return tl.maximum(a, b)',b'return a + b'),
                        (b'combine_fn=_cake_max_combine',b'combine_fn=tl.maximum'),
                        (b'axis=0',b'axis=True'),
                        (b'reverse=False',b'reverse=0'),
                        (b'    prefix =',b'    other = _cake_max_combine\n    prefix =')):
            with self.subTest(new=new),self.assertRaises(ValueError):
                validate_triton_kernel(source.replace(old,new),lowered.toolchain_requirements)
        with self.assertRaises(ValueError):
            validate_triton_kernel(source+MAX_SCAN_COMBINE_SOURCE.encode(),lowered.toolchain_requirements)

class ReductionLoopWidth(unittest.TestCase):
    def test_resident_and_carried_reductions_keep_graph_and_emitted_body(self):
        from tests.contracts.test_triton_loop_scopes import _reduction, _scalar_store
        c=Compiler.load(ROOT,ROOT/'compiler/revision.json')
        for doc in [_reduction('sum',resident=True),_reduction('max',resident=True),
                    _scalar_store('carried',op='sum'),_scalar_store('carried',op='max')]:
            doc.pop('residency',None)
            before=deepcopy(doc)
            a=c.assess(doc);self.assertTrue(a.lowering_eligible,a.findings)
            wanted=2 if len(doc['roles'][0]['execution_groups'])!=2 else 4
            result=c.specialize_triton_warps(doc,num_warps=wanted,schedule_id='reduced-width',entry_point='reduced_width')
            self.assertTrue(result.applied,(result.reason,result.message));self.assertEqual(doc,before)
            for key in doc.keys()-{'roles','lowering','schedule_id'}:
                self.assertEqual(result.schedule[key],doc[key])
            def kernel(assessment):
                lowering=c.lower(assessment)
                n=next(n for n in ast.parse(lowering.source).body if isinstance(n,ast.FunctionDef)
                       and n.name==lowering.toolchain_requirements['kernel_entry_point'])
                n.name='same_kernel';return ast.dump(n,include_attributes=False)
            self.assertEqual(kernel(a),kernel(result.assessment))
        nested=_reduction();nested.pop('residency',None)
        result=c.specialize_triton_warps(nested,num_warps=2,schedule_id='nested-width',entry_point='nested_width')
        self.assertEqual(result.reason,'loop_domain')
