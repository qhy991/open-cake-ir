"""Typed resident scans: CPU source semantics are distinct from device qualification."""
from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.frontend import parse
from open_cake_ir.compiler.ir import Schedule

ROOT = Path(__file__).resolve().parents[2]

SOURCE = '''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="resident-prefix", target="sm_103a", backend="triton", entry_point="run")
def candidate(lm, x: cake.Tensor((2, 8), "int32"), y: cake.Tensor((2, 8), "int32", mode="output")):
    row = lm.program(x, axis=0, dimension=0, tile=1)
    compute = lm.role(execution_groups=[0])
    with compute:
        values = lm.load(x[row, :])
        prefix = lm.scan(values, op="sum", axis=0, direction="forward")
        lm.store(y[row, :], prefix, coalesced=False)
'''


class ResidentIntegerScan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT/'compiler/revision.json')

    def lower(self, document):
        assessment = self.compiler.assess(document)
        self.assertEqual(assessment.findings, ())
        return self.compiler.lower(assessment).source

    def test_frontend_result_dtype_and_float_promotion_agree_with_verifier(self):
        for dtype, result in [('int32','int32'), ('fp32','fp32'), ('fp16','fp32'), ('bf16','fp32')]:
            source = SOURCE.replace('"int32"', f'"{dtype}"', 1)
            source = source.replace('y: cake.Tensor((2, 8), "int32"', f'y: cake.Tensor((2, 8), "{result}"')
            d = parse(source).document
            self.assertEqual(Schedule.from_dict(d).buffer('prefix').dtype.value, result)
            emitted = self.lower(d)
            native = 'int32' if dtype=='int32' else 'float32'
            self.assertIn(f'tl.cumsum(values.to(tl.{native}), axis=0, reverse=False)', emitted)

    def test_generated_operation_preserves_large_integers_signed_wrap_and_direction(self):
        # Execute the actual emitted statement through a typed array adapter. This
        # catches an implicit FP32 conversion; it makes no Triton/device claim.
        class Value:
            def __init__(self, value): self.value = np.asarray(value, dtype=np.int32)
            def to(self, dtype): return self.value.astype(dtype)
        class TL:
            int32, float32 = np.int32, np.float32
            @staticmethod
            def cumsum(value, axis, reverse):
                if reverse: value=np.flip(value,axis)
                result=np.cumsum(value,axis=axis,dtype=value.dtype)
                return np.flip(result,axis) if reverse else result
        values = [16777217, 1, -3, 2147483647, 1, -2147483648, -1, 2]
        for direction in ['forward','reverse']:
            d=parse(SOURCE.replace('direction="forward"', f'direction="{direction}"')).document
            generated=self.lower(d)
            statement=next(line.strip() for line in generated.splitlines() if line.strip().startswith('prefix ='))
            namespace={'tl':TL,'values':Value(values)}
            exec(statement, namespace)
            indices=list(range(len(values)))
            if direction=='reverse': indices.reverse()
            expected=[0]*len(values)
            total=0
            for i in indices:
                total+=values[i]
                expected[i]=(total+2**31) % 2**32 - 2**31
            self.assertEqual(namespace['prefix'].tolist(),expected)
            self.assertEqual(namespace['prefix'].dtype,np.int32)

    def test_scalar_prefix_is_identity_for_native_rank_zero_load(self):
        source=SOURCE.replace('(2, 8)', '(2,)').replace('x[row, :]', 'x[row]').replace('y[row, :]', 'y[row]')
        generated=self.lower(parse(source).document)
        self.assertIn('prefix = values.to(tl.int32)',generated)
        self.assertNotIn('tl.cumsum(',generated)

    def test_dtype_shape_axis_and_nonresident_values_fail_by_the_scan_rule(self):
        base=parse(SOURCE).document
        variants=[]
        for dtype in ['fp32','bf16']:
            d=deepcopy(base)
            next(b for b in d['buffers'] if b['name']=='prefix')['dtype']=dtype
            variants.append((d,'SCAN_DTYPE_MISMATCH'))
        d=deepcopy(base)
        next(b for b in d['buffers'] if b['name']=='prefix')['shape']=[4]
        variants.append((d,'SCAN_SHAPE_MISMATCH'))
        d=deepcopy(base)
        next(o for o in d['operations'] if o['id']=='prefix')['parameters']['axis']=1
        variants.append((d,'SCAN_AXIS_OUT_OF_RANGE'))
        d=deepcopy(base)
        next(b for b in d['buffers'] if b['name']=='prefix')['space']='global'
        variants.append((d,'SCAN_VALUE_SPACE'))
        d=deepcopy(base)
        next(o for o in d['operations'] if o['id']=='prefix')['reads'].append('values')
        variants.append((d,'SCAN_ARITY'))
        for document, code in variants:
            with self.subTest(code=code):
                self.assertIn(code,{f.code for f in self.compiler.assess(document).findings})

    def test_no_backend_fallback_is_added_for_integer_scan(self):
        for backend in ['metal','native_cuda','cutlass_cute_dsl']:
            d=parse(SOURCE).document
            d['lowering']['backend']=backend
            assessment=self.compiler.assess(d)
            self.assertFalse(assessment.lowering_eligible,backend)
