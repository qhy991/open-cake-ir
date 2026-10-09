"""Discrete storage retains exact values through IR, source and the tensor ABI.

Source execution uses a CPU value model; it grants no native-device qualification.
"""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.compiler.backends import triton
from open_cake_ir.compiler.ir import DType, Schedule
from open_cake_ir.compiler.ir.operations import cast_supported
from open_cake_ir.compiler.target import declared_target
from open_cake_ir.compiler.toolchain import project_triton_kernel
from open_cake_ir.evaluation.core import (TensorLaunchManifest, LoadedTorchTensorCandidate,
    compare_tile_output_values, _output_poison, _same_tensor_inputs)
from open_cake_ir.evaluation.workload import WorkloadContract
from tests.contracts.test_triton_loop_scopes import _TL, _Tile, _execute
from tests.contracts import test_metax_driver

ROOT = Path(__file__).resolve().parents[2]


def source(dtype, result_dtype=None, operation=None, *, target='xcore1002', backend='triton'):
    body = (f'        result = {operation}\n' if operation else '')
    return f'''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="discrete-storage", target="{target}", backend="{backend}", entry_point="run")
def candidate(lm, x: cake.Tensor((8,), "{dtype}"),
              y: cake.Tensor((8,), "{result_dtype or dtype}", mode="output")):
    block = lm.program(y, axis=0, dimension=0, tile=8)
    compute = lm.role(execution_groups=[0])
    with compute:
        values = lm.load(x[block], id="load_x")
{body}        lm.store(y[block], {'result' if operation else 'values'}, coalesced=False, id="store_y")
'''


def integer_fp32(value):
    """CPU conversion model without an intermediate FP64 rounding step."""
    sign = -1 if value < 0 else 1
    magnitude = abs(int(value))
    shift = max(magnitude.bit_length() - 24, 0)
    if not shift:
        return float(value)
    high, low = divmod(magnitude, 1 << shift)
    midpoint = 1 << (shift - 1)
    high += low > midpoint or low == midpoint and high % 2 == 1
    return float(sign * (high << shift))


def run_source(emission, values, **inputs):
    int64, boolean = object(), object()
    def convert(tile, dtype):
        if dtype is _TL.float32:
            data = [integer_fp32(v) if isinstance(v, int) else float(v) for v in tile.values]
        elif dtype is boolean:
            data = [bool(v) for v in tile.values]
        else:
            data = [int(v) for v in tile.values]
        return _Tile(tile.shape, data)
    output = [None] * len(values)
    with patch.object(_TL, 'int64', int64, create=True), \
         patch.object(_TL, 'int1', boolean, create=True), patch.object(_Tile, 'to', convert), \
         patch.object(_Tile, '__ge__', lambda tile,value: tile.binary(value, lambda a,b:a>=b), create=True):
        _execute(emission, {'x':list(values), 'y':output, **inputs})
    return output


class DiscreteStorageCompiler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT/'compiler/revision.json')

    def emission(self, text):
        document = frontend.parse(text).document
        assessment = self.compiler.assess(document)
        self.assertTrue(assessment.lowering_eligible, assessment.findings)
        lowered = self.compiler.lower(assessment)
        project_triton_kernel(lowered.source.encode(), lowered.toolchain_requirements)
        return triton.emit(Schedule.from_dict(document), declared_target(document['target']))

    def test_discrete_copy_preserves_values_that_float_transport_would_destroy(self):
        cases = {
            'int64':[-2**63, -2**60-1, -1, 0, 1, 2**53+1, 2**60+1, 2**63-1],
            'bool':[False, True, True, False, False, True, False, True],
        }
        for dtype, values in cases.items():
            with self.subTest(dtype=dtype):
                emission = self.emission(source(dtype))
                self.assertEqual(run_source(emission, values), values)
                self.assertEqual(emission.toolchain['signature']['x'], '*i64' if dtype=='int64' else '*i1')
                schedule = Schedule.from_dict(frontend.parse(source(dtype)).document)
                self.assertEqual(schedule.buffer('x').size_bytes, 64 if dtype=='int64' else 8)

    def test_explicit_casts_have_direct_cpu_source_semantics(self):
        values = [2**24+1, 2**24+3, -(2**24+1), -(2**24+3),
                  2**62+2**38+1, -(2**62+2**38+1), -2**63, 2**63-1]
        expected = [float(2**24), float(2**24+4), -float(2**24), -float(2**24+4),
                    float(2**62+2**39), -float(2**62+2**39), -float(2**63), float(2**63)]
        emission = self.emission(source('int64', 'fp32', 'lm.cast(values, to="fp32")'))
        self.assertEqual(run_source(emission, values), expected)
        flags = [False, True]*4
        for dtype in ('int32','fp32'):
            emission = self.emission(source('bool', dtype, f'lm.cast(values, to="{dtype}")'))
            self.assertEqual(run_source(emission, flags), [0,1]*4)
        emission = self.emission(source('int32', 'int64', 'lm.cast(values, to="int64")'))
        values = [-2**31, 2**31-1, -1, 0, 1, 2, -2, 17]
        self.assertEqual(run_source(emission, values), values)

    def test_int64_compare_keeps_exact_integer_literals(self):
        text=source('int64', 'int32', 'lm.compare(values, 1152921504606846977, op="eq")')
        output=run_source(self.emission(text), [2**60, 2**60+1]*4)
        self.assertEqual(output,[0,1]*4)

    def test_bool_select_and_indexed_int64_load_keep_discrete_domains(self):
        text = source('bool', 'bool', 'lm.select(values, values, 0)')
        flags = [False, True] * 4
        self.assertEqual(run_source(self.emission(text), flags), flags)
        text = source('int64').replace('y: cake.Tensor',
            'indices: cake.Tensor((8,), "int64"), y: cake.Tensor')
        text = text.replace('values = lm.load(x[block], id="load_x")',
            'index = lm.load(indices[block])\n        values = lm.load(x[index], id="load_x")')
        values = [2**60+i for i in range(8)]
        indices = [7, 0, -1, 2**63-1, 1, 2, 5, 6]
        self.assertEqual(run_source(self.emission(text), values, indices=indices),
                         [values[7], values[0], 0, 0, values[1], values[2], values[5], values[6]])
        text = text.replace('indices: cake.Tensor((8,), "int64")',
                            'indices: cake.Tensor((8,), "bool")')
        with self.assertRaisesRegex(frontend.FrontendError, 'gather indices'):
            frontend.parse(text)

    def test_bool_does_not_acquire_arithmetic_or_integer_addresses(self):
        for expression in ('values + 1','lm.square(values)'):
            with self.assertRaises(frontend.FrontendError):
                frontend.parse(source('bool','bool',expression))
        doc = frontend.parse(source('bool','int32','lm.compare(values, 0, op="lt")')).document
        self.assertIn('VALUE_OPERATION_TYPE', {f.code for f in self.compiler.assess(doc).findings})
        for expression in ('lm.reduce(values, op="sum", axis=0, scope="cta", across_loop=False)',):
            doc = frontend.parse(source('bool','bool',expression)).document
            self.assertIn('REDUCE_DTYPE_MISMATCH', {f.code for f in self.compiler.assess(doc).findings})

    def test_narrowing_and_numeric_to_bool_have_their_own_refusal(self):
        for src, dst in [('int64','int32'),('fp32','int64'),('fp32','bool'),('int64','bool')]:
            with self.subTest(src=src,dst=dst):
                self.assertFalse(cast_supported(DType(src),DType(dst)))
                with self.assertRaises(frontend.FrontendError) as caught:
                    frontend.parse(source(src,dst,f'lm.cast(values,to="{dst}")'))
                self.assertEqual(caught.exception.code, 'CAST_DTYPE_UNSUPPORTED')
        doc=frontend.parse(source('int32','fp32','lm.cast(values,to="fp32")')).document
        next(b for b in doc['buffers'] if b['name']=='values')['dtype']='int64'
        next(b for b in doc['buffers'] if b['name']=='result')['dtype']='int32'
        next(op for op in doc['operations'] if op['kind']=='cast')['parameters']['to']='int32'
        self.assertIn('CAST_DTYPE_UNSUPPORTED',{f.code for f in self.compiler.assess(doc).findings})

    def test_other_emitters_refuse_the_new_storage_by_name(self):
        for target,backend in [('sm_100a','cutlass_cute_dsl'),('sm_100a','native_cuda'),('apple_gpu_family7','metal')]:
            for dtype in ('int64','bool'):
                doc=frontend.parse(source(dtype,target=target,backend=backend)).document
                result=self.compiler.assess(doc)
                self.assertFalse(result.lowering_eligible)
                self.assertIn('BACKEND_DTYPE_UNEMITTABLE',{f.code for f in result.findings})


class DiscreteRuntimeABI(unittest.TestCase):
    def workload(self,dtype):
        return WorkloadContract({'workload_id':'discrete-storage',
            'cases':[{'case_id':'primary','shape':{'N':8},'seed':200,'mode':'fixed'}],
            'tensors':{name:{'shape':['N'],'dtype':dtype,'layout':'contiguous_row_major'} for name in ('x','y')},
            'semantics':{'target':'xcore1002','candidate_abi':{'inputs':['x'],'outputs':['y']}},
            'validation':{'comparison':'elementwise_atol_rtol','atol':1e20,'rtol':1e20}})

    def test_workload_names_original_storage_without_narrowing(self):
        for dtype in ('int64','bool'):
            workload=self.workload(dtype)
            self.assertEqual([arg.dtype for arg in workload.tensor_abi('primary')],[dtype,dtype])
            manifest = TensorLaunchManifest.for_workload(workload, 'primary', target='xcore1002',
                kernel_name='run', grid=[1,1,1], block=[64,1,1], dynamic_shared_memory_bytes=0,
                hidden_null_pointer_parameters=0)
            self.assertEqual([row[2] for row in manifest.tensor_abi],[dtype,dtype])
            self.assertEqual(manifest.tensors[0][2], 'torch.' + dtype)

    def test_discrete_comparison_is_exact_even_when_float_tolerance_is_large(self):
        for dtype, expected, different in [('int64',2**60+1,2**60),('bool',True,False)]:
            workload=self.workload(dtype)
            self.assertTrue(compare_tile_output_values(workload,{'y':[expected]},{'y':[expected]})[0])
            self.assertFalse(compare_tile_output_values(workload,{'y':[expected]},{'y':[different]})[0])
            wrong_type=float(expected) if dtype=='int64' else 1
            self.assertFalse(compare_tile_output_values(workload,{'y':[expected]},{'y':[wrong_type]})[0])

    def test_poison_does_not_call_finfo_on_discrete_types(self):
        for finite in (False,True):
            self.assertEqual(_output_poison('int64',finite=finite),-2**63)
            self.assertIs(_output_poison('bool',finite=finite),True)

    def test_metax_requires_the_declared_storage_width_before_dispatch(self):
        fixture=test_metax_driver.MetaxDriverTests();fixture.setUp()
        for dtype, runtime, width in [('int64','torch.int64',8),('bool','torch.bool',1)]:
            fixture.manifest.tensor_abi=(('x',(128,),dtype,'input'),)
            loaded=fixture.load()
            arg=fixture.argument(dtype=runtime,element_size=lambda:width)
            loaded.launch([arg],tensor_contract=fixture.manifest)
            with self.assertRaisesRegex(ValueError,'storage width'):
                loaded.launch([fixture.argument(dtype=runtime,element_size=lambda:4)],tensor_contract=fixture.manifest)
            loaded.close()
