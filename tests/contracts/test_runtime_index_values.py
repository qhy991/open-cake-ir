"""Runtime coordinates keep exact integer types, scalar domains and visible dependencies."""
from copy import deepcopy
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.frontend import parse, FrontendError
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.schema import schedule_schema

ROOT = Path(__file__).resolve().parents[2]
SOURCE = '''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="runtime-values", target="sm_103a", backend="triton", entry_point="run")
def candidate(lm, x: cake.Tensor((4, 8), "fp32"), indices: cake.Tensor((4,), "int32"), y: cake.Tensor((4, 8), "fp32", mode="output")):
    row = lm.program(y, axis=0, dimension=0, tile=1)
    compute = lm.role(execution_groups=[0])
    with compute:
        rid = lm.coordinate(source="program", name="row")
        ix = lm.load(indices[row])
        shifted = ix + 1
        floor = shifted // 2
        val = lm.load(x[lm.scalar_index(floor), :])
        positive = lm.compare(val, 0.0, op="gt")
        safe = lm.select(positive, val, 1.0)
        result = lm.log2(safe)
        lm.store(y[row, :], result, coalesced=False)
'''


class RuntimeIndexValues(unittest.TestCase):
    def setUp(self):
        self.compiler = Compiler.load(ROOT, ROOT/'compiler/revision.json')
        self.document = parse(SOURCE).document

    def codes(self, document):
        return {f.code for f in self.compiler.assess(document).findings}

    def test_scalar_gather_and_typed_values_lower_and_match_schema(self):
        import jsonschema
        jsonschema.validate(self.document, schedule_schema())
        assessment = self.compiler.assess(self.document)
        self.assertEqual(assessment.findings, ())
        source = self.compiler.lower(assessment).source
        self.assertIn('tl.log2(safe)', source)
        self.assertIn('floor', source)
        load = next(op for op in self.document['operations'] if op['id']=='val')
        self.assertEqual(load['reads'], ['x','floor'])
        self.assertIn('floor', load['depends_on'])
        result = Schedule.from_dict(self.document).buffer('val')
        self.assertEqual(result.shape, (8,))

    def test_scalar_only_load_stays_one_value(self):
        source = SOURCE.replace('x: cake.Tensor((4, 8)', 'x: cake.Tensor((4,)')
        source = source.replace('y: cake.Tensor((4, 8)', 'y: cake.Tensor((4,)')
        source = source.replace('lm.scalar_index(floor), :', 'lm.scalar_index(floor)')
        source = source.replace('y[row, :]', 'y[row]')
        document = parse(source).document
        self.assertEqual(Schedule.from_dict(document).buffer('val').shape, (1,))
        self.assertEqual(self.compiler.assess(document).findings, ())

    def test_vector_cannot_be_claimed_as_scalar_coordinate(self):
        source = SOURCE.replace('ix = lm.load(indices[row])', 'ix = lm.load(indices[:])')
        with self.assertRaisesRegex(FrontendError, 'scalar_index'):
            parse(source)

    def test_scalar_index_dependency_cannot_be_omitted(self):
        document = deepcopy(self.document)
        load = next(op for op in document['operations'] if op['id']=='val')
        load['reads'] = ['x']
        self.assertIn('ACCESS_INDEX_BUFFER_READS', self.codes(document))

    def test_scalar_index_store_has_no_borrowed_ownership(self):
        document = deepcopy(self.document)
        access = next(a for a in document['access_maps'] if a['operation']=='store_y')
        access['indices'][0] = {'source':'scalar_buffer','name':'floor'}
        self.assertIn('SCALAR_INDEX_WRITE_UNPROVEN', self.codes(document))

    def test_float_cannot_be_used_as_an_integer_address(self):
        document = deepcopy(self.document)
        next(b for b in document['buffers'] if b['name']=='floor')['dtype'] = 'fp32'
        self.assertIn('ACCESS_INDEX_BUFFER_DTYPE', self.codes(document))

    def test_bad_predicate_dtype_and_coordinate_scope_refuse(self):
        document = deepcopy(self.document)
        next(op for op in document['operations'] if op['id']=='safe')['reads'][0] = 'val'
        self.assertIn('VALUE_OPERATION_TYPE', self.codes(document))
        document = deepcopy(self.document)
        next(op for op in document['operations'] if op['id']=='rid')['parameters']['name'] = 'missing'
        self.assertIn('VALUE_OPERATION_TYPE', self.codes(document))

    def test_integer_divisor_must_be_positive_and_integral(self):
        for divisor in [0, -1, 0.5]:
            document = deepcopy(self.document)
            next(op for op in document['operations'] if op['id']=='floor')['parameters']['scalar'] = divisor
            self.assertTrue({'INTEGER_SCALAR_INVALID','INTEGER_DIVISOR_INVALID'} & self.codes(document))

    def test_unimplemented_backends_refuse_new_values(self):
        for backend in ['metal','cutlass_cute_dsl','native_cuda']:
            document = deepcopy(self.document)
            document['lowering']['backend'] = backend
            assessment = self.compiler.assess(document)
            self.assertFalse(assessment.lowering_eligible, backend)

    def test_fp8_cast_requires_an_explicit_conversion(self):
        source = SOURCE.replace('"fp32"), indices:', '"fp8_e4m3"), indices:')
        source = source.replace('val = lm.load(', 'raw = lm.load(').replace(
            '        positive =', '        val = lm.cast(raw, to="fp32")\n        positive =')
        assessment = self.compiler.assess(parse(source).document)
        self.assertEqual(assessment.findings, ())
        self.compiler.lower(assessment)
