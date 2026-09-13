"""Output-column specialization semantics and adversarial boundaries; no GPU qualification.

The structural assertions mirror the retained F-2026-09-11-004 mechanism evidence
(candidate 450d8350: one row program materializing B[K,N] and N outputs became
row-by-column programs loading B[:,col] producing one scalar). Numerical evidence runs
the emitted SIMD body through the existing CPU intrinsic adapter; it is CPU semantic
evidence, not GPU, timing or performance qualification.
"""

from copy import deepcopy
import json
import math
from pathlib import Path
import random
import unittest

from open_cake_ir.compiler import Compiler, frontend

# One authority for the generated-body CPU adapter and its draft compiler loader.
from tests.contracts import test_metal

ROOT = Path(__file__).resolve().parents[2]

ROWS, WIDTH, COLUMNS = 4, 6, 5

_HEADER = '''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="{name}", target="apple_gpu_family7", backend="metal", entry_point="{entry}"{metadata})
def candidate(lm, a: cake.Tensor(({rows}, {width}), "fp32"), b: cake.Tensor(({width}, {columns}), "fp32"), bias: cake.Tensor(({columns},), "fp32"), out: cake.Tensor(({rows}, {columns}), "fp32", mode="output"){extra_args}):
    compute = lm.role(warps=[0])
    row = lm.program(a, axis=0, dimension=0, tile=1)
{axes}    with compute:
        a_row = lm.load(a[row, :], id="load_a")
        b_tile = lm.load(b[:, :], id="load_b")
        biases = lm.load(bias[:], id="load_bias")
        products = b_tile * lm.broadcast(a_row, axis=0)
        totals = lm.reduce(products, op="sum", axis=0, scope="cta", across_loop=False, id="sum_k")
{body}        lm.store(out[row, :], {final}, coalesced=False, id="store_out")
'''

_PLAIN_STORE = ('shifted', '''        shifted = totals + biases
''')
_SILU = ('gated', '''        shifted = totals + biases
        negated = shifted * -1.0
        decayed = lm.exp(negated, id="exp")
        gate = lm.reciprocal(decayed + 1.0, id="reciprocal")
        gated = shifted * gate
''')


def gemm_document(metadata=False):
    text = _HEADER.format(name='gemm-row-programmed', entry='cake_gemm_rows',
        metadata=',\n               metadata={"workload_contract_sha256": "' + '1'*64 + '"}' if metadata else '',
        rows=ROWS, width=WIDTH, columns=COLUMNS, extra_args='', axes='', body=_PLAIN_STORE[1], final=_PLAIN_STORE[0])
    return frontend.parse(text).document


def gemm_silu_document():
    text = _HEADER.format(name='gemm-silu-row-programmed', entry='cake_gemm_silu_rows',
        metadata='', rows=ROWS, width=WIDTH, columns=COLUMNS, extra_args='', axes='', body=_SILU[1], final=_SILU[0])
    return frontend.parse(text).document


def coupled_document():
    """A row-wide fold of the epilogue vector: the softmax-coupling anti-case."""
    text = _HEADER.format(name='coupled-row-fold', entry='cake_coupled',
        metadata='', rows=ROWS, width=WIDTH, columns=COLUMNS,
        extra_args=', rowsum: cake.Tensor((%d,), "fp32", mode="output")' % ROWS, axes='',
        body='''        scales = lm.exp(totals, id="exp_row")
        denom = lm.reduce(scales, op="sum", axis=0, scope="cta", across_loop=False, id="sum_n")
        lm.store(rowsum[row], denom, coalesced=False, id="store_rowsum")
''' + _SILU[1], final=_SILU[0])
    return frontend.parse(text).document


def sliced_document():
    text = _HEADER.format(name='sliced-second-operand', entry='cake_sliced',
        metadata='', rows=ROWS, width=WIDTH, columns=COLUMNS, extra_args='', axes='',
        body=_PLAIN_STORE[1], final=_PLAIN_STORE[0])
    text = text.replace('b[:, :]', 'b[:, :3]').replace('bias[:]', 'bias[:3]').replace('out[row, :]', 'out[row, :3]')
    return frontend.parse(text).document


def column_programmed_document():
    """Already column-programmed: valid, but the mechanism has nothing to add."""
    text = '''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="already-columns", target="apple_gpu_family7", backend="metal", entry_point="cake_columns")
def candidate(lm, a: cake.Tensor((4, 6), "fp32"), b: cake.Tensor((6, 5), "fp32"), bias: cake.Tensor((5,), "fp32"), out: cake.Tensor((4, 5), "fp32", mode="output")):
    compute = lm.role(warps=[0])
    row = lm.program(a, axis=0, dimension=0, tile=1)
    col = lm.program(b, axis=1, dimension=1, tile=1)
    with compute:
        a_row = lm.load(a[row, :], id="load_a")
        b_col = lm.load(b[:, col], id="load_b")
        products = a_row * b_col
        totals = lm.reduce(products, op="sum", axis=0, scope="cta", across_loop=False, id="sum_k")
        biases = lm.load(bias[col], id="load_bias")
        shifted = totals + biases
        lm.store(out[row, col], shifted, coalesced=False, id="store_out")
'''
    return frontend.parse(text).document


def two_store_document():
    text = _HEADER.format(name='two-stores', entry='cake_two_stores',
        metadata='', rows=ROWS, width=WIDTH, columns=COLUMNS,
        extra_args=', out2: cake.Tensor((%d, %d), "fp32", mode="output")' % (ROWS, COLUMNS), axes='',
        body=_PLAIN_STORE[1] + '''        lm.store(out2[row, :], shifted, coalesced=False, id="store_out2")
''', final='shifted')
    return frontend.parse(text).document


class OutputColumnSpecializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Author checks run against the explicit working draft, never a stale release.
        cls.compiler = Compiler.load(ROOT, ROOT/'compiler/revision.json')

    def specialize(self, document, **kwargs):
        return self.compiler.specialize_output_columns(document,
            schedule_id=kwargs.get('schedule_id', 'column_specialized_test'),
            entry_point=kwargs.get('entry_point', 'cake_column_specialized'))

    # The CPU generated-body adapter is owned by the Metal contracts; borrow it rather
    # than forking the intrinsic ABI.
    lower = test_metal.MetalTests.lower
    execute_body = test_metal.MetalTests.execute_body

    def assert_refused(self, document, reason):
        self.assertTrue(self.compiler.assess(document).lowering_eligible)
        before = deepcopy(document)
        result = self.specialize(document)
        self.assertFalse(result.applied)
        self.assertIsNone(result.schedule)
        self.assertEqual(result.reason, reason, result.message)
        self.assertEqual(document, before)

    def test_candidate_matches_the_qualified_evidence_mechanism(self):
        for document in (gemm_document(metadata=True), gemm_silu_document()):
            with self.subTest(schedule=document['schedule_id']):
                before = deepcopy(document)
                result = self.specialize(document)
                self.assertTrue(result.applied, result.message)
                self.assertEqual(document, before)
                candidate = result.schedule
                axes = candidate['program_map']['axes']
                self.assertEqual(axes[0], before['program_map']['axes'][0])
                self.assertEqual(axes[1], {'name': 'col', 'axis': 1, 'buffer': 'b', 'dimension': 1, 'tile': 1})
                access = {a['buffer']: a for a in candidate['access_maps']}
                self.assertEqual(access['b']['indices'],
                                 [{'source': 'dimension', 'dimension': 0}, {'source': 'program', 'name': 'col'}])
                self.assertEqual(access['bias']['indices'], [{'source': 'program', 'name': 'col'}])
                self.assertEqual(access['out']['indices'],
                                 [{'source': 'program', 'name': 'row'}, {'source': 'program', 'name': 'col'}])
                self.assertEqual(access['a']['indices'], next(a for a in before['access_maps'] if a['buffer'] == 'a')['indices'])
                shapes = {b['name']: b['shape'] for b in candidate['buffers']}
                self.assertEqual(shapes['b_tile'], [WIDTH])
                self.assertEqual(shapes['products'], [WIDTH])
                self.assertEqual(shapes['totals'], [1])
                self.assertEqual(shapes['biases'], [1])
                self.assertEqual(shapes['shifted'], [1])
                self.assertTrue(all('broadcast_axis' not in op['parameters'] for op in candidate['operations']))
                self.assertEqual(candidate['metadata'], before['metadata'])
                self.assertEqual(candidate['outputs'], before['outputs'])
                assessment, lowering = self.lower(candidate)
                self.assertEqual(lowering.toolchain_requirements['threadgroups_per_grid'], [ROWS, COLUMNS, 1])
                self.assertEqual(lowering.toolchain_requirements['threads_per_threadgroup'], [32, 1, 1])
                # A projection, not a second authority: mutating it cannot change the result.
                candidate['outputs'].clear()
                self.assertTrue(result.schedule['outputs'])

    def test_row_and_column_programs_agree_on_cpu(self):
        rng = random.Random(9943)
        cases = [
            {'a': [rng.uniform(-2, 2) for _ in range(ROWS*WIDTH)],
             'b': [rng.uniform(-2, 2) for _ in range(WIDTH*COLUMNS)],
             'bias': [rng.uniform(-1, 1) for _ in range(COLUMNS)]},
            {'a': [0.0]*(ROWS*WIDTH), 'b': [0.0]*(WIDTH*COLUMNS), 'bias': [0.0]*COLUMNS},
            {'a': [rng.uniform(-8, 8) for _ in range(ROWS*WIDTH)],
             'b': [4.0]*(WIDTH*COLUMNS), 'bias': [1.0]*COLUMNS},
        ]
        for document in (gemm_document(), gemm_silu_document()):
            with self.subTest(schedule=document['schedule_id']):
                result = self.specialize(document)
                self.assertTrue(result.applied, result.message)
                for inputs in cases:
                    expected = self.execute_body(document, inputs)['out']
                    observed = self.execute_body(result.schedule, inputs)['out']
                    self.assertEqual(len(expected), ROWS*COLUMNS)
                    for got, want in zip(observed, expected):
                        self.assertLessEqual(abs(got - want), 1e-5 * max(1.0, abs(want)))

    def test_refused_coupling_is_exactly_the_reassociating_epilogue(self):
        self.assert_refused(coupled_document(), 'coupled_output_axis')
        # Why the guard is load-bearing: a per-column program cannot see the other
        # columns, so folding there reassociates the computation and changes values.
        totals = [3.0, -1.0, 2.0, 0.5, -4.0]
        row_denominator = math.fsum(math.exp(value) for value in totals)
        per_column_denominator = [math.exp(value) for value in totals]
        self.assertNotAlmostEqual(row_denominator, per_column_denominator[0])
        # Pushing the pointwise epilogue inside the fold would reassociate too; the
        # candidate keeps the epilogue after the K fold, and the CPU agreement test
        # above detects any such mutation.
        bias, products = 0.25, [1.5, -0.5, 2.0, 0.0, -1.0, 0.75]
        self.assertNotEqual(math.fsum(value + bias for value in products),
                            math.fsum(products) + bias)

    def test_sliced_second_operand_is_refused(self):
        self.assert_refused(sliced_document(), 'access_domain')

    def test_nonmetal_route_is_refused(self):
        document = frontend.parse((ROOT/'examples/python/b300_gemm_bias.py').read_text()).document
        self.assert_refused(document, 'target_route')

    def test_already_column_programmed_is_refused(self):
        self.assert_refused(column_programmed_document(), 'program_shape')

    def test_multiple_stores_are_refused(self):
        self.assert_refused(two_store_document(), 'operation_domain')

    def test_identity_must_be_fresh_and_input_is_never_repaired(self):
        document = gemm_document()
        self.assertEqual(self.specialize(document, schedule_id=document['schedule_id']).reason, 'result_identity')
        self.assertEqual(self.specialize(document, entry_point='').reason, 'result_identity')
        broken = gemm_document()
        broken['operations'][-1]['reads'] = ['missing_buffer']
        before = deepcopy(broken)
        result = self.specialize(broken)
        self.assertFalse(result.applied)
        self.assertEqual(result.reason, 'input_refused')
        self.assertEqual(broken, before)


class ColumnSpecializationCliTests(unittest.TestCase):
    def invoke(self, output, schedule_path):
        from unittest.mock import patch
        import contextlib
        import io
        from tools import apply_column_specialization as cli
        compiler = Compiler.load(ROOT, ROOT/'compiler/revision.json')
        args = ['apply_column_specialization.py', '--schedule', str(schedule_path),
                '--schedule-id', 'cli_columns', '--entry-point', 'cli_columns', '--output', str(output)]
        with patch.object(cli.Compiler, 'load', return_value=compiler), patch('sys.argv', args), \
                contextlib.redirect_stdout(io.StringIO()):
            return cli.main()

    def test_applied_cli_writes_complete_candidate(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            schedule = Path(root)/'schedule.py'
            schedule.write_text(_HEADER.format(name='cli-gemm', entry='cake_cli',
                metadata='', rows=ROWS, width=WIDTH, columns=COLUMNS, extra_args='', axes='',
                body=_PLAIN_STORE[1], final=_PLAIN_STORE[0]))
            out = Path(root).resolve()/'candidate'
            self.assertEqual(self.invoke(out, schedule), 0)
            report = json.loads((out/'result.json').read_text())
            self.assertTrue(report['applied'])
            self.assertIn('compiler_revision_id', report)
            self.assertTrue((out/'schedule.json').is_file())
            self.assertTrue((out/'lowered.py').is_file())
            self.assertIn('preserved', report['workload_binding'])

    def test_refused_cli_retains_reason_without_a_candidate(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            schedule = Path(root)/'schedule.py'
            schedule.write_text(_HEADER.format(name='cli-sliced', entry='cake_cli',
                metadata='', rows=ROWS, width=WIDTH, columns=COLUMNS, extra_args='', axes='', body=_PLAIN_STORE[1], final=_PLAIN_STORE[0])
                .replace('b[:, :]', 'b[:, :3]').replace('bias[:]', 'bias[:3]').replace('out[row, :]', 'out[row, :3]'))
            out = Path(root).resolve()/'refused'
            self.assertEqual(self.invoke(out, schedule), 2)
            report = json.loads((out/'result.json').read_text())
            self.assertFalse(report['applied'])
            self.assertEqual(report['reason'], 'access_domain')
            self.assertEqual({p.name for p in out.iterdir()}, {'result.json'})
