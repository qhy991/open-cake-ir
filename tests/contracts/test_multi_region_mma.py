"""Multi-region MMA provenance and emitted-source semantics; no GPU precision claim."""
import ast
from copy import deepcopy
import math
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.compiler.backends.triton import emit, preflight
from open_cake_ir.compiler.ir import Schedule
from tests.contracts.test_triton_loop_scopes import _execute, _gemm

ROOT = Path(__file__).resolve().parents[2]


def attention_source(rows=17, features=33, keys=32, columns=49, tile=16):
    """Retained flash16 structure, with independent feature and output dimensions."""
    return f'''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="multi-region-attention", target="gfx938", backend="triton", entry_point="multi_region_attention")
def candidate(lm, q: cake.Tensor(({rows}, {features}), "fp32"), k: cake.Tensor(({keys}, {features}), "fp32"), v: cake.Tensor(({keys}, {columns}), "fp32"), out: cake.Tensor(({rows}, {columns}), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0, 1, 2, 3])
    row_block = lm.program(q, axis=0, dimension=0, tile=16)
    for kk in lm.range(q, name="k_loop", dimension=1, tile={tile}, num_stages=2, loop_unroll_factor=1, disallow_acc_multi_buffer=True):
        with compute:
            query = lm.load(q[row_block, kk], reuse="streamed", id="load_q")
            keys = lm.load(k[:, kk], reuse="streamed", id="load_k")
            scores = lm.mma(query, keys, instruction={{"contract": "triton.dot.fp32_ieee"}}, tile_shape=[16, {keys}, {tile}], id="qk_dot")
    with compute:
        logits = scores * 0.0625
        peak = lm.reduce(logits, op="max", axis=1, scope="cta", across_loop=False, id="max_logit")
        shifted = logits - lm.broadcast(peak, axis=0)
        weights = lm.exp(shifted, id="exp")
        total = lm.reduce(weights, op="sum", axis=1, scope="cta", across_loop=False, id="sum_exp")
    for kv in lm.range(v, name="vo_loop", dimension=1, tile={tile}, num_stages=2, loop_unroll_factor=1, disallow_acc_multi_buffer=True):
        with compute:
            values = lm.load(v[:, kv], reuse="streamed", id="load_v")
            values_t = lm.transpose(values, id="transpose_v")
            chunk = lm.mma(weights, values_t, instruction={{"contract": "triton.dot.fp32_ieee"}}, tile_shape=[16, {tile}, {keys}], id="av_dot")
            lm.store(out[row_block, kv], chunk / lm.broadcast(total, axis=0), coalesced=True, id="store_out")
'''


def nested_cast_source():
    return '''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="nested-cast-kn-gemm", target="gfx938", backend="triton", entry_point="nested_cast_kn_gemm")
def candidate(lm, a: cake.Tensor((35, 33), "bf16"), b: cake.Tensor((33, 49), "bf16"), out: cake.Tensor((35, 49), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0])
    n = lm.program(b, axis=0, dimension=1, tile=32)
    with compute:
        for m in lm.range(a, name="m_loop", dimension=0, tile=16):
            for k in lm.range(a, name="k_loop", dimension=1, tile=16):
                av = lm.load(a[m,k], id="load_a")
                bv = lm.load(b[k,n], id="load_b")
                af = lm.cast(av, to="fp32", id="cast_a")
                bf = lm.cast(bv, to="fp32", id="cast_b")
                bt = lm.transpose(bf, id="transpose_b")
                result = lm.mma(af, bt, instruction={"contract":"triton.dot.fp32_ieee"}, tile_shape=[16,32,16], id="dot")
            lm.store(out[m,n], result, id="store_out")
'''


class MultiRegionMMA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)
        cls.target = cls.compiler._revision.targets['gfx938']

    def lower(self, document):
        assessment = self.compiler.assess(document)
        self.assertTrue(assessment.accepted, assessment.findings)
        self.assertTrue(assessment.lowering_eligible, assessment.findings)
        return emit(Schedule.from_dict(document), self.target)

    def refuse(self, document, code):
        assessment = self.compiler.assess(document)
        self.assertFalse(assessment.lowering_eligible)
        self.assertIn(code, [finding.code for finding in assessment.findings])
        self.assertTrue(all(f.path for f in assessment.findings if f.code == code))

    def test_retained_flash16_shape_lowers_without_changing_precision_or_carry(self):
        document = frontend.parse(attention_source(1024, 256, 64, 256, 64)).document
        emission = self.lower(document)
        source = emission.source
        self.assertIn('scores += tl.dot(query, tl.trans(keys), input_precision="ieee")', source)
        self.assertIn('chunk = tl.dot(weights, tl.trans(values_t), input_precision="ieee")', source)
        self.assertNotIn('chunk +=', source)
        self.assertNotIn('tf32', source)
        kernel = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef))
        loops = [node for node in kernel.body if isinstance(node, ast.For)]
        self.assertEqual([loop.target.id for loop in loops], ['kk', 'kv'])
        self.assertIn('weights', {node.targets[0].id for node in kernel.body
                                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)})
        self.assertFalse(any(isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                             and node.targets[0].id == 'chunk' for node in kernel.body))

    def test_chunked_attention_matches_unchunked_math_with_row_k_and_output_tails(self):
        for rows, features, keys, columns in [(17, 33, 32, 49), (3, 19, 16, 35)]:
            with self.subTest(shape=(rows, features, keys, columns)):
                document = frontend.parse(attention_source(rows, features, keys, columns)).document
                schedule = Schedule.from_dict(document)
                self.assertTrue(schedule.mma_accumulates_over(schedule.operation('qk_dot'), schedule.tile_loop('k_loop')))
                self.assertFalse(schedule.mma_accumulates_over(schedule.operation('av_dot'), schedule.tile_loop('vo_loop')))
                q = [((i * 3 + t) % 7 - 3) / 8 for i in range(rows) for t in range(features)]
                k = [((j + t * 2) % 5 - 2) / 4 for j in range(keys) for t in range(features)]
                v = [(j * 7 + c * 3) % 19 - 9 for j in range(keys) for c in range(columns)]
                memories = {'q': q[:], 'k': k[:], 'v': v[:], 'out': [None] * (rows * columns)}
                observer = _execute(self.lower(document), memories)
                expected = []
                for i in range(rows):
                    logits = [sum(q[i * features + t] * k[j * features + t] for t in range(features)) * 0.0625 for j in range(keys)]
                    peak = max(logits)
                    weights = [math.exp(x - peak) for x in logits]
                    denominator = sum(weights)
                    expected.extend(sum(weights[j] * v[j * columns + c] for j in range(keys)) / denominator for c in range(columns))
                for actual, wanted in zip(memories['out'], expected, strict=True):
                    self.assertAlmostEqual(actual, wanted, places=12)
                self.assertEqual([memories[name] for name in ('q', 'k', 'v')], [q, k, v])
                self.assertEqual(set(observer.stores.values()), {1})
                self.assertEqual(len(observer.stores), rows * columns)
                # Invariant QK/softmax work stays outside the AV loop.
                self.assertEqual(observer.loads[id(memories['q'])], math.ceil(rows / 16) * math.ceil(features / 16))

    def test_nested_cast_transpose_chain_retains_inner_k_and_resets_for_outer_rows(self):
        document = frontend.parse(nested_cast_source()).document
        emission = self.lower(document)
        a = [(i * 3 + t) % 7 - 3 for i in range(35) for t in range(33)]
        b = [(t * 2 + j) % 5 - 2 for t in range(33) for j in range(49)]
        memories = {'a': a[:], 'b': b[:], 'out': [None] * (35 * 49)}
        observer = _execute(emission, memories)
        self.assertEqual(memories['out'], [sum(a[i * 33 + t] * b[t * 49 + j] for t in range(33)) for i in range(35) for j in range(49)])
        self.assertEqual(set(observer.stores.values()), {1})
        self.assertIn('af = av.to(tl.float32)', emission.source)
        self.assertIn('bt = tl.trans(bf)', emission.source)

    def test_malformed_aliases_supply_no_operand_proof(self):
        for change in ('shape', 'dtype', 'storage', 'cycle', 'multiple_writers', 'cast_dtype'):
            document = frontend.parse(nested_cast_source()).document
            op = next(op for op in document['operations'] if op['id'] == 'transpose_b')
            result = next(buf for buf in document['buffers'] if buf['name'] == 'bt')
            if change == 'shape': result['shape'] = [16, 32]
            elif change == 'dtype': result['dtype'] = 'bf16'
            elif change == 'storage': result['space'] = 'global'
            elif change == 'cycle': op['reads'] = ['bt']
            elif change == 'cast_dtype': next(op for op in document['operations'] if op['id'] == 'cast_b')['parameters']['to'] = 'bf16'
            else:
                extra = deepcopy(op); extra['id'] = 'ambiguous'; document['operations'].append(extra)
            with self.subTest(change=change):
                schedule = Schedule.from_dict(document)
                self.assertIsNone(schedule.staged_operand_provenance('bt'))
                self.assertIn('TRITON_NESTED_MMA_OPERAND', [f.code for f in preflight(schedule, self.target)])
                self.assertFalse(self.compiler.assess(document).accepted)

    def test_varying_arithmetic_is_not_misclassified_as_an_invariant(self):
        source = attention_source().replace('values_t = lm.transpose(values,', 'scaled = values * 2.0\n            values_t = lm.transpose(scaled,')
        document = frontend.parse(source).document
        self.assertTrue(self.compiler.assess(document).accepted)
        self.refuse(document, 'TRITON_NESTED_MMA_OPERAND')

    def test_noncarry_value_cannot_escape_first_region(self):
        source = attention_source(keys=16).replace('logits = scores * 0.0625', 'logits = query * 0.0625')
        self.refuse(frontend.parse(source).document, 'BUFFER_ESCAPES_LOOP')

    def test_output_axis_result_cannot_be_reused_after_its_region(self):
        source = attention_source() + '    with compute:\n        escaped = chunk * 2.0\n'
        self.refuse(frontend.parse(source).document, 'BUFFER_ESCAPES_LOOP')

    def test_invariant_use_before_definition_and_input_writes_remain_invalid(self):
        document = frontend.parse(attention_source()).document
        exp = next(op for op in document['operations'] if op['id'] == 'exp')
        document['operations'].remove(exp); document['operations'].append(exp)
        self.refuse(document, 'OP_READ_BEFORE_WRITE')
        document = frontend.parse(attention_source()).document
        next(buf for buf in document['buffers'] if buf['name'] == 'out')['mode'] = 'input'
        self.refuse(document, 'INPUT_WRITTEN')

    def test_ancestor_contraction_and_missing_output_axis_ownership_stay_refused(self):
        document = _gemm()
        document['access_maps'][0]['indices'].reverse()
        self.refuse(document, 'TRITON_MMA_ANCESTOR_CARRY')
        document = frontend.parse(attention_source()).document
        store = next(a for a in document['access_maps'] if a['operation'] == 'store_out')
        store['indices'][1] = {'source': 'dimension', 'dimension': 1, 'extent': 16}
        self.refuse(document, 'TRITON_LOOP_STORE_OWNERSHIP')


if __name__ == '__main__':
    unittest.main()
