"""Guarded full-graph MMA candidates; CPU source semantics, not GPU qualification."""
from copy import deepcopy
import math
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Program, frontend
from open_cake_ir.compiler.program_passes import TRANSFORMATIONS
from open_cake_ir.tasks.workloads import create_task
from tests.contracts.test_epilogue_fusion import execute

ROOT = Path(__file__).resolve().parents[2]


def document(rows=17, depth=64, columns=32, *, transposed=False, silu=False, residual=False):
    right_shape = (columns, depth) if transposed else (depth, columns)
    axis = 1 if transposed else 0
    epilogue = '        shifted = totals + bias_values\n'
    if residual:
        epilogue += '        residual_values = lm.load(residual[row, :], id="load_residual")\n        shifted2 = shifted + residual_values\n'
    value = 'shifted2' if residual else 'shifted'
    if silu:
        epilogue += f'        negated = {value} * -1.0\n        exponential = lm.exp(negated)\n        gate = lm.reciprocal(exponential + 1.0)\n        activated = {value} * gate\n'
        value = 'activated'
    return frontend.parse(f'''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="row_dot", target="gfx938", backend="triton", entry_point="row_dot",
               metadata={{"workload_contract_sha256": "{'1' * 64}"}})
def candidate(lm, a: cake.Tensor(({rows}, {depth}), "fp32"), b: cake.Tensor({right_shape}, "fp32"), bias: cake.Tensor(({columns},), "fp32"), out: cake.Tensor(({rows}, {columns}), "fp32", mode="output"){f', residual: cake.Tensor(({rows}, {columns}), "fp32")' if residual else ''}):
    compute = lm.role(execution_groups=[0])
    row = lm.program(a, axis=0, dimension=0, tile=1)
    with compute:
        left = lm.load(a[row, :], id="load_a")
        right = lm.load(b[:, :], id="load_b")
        bias_values = lm.load(bias[:], id="load_bias")
        products = right * lm.broadcast(left, axis={axis})
        totals = lm.reduce(products, op="sum", axis={axis}, scope="cta", across_loop=False, id="sum_k")
{epilogue}        lm.store(out[row, :], {value}, coalesced=False, id="store_out")
''').document


class FP32ContractionRewrite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')

    def apply(self, source, **parameters):
        return self.compiler.specialize_fp32_contraction(source, **(dict(
            row_tile=16, column_tile=16, k_tile=32, num_warps=4, num_stages=1,
            schedule_id='tiled_ieee', entry_point='tiled_ieee') | parameters))

    def test_canonical_task_graphs_lower_and_keep_workload_public_tensors_and_epilogue(self):
        for task in ('gemm', 'gemm_silu', 'aka_gemm_nt_bias'):
            _, source = create_task(task, backend='triton-dcu', rows=1024, columns=64, depth=256)
            original = frontend.parse(source).document
            saved = deepcopy(original)
            result = self.apply(original, row_tile=32, column_tile=32, k_tile=64, num_stages=3)
            self.assertTrue(result.applied, (task, result.reason, result.message))
            self.assertEqual(original, saved)
            output = result.schedule
            self.assertEqual(output['metadata'], original['metadata'])
            self.assertEqual(output['outputs'], original['outputs'])
            self.assertEqual([b for b in output['buffers'] if b['space'] == 'global'],
                             [b for b in original['buffers'] if b['space'] == 'global'])
            source = self.compiler.lower(result.assessment).source
            self.assertIn('tl.dot(', source)
            self.assertIn('input_precision="ieee"', source)
            self.assertNotIn('input_precision="tf32"', source)
            self.assertNotIn('tl.sum(', source)
            self.assertIn('num_stages=3', source)
            self.assertEqual(sum(op['kind'] == 'transpose' for op in output['operations']),
                             0 if task == 'aka_gemm_nt_bias' else 1)

    def test_emitted_non_square_layouts_and_masked_tails_match_independent_oracle(self):
        # Unequal K/N catches swapping the right operand axes. Odd extents and
        # M=1 exercise ownership/masks without asserting physical dot behavior.
        for rows, depth, columns in ((17, 64, 32), (1, 19, 23)):
            for nt in (False, True):
                for silu in (False, True):
                    with self.subTest(rows=rows, nt=nt, silu=silu):
                        original = document(rows, depth, columns, transposed=nt, silu=silu, residual=True)
                        result = self.apply(original, k_tile=16)
                        self.assertTrue(result.applied, (result.reason, result.message))
                        a = [((i * 3 + k) % 11 - 5) / 8 for i in range(rows) for k in range(depth)]
                        b_kn = [[((k + 2 * j) % 7 - 3) / 4 for j in range(columns)] for k in range(depth)]
                        bias = [(j % 5 - 2) / 8 for j in range(columns)]
                        residual = [(i % 7 - 3) / 4 for i in range(rows * columns)]
                        b = ([b_kn[k][j] for j in range(columns) for k in range(depth)] if nt
                             else [value for row in b_kn for value in row])
                        expected = []
                        for i in range(rows):
                            for j in range(columns):
                                value = math.fsum(a[i * depth + k] * b_kn[k][j] for k in range(depth)) + bias[j] + residual[i * columns + j]
                                expected.append(value / (1 + math.exp(-value)) if silu else value)
                        actual, trace = execute(result.schedule, dict(a=a, b=b, bias=bias, residual=residual))
                        self.assertEqual(set(trace.stores.values()), {1})
                        self.assertEqual(len(trace.stores), rows * columns)
                        for observed, reference in zip(actual['out'], expected):
                            self.assertAlmostEqual(observed, reference, places=6)

    def test_full_program_path_preserves_bindings_and_validates_author_parameters(self):
        program = Program.from_schedule(document())
        params = dict(stage=program.stages[0].name, row_tile=16, column_tile=16, k_tile=32,
                      num_warps=4, num_stages=1, schedule_id='program_mma', entry_point='program_mma')
        name = 'specialize_fp32_contraction'
        self.assertIn(name, [item.name for item in TRANSFORMATIONS])
        result = self.compiler.rewrite_program(program, name, params)
        self.assertTrue(result.applied, result.message)
        for field in ('inputs', 'outputs', 'tensors', 'target'):
            self.assertEqual(result.program.document[field], program.document[field])
        self.assertEqual(result.program.document['stages'][0]['bindings'], program.document['stages'][0]['bindings'])
        self.compiler.lower_program(result.program)
        for wrong in (params | {'row_tile': True}, params | {'num_stages': 0}, params | {'num_warps': 32}):
            self.assertFalse(self.compiler.rewrite_program(program, name, wrong).applied)
        del params['k_tile']
        self.assertEqual(self.compiler.rewrite_program(program, name, params).reason, 'transform_parameters')
        tailed = Program.from_schedule(document(1, 19, 23))
        params.update(stage=tailed.stages[0].name, k_tile=16)
        self.assertTrue(self.compiler.rewrite_program(tailed, name, params).applied)

    def test_legal_non_dot_reduction_is_refused_by_the_contraction_guard(self):
        _, source = create_task('pairwise_sqdist', backend='triton-dcu', rows=32, columns=32, depth=64)
        original = frontend.parse(source).document
        self.assertTrue(self.compiler.assess(original).lowering_eligible)
        self.assertEqual(self.apply(original).reason, 'contraction_shape')
        _, source = create_task('attention_decode', backend='triton-dcu', rows=32, columns=32, depth=64)
        original = frontend.parse(source).document
        self.assertTrue(self.compiler.assess(original).lowering_eligible)
        self.assertEqual(self.apply(original).reason, 'contraction_shape')

    def test_existing_loop_and_persistent_mapping_are_not_silently_replaced(self):
        candidate = self.apply(document()).schedule
        self.assertEqual(self.apply(candidate, schedule_id='twice').reason, 'execution_commitments')
        original = document()
        original['program_map']['persistent'] = True
        original['residency'] = {'ctas_per_multiprocessor': 1}
        self.assertTrue(self.compiler.assess(original).lowering_eligible)
        self.assertEqual(self.apply(original).reason, 'execution_commitments')

    def test_rounding_or_alias_storage_is_outside_the_rewrite(self):
        original = document()
        next(b for b in original['buffers'] if b['name'] == 'a')['byte_offset'] = 4
        self.assertTrue(self.compiler.assess(original).lowering_eligible)
        self.assertEqual(self.apply(original).reason, 'storage_domain')

        # This BF16 load plus explicit widening is legal input. Its dtype and
        # rounding seam are outside the pass, not an incidental invalid reduce.
        original = document()
        next(b for b in original['buffers'] if b['name'] == 'a')['dtype'] = 'bf16'
        raw = deepcopy(next(b for b in original['buffers'] if b['name'] == 'left'))
        raw.update(name='left_narrow', dtype='bf16')
        original['buffers'].append(raw)
        original['operations'][0]['writes'] = ['left_narrow']
        original['operations'].insert(1, dict(id='widen_left', kind='cast', role='compute',
            reads=['left_narrow'], writes=['left'], parameters={'to': 'fp32'}, depends_on=['load_a']))
        product = next(op for op in original['operations'] if op['id'] == 'products')
        product['depends_on'] = ['widen_left', 'load_b']
        self.assertTrue(self.compiler.assess(original).lowering_eligible)
        self.assertEqual(self.apply(original).reason, 'storage_domain')

        original = document()
        original['buffers'][0]['mode'] = 'state'
        self.assertFalse(self.apply(original).applied)

    def test_producer_values_cannot_escape_and_extra_ordering_is_preserved_by_refusal(self):
        original = document()
        bias = next(op for op in original['operations'] if op['id'] == 'load_bias')
        bias['depends_on'] = ['load_a']
        self.assertTrue(self.compiler.assess(original).lowering_eligible)
        self.assertEqual(self.apply(original).reason, 'execution_commitments')
        # Equal vector lengths keep the modified source type-correct while
        # exposing a contraction load to the epilogue.
        original = document(depth=32, columns=32)
        shifted = next(op for op in original['operations'] if op['id'] == 'shifted')
        shifted['reads'][1] = 'left'
        shifted['depends_on'] = ['sum_k', 'load_a']
        self.assertTrue(self.compiler.assess(original).lowering_eligible)
        self.assertEqual(self.apply(original).reason, 'value_ownership')

    def test_target_and_identity_refusals_are_explicit(self):
        original = document()
        original['target'] = 'sm_100a'
        self.assertTrue(self.compiler.assess(original).lowering_eligible)
        self.assertEqual(self.apply(original).reason, 'target_route')
        self.assertEqual(self.apply(document(), schedule_id='row_dot').reason, 'result_identity')
        for value in (1, 8, 24, False):
            self.assertEqual(self.apply(document(), row_tile=value).reason, 'tile_extent')
        self.assertEqual(self.apply(document(), k_tile=64).reason, 'tile_extent')
        self.assertEqual(self.apply(None).reason, 'input_refused')

    def test_register_grid_anchor_cannot_shrink_the_public_row_domain(self):
        original = document(rows=64, depth=64, columns=32)
        original['program_map']['axes'][0]['buffer'] = 'left'
        self.assertTrue(self.compiler.assess(original).lowering_eligible)
        self.assertEqual(self.apply(original).reason, 'program_shape')


if __name__ == '__main__':
    unittest.main()
