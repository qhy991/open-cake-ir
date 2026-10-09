"""Pure pointwise width changes retain typed values, effects and rounding."""
import ast
from copy import deepcopy
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, frontend

ROOT = Path(__file__).resolve().parents[2]


def document(*, loop=True, target='gfx938', input_dtype='fp32', output_dtype='fp32',
             intermediate_narrow=False):
    columns = 70 if loop else 64
    index = 'col' if loop else ':'
    body = [f'raw = lm.load(x[row, {index}], id="load_x")']
    value = 'raw'
    if input_dtype != 'fp32':
        body.append('values = lm.cast(raw, to="fp32", id="widen")')
        value = 'values'
    body.extend([
        f'slopes = lm.load(slope[{index}], reuse="reused", id="load_slope")',
        f'scaled = {value} * slopes',
        f'positive = lm.compare({value}, 0.0, op="gt", id="compare")',
        f'selected = lm.select(positive, {value}, scaled, id="select")',
    ])
    result = 'selected'
    if intermediate_narrow:
        body.extend(['rounded = lm.cast(selected, to="fp16", id="round")',
                     'restored = lm.cast(rounded, to="fp32", id="restore")'])
        result = 'restored'
    elif output_dtype != 'fp32':
        body.append(f'rounded = lm.cast(selected, to="{output_dtype}", id="round")')
        result = 'rounded'
    body.append(f'lm.store(out[row, {index}], {result}, id="store_out")')
    indent = '            ' if loop else '        '
    body = '\n'.join(indent + line for line in body)
    if loop:
        body = ('        for col in lm.range(slope, dimension=0, tile=32, '
                'name="col", num_stages=2):\n' + body)
    return frontend.parse(f'''
from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="pure-width", target="{target}", backend="triton", entry_point="pure_width")
def candidate(lm, x: cake.Tensor((3, {columns}), "{input_dtype}"),
              slope: cake.Tensor(({columns},), "fp32"),
              out: cake.Tensor((3, {columns}), "{output_dtype}", mode="output")):
    compute = lm.role(execution_groups=[0])
    row = lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
{body}
''').document


class PureWidthSpecializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)

    def apply(self, source):
        return self.compiler.specialize_triton_warps(source, num_warps=2,
            schedule_id='pure-width-specialized', entry_point='pure_width_specialized')

    def check_preserved(self, source):
        original = deepcopy(source)
        before = self.compiler.assess(source)
        self.assertTrue(before.lowering_eligible, before.findings)
        result = self.apply(source)
        self.assertTrue(result.applied, (result.reason, result.message))
        self.assertEqual(source, original)
        for field in original.keys() - {'roles', 'schedule_id', 'lowering'}:
            self.assertEqual(result.schedule[field], original[field], field)
        self.assertEqual(result.schedule['roles'][0]['execution_groups'], [0, 1])

        def kernel(lowering):
            node = next(node for node in ast.parse(lowering.source).body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == lowering.toolchain_requirements['kernel_entry_point'])
            node.name = 'same_kernel'
            return ast.dump(node, include_attributes=False)

        emitted = self.compiler.lower(result.assessment)
        self.assertEqual(kernel(self.compiler.lower(before)), kernel(emitted))
        self.assertEqual(emitted.toolchain_requirements['compile_options']['num_warps'], 2)
        return emitted.source

    def test_typed_selection_and_fixed_pointwise_loop_preserve_graph_and_tails(self):
        for target in ('gfx938', 'sm_103a'):
            for loop in (False, True):
                with self.subTest(target=target, loop=loop):
                    source = self.check_preserved(document(target=target, loop=loop))
                    self.assertIn('tl.where(', source)
                    if loop:
                        self.assertIn('mask=', source)
                        self.assertIn('num_stages=2', source)

    def test_width_change_does_not_admit_undeclared_target_operations(self):
        source = document(target='sm_100a')
        assessment = self.compiler.assess(source)
        self.assertFalse(assessment.lowering_eligible)
        self.assertIn('TARGET_OPERATION_UNSUPPORTED', [f.code for f in assessment.findings])
        self.assertEqual(self.apply(source).reason, 'input_refused')

    def test_widening_and_terminal_narrowing_keep_explicit_rounding(self):
        for dtype in ('fp16', 'bf16'):
            with self.subTest(dtype=dtype):
                self.check_preserved(document(input_dtype=dtype))
                source = self.check_preserved(document(output_dtype=dtype))
                self.assertIn('.to(tl.' + ('float16' if dtype == 'fp16' else 'bfloat16') + ')', source)

    def test_valid_intermediate_narrowing_and_integer_cast_remain_unqualified(self):
        for source in (document(intermediate_narrow=True), document(input_dtype='int32')):
            with self.subTest(operations=source['operations']):
                self.assertTrue(self.compiler.assess(source).lowering_eligible)
                self.assertEqual(self.apply(source).reason, 'loop_domain')

    def test_non_int32_predicate_still_fails_canonical_typing(self):
        source = document()
        select = next(op for op in source['operations'] if op['id'] == 'select')
        select['reads'][0] = 'raw'
        assessment = self.compiler.assess(source)
        self.assertIn('VALUE_OPERATION_TYPE', [f.code for f in assessment.findings])
        self.assertEqual(self.apply(source).reason, 'input_refused')

    def test_repeated_output_store_still_fails_backend_ownership(self):
        source = document()
        next(buffer for buffer in source['buffers'] if buffer['name'] == 'out')['shape'][1] = 32
        store = next(access for access in source['access_maps'] if access['operation'] == 'store_out')
        store['indices'][1] = {'source': 'dimension', 'dimension': 1}
        assessment = self.compiler.assess(source)
        self.assertFalse(assessment.lowering_eligible)
        self.assertIn('TRITON_LOOP_STORE_OWNERSHIP', [f.code for f in assessment.findings])
        self.assertEqual(self.apply(source).reason, 'input_refused')

    def test_dynamic_loop_and_warp_specialization_remain_outside_domain(self):
        for change in ('stop', 'warp_specialize'):
            with self.subTest(change=change):
                source = document()
                if change == 'stop':
                    source['tile_loops'][0]['stop'] = {'program': 'row', 'add': 1, 'floor_div': 1}
                else:
                    source['tile_loops'][0]['range_options']['warp_specialize'] = True
                assessment = self.compiler.assess(source)
                if change == 'stop':
                    self.assertFalse(assessment.lowering_eligible)
                    self.assertIn('TRITON_LOOP_STOP_UNSUPPORTED', [f.code for f in assessment.findings])
                    self.assertEqual(self.apply(source).reason, 'input_refused')
                else:
                    self.assertTrue(assessment.lowering_eligible)
                    self.assertEqual(self.apply(source).reason, 'loop_domain')

    def test_persistent_program_and_residency_commitment_are_not_rewritten(self):
        for change in ('persistent', 'residency'):
            with self.subTest(change=change):
                source = document(loop=False)
                source['residency'] = {'ctas_per_multiprocessor': 1}
                if change == 'persistent':
                    source['program_map']['persistent'] = True
                self.assertTrue(self.compiler.assess(source).lowering_eligible)
                self.assertEqual(self.apply(source).reason, 'execution_commitments')


if __name__ == '__main__':
    unittest.main()
