"""Execute emitted sequential loop memory/carry semantics on CPU, not GPU qualification."""
import ast
from copy import deepcopy
import math
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.compiler.backends.triton import emit, preflight
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import Target
from tests.contracts.test_triton_loop_scopes import _execute

ROOT = Path(__file__).resolve().parents[2]


def softmax_source(*, rows=2, columns=7, tile=4, target='xcore1002'):
    return f'''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="three-pass-softmax", target="{target}", backend="triton", entry_point="three_pass_softmax")
def candidate(lm, x: cake.Tensor(({rows}, {columns}), "fp32"), out: cake.Tensor(({rows}, {columns}), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0])
    row = lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
        for max_col in lm.range(x, dimension=1, tile={tile}, name="pass_max"):
            indices_max = lm.coordinate(source="loop_tile", name="max_col", id="max_indices")
            valid_max = lm.compare(indices_max, {columns}, op="lt", id="max_valid")
            values_max = lm.load(x[row, max_col], id="load_max")
            guarded_max = lm.select(valid_max, values_max, "negative_infinity", id="guard_max")
            maximum = lm.reduce(guarded_max, op="max", axis=0, scope="cta", id="maximum")
        for sum_col in lm.range(x, dimension=1, tile={tile}, name="pass_sum"):
            indices_sum = lm.coordinate(source="loop_tile", name="sum_col", id="sum_indices")
            valid_sum = lm.compare(indices_sum, {columns}, op="lt", id="sum_valid")
            values_sum = lm.load(x[row, sum_col], id="load_sum")
            shifted_sum = values_sum - maximum
            exp_sum = lm.exp(shifted_sum, id="exp_sum")
            guarded_sum = lm.select(valid_sum, exp_sum, 0.0, id="guard_sum")
            total = lm.reduce(guarded_sum, op="sum", axis=0, scope="cta", id="total")
        inverse = lm.reciprocal(total, id="inverse")
        for out_col in lm.range(x, dimension=1, tile={tile}, name="pass_out"):
            values_out = lm.load(x[row, out_col], id="load_out")
            shifted_out = values_out - maximum
            exp_out = lm.exp(shifted_out, id="exp_out")
            result = exp_out * inverse
            lm.store(out[row, out_col], result, coalesced=True, id="store_out")
'''


class SequentialRegions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT/'compiler/revision.json')

    def test_three_pass_carries_tail_masks_and_emitted_source_agree_with_external_math(self):
        for target in ('sm_100a', 'xcore1002', 'gfx938', 'gfx1151'):
            for columns in (1, 7, 8, 17):
                with self.subTest(target=target,columns=columns):
                    d=frontend.parse(softmax_source(columns=columns,target=target)).document
                    a=self.compiler.assess(d)
                    self.assertTrue(a.lowering_eligible, a.findings)
                    result=self.compiler.lower(a)
                    values=[-1000.0 + (i % 5) for i in range(2*columns)]
                    memory={'x':values[:], 'out':[None]*(2*columns)}
                    emission=emit(Schedule.from_dict(d),self.compiler._revision.targets[target])
                    observer=_execute(emission,memory)
                    expected=[]
                    for row in range(2):
                        v=values[row*columns:(row+1)*columns];m=max(v)
                        denominator=sum(math.exp(x-m) for x in v)
                        expected.extend(math.exp(x-m)/denominator for x in v)
                    for actual,reference in zip(memory['out'],expected,strict=True):
                        self.assertAlmostEqual(actual,reference,places=12)
                    self.assertEqual(memory['x'],values)
                    self.assertEqual(set(observer.stores.values()),{1})
                    self.assertEqual(sum(isinstance(n,ast.For) for n in ast.walk(ast.parse(result.source))),3)
                    self.assertIn('inverse',result.source_map)

    def test_four_sibling_regions_are_ordered_by_operations_not_declaration_count(self):
        source=softmax_source().replace('        inverse = lm.reciprocal(total, id="inverse")', '''        for repeat_col in lm.range(x, dimension=1, tile=4, name="pass_repeat"):
            repeated = lm.load(x[row, repeat_col], id="load_repeat")
            repeated_sum = lm.reduce(repeated, op="sum", axis=0, scope="cta", id="repeated_sum")
        inverse = lm.reciprocal(total, id="inverse")''')
        d=frontend.parse(source).document
        d['tile_loops'].reverse()
        a=self.compiler.assess(d);self.assertTrue(a.lowering_eligible,a.findings)
        emission=emit(Schedule.from_dict(d),self.compiler._revision.targets['xcore1002'])
        memory={'x':[float(i) for i in range(14)],'out':[None]*14}
        observer=_execute(emission,memory)
        self.assertEqual(set(observer.stores.values()),{1})
        self.assertLess(emission.source.index('for max_col'),emission.source.index('for sum_col'))
        self.assertLess(emission.source.index('for repeat_col'),emission.source.index('for out_col'))

    def test_early_live_out_use_is_refused_by_the_common_dependency_or_scope_owner(self):
        d=frontend.parse(softmax_source()).document
        inverse=next(op for op in d['operations'] if op['id']=='inverse')
        d['operations'].remove(inverse);d['operations'].insert(0,inverse)
        a=self.compiler.assess(d)
        self.assertFalse(a.accepted)
        self.assertTrue(any(f.blocks_acceptance and f.path for f in a.findings),a.findings)

    def test_query_stop_and_unimplemented_options_remain_explicit_refusals(self):
        for change,code in (({'stop':{'program':'row','add':1,'floor_div':1}},'TRITON_NESTED_LOOP_STOP'),
                            ({'flatten':True},'TRITON_NESTED_LOOP_OPTION')):
            d=frontend.parse(softmax_source(target='sm_100a')).document
            if 'stop' in change:d['tile_loops'][0]['stop']=change['stop']
            else:d['tile_loops'][0]['range_options'].update(change)
            findings=preflight(Schedule.from_dict(d),Target.load(ROOT/'compiler/targets/sm_100a.json'))
            self.assertTrue(any(f.code==code for f in findings),findings)
