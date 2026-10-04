"""The output domain, never zero-filled input rows, authorizes work pruning."""

import ast
from copy import deepcopy
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler.backends.triton import emit, preflight
from open_cake_ir.compiler.backends.triton_output_domain import output_tile_domain
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.performance.work import work_bound
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.verifier import verify

ROOT = Path(__file__).resolve().parents[2]
TARGET = Target.load(ROOT / 'compiler/targets/sm_100a.json')


def document():
    return json.loads((ROOT / 'corpus/schedules/ragged-grouped-gemm-b1-smoke.json').read_text())


def prefix_document():
    d = document()
    d['buffers'][3]['valid_extent'] = dict(d['buffers'][0]['valid_extent'])
    return d


class EmptyOutputTileContract(unittest.TestCase):
    def test_other_backend_retains_declared_work(self):
        d = prefix_document()
        d['lowering']['backend'] = 'native_cuda'
        s = Schedule.from_dict(d)
        self.assertIsNone(output_tile_domain(s))
        self.assertTrue(work_bound(s).flops_exact)

    def test_dense_result_is_observable_even_with_masked_inputs(self):
        s = Schedule.from_dict(document())
        self.assertIsNone(output_tile_domain(s))
        self.assertTrue(work_bound(s).flops_exact)
        tree = ast.parse(emit(s, TARGET).source)
        kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        self.assertFalse(any(isinstance(n, ast.If) for n in ast.walk(kernel)))

    def test_output_prefix_guards_all_computation_and_keeps_store_mask(self):
        s = Schedule.from_dict(prefix_document())
        self.assertFalse(any(f.blocks_acceptance for f in verify(s, TARGET)))
        self.assertFalse(preflight(s, TARGET))
        source = emit(s, TARGET).source
        kernel = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
        guards = [n for n in kernel.body if isinstance(n, ast.If)]
        self.assertEqual(len(guards), 1)
        calls = [n for n in ast.walk(guards[0]) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)]
        self.assertTrue(any(n.func.attr == 'dot' for n in calls))
        stores = [n for n in calls if n.func.attr == 'store']
        self.assertEqual(len(stores), 1)
        mask = next(k.value for k in stores[0].keywords if k.arg == 'mask')
        self.assertIn('store_c_c_valid_extent', ast.unparse(mask))
        self.assertFalse(work_bound(s).flops_exact)
        self.assertIn('dot', work_bound(s).uncounted_arithmetic)

    def test_runtime_index_does_not_authorize_skipping_whole_program(self):
        d = prefix_document()
        d['access_maps'][-1]['indices'][1] = {'source': 'loop_tile', 'name': 'k'}
        self.assertIsNone(output_tile_domain(Schedule.from_dict(d)))

    def test_one_dense_output_keeps_otherwise_ragged_program_live(self):
        d = prefix_document()
        dense = deepcopy(d['buffers'][3])
        dense['name'] = 'dense'
        del dense['valid_extent']
        d['buffers'].append(dense)
        store = deepcopy(d['operations'][-1])
        store['id'] = 'store_dense'
        store['writes'] = ['dense']
        d['operations'].append(store)
        access = deepcopy(d['access_maps'][-1])
        access.update(operation='store_dense', buffer='dense')
        d['access_maps'].append(access)
        d['outputs'].append('dense')
        s = Schedule.from_dict(d)
        self.assertFalse(any(f.blocks_acceptance for f in verify(s, TARGET)))
        self.assertIsNone(output_tile_domain(s))
        self.assertTrue(work_bound(s).flops_exact)

    def test_atomic_effect_keeps_unconditional_execution(self):
        d = prefix_document()
        atomic = json.loads((ROOT / 'corpus/schedules/atomic-reservation-b8-smoke.json').read_text())
        # The pure-output proof must notice effects independently of later typing.
        d['operations'].append(next(o for o in atomic['operations'] if o['kind'] == 'atomic_rmw'))
        self.assertIsNone(output_tile_domain(Schedule.from_dict(d)))

    def test_compiler_owned_guard_cannot_shadow_authored_pointer(self):
        d = prefix_document()
        d['buffers'].append({'name':'_cake_output_length', 'space':'global',
                             'dtype':'fp32', 'shape':[4], 'mode':'input'})
        codes = {f.code for f in preflight(Schedule.from_dict(d), TARGET)}
        self.assertIn('BACKEND_IDENTIFIER_COLLISION', codes)


if __name__ == '__main__':
    unittest.main()
