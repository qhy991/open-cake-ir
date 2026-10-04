"""Execute the emitted loop control; preserve reduction state and strict refusals."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.compiler.backends.common import EmitError
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.backends import metax
from open_cake_ir.compiler.target import declared_target

ROOT = Path(__file__).resolve().parents[2]


def reduction_document(depth=255, factor=2):
    return frontend.parse(f'''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="c550-unroll-reduction", target="xcore1002", backend="triton", entry_point="kernel")
def candidate(lm, x: cake.Tensor((3, {depth}), "fp32"), out: cake.Tensor((3,), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0, 1, 2, 3])
    row = lm.program(x, axis=0, dimension=0, tile=1)
    for k in lm.range(x, name="depth_loop", dimension=1, tile=64, loop_unroll_factor={factor}):
        with compute:
            value = lm.load(x[row, k], id="load")
            result = lm.reduce(value, op="sum", axis=1, id="sum")
    with compute:
        lm.store(out[row], result, id="store")
''').document


def traced_iterations(source, extent, tile):
    """Run actual emitted control statements, observing each operation body's index.

    Replace tensor operations with a visit, not loop bounds or iterator math.
    This is a CPU semantic check of grouping; it makes no JIT/device claim.
    """
    kernel = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'kernel')
    outer = copy.deepcopy(next(n for n in kernel.body if isinstance(n, ast.For)))
    inner = next((n for n in outer.body if isinstance(n, ast.For)), None)
    visit = ast.Expr(ast.Call(ast.Name('visit', ast.Load()), [ast.Name('k', ast.Load())], []))
    if inner is not None:
        inner.body = [inner.body[0], visit]  # actual k = base + lane * tile
    else:
        outer.body = [visit]
    rows = []
    env = {'tl': SimpleNamespace(range=range, static_range=range), 'visit': rows.append}
    # Ordinary tl.range carries num_stages; it has no bearing on index order.
    env['tl'].range = lambda *args, **kwargs: range(*args)
    for n in ast.walk(outer):
        if isinstance(n, ast.Name) and n.id.startswith('N_'):
            env[n.id] = extent
        elif isinstance(n, ast.Name) and n.id.startswith('BLOCK_'):
            env[n.id] = tile
    exec(compile(ast.fix_missing_locations(ast.Module([outer], type_ignores=[])), '<emitted-loop>', 'exec'), env)
    return rows, kernel, outer


class MetaxPartialUnroll(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT/'compiler/revision.json')

    def test_grouping_preserves_each_original_tile_and_tail_without_extra_iterations(self):
        for depth, factor in ((255, 2), (256, 2), (511, 2), (511, 4), (767, 3)):
            with self.subTest(depth=depth, factor=factor):
                candidate = reduction_document(depth, factor)
                before = copy.deepcopy(candidate)
                assessment = self.compiler.assess(candidate)
                self.assertTrue(assessment.accepted, assessment.findings)
                self.assertTrue(assessment.lowering_eligible, assessment.findings)
                source = self.compiler.lower(assessment).source
                observed, kernel, outer = traced_iterations(source, depth, 64)
                self.assertEqual(observed, list(range(0, depth, 64)))
                self.assertEqual(candidate, before)
                self.assertNotIn('loop_unroll_factor=', source)
                # The carried result is initialized once before the outer group,
                # and the original tail mask is still used in its load.
                assigned_before = [n for n in kernel.body[:kernel.body.index(next(n for n in kernel.body if isinstance(n, ast.For)))] if isinstance(n, ast.Assign)]
                self.assertTrue(any(any(isinstance(t, ast.Name) and t.id == 'result' for t in n.targets) for n in assigned_before))
                self.assertIn('k_offsets < N_X_D1', source)
                self.assertIsInstance(outer.body[0], ast.For)

    def test_full_unroll_and_other_targets_keep_their_original_routes(self):
        d = reduction_document(255, 4)
        source = self.compiler.lower(self.compiler.assess(d)).source
        rows, _, outer = traced_iterations(source, 255, 64)
        self.assertEqual(rows, [0, 64, 128, 192])
        self.assertIn('tl.static_range', source)
        self.assertFalse(any(isinstance(n, ast.For) for n in outer.body))
        for target in ('sm_103a', 'gfx938'):
            d = reduction_document(255, 2); d['target'] = target
            result = self.compiler.assess(d)
            self.assertTrue(result.lowering_eligible, result.findings)
            other = self.compiler.lower(result).source
            self.assertIn('loop_unroll_factor=2', other)
            self.assertNotIn('_maca_unroll_', other)

    def test_dynamic_stop_is_refused_by_unroll_even_when_its_static_cap_divides(self):
        d = reduction_document()
        d['tile_loops'][0]['stop'] = dict(program='row', add=1, floor_div=1)
        findings = metax.preflight(Schedule.from_dict(d), declared_target('xcore1002'))
        self.assertIn('MACA_LOOP_UNROLL_UNSUPPORTED', [f.code for f in findings])

    def test_generated_group_names_cannot_shadow_an_authored_buffer(self):
        d = reduction_document()
        # Rename only a scratch value. The emitted-source namespace check must
        # own the collision, rather than accidentally changing workload semantics.
        for b in d['buffers']:
            if b['name'] == 'value': b['name'] = '_maca_unroll_k_base'
        for op in d['operations']:
            for key in ('reads','writes'):
                op[key] = ['_maca_unroll_k_base' if v == 'value' else v for v in op[key]]
        with self.assertRaisesRegex(EmitError, 'compiler-owned binding'):
            self.compiler.lower(self.compiler.assess(d))
