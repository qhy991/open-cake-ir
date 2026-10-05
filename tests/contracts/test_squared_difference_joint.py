"""Joint candidates: complete ABI, masked tails, reduction carry, owned refusals.

CPU execution below checks emitted control flow with exactly representable small
integers. It makes no claim about floating-point error or device performance.
"""
import copy
import itertools
from types import SimpleNamespace
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Program
from open_cake_ir.compiler.frontend import parse
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.reduction_tiling import specialize_squared_difference
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.contraction.authoring import starter_source
from open_cake_ir.tasks.contraction.workload import workload_document
from open_cake_ir.tasks.contraction.arithmetic import validate_candidate_arithmetic
from tests.contracts.test_triton_loop_scopes import _execute

ROOT = Path(__file__).resolve().parents[2]


class JointSquaredDifference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')

    def seed(self, rows=3, depth=16, columns=8):
        workload = WorkloadContract(workload_document('pairwise_sqdist', rows=rows,
            depth=depth, columns=columns, backend='triton-metax'))
        return parse(starter_source(workload)).document, workload

    def candidate(self, schedule, **changes):
        parameters = dict(output_tile=4, k_tile=4, loop_unroll_factor=2,
                          schedule_id='joint', entry_point='joint')
        return specialize_squared_difference(self.compiler, schedule, **(parameters | changes))

    def test_experiment_matrix_has_one_joint_construction_and_preserves_abi(self):
        for rows, columns in ((128, 32), (256, 64)):
            seed, workload = self.seed(rows=rows, depth=256, columns=columns)
            original = copy.deepcopy(seed)
            program = Program.from_schedule(seed)
            for output_tile, k_tile, factor in itertools.product((columns, 4), (32, 64), (1, 2)):
                with self.subTest(rows=rows, output_tile=output_tile, k_tile=k_tile, factor=factor):
                    result = self.compiler.rewrite_program(program, 'specialize_squared_difference',
                        dict(stage=program.stages[0].name, output_tile=output_tile,
                             k_tile=k_tile, loop_unroll_factor=factor, schedule_id='joint', entry_point='joint'))
                    self.assertTrue(result.applied, result.message)
                    for key in ('inputs', 'outputs', 'tensors', 'target'):
                        self.assertEqual(result.program.document[key], program.document[key])
                    self.assertEqual(result.program.stages[0].bindings, program.stages[0].bindings)
                    schedule = result.program.document['stages'][0]['schedule']
                    validate_candidate_arithmetic(schedule, workload)
                    self.assertEqual(schedule['tile_loops'][0]['range_options']['loop_unroll_factor'], factor)
                    self.compiler.lower_program(result.program)
            self.assertEqual(seed, original)

    def test_tail_masks_carry_reset_and_exactly_one_store_on_both_routes(self):
        # K=15 has four ceil trips at tile4; N=7 has a partial final output tile.
        for depth, columns in ((16, 8), (15, 7), (9, 5)):
            seed, _ = self.seed(depth=depth, columns=columns)
            for target, factor in itertools.product(('xcore1002', 'sm_100a'), (1, 2)):
                if ((depth + 3) // 4) % factor:
                    continue
                with self.subTest(depth=depth, columns=columns, target=target, factor=factor):
                    seed['target'] = target
                    # Source-wide arange refusals alone are repairable; the whole
                    # Program API must still reach masked joint candidates.
                    program = Program.from_schedule(seed)
                    result = self.compiler.rewrite_program(program, 'specialize_squared_difference',
                        dict(stage=program.stages[0].name, output_tile=4, k_tile=4,
                             loop_unroll_factor=factor, schedule_id='joint', entry_point='joint'))
                    self.assertTrue(result.applied, result.message)
                    schedule = result.program.document['stages'][0]['schedule']
                    assessment = self.compiler.assess(schedule)
                    emission = self.compiler.lower(assessment)
                    x, c, _, _, _, store = schedule['operations']
                    x_name, c_name, out_name = x['reads'][0], c['reads'][0], store['writes'][0]
                    xv = [(i * 3) % 11 - 5 for i in range(3 * depth)]
                    cv = [(i * 7) % 13 - 6 for i in range(columns * depth)]
                    memories = {x_name: xv, c_name: cv, out_name: [None] * (3 * columns)}
                    observed = _execute(SimpleNamespace(source=emission.source,
                        toolchain=emission.toolchain_requirements), memories)
                    expected = [sum((cv[n*depth+k] - xv[r*depth+k])**2 for k in range(depth))
                                for r in range(3) for n in range(columns)]
                    self.assertEqual(memories[out_name], expected)
                    self.assertEqual(len(observed.stores), 3 * columns)
                    self.assertEqual(set(observed.stores.values()), {1})

    def test_full_extents_keep_loop_free_seed_and_structural_feedback_is_not_a_score(self):
        seed, _ = self.seed(rows=128, depth=256, columns=32)
        result = self.candidate(seed, output_tile=32, k_tile=256, loop_unroll_factor=1)
        self.assertTrue(result.applied, result.message)
        self.assertEqual(result.schedule['operations'], seed['operations'])
        self.assertEqual(result.schedule['access_maps'], seed['access_maps'])
        self.assertEqual(result.schedule['tile_loops'], [])
        tiled = self.candidate(seed, output_tile=4, k_tile=64)
        self.assertTrue(tiled.applied, tiled.message)
        self.assertIn('Intermediate tile [32, 256] -> [4, 64]', tiled.message)
        self.assertIn('Static CTAs 128 -> 1024', tiled.message)
        self.assertIn('row-input load multiplicity 1 -> 8', tiled.message)
        self.assertIn('do not predict', tiled.message)
        self.assertEqual(self.candidate(seed, k_tile=256).reason, 'unroll_extent')

    def test_parameter_and_graph_counterexamples_are_refused_by_their_owners(self):
        seed, _ = self.seed()
        for changes, reason in (({'k_tile': True}, 'tile_extent'), ({'k_tile': 3}, 'tile_extent'),
            ({'k_tile': 32}, 'tile_extent'), ({'output_tile': 3}, 'output_extent'),
            ({'output_tile': 16}, 'output_extent'), ({'loop_unroll_factor': True}, 'unroll_extent'),
            ({'loop_unroll_factor': 0}, 'unroll_extent'), ({'loop_unroll_factor': 3}, 'unroll_extent')):
            with self.subTest(changes=changes):
                self.assertEqual(self.candidate(seed, **changes).reason, reason)
        for mutate, reason in ((lambda d: d['operations'][2]['parameters'].update(op='add'), 'arithmetic_domain'),
            (lambda d: d['buffers'][3].update(dtype='bf16'), 'storage_domain'),
            (lambda d: d['access_maps'][1]['indices'][1].update(offset=1), 'access_domain')):
            d = copy.deepcopy(seed); mutate(d)
            self.assertEqual(self.candidate(d).reason, reason)
        tiled = self.candidate(seed)
        refused = self.candidate(tiled.schedule, schedule_id='another')
        self.assertEqual(refused.reason, 'execution_commitments')
        self.assertIn('tile_loops:', refused.message)
        self.assertEqual(self.candidate(seed, schedule_id=seed['schedule_id']).reason, 'result_identity')

    def test_generated_names_avoid_existing_buffers_and_loop_names(self):
        seed, _ = self.seed()
        # Rename an intermediate into each generated namespace, preserving uses.
        for old, new in zip([b['name'] for b in seed['buffers'] if b['space'] == 'register'],
                            ('output_columns', 'contracted_tile', 'contracted_tile__loop')):
            for b in seed['buffers']:
                if b['name'] == old: b['name'] = new
            for op in seed['operations']:
                for field in ('reads', 'writes'):
                    op[field] = [new if name == old else name for name in op[field]]
        result = self.candidate(seed)
        self.assertTrue(result.applied, result.message)
        Schedule.from_dict(result.schedule)
        self.assertEqual(result.schedule['tile_loops'][0]['iterator'], 'contracted_tile_')
        self.assertNotEqual(result.schedule['tile_loops'][0]['name'], 'contracted_tile__loop')

    def test_backend_refusal_retains_category_path_and_explanation(self):
        seed, _ = self.seed()
        # The existing backend, not this pass, owns identifier safety.
        result = self.candidate(seed, entry_point='class')
        self.assertFalse(result.applied)
        self.assertEqual(result.reason, 'result_refused')
        self.assertIn('BACKEND_IDENTIFIER_UNSAFE', result.message)
        self.assertIn('lowering.entry_point', result.message)
        self.assertIn('hardware_conformance', result.message)


if __name__ == '__main__':
    unittest.main()
