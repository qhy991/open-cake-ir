"""Logical K contribution contracts and CPU/source boundaries, without GPU claims."""
from __future__ import annotations

import ast
import copy
import json
import re
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.backends import cutedsl, metal, native_cuda, triton
from open_cake_ir.compiler.backends.common import EmitError
from open_cake_ir.compiler.frontend import parse
from open_cake_ir.compiler.ir import DType, MmaParameters, OperationKind, Schedule, ScheduleParseError
from open_cake_ir.compiler.performance.work import work_bound
from open_cake_ir.compiler.schema import schedule_schema
from open_cake_ir.compiler.target import Target

ROOT = Path(__file__).resolve().parents[2]


def document(backend='native'):
    return json.loads((ROOT / f'examples/schedules/{backend}/kmeans-partials.json').read_text())


def mma(doc, index=0):
    return [op for op in doc['operations'] if op['kind'] == 'mma'][index]


class KRangeContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')

    def lower(self, doc):
        assessment = self.compiler.assess(doc)
        self.assertTrue(assessment.lowering_eligible, assessment.findings)
        return self.compiler.lower(assessment)

    def test_shared_typed_owner_normalizes_without_reordering(self):
        p = MmaParameters(DType.FP32, None, (16, 16, 128), ((0, 16), (16, 32), (64, 96)))
        self.assertEqual(p.k_ranges, ((0, 32), (64, 96)))
        self.assertEqual(p.contribution_ranges, p.k_ranges)
        self.assertEqual(p.selected_k, 64)
        full = replace(p, k_ranges=((0, 32), (32, 128)))
        self.assertIsNone(full.k_ranges)
        self.assertEqual(full.contribution_ranges, ((0, 128),))
        self.assertEqual(full.selected_k, 128)

    def test_public_parser_refuses_invalid_ranges_at_owning_operation(self):
        invalid = [None, [], {}, '0:16', [0, 16], [[0]], [[0, 16, 32]],
                   [[False, 16]], [[0, True]], [[0.0, 16]], [[-1, 16]],
                   [[0, 0]], [[16, 0]], [[0, 65]], [[0, 32], [16, 48]],
                   [[32, 48], [0, 16]], [[16, 32], [16, 32]]]
        for value in invalid:
            with self.subTest(value=value):
                d = document(); mma(d)['parameters']['k_ranges'] = value
                with self.assertRaises(ScheduleParseError): Schedule.from_dict(d)
                a = self.compiler.assess(d)
                self.assertFalse(a.accepted)
                self.assertEqual(a.findings[0].code, 'SCHEDULE_STRUCTURE')
                self.assertTrue(a.findings[0].path.startswith('schedule.operations[2].parameters.k_ranges'), a.findings)

    def test_ranges_require_explicit_input_tile(self):
        d = document(); del mma(d)['parameters']['tile_shape']
        with self.assertRaisesRegex(ScheduleParseError, 'k_ranges requires tile_shape'):
            Schedule.from_dict(d)

    def test_full_ranges_preserve_original_assessment_and_source(self):
        for path in ['examples/schedules/native/gemm-bias.json', 'examples/schedules/native/two-mma.json',
                     'examples/schedules/native/kmeans.json', 'corpus/schedules/flash-kmeans-b32-smoke-v2.json']:
            with self.subTest(path=path):
                original = json.loads((ROOT / path).read_text()); explicit = copy.deepcopy(original)
                for op in explicit['operations']:
                    if op['kind'] == 'mma':
                        end = op['parameters']['tile_shape'][2]
                        op['parameters']['k_ranges'] = [[0, end // 2], [end // 2, end]]
                before = copy.deepcopy(explicit)
                self.assertEqual(self.compiler.assess(original), self.compiler.assess(explicit))
                self.assertEqual(self.lower(original).source, self.lower(explicit).source)
                self.assertEqual(explicit, before)

    def test_adjacent_ranges_have_one_public_document(self):
        canonical = document(); alternate = copy.deepcopy(canonical)
        mma(alternate)['parameters']['k_ranges'] = [[0, 8], [8, 16], [32, 40], [40, 48]]
        a = self.compiler.assess(canonical); b = self.compiler.assess(alternate)
        self.assertEqual(a, b)
        self.assertEqual(self.lower(canonical).source, self.lower(alternate).source)

    def test_python_authoring_uses_the_same_canonical_projection(self):
        source = (ROOT / 'examples/python/kmeans_pipeline.py').read_text()
        original = parse(source).document
        full = parse(source.replace('tile_shape=(128, 256, 64)',
                                    'tile_shape=(128, 256, 64), k_ranges=[[0, 32], [32, 64]]')).document
        self.assertEqual(original, full)
        partial = parse(source.replace('tile_shape=(128, 256, 64)',
                                       'tile_shape=(128, 256, 64), k_ranges=[[0, 8], [8, 16], [32, 48]]')).document
        self.assertEqual(mma(partial)['parameters']['k_ranges'], [[0, 16], [32, 48]])

    def test_authoring_schema_requires_tile_and_interval_shape(self):
        # Schema structure is inspected because jsonschema is optional in the CPU environment.
        variants = schedule_schema()['properties']['operations']['items']['allOf']
        entry = next(row for row in variants if row['if']['properties']['kind'].get('const') == 'mma')
        parameters = entry['then']['properties']['parameters']
        self.assertEqual(parameters['dependentRequired'], {'k_ranges': ['tile_shape']})
        intervals = parameters['properties']['k_ranges']
        self.assertEqual(intervals['minItems'], 1)
        self.assertEqual(intervals['items']['minItems'], 2)
        self.assertEqual(intervals['items']['maxItems'], 2)

    def test_full_input_domain_cannot_be_replaced_by_selected_extent(self):
        for backend in ('native', 'triton'):
            d = document(backend); mma(d)['parameters']['tile_shape'][2] *= 2
            a = self.compiler.assess(d)
            self.assertFalse(a.accepted)
            self.assertIn('MMA_INPUT_TILE_DOMAIN', [f.code for f in a.findings])
            module = native_cuda if backend == 'native' else triton
            with self.assertRaises(EmitError):
                module.emit(Schedule.from_dict(d), Target.load(ROOT / f"compiler/targets/{d['target']}.json"))

    def test_full_alias_cannot_erase_a_mismatched_input_domain(self):
        for backend in ('native', 'triton'):
            for spelling in ('omitted', 'full', 'adjacent_full'):
                with self.subTest(backend=backend, spelling=spelling):
                    d = document(backend); p = mma(d)['parameters']; p['tile_shape'][2] //= 2
                    k = p['tile_shape'][2]
                    if spelling == 'omitted': del p['k_ranges']
                    else: p['k_ranges'] = [[0, k]] if spelling == 'full' else [[0, k//2], [k//2, k]]
                    s = Schedule.from_dict(d)
                    self.assertIsNone(next(op for op in s.operations if op.kind is OperationKind.MMA).parameters.k_ranges)
                    assessment = self.compiler.assess(d)
                    self.assertFalse(assessment.accepted)
                    self.assertIn('MMA_INPUT_TILE_DOMAIN', [f.code for f in assessment.findings])
                    module = native_cuda if backend == 'native' else triton
                    with self.assertRaises(EmitError):
                        module.emit(s, Target.load(ROOT / f"compiler/targets/{d['target']}.json"))

    def test_direct_emission_cannot_write_beyond_the_declared_result(self):
        for backend in ('native', 'triton'):
            d = document(backend); result = next(b for b in d['buffers'] if b['name'] == mma(d)['writes'][0])
            result['shape'][1] //= 2
            s = Schedule.from_dict(d); target = Target.load(ROOT / f"compiler/targets/{d['target']}.json")
            module = native_cuda if backend == 'native' else triton
            prefix = 'NATIVE' if backend == 'native' else 'TRITON'
            self.assertIn(prefix + '_MMA_TILE_DOMAIN', [f.code for f in module.preflight(s, target)])
            with self.assertRaises(EmitError): module.emit(s, target)

    def test_alignment_is_checked_per_endpoint_not_total_size(self):
        d = document(); mma(d)['parameters']['k_ranges'] = [[1, 17], [33, 49]]
        a = self.compiler.assess(d)
        self.assertFalse(a.accepted)
        self.assertEqual([f.code for f in a.findings], ['MMA_K_RANGES_INSTRUCTION_MISMATCH'])
        with self.assertRaises(EmitError):
            native_cuda.emit(Schedule.from_dict(d), Target.load(ROOT / 'compiler/targets/sm_103a.json'))

    def test_native_first_selected_atom_initializes_each_output_tile(self):
        source = self.lower(document()).source
        for op_id, expected in [('mma', [0, 2]), ('mma2', [1, 3])]:
            body = source.split(f'// CAKE_OP: {op_id}\n', 1)[1].split('// CAKE_OP:', 1)[0]
            loops = re.findall(r'for \(int atom=(\d+); atom<(\d+); \+\+atom\)', body)
            atoms = [a for start, end in loops for a in range(int(start), int(end))]
            self.assertEqual(atoms, expected)
            first = int(re.search(r'\|\| atom != (\d+)\);', body).group(1))
            # The emitted predicate is evaluated for two K stages and two output tiles.
            for output_tile in range(2):
                use_d = [iteration != 0 or atom != first for iteration in range(2) for atom in atoms]
                self.assertEqual(use_d, [False, True, True, True])
        self.assertLess(source.index('// CAKE_OP: mma2'), source.index('cake_commit(free0+stage)'))
        self.assertIn('cake_commit(bar1);', source)
        self.assertIn('cake_commit(bar2);', source)
        self.assertIn('cake_wait(bar1, 0);', source)
        self.assertIn('cake_wait(bar2, 0);', source)
        self.assertIn('cake_inval(bar1);', source)
        self.assertIn('cake_inval(bar2);', source)
        self.assertIn('tcgen05.dealloc.cta_group::1.sync.aligned.b32', source)

    def test_native_independent_completions_and_tmem_regions_remain_required(self):
        d = document(); mma(d, 1)['signals'] = mma(d)['signals']
        a = self.compiler.assess(d)
        self.assertFalse(a.lowering_eligible)
        d = document(); next(b for b in d['buffers'] if b['name'] == 'acc2')['byte_offset'] = 0
        self.assertFalse(self.compiler.assess(d).lowering_eligible)

    def test_a_selector_never_creates_multiple_results(self):
        for backend in ('native', 'triton'):
            d = document(backend)
            extra = copy.deepcopy(next(b for b in d['buffers'] if b['name'] == mma(d)['writes'][0]))
            extra['name'] = 'unrequested_second_result'; d['buffers'].append(extra)
            mma(d)['writes'].append(extra['name'])
            a = self.compiler.assess(d)
            self.assertFalse(a.accepted)
            self.assertIn('MMA_RESULT_COUNT', [f.code for f in a.findings])

    def test_complementary_work_counts_full_products_plus_explicit_add(self):
        for backend, original_path in [('native', 'examples/schedules/native/kmeans.json'),
                                       ('triton', 'corpus/schedules/flash-kmeans-b32-smoke-v2.json')]:
            partial = Schedule.from_dict(document(backend)); bound = work_bound(partial)
            full = work_bound(Schedule.from_dict(json.loads((ROOT / original_path).read_text())))
            self.assertEqual(bound.mma_flops, full.mma_flops)
            repetitions = {r.operation: r.whole_grid for r in bound.operation_repetitions}
            combine = next(op for op in partial.operations if op.kind is OperationKind.ELEMENTWISE
                           and len(op.reads) == 2 and all(partial.buffer(n).shape == partial.buffer(op.writes[0]).shape for n in op.reads))
            self.assertEqual(bound.flops - full.flops,
                             partial.buffer(combine.writes[0]).elements * repetitions[combine.op_id])
            self.assertEqual(bound.scheduled_read_bytes_upper_bound, full.scheduled_read_bytes_upper_bound)
            d = document(backend); mma(d)['parameters']['k_ranges'] = [[0, 16]]
            smaller = work_bound(Schedule.from_dict(d))
            p = mma(d)['parameters']; old_k = 32 if backend == 'native' else 64
            self.assertEqual(bound.mma_flops - smaller.mma_flops,
                             2 * p['tile_shape'][0] * p['tile_shape'][1] * (old_k - 16) * repetitions[mma(d)['id']])

    def test_triton_refuses_unproven_size_dtype_target_and_carry(self):
        for change in ('size', 'dtype', 'target', 'carry'):
            with self.subTest(change=change):
                d = document('triton')
                if change == 'size': mma(d)['parameters']['k_ranges'] = [[0, 32]]
                elif change == 'dtype':
                    mma(d)['parameters']['instruction']['contract'] = 'triton.dot.fp32_ieee'
                    for b in d['buffers']:
                        if b['dtype'] == 'bf16': b['dtype'] = 'fp32'
                elif change == 'target': d['target'] = 'apple_gpu_family8'
                else:
                    loop = d['tile_loops'][0]; loop.update(buffer='tokens', dimension=2, tile=128)
                    for access in d['access_maps']:
                        if access['operation'] in ('load_tokens', 'load_centroids'):
                            if access['operation'] == 'load_centroids':
                                access['indices'][1] = {'source': 'dimension', 'dimension': 1, 'extent': 64}
                            access['indices'][-1] = {'source': 'loop_tile', 'name': loop['iterator']}
                s = Schedule.from_dict(d); target = Target.load(ROOT / f"compiler/targets/{d['target']}.json")
                findings = triton.preflight(s, target)
                self.assertIn('TRITON_MMA_K_RANGES_UNSUPPORTED', [f.code for f in findings])
                with self.assertRaises(EmitError): triton.emit(s, target)

    def test_cute_and_metal_directly_refuse_the_refinement(self):
        d = json.loads((ROOT / 'corpus/schedules/cute-k-ranges-unsupported.json').read_text())
        for module, route, code in [(cutedsl, 'cutlass_cute_dsl', 'CUTE_MMA_K_RANGES_UNSUPPORTED'),
                                    (metal, 'metal', 'METAL_MMA_K_RANGES_UNSUPPORTED')]:
            d['lowering']['backend'] = route
            if route == 'metal': d['target'] = 'apple_gpu_family8'
            s = Schedule.from_dict(d); target = Target.load(ROOT / f"compiler/targets/{d['target']}.json")
            self.assertIn(code, [f.code for f in module.preflight(s, target)])
            with self.assertRaises(EmitError): module.emit(s, target)
            self.assertIn(code, [f.code for f in self.compiler.assess(d).findings])

    def test_emitted_triton_selects_the_same_logical_coordinates_from_both_operands(self):
        class LogicalTL:
            float32 = np.float32
            arange = staticmethod(np.arange)
            where = staticmethod(np.where)
            broadcast_to = staticmethod(np.broadcast_to)
            trans = staticmethod(np.transpose)
            gather = staticmethod(lambda a, indices, axis: np.take_along_axis(a, indices, axis))
            dot = staticmethod(lambda a, b, **kw: a @ b)
            inline_asm_elementwise = staticmethod(lambda *a, args, **kw: args[0])
        d = document('triton')
        # Nonuniform intervals exercise logical selection, without native atom assumptions.
        mma(d)['parameters']['k_ranges'] = [[0, 15], [32, 81]]
        mma(d, 1)['parameters']['k_ranges'] = [[15, 32], [81, 128]]
        source = self.lower(d).source; tree = ast.parse(source)
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                       and isinstance(node.targets[0], ast.Name)
                       and node.targets[0].id in {'cross_even', 'cross_odd', 'cross'}]
        rng = np.random.default_rng(42)
        a = rng.integers(-2, 3, size=(256, 128)).astype(np.float32)
        b = rng.integers(-2, 3, size=(64, 128)).astype(np.float32)
        env = {'tl': LogicalTL, 'token_tile': a, 'centroid_tile': b}
        for node in assignments:
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<emitted logical MMA>', 'exec'), env)
        even = [*range(15), *range(32, 81)]
        odd = [*range(15, 32), *range(81, 128)]
        np.testing.assert_array_equal(env['cross_even'], a[:, even] @ b[:, even].T)
        np.testing.assert_array_equal(env['cross_odd'], a[:, odd] @ b[:, odd].T)
        np.testing.assert_array_equal(env['cross'], a @ b.T)
        # This CPU execution verifies selection only. Actual compiler IR and GPU
        # rounding/materialization qualification are retained outside the checkout.
        self.assertEqual(source.count('constraints="=f,f"'), 2)


if __name__ == '__main__': unittest.main()
