"""Broader explicit candidate construction; no target qualification is fabricated."""
import math
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, frontend
from tools.benchmarks.c550.width_fixtures import extension_cases, ieee_output_loop_cases, nested_output_loop_cases
from tools.qualify_c550_widths import CASES, width_source, WidthCandidateRefused
from tests.contracts.test_epilogue_fusion import execute

ROOT = Path(__file__).resolve().parents[2]


class C550WidthExtension(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)
        cls.cases = extension_cases()

    def test_41_successor_candidates_leave_sealed_first_block_out(self):
        original_names = {task+'-w'+str(w) for task, _, _, widths in CASES for w in widths}
        names = set()
        refused = 0
        for row in self.cases:
            original = frontend.parse(row['source']).document
            for width in row['widths']:
                with self.subTest(case=row['name'], width=width):
                    if width == 16:
                        with self.assertRaises(WidthCandidateRefused) as caught:
                            width_source(self.compiler, row['source'], width)
                        self.assertEqual(caught.exception.codes, ('MACA_WARP_COUNT_UNQUALIFIED',))
                        refused += 1
                        continue
                    source = width_source(self.compiler, row['source'], width)
                    document = frontend.parse(source).document
                    self.assertTrue(self.compiler.assess(document).lowering_eligible)
                    self.assertEqual(document['roles'][0]['execution_groups'], list(range(width)))
                    for key in original.keys() - {'roles'}:
                        self.assertEqual(original[key], document[key], key)
                    names.add(row['name']+'-w'+str(width))
        self.assertEqual(len(names), 31)
        self.assertEqual(refused, 10)
        self.assertFalse(names & original_names)

    def test_mma_variants_compute_original_nt_geometry_and_store_each_output_once(self):
        for row in self.cases:
            if not row['name'].startswith('ieee_mma_'):
                continue
            with self.subTest(case=row['name']):
                document = frontend.parse(row['source']).document
                left = [((i + k) % 7 - 3)/8 for i in range(24) for k in range(64)]
                right = [((j + 2*k) % 9 - 4)/8 for j in range(32) for k in range(64)]
                bias = [(j % 3 - 1)/4 for j in range(32)]
                output, trace = execute(document, dict(input=left, weight_nt=right, bias=bias))
                expected = [math.fsum(left[i*64+k]*right[j*64+k] for k in range(64))+bias[j]
                            for i in range(24) for j in range(32)]
                self.assertEqual(output['output'], expected)
                self.assertEqual(len(trace.stores), 24*32)
                self.assertEqual(set(trace.stores.values()), {1})

    def test_pointwise_loop_keeps_existing_silu_computation(self):
        row = next(r for r in self.cases if r['name']=='silu_fixed_loop')
        values = [((i % 23)-11)/8 for i in range(4*128)]
        output, trace = execute(frontend.parse(row['source']).document, dict(x=values))
        for observed, x in zip(output['out'], values):
            self.assertAlmostEqual(observed, x/(1+math.exp(-x)), places=7)
        self.assertEqual(len(trace.stores), 4*128)
        self.assertEqual(set(trace.stores.values()), {1})

    def test_rounded_mma_carry_covers_full_k_and_masks_row_tail(self):
        row = next(r for r in self.cases if r['name']=='rounded_mma_k_loop')
        left = [(i % 3)/16 for i in range(5) for k in range(2048)]
        right = [(j % 5)/8 for j in range(128) for k in range(2048)]
        observed, trace = execute(frontend.parse(row['source']).document, dict(a=left, b=right))
        expected = [2048*((i % 3)/16)*((j % 5)/8) for i in range(5) for j in range(128)]
        self.assertEqual(observed['out'], expected)
        self.assertEqual(len(trace.stores), 5*128)
        self.assertEqual(set(trace.stores.values()), {1})

    def test_carried_reduction_uses_all_chunks(self):
        from unittest.mock import patch
        from tests.contracts.test_triton_loop_scopes import _TL, _Tile
        row = next(r for r in self.cases if r['name']=='rms_carried_loop')
        values = [(i % 13 - 6)/8 for i in range(4*256)]
        weights = [(j % 7)/8 for j in range(256)]
        with patch.object(_TL, 'rsqrt', staticmethod(lambda a:_Tile(a.shape,[1/math.sqrt(v) for v in a.values])),create=True):
            observed, trace = execute(frontend.parse(row['source']).document, dict(x=values, weight=weights))
        expected = []
        for i in range(4):
            inputs = values[i*256:(i+1)*256]
            inverse = 1/math.sqrt(math.fsum(v*v for v in inputs)/256+1e-5)
            expected += [v*inverse*w for v,w in zip(inputs, weights)]
        for actual, wanted in zip(observed['out'], expected):
            self.assertAlmostEqual(actual, wanted, places=7)
        self.assertEqual(len(trace.stores), 4*256)
        self.assertEqual(set(trace.stores.values()), {1})

    def test_ieee_output_loop_keeps_original_workload_and_full_contraction(self):
        row = ieee_output_loop_cases()[0]
        original = next(r for r in self.cases if r['name']=='rounded_mma_k_loop')
        self.assertEqual(row['workload'], original['workload'])
        document = frontend.parse(row['source']).document
        self.assertEqual(next(op for op in document['operations'] if op['kind']=='mma')['parameters']['tile_shape'], [16,16,2048])
        left = [(i % 3)/16 for i in range(5) for k in range(2048)]
        right = [(j % 5)/8 for j in range(128) for k in range(2048)]
        observed, trace = execute(document, dict(a=left, b=right))
        expected = [2048*((i % 3)/16)*((j % 5)/8) for i in range(5) for j in range(128)]
        self.assertEqual(observed['out'], expected)
        self.assertEqual(len(trace.stores), 5*128)
        self.assertEqual(set(trace.stores.values()), {1})

    def test_nested_output_loop_preserves_full_original_gemm(self):
        row = nested_output_loop_cases()[0]
        self.assertEqual(row['workload'], ieee_output_loop_cases()[0]['workload'])
        left = [(i % 3)/16 for i in range(5) for k in range(2048)]
        right = [(j % 5)/8 for j in range(128) for k in range(2048)]
        output, trace = execute(frontend.parse(row['source']).document, dict(a=left,b=right))
        self.assertEqual(output['out'], [2048*((i%3)/16)*((j%5)/8) for i in range(5) for j in range(128)])
        self.assertEqual(len(trace.stores),640)
        self.assertEqual(set(trace.stores.values()),{1})


if __name__ == '__main__':
    unittest.main()
