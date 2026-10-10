"""Broader explicit candidate construction; no target qualification is fabricated."""
import math
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, frontend
from tools.benchmarks.c550.width_fixtures import extension_cases
from tools.qualify_c550_widths import CASES, width_source
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
        for row in self.cases:
            original = frontend.parse(row['source']).document
            for width in row['widths']:
                with self.subTest(case=row['name'], width=width):
                    source = width_source(self.compiler, row['source'], width)
                    document = frontend.parse(source).document
                    self.assertTrue(self.compiler.assess(document).lowering_eligible)
                    self.assertEqual(document['roles'][0]['execution_groups'], list(range(width)))
                    for key in original.keys() - {'roles'}:
                        self.assertEqual(original[key], document[key], key)
                    names.add(row['name']+'-w'+str(width))
        self.assertEqual(len(names), 41)
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


if __name__ == '__main__':
    unittest.main()
