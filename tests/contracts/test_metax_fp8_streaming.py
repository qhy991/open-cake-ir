"""Explicit K-streaming keeps compensation state at the owning loop boundary."""
import ast
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.tasks.workloads import load_workload
from open_cake_ir.tasks.metax_fp8_gemm import streaming_source

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'examples/python/xcore1002_fp8_streaming.py'


class MetaxStreamingFP8(unittest.TestCase):
    def test_task_projection_binds_the_same_nt_workload(self):
        workload = load_workload(ROOT / 'contracts/workloads/metax-fp8-e4m3-gemm-fp32-xcore1002-m64-n64-k64-v2.json')
        document = frontend.parse(streaming_source(workload)).document
        self.assertEqual(document['metadata']['workload_contract_sha256'], workload.canonical_sha256)
        globals_ = [(buffer['name'], tuple(buffer['shape']), buffer['dtype'], buffer['mode'])
                    for buffer in document['buffers'] if buffer['space'] == 'global']
        self.assertEqual(globals_, [(arg.name, arg.shape, arg.dtype, arg.mode) for arg in workload.tensor_abi('primary')])

    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')
        cls.source = SOURCE.read_text()

    def test_state_initialized_before_loop_and_finalized_after_it(self):
        assessment = self.compiler.assess(frontend.parse(self.source).document)
        self.assertTrue(assessment.accepted, assessment.findings)
        self.assertTrue(assessment.lowering_eligible, assessment.findings)
        lowering = self.compiler.lower(assessment)
        kernel = next(node for node in ast.parse(lowering.source).body
                      if isinstance(node, ast.FunctionDef) and node.name.endswith('_kernel'))
        loop = next(node for node in kernel.body if isinstance(node, ast.For))
        init = [node for node in kernel.body if isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in {'_maca_fp8_stream_total', '_maca_fp8_stream_correction'}]
        final = next(node for node in kernel.body if isinstance(node, ast.Assign)
                     and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'result')
        self.assertEqual(len(init), 2)
        self.assertTrue(all(node.lineno < loop.lineno for node in init))
        self.assertGreater(final.lineno, loop.end_lineno)
        self.assertIn('_maca_fp8_stream_correction = _maca_fp8_stream_correction + _maca_fp8_stream_error', lowering.source)
        self.assertIn('tl.trans(right.to(tl.float32))', lowering.source)
        self.assertNotIn('tl.dot(', lowering.source)
        self.assertNotIn('tl.float64', lowering.source)

    def test_legal_counterexamples_are_refused_by_streaming_scope(self):
        cases = {
            'live_result_consumer': self.source.replace(
                "        lm.store(out[row,:], result, id='store_out')",
                "            observed = lm.add(result, 1.0, id='observe')\n"
                "        lm.store(out[row,:], result, id='store_out')"),
            'K2': self.source.replace('dimension=1, tile=1', 'dimension=1, tile=2')
                              .replace('tile_shape=(2,64,1)', 'tile_shape=(2,64,2)'),
            'eight_groups': self.source.replace('[0,1,2,3]', '[0,1,2,3,4,5,6,7]'),
            'full_unroll': self.source.replace('tile=1, num_stages=1',
                                               'tile=1, num_stages=1, loop_unroll_factor=64'),
        }
        for name, source in cases.items():
            with self.subTest(name=name):
                assessment = self.compiler.assess(frontend.parse(source).document)
                self.assertTrue(assessment.accepted, assessment.findings)
                self.assertFalse(assessment.lowering_eligible)
                blockers = [f.code for f in assessment.findings if f.blocks_lowering]
                self.assertIn('MACA_FP8_COMPENSATED_STREAM_UNQUALIFIED', blockers)


if __name__ == '__main__':
    unittest.main()
