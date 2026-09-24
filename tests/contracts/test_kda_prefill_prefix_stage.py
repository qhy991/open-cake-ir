"""Exact-B300 KDA decay-prefix component; no complete Workload claim."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Schedule, Target
from open_cake_ir.compiler.performance.work import work_bound
from open_cake_ir.compiler.verifier import verify

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'tests/fixtures/kda-decay-prefix32-sm103a.json'


class KdaDecayPrefixStage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')

    def test_mathematics_and_chunk_reset_are_visible_in_lowering(self):
        value = json.loads(FIXTURE.read_text())
        assessment = self.compiler.assess(value)
        self.assertTrue(assessment.lowering_eligible,
                        [(finding.code, finding.path) for finding in assessment.findings
                         if finding.blocks_lowering])
        lowering = self.compiler.lower(assessment)
        source = lowering.source
        self.assertEqual(value['target'], 'sm_103a')
        self.assertEqual(value['program_map']['axes'][0]['tile'], 32)
        self.assertEqual(value['outputs'], ['prefix_out'])
        self.assertIn('tl.cumprod(decay.to(tl.float32), axis=0, reverse=False)', source)
        self.assertIn('_cake_kda_b300_decay_prefix32_kernel[(256, 64, 1)]', source)
        self.assertLess(source.index('# CAKE_OP:rate_exp'), source.index('# CAKE_OP:scale_gate'))
        self.assertLess(source.index('# CAKE_OP:exp_decay'), source.index('# CAKE_OP:prefix_mul'))
        self.assertLess(source.index('# CAKE_OP:prefix_mul'), source.index('# CAKE_OP:store_prefix'))
        self.assertIn('prefix_mul', work_bound(Schedule.from_dict(value)).uncounted_arithmetic)

    def test_scan_typing_still_rejects_wrong_intermediate_precision(self):
        value = json.loads(FIXTURE.read_text())
        next(buffer for buffer in value['buffers']
             if buffer['name'] == 'prefix_tile')['dtype'] = 'bf16'
        findings = verify(Schedule.from_dict(value),
                          Target.load(ROOT / 'compiler/targets/sm_103a.json'))
        self.assertIn('SCAN_DTYPE_MISMATCH', {finding.code for finding in findings})


if __name__ == '__main__':
    unittest.main()
