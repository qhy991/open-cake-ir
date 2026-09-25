"""A model-width expert projection tile reaches the exact B300 native route."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / 'examples/schedules/native/weave-model-upgate-tile-b300.json'


class WeaveModelUpgateTile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)

    def test_complete_b300_tensor_tile_lowers_with_visible_resources(self):
        document = json.loads(DOCUMENT.read_text())
        assessment = self.compiler.assess(document)
        self.assertFalse(any(finding.blocks_lowering or finding.blocks_acceptance
                             for finding in assessment.findings))
        lowered = self.compiler.lower(assessment)
        req = lowered.toolchain_requirements
        self.assertEqual(req['target'], 'sm_103a')
        self.assertEqual(req['grid'], [1, 24, 1])
        self.assertEqual(req['threads_per_cta'], 192)
        self.assertEqual(req['dynamic_shared_bytes'], 49200)
        self.assertEqual(req['argument_order'], ['a', 'b', 'c'])
        self.assertEqual([row['shape'] for row in req['arguments']],
                         [[128, 2048], [1536, 2048], [128, 1536]])
        self.assertIn('tcgen05.mma.cta_group::1', lowered.source)
        self.assertIn('cp.async.bulk.tensor', lowered.source)

    def test_wrong_weight_dtype_is_refused_by_the_load_contract(self):
        document = json.loads(DOCUMENT.read_text())
        next(row for row in document['buffers'] if row['name'] == 'b')['dtype'] = 'fp32'
        findings = self.compiler.assess(document).findings
        self.assertTrue(any(finding.code == 'LOAD_DTYPE_MISMATCH'
                            and finding.blocks_lowering for finding in findings))


if __name__ == '__main__':
    unittest.main()
