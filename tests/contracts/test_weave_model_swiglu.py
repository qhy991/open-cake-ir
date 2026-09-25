"""Model-width SwiGLU keeps the BF16 handoff explicit in Cake and CUDA."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / 'examples/schedules/native/weave-model-swiglu-bf16-b300.json'


class WeaveModelSwiGLU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)

    def test_complete_model_activation_lowers_with_one_bf16_round(self):
        document = json.loads(DOCUMENT.read_text())
        assessment = self.compiler.assess(document)
        self.assertFalse(any(finding.blocks_lowering or finding.blocks_acceptance
                             for finding in assessment.findings))
        lowered = self.compiler.lower(assessment)
        requirements = lowered.toolchain_requirements
        self.assertEqual(requirements['target'], 'sm_103a')
        self.assertEqual(requirements['grid'], [128, 1, 1])
        self.assertEqual(requirements['block'], [32, 1, 1])
        self.assertEqual([(row['dtype'], row['shape'])
                          for row in requirements['arguments']],
                         [('fp32', [128, 1536]), ('bf16', [128, 768])])
        self.assertIn('for (int cake_feature=cake_lane; cake_feature<768;',
                      lowered.source)
        self.assertIn('expf(', lowered.source)
        self.assertEqual(lowered.source.count('__float2bfloat16_rn('), 1)
        self.assertTrue({'load_up', 'load_gate', 'round_bf16', 'store'}
                        <= set(lowered.source_map))

    def test_model_mapping_and_store_contracts_refuse_other_realizations(self):
        for name, change, expected in (
            ('two rows per CTA',
             lambda document: document['program_map']['axes'][0].update(tile=2),
             'NATIVE_MODEL_ACTIVATION_MAP'),
            ('noncoalesced store',
             lambda document: next(row for row in document['operations']
                                   if row['id'] == 'store')['parameters'].update(
                                       coalesced=False),
             'NATIVE_MODEL_ACTIVATION_STORE'),
        ):
            with self.subTest(realization=name):
                document = json.loads(DOCUMENT.read_text())
                change(document)
                findings = self.compiler.assess(document).findings
                self.assertTrue(any(finding.code == expected and finding.blocks_lowering
                                    for finding in findings))


if __name__ == '__main__':
    unittest.main()
