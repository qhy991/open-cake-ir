"""The B300 bin-pack Schedule has one typed atomic owner for both row stores."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Schedule, Target
from open_cake_ir.compiler.backends.native_cuda_expert_bin import preflight


ROOT = Path(__file__).resolve().parents[2]
DOCUMENT = ROOT / 'examples/schedules/native/weave-model-expert-bin-pack-b300.json'
ATOM = 'ptx.atom.relaxed.gpu.global.add.s32'


class NativeModelExpertBin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)
        cls.document = json.loads(DOCUMENT.read_text())

    def test_complete_dynamic_bf16_copy_lowers_with_runtime_domain_gate(self):
        assessment = self.compiler.assess(self.document)
        self.assertTrue(assessment.lowering_eligible,
                        [(f.code, f.path) for f in assessment.findings
                         if f.blocks_lowering])
        lowered = self.compiler.lower(assessment)
        req = lowered.toolchain_requirements
        self.assertEqual(req['target'], 'sm_103a')
        self.assertEqual(req['grid'], [2048, 8, 1])
        self.assertEqual(req['block'], [256, 1, 1])
        self.assertIs(req['input_domain_runtime_check'], True)
        self.assertEqual(req['state_reset'], 'zero_counts_before_launch')
        self.assertEqual(req['bin_capacity_rows_per_expert'], 2048)
        self.assertIn('atom.relaxed.gpu.global.add.s32', lowered.source)
        self.assertIn('if (seen & bit) return int(cudaErrorInvalidValue)',
                      lowered.source)
        for operation in self.document['operations']:
            self.assertIn(operation['id'], lowered.source_map)

    def test_second_store_cannot_escape_returned_old_row(self):
        changed = deepcopy(self.document)
        store = next(op for op in changed['operations']
                     if op['id'] == 'store_hidden')
        store['reads'][2] = 'key'
        assessment = self.compiler.assess(changed)
        self.assertFalse(assessment.lowering_eligible)
        self.assertIn('STORE_INDEX_BUFFER_READS',
                      [f.code for f in assessment.findings])
        self.assertIn('NATIVE_BIN_DATAFLOW', [f.code for f in preflight(
            Schedule.from_dict(changed),
            Target.load(ROOT / 'compiler/targets/sm_103a.json'))])

    def test_wrong_target_contract_and_weight_dtype_are_refused(self):
        schedule = Schedule.from_dict(self.document)
        target = Target.load(ROOT / 'compiler/targets/sm_103a.json')
        missing = replace(target, instruction_contracts=
                          target.instruction_contracts - {ATOM})
        self.assertIn('NATIVE_BIN_TARGET',
                      [f.code for f in preflight(schedule, missing)])
        changed = deepcopy(self.document)
        next(buffer for buffer in changed['buffers']
             if buffer['name'] == 'packed')['dtype'] = 'fp32'
        assessment = self.compiler.assess(changed)
        self.assertFalse(assessment.lowering_eligible)
        self.assertIn('STORE_DTYPE_UNSUPPORTED',
                      [f.code for f in assessment.findings])
        self.assertIn('NATIVE_BIN_GLOBALS', [f.code for f in preflight(
            Schedule.from_dict(changed), target)])


if __name__ == '__main__':
    unittest.main()
