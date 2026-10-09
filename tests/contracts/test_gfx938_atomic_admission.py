"""Exact gfx938 atomic admission; unsafe edges remain refused by their owner."""
import copy
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.backends.triton import preflight
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import Target

ROOT = Path(__file__).resolve().parents[2]

class Gfx938AtomicAdmission(unittest.TestCase):
    def setUp(self):
        self.compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')
        self.document = json.loads((ROOT / 'corpus/schedules/gfx938-atomic-reservation-b8-smoke.json').read_text())

    def test_exact_relaxed_device_scope_fetch_add_lowers(self):
        assessment = self.compiler.assess(self.document)
        self.assertTrue(assessment.lowering_eligible)
        source = self.compiler.lower(assessment).source
        self.assertIn('tl.atomic_add(', source)
        self.assertIn('sem="relaxed"', source)
        self.assertIn('scope="gpu"', source)
        self.assertNotIn('maxnreg=', source)

    def test_missing_exact_contract_is_not_inherited(self):
        target = json.loads((ROOT / 'compiler/targets/gfx938.json').read_text())
        target['instruction_contracts'].remove('triton.atomic_add.i32.relaxed.gpu')
        findings = preflight(Schedule.from_dict(self.document), Target.from_dict(target))
        self.assertIn('TRITON_ATOMIC_CONTRACT_UNSUPPORTED', {f.code for f in findings})

    def test_retained_histogram_edge_mistakes_still_refuse(self):
        invalid = copy.deepcopy(self.document)
        atomic = next(o for o in invalid['operations'] if o['kind'] == 'atomic_rmw')
        atomic['writes'] = ['counts']
        invalid['outputs'] = ['counts']
        assessment = self.compiler.assess(invalid)
        codes = {f.code for f in assessment.findings}
        self.assertFalse(assessment.lowering_eligible)
        self.assertIn('ATOMIC_EDGE_COUNT', codes)
        self.assertIn('OUTPUT_MODE', codes)
        self.assertNotIn('TARGET_OPERATION_UNSUPPORTED', codes)
