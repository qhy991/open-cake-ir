"""C550 exact-layout admission and CPU semantics; device qualification is separate."""
from copy import deepcopy
import math
import json
from pathlib import Path
import unittest
from open_cake_ir.compiler import Compiler
from tests.contracts.test_fp32_contraction_rewrite import document
from tests.contracts.test_epilogue_fusion import execute
from tools.prepare_c550_fp32_contraction import candidates, PARAMETERS

ROOT = Path(__file__).resolve().parents[2]


class MetaXFP32Contraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)

    def apply(self, source, **kwargs):
        return self.compiler.specialize_fp32_contraction(source,
            **(PARAMETERS | dict(schedule_id='metax_ieee', entry_point='metax_ieee') | kwargs))

    def test_prepared_programs_keep_original_contracts_and_ieee_route(self):
        rows = list(candidates(self.compiler))
        self.assertEqual(len(rows), 4)
        for name, workload, original, program in rows:
            self.assertEqual(workload['validation']['atol'], 2e-5)
            self.assertEqual(workload['validation']['rtol'], 2e-5)
            self.assertEqual(len(workload['cases']), 5)
            for stage in program.stages:
                schedule = json.loads(stage.schedule_bytes)
                emitted = self.compiler.lower(self.compiler.assess(schedule)).source
                if name.startswith('mma'):
                    self.assertIn('input_precision="ieee"', emitted)
                    self.assertFalse(any(op['kind'] == 'transpose' for op in schedule['operations']))
                    self.assertNotIn('input_precision="tf32"', emitted)

    def test_layout_gate_refuses_legal_kn_before_unqualified_transpose(self):
        original = document()
        original['target'] = 'xcore1002'
        self.assertTrue(self.compiler.assess(original).lowering_eligible)
        self.assertEqual(self.apply(original).reason, 'operand_layout')
        original['target'] = 'gfx938'
        self.assertTrue(self.apply(original).applied)

    def test_tail_semantics_keep_residual_and_silu(self):
        # Odd K/N are a CPU semantic check; the current task factory does not
        # admit them as C550 registered task Workloads.
        original = document(rows=5, depth=19, columns=7, transposed=True, silu=True, residual=True)
        original['target'] = 'xcore1002'
        saved = deepcopy(original)
        result = self.apply(original)
        self.assertTrue(result.applied, (result.reason,result.message))
        self.assertEqual(original,saved)
        a=[((i*3+k)%11-5)/8 for i in range(5) for k in range(19)]
        b=[((k+2*j)%7-3)/4 for j in range(7) for k in range(19)]
        bias=[(j-3)/8 for j in range(7)]
        residual=[(i%7-3)/4 for i in range(35)]
        actual,trace=execute(result.schedule,dict(a=a,b=b,bias=bias,residual=residual))
        self.assertEqual(set(trace.stores.values()),{1})
        self.assertEqual(len(trace.stores),35)
        for i in range(5):
            for j in range(7):
                value=math.fsum(a[i*19+k]*b[j*19+k] for k in range(19))+bias[j]+residual[i*7+j]
                self.assertAlmostEqual(actual['out'][i*7+j],value/(1+math.exp(-value)),places=6)

    def test_metax_keeps_existing_execution_storage_and_width_guards(self):
        original=document(transposed=True)
        original['target']='xcore1002'
        self.assertTrue(self.apply(original).applied)
        self.assertEqual(self.apply(original,num_warps=16).reason,'result_refused')
        result=self.apply(original,num_warps=16)
        self.assertIn('MACA_WARP_COUNT_UNQUALIFIED',result.message)
        original['residency']={'ctas_per_multiprocessor':1}
        self.assertTrue(self.compiler.assess(original).lowering_eligible)
        self.assertEqual(self.apply(original).reason,'execution_commitments')

if __name__=='__main__':
    unittest.main()
