"""FlashInfer GEMM integration: exact axes, FP16 ABI, independent oracle and lowering."""
from copy import deepcopy
import math
import unittest
from pathlib import Path

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.frontend import parse
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.solx_fib import gemm
from open_cake_ir.tasks.tiles.workload import _round
from open_cake_ir.tasks.workloads import create_task, reference_outputs, load_workload
from open_cake_ir.tasks.efficiency import task_work

ROOT = Path(__file__).resolve().parents[2]


class FlashInferGemmTests(unittest.TestCase):
    def test_all_eight_exact_tasks_have_fp16_abi_and_lower_on_b300(self):
        compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')
        self.assertEqual(len(gemm.TASKS), 8)
        for task, spec in gemm.SPECS.items():
            doc, source = create_task(task, backend='triton-b300', rows=min(spec['batches']),
                                      columns=spec['N'], depth=spec['K'])
            workload = WorkloadContract(doc)
            self.assertEqual(workload.target, 'sm_103a')
            self.assertEqual([a.dtype for a in workload.tensor_abi('primary')], ['fp16']*3)
            self.assertEqual(workload.tensor_abi('primary')[1].shape, (spec['N'], spec['K']))
            assessment = compiler.assess(parse(source).document)
            self.assertEqual(assessment.findings, (), task)
            self.assertEqual(compiler.lower(assessment).target, 'sm_103a')
            committed = load_workload(ROOT/'contracts/workloads'/f"{doc['workload_id']}.json")
            self.assertEqual(committed.document, doc)
            work = task_work(workload, 'primary')
            m = min(spec['batches'])
            self.assertEqual(work['logical_io_bytes'], 2*(m*spec['K'] + spec['N']*spec['K'] + m*spec['N']))

    def test_oracle_observes_b_transpose_and_final_half_rounding(self):
        task = 'fib_gemm_n128_k2048'
        w = WorkloadContract(gemm.workload_document(task, rows=1))
        k, n = 2048, 128
        a = [0.0]*k
        a[0], a[-1] = 0.5, -1.0
        b = [0.0]*(n*k)
        for j in range(n):
            b[j*k], b[j*k+k-1] = _round(j/128, 'fp16'), 0.25
        actual = reference_outputs(w, 'primary', {'a': a, 'b': b})['out']
        self.assertEqual(actual, [_round(0.5*j/128-0.25, 'fp16') for j in range(n)])

    def test_original_N_and_K_cannot_be_silently_shrunk(self):
        for kwargs in ({'columns': 16}, {'depth': 16}, {'rows': 0}):
            with self.assertRaises(ValueError):
                gemm.workload_document('fib_gemm_n128_k2048', **kwargs)

    def test_contract_rejects_wrong_layout_dtype_or_weak_comparison(self):
        document = gemm.workload_document('fib_gemm_n128_k2048')
        for field,value in [('dtype','bf16'),('shape',['K','N'])]:
            bad = deepcopy(document); bad['tensors']['b'][field] = value
            with self.assertRaises(ValueError): gemm.validate_contract(bad)
        bad = deepcopy(document);bad['validation']['comparison'] = 'matched_ratio'
        with self.assertRaises(ValueError):gemm.validate_contract(bad)

    def test_materialized_fp16_inputs_are_bounded_and_representable(self):
        w = WorkloadContract(gemm.workload_document('fib_gemm_n128_k2048'))
        values = gemm.materialize_case(w, 'mixed_magnitude')
        self.assertEqual(set(values), {'a','b'})
        for vector in values.values():
            self.assertTrue(all(math.isfinite(v) and abs(v) <= 2 and _round(v,'fp16') == v
                                for v in vector))

    def test_half_rounding_is_not_fp32_rounding(self):
        self.assertEqual(_round(1 + 2**-11, 'fp16'), 1.0)
        self.assertEqual(_round(1 + 3*2**-11, 'fp16'), 1 + 2**-9)
        self.assertNotEqual(_round(1.0001, 'fp16'), _round(1.0001, 'fp32'))
