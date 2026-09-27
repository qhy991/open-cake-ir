"""FlashInfer GEMM integration: exact axes, FP16 ABI, independent oracle and lowering."""
from copy import deepcopy
import json
import math
import unittest
from pathlib import Path

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.frontend import parse
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.solx_fib import gemm
from open_cake_ir.tasks.solx_fib.b300_gemv004 import column_tiled_source, scalar_warp_program
from open_cake_ir.tasks.solx_fib.b300_tensorcore004 import tensorcore_source
from open_cake_ir.tasks.tiles.workload import _round
from open_cake_ir.tasks.workloads import create_task, reference_outputs, load_workload
from open_cake_ir.tasks.efficiency import task_work

ROOT = Path(__file__).resolve().parents[2]


class FlashInferGemmTests(unittest.TestCase):
    def test_004_tensorcore_route_covers_each_official_m_at_least_16(self):
        compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')
        rows_values = [rows for rows in gemm.SPECS['fib_gemm_n128_k2048']['batches']
                       if rows >= 16]
        self.assertEqual(len(rows_values), 19)
        for rows in rows_values:
            with self.subTest(rows=rows):
                workload = WorkloadContract(gemm.workload_document(
                    'fib_gemm_n128_k2048', rows=rows))
                schedule = parse(tensorcore_source(workload)).document
                mma = [op for op in schedule['operations'] if op['kind'] == 'mma']
                self.assertEqual(len(mma), 1)
                self.assertEqual(mma[0]['parameters']['instruction']['contract'],
                                 'triton.dot.fp16_fp32')
                assessment = compiler.assess(schedule)
                self.assertFalse([f for f in assessment.findings if f.blocks_lowering],
                                 assessment.findings)
                lowered = compiler.lower(assessment)
                self.assertEqual(lowered.target, 'sm_103a')
                self.assertIn('tl.dot(', lowered.source)
                self.assertIn('rounded = acc.to(tl.float16)', lowered.source)
        with self.assertRaises(ValueError):
            tensorcore_source(WorkloadContract(gemm.workload_document(
                'fib_gemm_n128_k2048', rows=8)))

    def test_004_scalar_warp_rewrite_keeps_small_m_outputs_independent(self):
        compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')
        for rows in (1, 2, 8):
            workload = WorkloadContract(gemm.workload_document('fib_gemm_n128_k2048', rows=rows))
            starter = parse(gemm.starter_source(workload)).document
            for warps in (4, 8, 16):
                program = scalar_warp_program(compiler, workload, num_warps=warps)
                self.assertEqual(len(program.stages), 1)
                self.assertEqual(set(program.inputs), {'a', 'b'})
                self.assertEqual(program.outputs, ('out',))
                schedule = json.loads(program.stages[0].schedule_bytes)
                self.assertEqual(schedule['operations'], starter['operations'])
                self.assertEqual(schedule['access_maps'], starter['access_maps'])
                self.assertEqual(schedule['roles'][0]['execution_groups'], list(range(warps)))
                lowered = compiler.lower_program(program)
                lowered.validate_binding()
                self.assertIn(f'_kernel[({rows}, 128, 1)]', lowered.lowerings[0].source)
                self.assertIn(f'num_warps={warps}', lowered.lowerings[0].source)
        with self.assertRaises(ValueError):
            scalar_warp_program(compiler, WorkloadContract(gemm.workload_document(
                'fib_gemm_n128_k2048', rows=16)), num_warps=4)

    def test_004_column_tiles_reuse_a_and_reduce_each_b_row(self):
        compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')
        workload = WorkloadContract(gemm.workload_document('fib_gemm_n128_k2048', rows=1))
        for tile in (2, 4, 8):
            source = column_tiled_source(workload, columns_per_cta=tile)
            schedule = parse(source).document
            self.assertEqual(sum(op['kind'] == 'store' for op in schedule['operations']), 1)
            assessment = compiler.assess(schedule)
            self.assertEqual(assessment.findings, ())
            lowered = compiler.lower(assessment)
            self.assertIn(f'_kernel[(1, {128 // tile}, 1)]', lowered.source)
            self.assertIn(f'BLOCK_COLUMN={tile}', lowered.source)
            self.assertIn('products = b32 * a32[None, :]', lowered.source)
            self.assertIn('totals = tl.sum(products.to(tl.float32), axis=1)', lowered.source)
            self.assertIn('num_warps=4', lowered.source)
        with self.assertRaises(ValueError):
            column_tiled_source(WorkloadContract(gemm.workload_document(
                'fib_gemm_n128_k2048', rows=2)))
        with self.assertRaises(ValueError):
            column_tiled_source(workload, columns_per_cta=3)

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
