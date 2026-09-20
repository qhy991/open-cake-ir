"""The task's mathematical tolerance must never replace its bitwise peer gate."""
from copy import deepcopy
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.evaluation.core import compare_tile_outputs
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.tinygemm import reproduction as task
from open_cake_ir.tasks.workloads import create_task, load_workload, reference_outputs

ROOT = Path(__file__).resolve().parents[2]


class TinyGemmReproduction(unittest.TestCase):
    def workload(self):
        return WorkloadContract(task.workload_document(rows=1, columns=16, depth=32))

    def test_cpu_math_covers_transposition_bias_and_bf16_rounding(self):
        workload = self.workload()
        inputs = {'x': [0.5] * 32, 'weight': [float(n) / 16 for n in range(16) for _ in range(32)],
                  'bias': [0.25] * 16}
        expected = [float(n) + 0.25 for n in range(16)]
        self.assertEqual(task.mathematical_reference(workload, 'primary', inputs), {'out': expected})

    def test_numeric_tolerance_pass_does_not_relax_candidate_bitwise_comparison(self):
        workload = self.workload()
        inputs = task.materialize_case(workload, 'zeros')
        # This one-ULP BF16 difference fits the mathematical tolerance, but the
        # candidate must still reproduce the external peer's bits exactly.
        inputs['bias'] = [1 / 8] * 16
        peer = {'out': [0.1259765625] * 16}
        with patch.object(task, 'peer_reference', return_value=peer):
            expected = reference_outputs(workload, 'zeros', inputs)
        before = {k: list(v) for k, v in inputs.items()}
        wrong = {'out': [0.125] * 16}
        passed, metrics = compare_tile_outputs(workload, before, expected, wrong, before)
        self.assertFalse(passed)
        self.assertEqual(metrics['output_mismatches'], 16)
        self.assertTrue(compare_tile_outputs(workload, before, expected, peer, before)[0])

    def test_bad_external_reference_is_rejected_before_it_can_be_an_oracle(self):
        workload = self.workload()
        inputs = task.materialize_case(workload, 'zeros')
        for peer in ({'out': [1.0] * 16}, {'out': [float('nan')] * 16}, {'out': [0.0] * 15}):
            with self.subTest(peer=peer), patch.object(task, 'peer_reference', return_value=peer):
                with self.assertRaisesRegex(ValueError, 'independent complete-output'):
                    task.reference_outputs(workload, 'zeros', inputs)

    def test_contract_refuses_tolerance_substitution_and_wrong_target(self):
        document = task.workload_document(rows=1, columns=16, depth=32)
        broken = deepcopy(document)
        broken['validation'].update(comparison='elementwise_atol_rtol', atol=0.01)
        with self.assertRaisesRegex(ValueError, 'mandatory bitwise'):
            task.validate_contract(broken)
        with self.assertRaisesRegex(ValueError, 'exact B200 or B300'):
            task.workload_document(backend='triton-dcu')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'workload.json'
            path.write_text(json.dumps(document))
            self.assertEqual(load_workload(path).workload_id, document['workload_id'])

    def test_all_public_shapes_and_both_stage_choices_lower_without_gpu_or_peer_calls(self):
        compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')
        with patch.object(task, 'peer_reference', side_effect=AssertionError('preparation must be CPU-only')):
            for batch, columns, depth in ((1, 128, 720), (16, 1024, 1024), (64, 4096, 3072)):
                document, _ = create_task(task.TASK, backend='triton-b300', rows=batch,
                                          columns=columns, depth=depth)
                workload = WorkloadContract(document)
                for stages in (4, 8):
                    with self.subTest(batch=batch, stages=stages):
                        source = task.starter_source(workload, stages=stages)
                        assessment = compiler.assess(frontend.parse(source).document)
                        self.assertTrue(assessment.lowering_eligible, assessment.findings)
                        lowering = compiler.lower(assessment)
                        self.assertIn('tl.bfloat16', lowering.source)
                        self.assertIn(f'num_stages={stages}', lowering.source)

    def test_partitioned_lowering_keeps_each_quarter_and_accumulates_across_trips(self):
        import numpy as np
        class LogicalTL:
            float32 = np.float32
            arange = staticmethod(np.arange)
            broadcast_to = staticmethod(np.broadcast_to)
            trans = staticmethod(np.transpose)
            gather = staticmethod(lambda a, i, axis: np.take_along_axis(a, i, axis))
            dot = staticmethod(lambda a, b, acc, **kw: acc + a @ b)
            inline_asm_elementwise = staticmethod(lambda *a, args, **kw: args[0])
        workload = WorkloadContract(task.workload_document(rows=16, columns=16, depth=3072))
        compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')
        document = frontend.parse(task.partitioned_source(workload)).document
        assessment = compiler.assess(document)
        self.assertTrue(assessment.lowering_eligible, assessment.findings)
        tree = ast.parse(compiler.lower(assessment).source)
        loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For))
        assignments = [n for n in loop.body if isinstance(n, ast.Assign)
                       and isinstance(n.targets[0], ast.Name) and n.targets[0].id in {'acc0','acc1','acc2','acc3'}]
        self.assertEqual(len(assignments), 8)
        rng = np.random.default_rng(7)
        inputs = [(rng.integers(-2, 3, (16, 1024)).astype(np.float32),
                   rng.integers(-2, 3, (16, 1024)).astype(np.float32)) for _ in range(3)]
        env = {'tl': LogicalTL, **{f'acc{i}': np.full((16, 16), i + 1, np.float32) for i in range(4)}}
        for a, b in inputs:
            env.update(a=a, b=b)
            for node in assignments:
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<selected K carry>', 'exec'), env)
        for i in range(4):
            expected = np.full((16, 16), i + 1, np.float32)
            for a, b in inputs:
                expected += a[:,i*256:(i+1)*256] @ b[:,i*256:(i+1)*256].T
            np.testing.assert_array_equal(env[f'acc{i}'], expected)
        # This checks selected coordinates/carry, not GPU instruction rounding.
        for change in ('target', 'unaligned_quarter'):
            bad = deepcopy(document)
            if change == 'target': bad['target'] = 'sm_100a'
            else:
                next(op for op in bad['operations'] if op['id']=='dot0')['parameters']['k_ranges'] = [[1,257]]
            refused = compiler.assess(bad)
            self.assertIn('TRITON_MMA_K_RANGES_UNSUPPORTED', [f.code for f in refused.findings])
