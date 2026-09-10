"""Semantic seams from OMOE; CPU and source checks do not qualify GPU execution."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.evaluation.core import compare_tile_outputs
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks import add_rmsnorm
from open_cake_ir.tasks.tiles.workload import _round
from open_cake_ir.tasks.workloads import create_task, load_workload, materialize_case, reference_outputs

ROOT = Path(__file__).resolve().parents[2]


class OmoeTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.lock.json')

    def task(self, columns=4, rows=1, backend='triton-b200'):
        document, source = create_task('add_rmsnorm_bf16', rows=rows, columns=columns, backend=backend)
        return WorkloadContract(document), source

    def test_dual_output_rounds_residual_before_normalization(self):
        workload, _ = self.task()
        inputs = {'delta': [2**-8]*4, 'residual': [1.0, 1.0+2**-7]*2, 'weight': [1.0]*4}
        output = reference_outputs(workload, 'primary', inputs)
        # Ties round down when the BF16 mantissa is even, up when it is odd.
        rounded = [1.0, 1.0+2**-6]*2
        self.assertEqual(output['residual_out'], rounded)
        inv = 1/math.sqrt(sum(x*x for x in rounded)/4 + 1e-6)
        self.assertEqual(output['out'], [_round(x*inv, 'bf16') for x in rounded])
        self.assertNotEqual(output['residual_out'], [d+r for d,r in zip(inputs['delta'],inputs['residual'])])

    def test_exact_residual_does_not_inherit_normalization_tolerance(self):
        workload, _ = self.task()
        before = {'delta': [0.0]*4, 'residual': [1.0]*4, 'weight': [1.0]*4}
        expected = reference_outputs(workload, 'primary', before)
        changed = deepcopy(expected)
        changed['out'][0] += 2**-7
        self.assertTrue(compare_tile_outputs(workload, before, expected, changed, before)[0])
        changed['residual_out'][0] += 2**-7
        self.assertFalse(compare_tile_outputs(workload, before, expected, changed, before)[0])
        for observed in ({'out': expected['out']}, {**expected, 'extra': [0.0]},
                         {**expected, 'residual_out': expected['residual_out'][:-1]},
                         {**expected, 'out': [float('nan')]*4}):
            self.assertFalse(compare_tile_outputs(workload, before, expected, observed, before)[0])

    def test_signed_zero_residual_and_unchanged_inputs_are_checked(self):
        workload, _ = self.task()
        inputs = materialize_case(workload, 'zeros')
        output = reference_outputs(workload, 'zeros', inputs)
        self.assertEqual(math.copysign(1.0, output['residual_out'][1]), -1.0)
        changed = deepcopy(output); changed['residual_out'][1] = 0.0
        self.assertFalse(compare_tile_outputs(workload, inputs, output, changed, inputs)[0])
        after = deepcopy(inputs); after['delta'][1] = 0.0
        self.assertFalse(compare_tile_outputs(workload, inputs, output, output, after)[0])

    def test_public_loader_rejects_contract_weakening(self):
        workload, _ = self.task()
        mutations = [lambda d: d['validation']['outputs']['residual_out'].update(comparison='elementwise_atol_rtol', atol=1.0),
                     lambda d: d['semantics'].update(epsilon=1e-3),
                     lambda d: d['semantics']['candidate_abi'].update(outputs=['out']),
                     lambda d: d['cases'].pop(),
                     lambda d: d['semantics']['arithmetic'].update(normalization_input='unrounded_sum')]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'workload.json'
            path.write_text(json.dumps(workload.document)); load_workload(path)
            for mutate in mutations:
                d = deepcopy(workload.document); mutate(d); path.write_text(json.dumps(d))
                with self.assertRaises(ValueError): load_workload(path)

    def test_comparison_refuses_partial_or_malformed_output_rules(self):
        workload, _ = self.task()
        inputs = materialize_case(workload, 'primary')
        expected = reference_outputs(workload, 'primary', inputs)
        for change in (lambda r: r.pop('residual_out'),
                       lambda r: r['residual_out'].update(atol=1.0),
                       lambda r: r['out'].update(rtol=float('inf')),
                       lambda r: r['out'].update(comparison='always_pass'),
                       lambda r: r['out'].update(atol=True)):
            d = deepcopy(workload.document); change(d['validation']['outputs'])
            # Core must refuse even a direct caller which bypassed task loading.
            if math.isinf(d['validation']['outputs']['out']['rtol']):
                with self.assertRaises(ValueError): WorkloadContract(d)
            else:
                with self.assertRaises(ValueError):
                    compare_tile_outputs(WorkloadContract(d), inputs, expected, expected, inputs)

    def test_inputs_remain_in_declared_bf16_domain(self):
        workload, _ = self.task()
        inputs = materialize_case(workload, 'primary')
        for bad in ([True]*4, [0.1]*4, [float('nan')]*4, [512.0]*4, []):
            with self.assertRaises(ValueError):
                reference_outputs(workload, 'primary', {**inputs, 'delta': bad})
        for case in workload.case_ids:
            values = materialize_case(workload, case)
            self.assertEqual(values, materialize_case(workload, case))
            out = reference_outputs(workload, case, values)
            self.assertTrue(compare_tile_outputs(workload, values, out, out, values)[0])

    def test_real_non_power_two_width_lowers_with_zero_masked_padding(self):
        for backend in add_rmsnorm.BACKENDS_SUPPORTED:
            for columns in (1, 7, 2559, 2560, 2561):
                with self.subTest(backend=backend, columns=columns):
                    workload, source = self.task(columns=columns, rows=2, backend=backend)
                    schedule = frontend.parse(source).document
                    assessed = self.compiler.assess(schedule)
                    self.assertTrue(assessed.lowering_eligible, assessed.findings)
                    lowered = self.compiler.lower(assessed).source
                    self.assertIn('other=0.0', lowered)
                    self.assertIn('mask=col_offsets < D_RESIDUAL_OUT_1', lowered)
                    self.assertIn('rounded = summed.to(tl.bfloat16)', lowered)
                    self.assertIn('z = rounded.to(tl.float32)', lowered)
                    self.assertIn(f'(2, {columns})', lowered)
                    self.assertEqual([a.name for a in workload.tensor_abi('primary') if a.mode=='output'], ['out','residual_out'])

    def test_unsupported_target_and_unbounded_shape_are_refused(self):
        for args in ({'backend':'metal-m2'}, {'columns':0}, {'columns':16385}, {'rows':True}):
            with self.assertRaises(ValueError): self.task(**args)


class OmoeCommonEvaluationTests(unittest.TestCase):
    def test_cuda_worker_and_common_assay_use_registered_task_semantics(self):
        from hashlib import sha256
        from open_cake_ir.evaluation import LaunchableCandidate
        from open_cake_ir.evaluation.core import EvaluationProtocol, TensorLaunchManifest
        from open_cake_ir.tasks.tiles.evaluation import evaluate_tile_workload
        from open_cake_ir.tasks import evaluate as worker
        document, _ = create_task('add_rmsnorm_bf16', rows=1, columns=7, backend='triton-b200')
        workload = WorkloadContract(document)
        manifest = TensorLaunchManifest.for_workload(workload, 'rounding_ties', target=workload.target,
            kernel_name='cpu_protocol_fixture', grid=[1,1,1], block=[128,1,1],
            dynamic_shared_memory_bytes=0, hidden_null_pointer_parameters=0)
        payloads = {'cubin': b'\x7fELFprotocol_fixture',
                    'launch_manifest': json.dumps(manifest.as_dict()).encode()}
        candidate = LaunchableCandidate('a'*64, workload.target, manifest.kernel_name,
            {k:sha256(v).hexdigest() for k,v in payloads.items()}, manifest.canonical_sha256, payloads)
        protocol = EvaluationProtocol('cpu_protocol_fixture', 'search', workload.canonical_sha256,
                                      'rounding_ties', 'none')
        class Launcher:
            corrupt = False
            def launch_tensors(self, candidate, manifest, inputs):
                # This fixture checks routing and validation, not GPU execution.
                out = reference_outputs(workload, 'rounding_ties', inputs)
                if self.corrupt: out['residual_out'][0] += 2**-7
                return out, deepcopy(inputs), {'candidate_sha256':candidate.candidate_sha256,
                                              'kernel_calls':1, 'fallback_calls':0}
        launcher = Launcher()
        self.assertTrue(evaluate_tile_workload(candidate,workload,protocol,launcher).correctness_passed)
        launcher.corrupt = True
        self.assertFalse(evaluate_tile_workload(candidate,workload,protocol,launcher).correctness_passed)
        inputs = worker.materialize_case(workload, 'rounding_ties')
        self.assertEqual(inputs, materialize_case(workload, 'rounding_ties'))
        self.assertEqual(worker.reference_outputs(workload, 'rounding_ties',inputs)['residual_out'][0],1.0)
