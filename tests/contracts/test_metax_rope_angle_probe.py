"""Angle-probe source and sealed handoff contracts; CPU fixtures are not device code."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import UUID

from open_cake_ir.compiler import Compiler, Target
from open_cake_ir.evaluation.core import LaunchableCandidate
from open_cake_ir.evaluation.paired import candidate_identity
from open_cake_ir.lab import CandidateSubmission, OpenCakeEnvironment
from open_cake_ir.lab.build import BuildRequest, TritonToolchainBuilder
from open_cake_ir.serialization import canonical_json_bytes
from open_cake_ir.tasks.c550_bench.binding import BENCH_COMMIT, BenchProblem
from tests.contracts.test_c550_bench_binding import ORACLE_NUMERICS
from tests.contracts.test_portable_program_evaluation import NativeCompiler
from tests.contracts.test_triton_loop_scopes import _execute
from tools import qualify_metax_rope_angles as probe

ROOT = Path(__file__).resolve().parents[2]


def problem():
    tensor = lambda dtype: SimpleNamespace(dtype=SimpleNamespace(value=dtype))
    definition = SimpleNamespace(reference='def run(*args): return None\n', custom_inputs_entrypoint=None,
        inputs={name: tensor(dtype) for name, dtype in [('position_ids', 'int64'),
                ('inv_freq', 'float32'), ('attention_scaling', 'float32')]},
        outputs={'cos_sin': tensor('bfloat16')},
        get_input_shapes=lambda axes: {'position_ids': (axes['batch_size'], axes['seq_len']),
                                      'inv_freq': (64,), 'attention_scaling': None},
        get_output_shapes=lambda axes: {'cos_sin': (axes['batch_size'], axes['seq_len'], 128, 2)})
    tolerance = SimpleNamespace(model_dump=lambda: {'max_atol': 1e-5, 'max_rtol': .05,
                                                    'required_matched_ratio': .99, 'max_error_cap': None})
    workloads = tuple(SimpleNamespace(uuid=str(UUID(int=index + 1)),
        axes={'batch_size': b, 'seq_len': s}, tolerance=tolerance)
        for index, (b, s) in enumerate(probe.SHAPES))
    raw = tuple({'uuid': row.uuid, 'inputs': {'attention_scaling': {'type': 'scalar', 'value': 1.0}},
                 'tolerance': {}} for row in workloads)
    return BenchProblem(Path('/fixture/original-bench'), probe.TASK,
        SimpleNamespace(document=lambda name: {'seed': 200}), definition, workloads, raw,
        Path('/fixture/original-bench/rope'))


class AngleSource(unittest.TestCase):
    def test_original_shapes_and_explicit_policy_are_the_only_probe_question(self):
        original = problem()
        self.assertEqual([(row['batch'], row['sequence']) for row in probe.select_cases(original)], list(probe.SHAPES))
        changed = replace(original, workloads=original.workloads + (original.workloads[0],))
        with self.assertRaisesRegex(ValueError, 'one original workload'):
            probe.select_cases(changed)
        for field, value in [('float32_matmul_precision', 'highest'), ('allow_tf32', False),
                             ('initialization', {'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE': None})]:
            policy = deepcopy(ORACLE_NUMERICS); policy[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'observed original HIGH'):
                probe.hypothesis_policy(policy)
        with self.assertRaises(ValueError):
            probe.angle_source(2, 33, 'tf32')
        with self.assertRaises(ValueError):
            probe.angle_source(1, 2048, 'automatic')

    def test_both_emissions_keep_fp32_dot_direction_and_production_tf32_closed(self):
        compiler = Compiler.load(ROOT)
        for b, s in probe.SHAPES:
            for precision in probe.PRECISIONS:
                with self.subTest(shape=(b, s), precision=precision):
                    document, emission, target = probe.probe_emission(b, s, precision)
                    self.assertNotIn(probe.TF32, target.instruction_contracts)
                    self.assertIn(f'tl.dot(frequencies, tl.trans(positions), input_precision="{precision}")', emission.source)
                    self.assertNotIn('tl.float16', emission.source)
                    self.assertEqual(emission.toolchain['code_object'], 'mcfatbin')
                    self.assertEqual(emission.toolchain['compile_constants']['D_POSITION_IDS_2'], 1)
                    self.assertEqual(emission.toolchain['compile_constants']['D_INV_FREQ_1'], 1)
                    assessment = compiler.assess(document)
                    self.assertEqual(assessment.lowering_eligible, precision == 'ieee')
                    if precision == 'tf32':
                        self.assertEqual({f.code for f in assessment.findings if f.blocks_lowering}, {'TARGET_INSTRUCTION_UNSUPPORTED'})

    def test_actual_emitted_padding_and_sequence_tail_write_every_angle(self):
        # This interpreter does not model TF32. Exactly representable powers of two
        # isolate address, K-zero masking and transpose semantics from precision.
        _, emission, _ = probe.probe_emission(2, 131, 'tf32')
        positions = [100 * batch + row for batch in range(2) for row in range(131)]
        frequencies = [2. ** -(i % 16) for i in range(64)]
        memory = {'position_ids': positions[:], 'inv_freq': frequencies[:], 'angles': [None] * (2 * 131 * 64)}
        trace = _execute(emission, memory)
        self.assertEqual(memory['angles'], [float(position) * frequency for position in positions for frequency in frequencies])
        self.assertEqual(memory['position_ids'], positions)
        self.assertEqual(memory['inv_freq'], frequencies)
        self.assertEqual(len(trace.stores), len(memory['angles']))
        self.assertEqual(set(trace.stores.values()), {1})

    def test_bit_comparison_distinguishes_rounding_missing_output_and_bad_reference(self):
        count = 2 * 131 * 64
        expected = struct.pack('<f', 1.) * count
        self.assertTrue(probe.compare_angles(expected, expected, 2, 131)['bitwise_equal'])
        for word in (0x3f800001, 0x7fc00000, 0):
            observed = expected[:-4] + struct.pack('<I', word)
            check = probe.compare_angles(expected, observed, 2, 131)
            self.assertFalse(check['bitwise_equal'])
            self.assertEqual(check['bit_mismatches'], 1)
        with self.assertRaisesRegex(ValueError, 'complete declared'):
            probe.compare_angles(expected, expected[:-4], 2, 131)
        with self.assertRaisesRegex(ValueError, 'must be finite'):
            probe.compare_angles(struct.pack('<f', float('nan')) * count, expected, 2, 131)


class AngleBinding(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.problem = problem()
        self.policy = deepcopy(ORACLE_NUMERICS)
        self.commit = '1' * 40
        self.cases = probe.select_cases(self.problem)
        probe.write(self.root / 'result.json', {'source_commit': self.commit, 'target': probe.TARGET,
            'passed': True, 'cases': [{**case, 'precisions': [{'precision': p, 'native_compiled': True}
                                                           for p in probe.PRECISIONS]} for case in self.cases]})
        probe.write(self.root / 'binding.json', {'bench_commit': BENCH_COMMIT, 'task': probe.TASK,
            'cases': self.cases, 'oracle_numerics': self.policy, 'precisions': list(probe.PRECISIONS),
            'scope': 'angle stage only; original RoPE Workload remains the parent authority'})
        self.candidates = {}
        for case in self.cases:
            directory = self.root / case['workload_uuid']; directory.mkdir()
            workload = probe.original_workload(self.problem, case, self.policy)
            probe.write(directory / 'original-workload.json', workload.document)
            for precision in probe.PRECISIONS:
                folder = directory / precision; folder.mkdir()
                document, emission, _ = probe.probe_emission(case['batch'], case['sequence'], precision)
                payload = emission.source.encode()
                requirements = {'compiler': 'triton', 'source_language': 'python', 'target': probe.TARGET, **emission.toolchain}
                probe.write(folder / 'requirements.json', requirements)
                submission = CandidateSubmission.seal(OpenCakeEnvironment.media_type, canonical_json_bytes(document))
                request = BuildRequest(submission.sha256, payload, 'lowered_source', sha256(payload).hexdigest(),
                                       probe.TARGET, requirements['kernel_entry_point'], requirements)
                builder = TritonToolchainBuilder(workload=workload, case_id='primary', isolated_compiler=NativeCompiler())
                candidate = builder.build_stage(request, probe.stage_abi(case))
                for role, data in candidate.artifact_payloads.items():
                    (folder / (role + '.bin')).write_bytes(data)
                probe.write(folder / 'candidate.json', candidate_identity(candidate))
                self.candidates[(case['workload_uuid'], precision)] = candidate
        self.case = self.cases[0]
        self.folder = self.root / self.case['workload_uuid'] / 'tf32'

    def refused(self, message):
        with patch.object(BenchProblem, 'open', return_value=self.problem), \
             patch.object(probe, 'numeric_policy', return_value=self.policy), \
             patch('open_cake_ir.evaluation.local_broker.admit_local_job') as lease, \
             self.assertRaisesRegex(ValueError, message):
            probe.evaluate(SimpleNamespace(built=self.root, bench_root=Path('/fixture'), oracle_numerics=Path('/fixture/policy')),
                           {'source_commit': self.commit})
        lease.assert_not_called()

    def reseal(self, change):
        candidate = self.candidates[(self.case['workload_uuid'], 'tf32')]
        payloads = dict(candidate.artifact_payloads)
        change(payloads)
        sealed = LaunchableCandidate(candidate.candidate_sha256, candidate.target, candidate.entry_point,
            {role: sha256(data).hexdigest() for role, data in payloads.items()},
            sha256(payloads['launch_manifest']).hexdigest(), payloads)
        for role, data in payloads.items():
            (self.folder / (role + '.bin')).write_bytes(data)
        (self.folder / 'candidate.json').write_bytes(canonical_json_bytes(candidate_identity(sealed)))

    def test_six_native_fixture_stages_bind_to_original_workloads_without_allocation(self):
        result = probe.bound_candidates(self.root, self.commit, self.problem, self.policy)
        self.assertEqual(len(result), 3)
        self.assertEqual([p for _, routes in result for p, _, _ in routes], list(probe.PRECISIONS) * 3)

    def test_stale_source_and_changed_original_workload_refuse_before_allocation(self):
        path = self.root / 'result.json'; original = path.read_bytes()
        value = json.loads(original); value['source_commit'] = '2' * 40
        path.write_text(json.dumps(value)); self.refused('exact source')
        path.write_bytes(original)
        path = self.root / self.case['workload_uuid'] / 'original-workload.json'
        value = json.loads(path.read_text()); value['validation']['raw_tolerance'] = {'max_rtol': 10.}
        path.write_text(json.dumps(value)); self.refused('original Workload differs')

    def test_swapped_precision_and_resealed_source_refuse_before_allocation(self):
        source = self.root / self.case['workload_uuid'] / 'ieee'
        for path in source.iterdir():
            shutil.copyfile(path, self.folder / path.name)
        self.refused('current original-case emission')

    def test_resealed_kernel_source_cannot_borrow_the_original_candidate(self):
        self.reseal(lambda p: p.update(lowered_source=p['lowered_source'] + b'\n# changed\n'))
        self.refused('current original-case emission')

    def test_native_pointer_count_is_checked_before_device_allocation(self):
        def change(payloads):
            manifest = json.loads(payloads['launch_manifest']); manifest['hidden_null_pointer_parameters'] = 2
            compilation = json.loads(payloads['stage_compilation']); compilation['hidden_null_pointer_parameters'] = 2
            payloads['launch_manifest'] = canonical_json_bytes(manifest)
            payloads['stage_compilation'] = canonical_json_bytes(compilation)
        self.reseal(change)
        self.refused('pointer ABI')

    def test_resealed_launch_grid_is_bound_to_the_emission(self):
        def change(payloads):
            manifest = json.loads(payloads['launch_manifest']); manifest['grid'][0] += 1
            payloads['launch_manifest'] = canonical_json_bytes(manifest)
        self.reseal(change)
        self.refused('current original-case emission')

    def test_case_directory_links_and_open_production_gate_refuse(self):
        directory = self.folder.parent
        saved = directory.with_name('saved-case'); directory.rename(saved); directory.symlink_to(saved, target_is_directory=True)
        self.refused('case directory')
        directory.unlink(); saved.rename(directory)
        target = Target.load(ROOT / 'compiler/targets/xcore1002.json')
        with patch.object(Target, 'load', return_value=replace(target, instruction_contracts=target.instruction_contracts | {probe.TF32})):
            self.refused('production TF32')

    def test_observed_policy_drift_uses_the_shared_owner_before_allocation(self):
        with patch.object(BenchProblem, 'open', return_value=self.problem), \
             patch.object(probe, 'numeric_policy', return_value=self.policy), \
             patch('open_cake_ir.lab.executor.ExecutorRevision.for_target',
                   return_value=SimpleNamespace(admit_host=lambda: {})), \
             patch('open_cake_ir.tasks.c550_bench.binding.require_oracle_numerics',
                   side_effect=ValueError('policy drift')) as require, \
             patch('open_cake_ir.evaluation.local_broker.admit_local_job') as lease, \
             self.assertRaisesRegex(ValueError, 'policy drift'):
            probe.evaluate(SimpleNamespace(built=self.root, bench_root=Path('/fixture'), oracle_numerics=Path('/fixture/policy')),
                           {'source_commit': self.commit})
        require.assert_called_once_with(self.policy, phase='before_angle_probe_allocation')
        lease.assert_not_called()


if __name__ == '__main__':
    unittest.main()
