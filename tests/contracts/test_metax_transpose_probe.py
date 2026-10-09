"""Qualification probes keep production admission closed and preserve movement."""
from dataclasses import replace
import importlib.util
from pathlib import Path
from hashlib import sha256
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler, Target, frontend
from open_cake_ir.compiler.backends.triton import emit, preflight
from open_cake_ir.compiler.ir import OperationKind, Schedule
from open_cake_ir.compiler.verifier import verify
from tests.contracts.test_triton_loop_scopes import _execute

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('transpose_probe', ROOT / 'tools/qualify_metax_transpose.py')
probe_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe_tool)


class MetaxTransposeProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)
        cls.target = Target.load(ROOT / 'compiler/targets/xcore1002.json')
        cls.probe = replace(cls.target, operation_kinds=cls.target.operation_kinds | {OperationKind.TRANSPOSE})

    def test_actual_emission_preserves_coordinates_tails_and_each_output(self):
        for dtype in probe_tool.DTYPES:
            for rows, columns in probe_tool.SHAPES:
                with self.subTest(dtype=dtype, shape=(rows, columns)):
                    raw = frontend.parse(probe_tool.source(rows, columns, dtype)).document
                    self.assertIn('TARGET_OPERATION_UNSUPPORTED',
                                  {f.code for f in self.compiler.assess(raw).findings})
                    schedule = Schedule.from_dict(raw)
                    findings = (*verify(schedule, self.probe), *preflight(schedule, self.probe))
                    self.assertFalse([f for f in findings if f.blocks_lowering])
                    values = ([bool(i % 2) for i in range(rows * columns)] if dtype == 'bool' else
                              [i + 2**54 for i in range(rows * columns)] if dtype == 'int64' else
                              [i % 47 - 23 for i in range(rows * columns)])
                    memory = {'x': values[:], 'out': [None] * len(values)}
                    trace = _execute(emit(schedule, self.probe), memory)
                    self.assertEqual(memory['out'], [values[r * columns + c] for c in range(columns) for r in range(rows)])
                    self.assertEqual(memory['x'], values)
                    self.assertEqual(set(trace.stores.values()), {1})

    def test_wrong_transpose_shape_and_fp8_refuse_at_their_owners(self):
        raw = frontend.parse(probe_tool.source(16, 32, 'fp32')).document
        next(b for b in raw['buffers'] if b['name'] == 'transposed')['shape'] = [16, 32]
        self.assertIn('VALUE_OPERATION_TYPE', {f.code for f in verify(Schedule.from_dict(raw), self.probe)})
        fp8 = Schedule.from_dict(frontend.parse(probe_tool.source(16, 32, 'fp8_e4m3')).document)
        self.assertIn('VALUE_OPERATION_TYPE', {f.code for f in verify(fp8, self.probe)})
        self.assertIn('MACA_FP8_OPERATION_UNQUALIFIED', {f.code for f in preflight(fp8, self.probe)})

    def test_storage_patterns_distinguish_every_coordinate(self):
        from open_cake_ir.compiler.ir import DType
        for dtype in probe_tool.DTYPES:
            for rows, columns in probe_tool.SHAPES:
                size = DType(dtype).itemsize
                patterns = probe_tool.storage_patterns(rows, columns, dtype)
                signatures = [tuple(raw[i * size:(i + 1) * size] for raw in patterns)
                              for i in range(rows * columns)]
                self.assertEqual(len(set(signatures)), rows * columns, (dtype, rows, columns))
        # These two actual past patterns cannot observe a whole-row swap.
        old_bool = bytes(i % 2 for i in range(16 * 32))
        old_i64 = bytes((i * 37) % 256 for i in range(16 * 32 * 8))
        self.assertEqual(old_bool[:32], old_bool[32:64])
        self.assertEqual(old_i64[:256], old_i64[256:512])


class DeviceBoundary(unittest.TestCase):
    def setUp(self):
        from open_cake_ir.evaluation.workload import WorkloadContract
        from open_cake_ir.lab import CandidateSubmission, OpenCakeEnvironment
        from open_cake_ir.lab.build import BuildRequest, TritonToolchainBuilder
        from open_cake_ir.serialization import canonical_json_bytes
        from tests.contracts.test_portable_program_evaluation import NativeCompiler
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.commit = '1' * 40
        target = Target.load(ROOT / 'compiler/targets/xcore1002.json')
        self.probe = replace(target, operation_kinds=target.operation_kinds | {OperationKind.TRANSPOSE})
        self.records = []
        for dtype in probe_tool.DTYPES:
            for rows, columns in probe_tool.SHAPES:
                raw = frontend.parse(probe_tool.source(rows, columns, dtype)).document
                emission = emit(Schedule.from_dict(raw), self.probe)
                source = emission.source.encode()
                requirements = {'compiler': 'triton', 'source_language': 'python', 'target': 'xcore1002',
                                **emission.toolchain}
                submission = CandidateSubmission.seal(OpenCakeEnvironment.media_type, canonical_json_bytes(raw))
                builder = TritonToolchainBuilder(workload=WorkloadContract(probe_tool.workload_document(rows, columns, dtype)),
                    case_id='primary', isolated_compiler=NativeCompiler())
                candidate = builder.build(BuildRequest(submission.sha256, source, 'lowered_source', sha256(source).hexdigest(),
                    'xcore1002', requirements['kernel_entry_point'], requirements))
                record = {'name': f'{dtype}-{rows}-{columns}', 'shape': [rows, columns],
                          'dtype': dtype, 'native_compiled': True}
                self.records.append(record)
                directory = self.root / record['name']
                directory.mkdir()
                self.save_candidate(directory, candidate)
                self.last = candidate, directory
        self.result = {'source_commit': self.commit, 'target': 'xcore1002', 'passed': True, 'cases': self.records,
            'refused_cases': [{'name': f'fp8_e4m3-{rows}-{columns}',
                'codes': ['MACA_FP8_OPERATION_UNQUALIFIED', 'VALUE_OPERATION_TYPE']} for rows, columns in probe_tool.SHAPES]}
        self.save_result()

    def save_result(self):
        (self.root / 'result.json').write_text(json.dumps(self.result))

    def save_candidate(self, directory, candidate):
        from open_cake_ir.evaluation.paired import candidate_identity
        for role, payload in candidate.artifact_payloads.items():
            (directory / (role + '.bin')).write_bytes(payload)
        (directory / 'candidate.json').write_text(json.dumps(candidate_identity(candidate)))

    def reseal_last(self, *, source=None, manifest=None):
        from open_cake_ir.evaluation.core import LaunchableCandidate
        from open_cake_ir.serialization import canonical_json_bytes
        candidate, directory = self.last
        payloads = dict(candidate.artifact_payloads)
        if source is not None:
            payloads['lowered_source'] = source
        if manifest is not None:
            payloads['launch_manifest'] = canonical_json_bytes(manifest)
        sealed = LaunchableCandidate(candidate.candidate_sha256, candidate.target, candidate.entry_point,
            {role: sha256(payload).hexdigest() for role, payload in payloads.items()},
            sha256(payloads['launch_manifest']).hexdigest(), payloads)
        self.save_candidate(directory, sealed)

    def refused_before_allocation(self, message):
        # A Torch import substitute does no device work. The real allocation
        # boundary must remain uncalled even for the last retained case.
        with patch.dict('sys.modules', {'torch': SimpleNamespace()}), \
             patch('open_cake_ir.evaluation.local_broker.admit_local_job') as allocate, \
             self.assertRaisesRegex(ValueError, message):
            probe_tool.evaluate(SimpleNamespace(built=self.root), {'source_commit': self.commit})
        allocate.assert_not_called()

    def test_complete_fixture_binds_without_device_work(self):
        self.assertEqual(len(probe_tool.bound_candidates(self.root, self.commit)), 18)

    def test_wrong_source_or_missing_fp8_control_refuses_before_allocation(self):
        self.result['source_commit'] = '2' * 40
        self.save_result()
        self.refused_before_allocation('exact source commit')
        self.result['source_commit'] = self.commit
        self.result['refused_cases'].pop()
        self.save_result()
        self.refused_before_allocation('FP8 refusal controls')

    def test_self_consistent_changed_last_source_is_not_this_probe(self):
        self.reseal_last(source=self.last[0].artifact_payloads['lowered_source'] + b'\n# another emission\n')
        self.refused_before_allocation('current fixed probe emission')

    def test_self_consistent_wrong_workload_manifest_refuses_before_allocation(self):
        manifest = json.loads(self.last[0].artifact_payloads['launch_manifest'])
        manifest['workload_sha256'] = '9' * 64
        self.reseal_last(manifest=manifest)
        self.refused_before_allocation('Workload|workload')

    def test_self_consistent_wrong_launch_block_refuses_before_allocation(self):
        manifest = json.loads(self.last[0].artifact_payloads['launch_manifest'])
        manifest['block'] = [128, 1, 1]
        self.reseal_last(manifest=manifest)
        self.refused_before_allocation('current fixed probe emission')

    def test_open_production_target_cannot_use_pre_admission_probe(self):
        with patch.object(Target, 'load', return_value=self.probe):
            self.refused_before_allocation('production Target to remain closed')


if __name__ == '__main__':
    unittest.main()
