"""CPU doubles for common Metal Evaluation receipts; no GPU observer invocation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import os
import platform
import subprocess
from pathlib import Path
import struct
from subprocess import CompletedProcess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from open_cake_ir.evaluation.core import EvaluationReceipt, LaunchableCandidate
from open_cake_ir.evaluation.metal_manifest import MetalTensorLaunchManifest, compile_options
from open_cake_ir.evaluation.metal_observations import METAL_PROFILE_KIND
from open_cake_ir.evaluation.paired import PAIRED_METAL_KIND, paired_protocol, validate_paired_broker
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.evaluation import metal_runtime
from open_cake_ir.tasks import evaluate, workloads

HOST = {'device_name': 'Apple M1 Pro', 'device_registry_id': '1234',
        'operating_system': 'explicit unit-test host', 'target': 'apple_gpu_family7'}
CASES = ['primary', 'zeros', 'near_zero', 'alternating', 'mixed_magnitude']
JOB = 'metal-123456789abc'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def workload_fixture():
    return WorkloadContract({'workload_id': 'metal-evaluation-unit-fixture',
        'semantics': {'target': HOST['target'], 'candidate_abi': {'inputs': ['x'], 'outputs': ['out']}},
        'cases': [{'case_id': name, 'shape': {'R': 1, 'C': 2}, 'seed': i, 'mode': name} for i, name in enumerate(CASES)],
        'tensors': {name: {'shape': ['R', 'C'], 'dtype': 'fp32', 'layout': 'contiguous_row_major'} for name in ('x', 'out')},
        'validation': {'all_cases_required': True, 'primary_case': 'primary', 'comparison': 'elementwise_atol_rtol', 'atol': 1e-5, 'rtol': 1e-5}})


def policy_fixture():
    return {'case_id': 'primary', 'validation_case_ids': CASES.copy(),
        'attribution_evaluation': 'correctness_then_profile_each_search_survivor',
        'search_evaluation': 'correctness_then_paired_metal',
        'confirmatory_evaluation': 'fresh_fixed_candidate_correctness_then_paired_metal',
        'paired_timing': {'kind': PAIRED_METAL_KIND, 'arms': ['candidate', 'baseline'],
            'pair_order': [['candidate', 'baseline'], ['baseline', 'candidate']],
            'samples_per_cohort': 2, 'route_calls_per_cohort': 3,
            'maximum_cv': 0.2, 'materiality_ratio': 1.01, 'required_pair_wins': 2}}


def candidate_fixture(workload, label):
    manifest = MetalTensorLaunchManifest.for_workload(workload, 'primary', target=HOST['target'],
        kernel_name='fixture_kernel', grid=[1, 1, 1], block=[32, 1, 1])
    payloads = {'metal_binary_archive': b'not executed: explicit CPU artifact double ' + label.encode(),
        'launch_manifest': canonical(manifest.as_dict()),
        'metal_build_report': canonical({'build': {'host': HOST}, 'archive_only_reload': {'host': HOST, 'source_library_rebuilt': False}, 'compile_options': compile_options()})}
    candidate = LaunchableCandidate(sha256(label.encode()).hexdigest(), HOST['target'], 'fixture_kernel',
        {role: sha256(data).hexdigest() for role, data in payloads.items()}, manifest.canonical_sha256, payloads)
    return candidate, manifest


def fake_observe(**kwargs):
    """Counter observations are explicit test doubles, never called native evidence."""
    launches = []
    for row in kwargs['launch_plan']:
        start = 100.0 + row['index']
        command = {'launch_index': row['index'], 'completed': True, 'timed': row['timed'],
                   'gpu_start_seconds': start, 'gpu_end_seconds': start + (0.001 if row['role'] == 'candidate' else 0.002)}
        observed = {**row, 'passed': True, 'metrics': {'output_mismatches': 0, 'max_abs_error': 0.0, 'inputs_unchanged': True}, 'command_buffer': command}
        if row['profile']:
            observed['profile_raw'] = {'counter_set': 'Timestamp', 'counter': 'GPUTimestamp', 'sampling_boundary': 'compute_stage',
                'units': 'raw_device_timestamp_units', 'sample_indices': [0, 1], 'resolved_bytes': 16,
                'timestamps': [100, 200], 'command_buffer': command}
        launches.append(observed)
    return {'host': HOST, 'module_loads': len(kwargs['candidates']), 'launches': launches}


class MetalEvaluationContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='cake-metal-evaluation-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workload = workload_fixture()
        self.candidate, self.manifest = candidate_fixture(self.workload, 'candidate')
        self.baseline, _ = candidate_fixture(self.workload, 'baseline')
        self.policy = policy_fixture()

    def authority(self, purpose='search'):
        return evaluate._Authority({'purpose': purpose, 'evaluation_protocol': self.policy}, self.root,
            SimpleNamespace(admit_host=lambda: {'kind': 'metal', 'host': HOST, 'observer_executable': '/unit-test/not-executed'}),
            self.workload, self.manifest, self.candidate, self.candidate.artifact_payloads, 'primary', self.baseline)

    def run_worker(self, purpose='search', observer=fake_observe):
        result = evaluate._base_result('unassigned')
        def inputs(workload, case_id):
            return {'x': [float(CASES.index(case_id)), 2.0]}
        with patch.dict(os.environ, {'METAL_JOB_ID': JOB}), \
             patch('open_cake_ir.evaluation.local_broker.observe_local_metal_job', return_value=JOB), \
             patch.object(workloads, 'materialize_case', side_effect=inputs, create=True) as materialize, \
             patch.object(workloads, 'reference_outputs', side_effect=lambda w,c,i: {'out': list(i['x'])}, create=True), \
             patch.object(metal_runtime, 'observe', side_effect=observer) as observe:
            evaluate._evaluate_metal_candidate(self.authority(purpose), result)
        self.assertEqual([call.args[1] for call in materialize.call_args_list], CASES)
        self.assertEqual(result['mode'], 'local_serialized')
        self.assertEqual(observe.call_count, 1)
        return result

    def receipt(self, result, *, payloads=None):
        value = result['receipt']
        data = payloads or {role: (self.root / name).read_bytes() for role, name in value['artifacts'].items()}
        return EvaluationReceipt(self.candidate.candidate_sha256, self.workload.canonical_sha256,
            sha256(canonical(self.policy)).hexdigest(), 'attribution' if 'profile' in data else 'search', 'primary',
            value['correctness_passed'], value['correctness'], 1, 0, sha256(data['launch_receipt']).hexdigest(), value['timing'], data)

    def test_real_worker_path_reuses_common_receipt_derivation_and_all_input_cases(self):
        result = self.run_worker()
        receipt = self.receipt(result)
        self.assertTrue(receipt.correctness_passed)
        self.assertTrue(receipt.timing['measurement_quality_passed'])
        self.assertGreater(receipt.timing['speedup'], 1.9)
        validate_paired_broker(receipt, JOB, result['counters'])
        self.assertEqual(result['counters']['compiler_invocations'], 0)
        self.assertEqual(result['counters']['module_loads'], 2)
        self.assertEqual(result['counters']['preflight_calls'], 10)
        self.assertEqual(result['counters']['kernel_calls'], 32)
        self.assertEqual(result['counters']['timing_samples'], 8)

    def test_raw_timestamp_and_validation_coverage_mutations_are_rejected(self):
        result = self.run_worker()
        base = {role: (self.root / name).read_bytes() for role, name in result['receipt']['artifacts'].items()}
        for role, mutate in (
            ('timing_samples', lambda d: d['measurements'][0]['arms']['candidate']['samples_ms'].__setitem__(0, 77.0)),
            ('timing_samples', lambda d: d.update(timer='CUPTI')),
            ('timing_samples', lambda d: d.update(allocation_mode='exclusive')),
            ('correctness_output', lambda d: d['participants']['candidate']['preflight']['launches'].pop()),
        ):
            changed = deepcopy(base)
            document = json.loads(changed[role]); mutate(document); changed[role] = canonical(document)
            with self.subTest(role=role), self.assertRaises(ValueError):
                self.receipt(result, payloads=changed)

    def test_a_failed_nonprimary_case_prevents_timing_and_retains_all_preflight_checks(self):
        def rejected(**kwargs):
            observation = fake_observe(**kwargs)
            observation['launches'] = [row for row in observation['launches'] if row['phase'] == 'preflight']
            row = next(row for row in observation['launches'] if row['role'] == 'candidate' and row['input_case_id'] == 'zeros')
            row['passed'] = False
            row['metrics']['output_mismatches'] = 1
            row['metrics']['max_abs_error'] = 1.0
            return observation
        result = self.run_worker(observer=rejected)
        receipt = self.receipt(result)
        self.assertFalse(receipt.correctness_passed)
        self.assertIsNone(receipt.timing)
        self.assertEqual(result['counters']['kernel_calls'], 10)
        self.assertEqual(result['counters']['timing_samples'], 0)
        validate_paired_broker(receipt, JOB, result['counters'])

    def test_profile_is_separate_raw_uint64_stage_timestamps_with_no_timing_claim(self):
        result = self.run_worker('attribution')
        receipt = self.receipt(result)
        self.assertIsNone(receipt.timing)
        self.assertEqual(receipt.attribution_feedback['kind'], METAL_PROFILE_KIND)
        self.assertEqual(receipt.attribution_feedback['timestamp_delta'], 100)
        self.assertEqual(receipt.attribution_feedback['units'], 'raw_device_timestamp_units')
        payloads = dict(receipt.artifact_payloads)
        profile = json.loads(payloads['profile']); profile['summary']['timestamp_delta'] = 999
        payloads['profile'] = canonical(profile)
        with self.assertRaises(ValueError):
            self.receipt(result, payloads=payloads)

    def test_profile_policy_identity_crosses_worker_and_loader_with_utf8_notes(self):
        base = self.root
        for index, note in enumerate(("baseline", "基线")):
            with self.subTest(note=note):
                self.root = base / str(index)
                self.root.mkdir()
                self.policy['note'] = note
                result = self.run_worker('attribution')
                receipt = self.receipt(result)
                self.assertEqual(receipt.attribution_feedback['kind'], METAL_PROFILE_KIND)
                payloads = dict(receipt.artifact_payloads)
                profile = json.loads(payloads['profile'])
                profile['evaluation_protocol']['note'] = 'different policy'
                payloads['profile'] = canonical(profile)
                with self.assertRaisesRegex(ValueError, 'Evaluation identity differs'):
                    self.receipt(result, payloads=payloads)

    def test_missing_broker_or_case_projection_refuses_before_observer(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(metal_runtime, 'observe') as observe:
            with self.assertRaisesRegex(ValueError, 'broker job'):
                evaluate._evaluate_metal_candidate(self.authority(), evaluate._base_result('unassigned'))
            observe.assert_not_called()
        self.policy['validation_case_ids'] = ['primary']
        with patch.dict(os.environ, {'METAL_JOB_ID': JOB}), patch.object(metal_runtime, 'observe') as observe:
            with self.assertRaisesRegex(ValueError, 'case projection'):
                evaluate._evaluate_metal_candidate(self.authority(), evaluate._base_result('unassigned'))
            observe.assert_not_called()

    def test_metal_policy_does_not_borrow_cupti_callback_counts_or_allow_unwarmed_samples(self):
        self.assertIsNotNone(paired_protocol(self.policy))
        wrong = deepcopy(self.policy); wrong['paired_timing']['route_calls_per_cohort'] = 2
        with self.assertRaisesRegex(ValueError, 'warmup'):
            paired_protocol(wrong)

    def test_missing_native_observer_never_compiles_or_dispatches(self):
        with patch.object(metal_runtime.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'observer executable'):
                metal_runtime.observe(workload=self.workload, candidates={'candidate': self.candidate},
                    manifests={'candidate': self.manifest}, input_cases={}, launch_plan=[],
                    observer_executable=self.root / 'missing', expected_host=HOST, directory=self.root / 'observation')
            run.assert_not_called()

    def test_observer_preparation_rejects_an_enclosing_parent_checkout(self):
        parent = self.root / 'parent-project'
        parent.mkdir()
        (parent / '.git').mkdir()
        output = parent / 'nested' / 'artifacts'
        with patch.object(metal_runtime.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'enclosing checkout'):
                metal_runtime.compile_observer(output)
            run.assert_not_called()
        self.assertFalse(output.exists())

    def test_observer_binary_snapshots_are_checked_by_independent_common_oracle(self):
        executable = self.root / 'observer-double'; executable.write_bytes(b'explicit test double; never executed')
        inputs = {'primary': {'inputs': {'x': [1.0, 2.0]}, 'expected': {'out': [1.0, 2.0]}}}
        plan = [{'index': 0, 'role': 'candidate', 'phase': 'preflight', 'input_case_id': 'primary', 'timed': False, 'profile': False}]
        def process(command, **kwargs):
            request = json.loads(Path(command[1]).read_text()); directory = Path(request['output_directory'])
            (directory / 'x.bin').write_bytes(struct.pack('<2f', 1.0, 2.0))
            (directory / 'out.bin').write_bytes(struct.pack('<2f', 1.0, 2.0))
            row = {'index': 0, 'role': 'candidate', 'phase': 'preflight', 'input_case_id': 'primary',
                'buffer_paths': {'x': 'x.bin', 'out': 'out.bin'}, 'preflight_guard_passed': True,
                'command_buffer': {'launch_index': 0, 'completed': True, 'timed': False, 'gpu_start_seconds': 1.0, 'gpu_end_seconds': 1.001}}
            report = {'status': 'completed', 'host': HOST, 'source_library_rebuilt': False,
                'archive_miss_policy': 'failOnBinaryArchiveMiss', 'module_loads': 1, 'launches': [row],
                'snapshot_persistence': {'condition': 'owned_snapshots_written_at_cohort_end',
                    'pending_payload_limit_bytes': 64 * 1024 * 1024, 'peak_pending_payload_bytes': 0, 'failed_writes': []}}
            return CompletedProcess(command, 0, canonical(report), b'')
        with patch.object(metal_runtime.subprocess, 'run', side_effect=process):
            result = metal_runtime.observe(workload=self.workload, candidates={'candidate': self.candidate},
                manifests={'candidate': self.manifest}, input_cases=inputs, launch_plan=plan,
                observer_executable=executable, expected_host=HOST, directory=self.root / 'observation')
        self.assertTrue(result['launches'][0]['passed'])
        request = json.loads((self.root / 'observation' / 'request.json').read_text())
        self.assertNotIn('source_path', request['participants'][0])
        self.assertEqual(request['participants'][0]['archive_path'].split('.')[-1], 'metallib')
        retained = json.loads((self.root / 'observation' / 'observer.stdout.json').read_text())
        self.assertEqual(retained['snapshot_persistence']['condition'], 'owned_snapshots_written_at_cohort_end')


_SNAPSHOT_ROOT = Path(__file__).resolve().parents[2]
_SNAPSHOT_SWIFTC = Path('/Library/Developer/CommandLineTools/usr/bin/swiftc')
_SNAPSHOT_SDK = Path('/Library/Developer/CommandLineTools/SDKs/MacOSX15.2.sdk')
_SNAPSHOT_SOURCE = _SNAPSHOT_ROOT / 'src/open_cake_ir/evaluation/metal/observer.swift'

_SNAPSHOT_HARNESS = r'''
import Foundation
func check(_ condition: Bool, _ message: String) throws {
    if !condition { throw NSError(domain: message, code: 1) }
}
func rejected(_ operation: () throws -> Void) throws {
    do { try operation() } catch { return }
    throw NSError(domain: "expected refusal", code: 2)
}
func participant(_ role: String = "candidate", shape: [Int] = [4]) -> Participant {
    Participant(role: role, archive_path: "unused", manifest: Manifest(target: "apple_gpu_family7", kernel_name: "unused",
        tensor_abi: [Tensor(name: "x", shape: shape, dtype: "fp32", mode: "input"),
                     Tensor(name: "out", shape: shape, dtype: "fp32", mode: "output")], grid: [1,1,1], block: [32,1,1]))
}
func launch(_ index: Int, role: String = "candidate", phase: String = "cohort", pair: Int? = 0, position: Int? = 0,
            input: String = "primary", timed: Bool = false) -> Launch {
    Launch(index: index, role: role, phase: phase, input_case_id: input, timed: timed,
           profile: phase == "profile", pair_index: pair, position: position)
}
func capture(_ writer: OwnedSnapshots, _ bytes: [UInt8], _ name: String, deferred: Bool = true) throws {
    _ = try bytes.withUnsafeBytes { try writer.capture($0.baseAddress!, count: bytes.count, name: name, deferred: deferred) }
}
let directory = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
switch CommandLine.arguments[1] {
case "owned":
    var written: [String: Data] = [:]
    let writer = OwnedSnapshots(directory: directory, limit: 16, write: { written[$1.lastPathComponent] = $0 })
    let raw = UnsafeMutableRawPointer.allocate(byteCount: 8, alignment: 8)
    defer { raw.deallocate() }
    raw.storeBytes(of: UInt64(17), as: UInt64.self)
    let first = try writer.capture(raw, count: 8, name: "first", deferred: true)
    raw.storeBytes(of: UInt64(99), as: UInt64.self)
    _ = try writer.capture(raw, count: 8, name: "second", deferred: true)
    raw.storeBytes(of: UInt64(255), as: UInt64.self)
    try check(written.isEmpty, "writes occurred inside cohort")
    try check(first.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) } == 17, "returned snapshot aliases reused memory")
    try writer.flush()
    try check(written["first"]!.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) } == 17, "first snapshot was mutated")
    try check(written["second"]!.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) } == 99, "second snapshot was mutated")
    try check(writer.pendingBytes == 0 && writer.peakPendingBytes == 16, "snapshot payload accounting differs")
case "boundaries":
    let launches = [launch(0, phase: "preflight", pair: nil, position: nil),
        launch(1), launch(2), launch(3, timed: true),
        launch(4, role: "baseline", position: 1), launch(5, role: "baseline", position: 1, timed: true),
        launch(6, phase: "postflight", pair: nil, position: nil),
        launch(7, phase: "profile", pair: nil, position: nil), launch(8, pair: 1)]
    let boundary = try snapshotFlushPlan(launches, participants: [participant(), participant("baseline")], limit: 96)
    try check(boundary == Set([0,3,5,6,7,8]), "cohort boundary differs from existing coordinates")
    var writes = 0
    let writer = OwnedSnapshots(directory: directory, write: { _,_ in writes += 1 })
    try capture(writer, [1,2], "preflight", deferred: false)
    try capture(writer, [3,4], "profile", deferred: false)
    try check(writes == 2 && writer.pendingBytes == 0, "non-cohort persistence was deferred")
case "plan_refusal":
    try rejected { _ = try snapshotFlushPlan([launch(0),launch(1),launch(2)], participants: [participant()], limit: 95) }
    try rejected { _ = try snapshotFlushPlan([launch(0, pair: nil)], participants: [participant()], limit: 96) }
    try rejected { _ = try snapshotFlushPlan([launch(0), launch(1, input: "another")], participants: [participant()], limit: 96) }
    try rejected { _ = try snapshotFlushPlan([launch(0), launch(1, phase: "preflight", pair: nil, position: nil),launch(2)], participants: [participant()], limit: 96) }
    try rejected { _ = try snapshotFlushPlan([launch(0)], participants: [participant(shape: [Int.max,2])], limit: snapshotPayloadLimit) }
    try check(try snapshotFlushPlan([launch(0),launch(1),launch(2)], participants: [participant()], limit: 96) == Set([2]), "exact memory boundary refused")
case "capture_refusal":
    var writes = 0
    let writer = OwnedSnapshots(directory: directory, limit: 4, write: { _,_ in writes += 1 })
    try capture(writer, [1,2,3,4], "one")
    try rejected { try capture(writer, [5], "two") }
    try check(writer.pendingBytes == 4 && writer.peakPendingBytes == 4 && writes == 0, "bound refusal changed pending evidence")
    try writer.flush(); try check(writes == 1, "prior snapshot was lost on memory refusal")
case "failure_flush":
    var writes: [String: Data] = [:]
    let writer = OwnedSnapshots(directory: directory, write: { writes[$1.lastPathComponent] = $0 })
    try capture(writer, [1,2], "one"); try capture(writer, [3,4], "two")
    let original = NSError(domain: "later_dispatch", code: 17)
    let failure = flushAfterFailure(writer, original: original)
    try check((failure.0 as NSError).domain == "later_dispatch" && failure.1 == nil, "original dispatch failure changed")
    try check(writes.count == 2 && writer.pendingBytes == 0, "failure did not flush previous callbacks")
case "write_failure":
    var attempts: [String] = []; var written: [String] = []
    let writer = OwnedSnapshots(directory: directory, write: { _,path in
        attempts.append(path.lastPathComponent)
        if path.lastPathComponent == "bad" { throw NSError(domain: "disk_write", code: 5) }
        written.append(path.lastPathComponent)
    })
    try capture(writer, [1], "bad"); try capture(writer, [2], "good")
    let original = NSError(domain: "later_dispatch", code: 17)
    let failure = flushAfterFailure(writer, original: original)
    try check((failure.0 as NSError).domain == "later_dispatch" && (failure.1 as NSError?)?.domain == "disk_write", "flush hid original error")
    try check(attempts == ["bad","good"] && written == ["good"] && writer.failedWrites == ["bad"], "write failure discarded other pending callbacks")
    try writer.flush()
    try check(attempts == ["bad","good"], "failed snapshot writes were retried")
case "files":
    let writer = OwnedSnapshots(directory: directory)
    try capture(writer, [7,8], "first.bin")
    try check(!FileManager.default.fileExists(atPath: directory.appendingPathComponent("first.bin").path), "file persisted early")
    try writer.flush()
    try check(try Data(contentsOf: directory.appendingPathComponent("first.bin")) == Data([7,8]), "persisted bytes differ")
    try capture(writer, [9], "first.bin")
    try rejected { try writer.flush() }
    try check(try Data(contentsOf: directory.appendingPathComponent("first.bin")) == Data([7,8]), "existing evidence overwritten")
default: throw NSError(domain: "unknown test", code: 3)
}
print("CPU snapshot behavior passed")
'''


@unittest.skipUnless(platform.system() == 'Darwin' and _SNAPSHOT_SWIFTC.is_file() and _SNAPSHOT_SDK.is_dir(),
                     'native Foundation snapshot tests require the declared macOS Swift/SDK tools')
class MetalCohortSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='cake-snapshot-tests-')
        directory = Path(cls.temporary.name)
        (directory / 'main.swift').write_text(_SNAPSHOT_SOURCE.read_text() + '\n' + _SNAPSHOT_HARNESS)
        cls.executable = directory / 'snapshot-tests'
        command = [str(_SNAPSHOT_SWIFTC), '-sdk', str(_SNAPSHOT_SDK), '-target', 'arm64-apple-macosx15.0', '-O', '-D', 'SNAPSHOT_TESTS',
                   str(directory / 'main.swift'), '-o', str(cls.executable)]
        build = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if build.returncode:
            raise AssertionError(build.stdout + build.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_behavior(self, name):
        with tempfile.TemporaryDirectory(prefix='cake-snapshot-files-') as directory:
            result = subprocess.run([str(self.executable), name, directory], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('CPU snapshot behavior passed', result.stdout)

    def test_reused_buffer_mutation_cannot_change_owned_snapshots(self): self.run_behavior('owned')
    def test_existing_coordinates_define_complete_and_partial_final_cohorts(self): self.run_behavior('boundaries')
    def test_memory_overflow_missing_coordinates_and_split_cohorts_refuse_before_dispatch(self): self.run_behavior('plan_refusal')
    def test_capture_bound_preserves_pending_callback_bytes(self): self.run_behavior('capture_refusal')
    def test_later_failure_flushes_all_owned_callback_bytes(self): self.run_behavior('failure_flush')
    def test_flush_failure_preserves_original_error_and_attempts_each_file_once(self): self.run_behavior('write_failure')
    def test_actual_files_keep_exact_bytes_and_refuse_overwrites(self): self.run_behavior('files')


if __name__ == '__main__':
    unittest.main()
