"""CPU execution/authority probes; these fixtures make no GPU timing claim."""
from __future__ import annotations

import copy
from contextlib import ExitStack
from dataclasses import replace
from hashlib import sha256
import json
import os
import grp
import pwd
from pathlib import Path
from tests.contracts._executor_fixture import compiler_reference
import tempfile
import shutil
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.evaluation.core import EvaluationReceipt, LaunchableCandidate, TensorLaunchManifest
from open_cake_ir.evaluation.paired import (
    candidate_identity, paired_protocol, paired_summary, validate_pair_candidates, validate_receipt_policy, validate_paired_broker,
)
from open_cake_ir.tasks.workloads import load_workload, materialize_case, reference_outputs
from open_cake_ir.tasks.runtime import TaskLab as Lab
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.evidence import EvidenceStore
from open_cake_ir.lab.bindings import external_file, load_baseline_bundle, resolve_execution_bindings
from open_cake_ir.lab.contracts import CampaignLock, StudyContract
from open_cake_ir.lab.archive import _validate_receipt_authority, _archive_evaluation_receipt
from open_cake_ir.lab.pairing import bind_baseline
from open_cake_ir.lab.runtime import CommandBrokerSubmitter
from tests.contracts.test_native_triton_pairing import DraftCompilerFixture
from open_cake_ir.tasks import evaluate as worker

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / 'contracts/studies/matched-search-triton-b300-optimization-template.json'


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def protocol():
    value = json.loads(TEMPLATE.read_bytes())['evaluation_protocol']
    value['case_id'] = 'tiny'
    value['note'] = '基线与候选的共同测量协议'
    return value


def sealed(workload, role, *, case_id='tiny'):
    manifest = TensorLaunchManifest.for_workload(workload, case_id, target='sm_103a',
        kernel_name=role, grid=[1, 1, 1], block=[128, 1, 1],
        dynamic_shared_memory_bytes=0, hidden_null_pointer_parameters=2)
    payloads = {'cubin': b'\x7fELF-CPU-fixture-' + role.encode(),
                'launch_manifest': encoded(manifest.as_dict())}
    return LaunchableCandidate(sha256(role.encode()).hexdigest(), 'sm_103a', role,
        {key: sha256(value).hexdigest() for key, value in payloads.items()},
        manifest.canonical_sha256, payloads), manifest


class PairedExecutionTests(unittest.TestCase):
    def setUp(self):
        self.workload = load_workload(ROOT / 'contracts/workloads/rmsnorm-fp32-v2.json')
        self.case_id = 'tiny'
        self.candidate, self.manifest = sealed(self.workload, 'candidate')
        self.baseline, _ = sealed(self.workload, 'baseline')
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name).resolve()
        self.protocol = protocol()
        self.events = []
        self.snapshots = []
        self.benchmarks = []
        self.bad_role = None
        self.bad_call = None
        self.bad_case = None
        self.created, self.closed = [], []
        self.fail_load_at = self.fail_close_at = None
        self.load_error = RuntimeError('test-only load failure')
        self.close_error = RuntimeError('test-only close failure')
        self.callback_count = 42
        self.latencies = {'candidate': 1.0, 'baseline': 1.0}

    def execute(self):
        owner = self
        class FakeLoaded:
            def __init__(self, candidate, manifest, inputs, admission):
                if len(owner.created) == owner.fail_load_at:
                    raise owner.load_error
                owner.created.append(self)
                self.candidate = candidate
                self.role = candidate.entry_point
                self.inputs = copy.deepcopy(inputs)
                self.expected = reference_outputs(owner.workload, owner.case_id, inputs)
                self.loaded = SimpleNamespace(launch_calls=0, resources={'fixture': True})
            def fresh_argument_sets(self, count):
                return [{'outputs': {key: [float('nan')] * len(values) for key, values in self.expected.items()}}
                        for _ in range(count)]
            def launch(self, values):
                self.loaded.launch_calls += 1
                owner.events.append(self.role)
                values['outputs'] = copy.deepcopy(self.expected)
                if owner.bad_role == self.role and self.loaded.launch_calls == owner.bad_call:
                    next(iter(values['outputs'].values()))[0] = float('nan')
                if (owner.bad_role == self.role and owner.bad_case is not None
                        and self.inputs == materialize_case(owner.workload, owner.bad_case)):
                    next(iter(values['outputs'].values()))[0] = float('nan')
            def snapshot(self, values):
                owner.snapshots.append(self.role)
                return values['outputs'], copy.deepcopy(self.inputs)
            def launch_tensors(self, candidate, manifest, inputs):
                values = self.fresh_argument_sets(1)[0]
                self.launch(values)
                observed, after = self.snapshot(values)
                return observed, after, {'candidate_sha256': candidate.candidate_sha256,
                    'kernel_calls': 1, 'fallback_calls': 0}
            def close(self):
                owner.closed.append(self)
                if owner.created.index(self) == owner.fail_close_at:
                    raise owner.close_error
        class Helper:
            def bench_gpu_time_with_cupti(self, callback, **kwargs):
                owner.benchmarks.append(kwargs)
                for _ in range(owner.callback_count):
                    callback()
                return [owner.latencies[owner.events[-1]]] * kwargs['repeat_iters']
            def bench_gpu_time_with_cuda_event(self, *args, **kwargs):
                raise AssertionError('event fallback')
            def bench_gpu_time_with_cudagraph(self, *args, **kwargs):
                raise AssertionError('graph fallback')
        authority = worker._Authority({'purpose': 'search', 'evaluation_protocol': self.protocol},
            self.output, None, self.workload, self.manifest, self.candidate,
            self.candidate.artifact_payloads, self.case_id, self.baseline)
        result = worker._base_result('gpuq-123456789abc')
        admission = SimpleNamespace(broker_job_id='gpuq-123456789abc', gpu_uuid='GPU-CPU-fixture')
        with patch.object(worker, 'LoadedTorchTensorCandidate', FakeLoaded):
            worker._evaluate_paired_tile(authority, result, Helper(), admission)
        self.result = result
        return self.receipt()

    def receipt(self, *, payloads=None, timing=None):
        values = self.result['receipt']
        if payloads is None:
            payloads = {role: (self.output / path).read_bytes() for role, path in values['artifacts'].items()}
        return EvaluationReceipt(self.candidate.candidate_sha256, self.workload.canonical_sha256,
            sha256(encoded(self.protocol)).hexdigest(), 'search', self.case_id, values['correctness_passed'],
            values['correctness'], 1, 0, sha256(payloads['launch_receipt']).hexdigest(),
            values['timing'] if timing is None else timing, payloads)

    def use_all_case_task(self):
        from open_cake_ir.tasks.workloads import create_task
        from open_cake_ir.tasks.normalization.study import evaluation_policy
        from open_cake_ir.evaluation.workload import WorkloadContract
        document, _ = create_task('silu', backend='triton-b300', rows=2, columns=8)
        self.workload = WorkloadContract(document)
        self.case_id = 'primary'
        self.candidate, self.manifest = sealed(self.workload, 'candidate', case_id=self.case_id)
        self.baseline, _ = sealed(self.workload, 'baseline', case_id=self.case_id)
        self.protocol = evaluation_policy(self.workload)

    def test_cuda_task_checks_every_case_but_times_only_the_primary_loaded_owner(self):
        self.use_all_case_task()
        receipt = self.execute()
        self.assertTrue(receipt.correctness_passed)
        self.assertTrue(receipt.timing['measurement_quality_passed'])
        correctness = json.loads(receipt.artifact_payloads['correctness_output'])
        for role in ('candidate', 'baseline'):
            for phase in ('preflight', 'postflight'):
                self.assertEqual([row['input_case_id'] for row in correctness['participants'][role][phase]['launches']],
                                 list(self.workload.case_ids))
        self.assertEqual(self.result['counters']['module_loads'], 10)
        self.assertEqual(self.result['counters']['preflight_calls'], 10)
        self.assertEqual(self.result['counters']['kernel_calls'], 860)
        self.assertEqual(self.result['counters']['timing_samples'], 500)
        self.assertEqual(sorted(x.loaded.launch_calls for x in self.created), [2]*8+[422]*2)
        self.assertEqual(self.closed, self.created)
        validate_paired_broker(receipt, 'gpuq-123456789abc', self.result['counters'])
        raw = json.loads(receipt.artifact_payloads['timing_samples'])
        self.assertEqual(raw['kind'], 'fixed_baseline_paired_cupti_v1')
        self.assertEqual(raw['case_id'], 'primary')
        self.assertNotIn('command_buffers', raw['measurements'][0]['arms']['candidate'])

    def test_cuda_multicase_receipt_refuses_missing_duplicate_reordered_and_forged_checks(self):
        self.use_all_case_task()
        receipt = self.execute()
        original = json.loads(receipt.artifact_payloads['correctness_output'])
        for phase in ('preflight', 'postflight'):
            for change in ('missing', 'duplicate', 'reordered', 'metrics'):
                with self.subTest(phase=phase, change=change):
                    raw = copy.deepcopy(original)
                    rows = raw['participants']['candidate'][phase]['launches']
                    if change == 'missing': rows.pop()
                    elif change == 'duplicate': rows[1] = copy.deepcopy(rows[0])
                    elif change == 'reordered': rows.reverse()
                    else: rows[-1]['metrics']['output_mismatches'] = 1
                    with self.assertRaisesRegex(ValueError, 'CUDA'):
                        self.receipt(payloads={**receipt.artifact_payloads, 'correctness_output': encoded(raw)})
        for field in ('module_loads', 'preflight_calls', 'kernel_calls'):
            with self.subTest(counter=field), self.assertRaisesRegex(ValueError, 'work counters'):
                validate_paired_broker(receipt, 'gpuq-123456789abc',
                                      {**self.result['counters'], field: self.result['counters'][field]-1})

    def test_nonprimary_cuda_failure_retains_all_preflight_cases_and_never_times(self):
        self.use_all_case_task()
        self.bad_role, self.bad_case = 'candidate', 'alternating'
        receipt = self.execute()
        self.assertFalse(receipt.correctness_passed)
        self.assertIsNone(receipt.timing)
        self.assertFalse(self.benchmarks)
        self.assertEqual(self.result['counters']['module_loads'], 10)
        self.assertEqual(self.result['counters']['kernel_calls'], 10)
        correctness = json.loads(receipt.artifact_payloads['correctness_output'])
        for role in ('candidate', 'baseline'):
            self.assertEqual([row['input_case_id'] for row in correctness['participants'][role]['preflight']['launches']],
                             list(self.workload.case_ids))
        self.assertEqual(self.closed, self.created)
        validate_paired_broker(receipt, 'gpuq-123456789abc', self.result['counters'])

    def test_cuda_multicase_projection_and_abi_are_checked_before_loading(self):
        self.use_all_case_task()
        self.protocol['validation_case_ids'] = list(self.workload.case_ids[:-1])
        with self.assertRaisesRegex(ValueError, 'case projection'):
            self.execute()
        self.assertFalse(self.created)
        self.protocol['validation_case_ids'] = list(self.workload.case_ids)
        changed = json.loads(encoded(self.workload.document))
        changed['cases'][-1]['shape']['R'] += 1
        from open_cake_ir.evaluation.workload import WorkloadContract
        self.workload = WorkloadContract(changed)
        self.candidate, self.manifest = sealed(self.workload, 'candidate', case_id=self.case_id)
        self.baseline, _ = sealed(self.workload, 'baseline', case_id=self.case_id)
        with self.assertRaises(ValueError):
            self.execute()
        self.assertFalse(self.created)

    def test_validation_case_admission_does_not_relabel_or_relax_the_primary_seal(self):
        self.use_all_case_task()
        original = self.candidate.artifact_payloads['launch_manifest']
        with self.assertRaisesRegex(ValueError, 'sealed launch ABI'):
            self.manifest.check_workload(self.workload, 'zeros')
        for case_id in self.workload.case_ids:
            self.manifest.check_validation_case(self.workload, case_id)
        self.assertEqual(self.manifest.case_id, 'primary')
        self.assertEqual(self.candidate.artifact_payloads['launch_manifest'], original)
        with self.assertRaises(ValueError):
            self.manifest.check_validation_case(self.workload, 'unlisted')

    def test_cuda_multicase_cleanup_attempts_every_owner_and_preserves_primary_error(self):
        self.use_all_case_task()
        self.fail_load_at, self.fail_close_at = 4, 1
        with self.assertRaises(RuntimeError) as caught:
            self.execute()
        self.assertIs(caught.exception, self.load_error)
        self.assertIs(caught.exception.__cause__, self.close_error)
        self.assertEqual(len(self.created), 4)
        self.assertEqual(self.closed, self.created)

    def test_actual_launcher_order_fresh_outputs_both_oracles_and_close_null(self):
        receipt = self.execute()
        order = paired_protocol(self.protocol).pair_order
        self.assertEqual(self.events, ['candidate', 'baseline'] +
            [role for pair in order for role in pair for _ in range(42)] + ['candidate', 'baseline'])
        self.assertEqual(self.snapshots, self.events)
        self.assertEqual(self.result['counters']['kernel_calls'], 844)
        self.assertEqual(self.result['counters']['timing_samples'], 500)
        self.assertTrue(receipt.correctness_passed)
        self.assertEqual(receipt.timing['classification'], 'close_null')
        self.assertTrue(receipt.timing['measurement_quality_passed'])
        for options in self.benchmarks:
            self.assertEqual(options, dict(dry_run_iters=11, repeat_iters=25,
                cold_l2_cache=True, use_cuda_graph=False))
        validate_receipt_policy(receipt, self.protocol, candidate_identity(self.baseline), self.candidate)
        validate_paired_broker(receipt, 'gpuq-123456789abc', self.result['counters'])
        with self.assertRaisesRegex(ValueError, 'job or work counters'):
            validate_paired_broker(receipt, 'gpuq-111111111111', self.result['counters'])
        with self.assertRaisesRegex(ValueError, 'job or work counters'):
            validate_paired_broker(receipt, 'gpuq-123456789abc', {**self.result['counters'], 'timing_samples': 499})

    def test_submitter_transports_both_sealed_bundles_to_worker_and_checks_actual_job(self):
        retained = self.execute()
        executor_reference = {'path':'runtime/executors/CPU-fixture.json',
                              'canonical_sha256':'a'*64, 'executor_id':'CPU-fixture'}
        executor = SimpleNamespace(reference=executor_reference, canonical_sha256='a'*64, executor_id='CPU-fixture', project_root=ROOT)
        requests = []
        def command(argv, **kwargs):
            request_path = Path(argv[argv.index('--request') + 1])
            result_path = Path(argv[argv.index('--output') + 1])
            document = json.loads(request_path.read_bytes())
            requests.append(document)
            with patch.object(worker.ExecutorRevision, 'load_reference', return_value=executor):
                authority = worker._load_authority(request_path)
            self.assertEqual(authority.candidate, self.candidate)
            self.assertEqual(authority.baseline, self.baseline)
            self.assertEqual(authority.request['evaluation_protocol'], self.protocol)
            for role, name in self.result['receipt']['artifacts'].items():
                (request_path.parent / name).write_bytes(retained.artifact_payloads[role])
            result = {**self.result, 'admitted':True}
            result_path.write_bytes(encoded(result))
            return SimpleNamespace(returncode=0, stdout=b'',
                stderr=b'[gpu-run] accepted job gpuq-123456789abc\n')
        from tests.contracts._executor_fixture import compiler_reference
        submitter = CommandBrokerSubmitter(compiler_reference=compiler_reference(ROOT), command=('CPU-fixture',),
            workload_path=ROOT / 'contracts/workloads/rmsnorm-fp32-v2.json',
            workload_sha256=self.workload.canonical_sha256,
            protocol_sha256=sha256(encoded(self.protocol)).hexdigest(), cwd=ROOT,
            executor=executor, service_user=pwd.getpwuid(os.geteuid()).pw_name,
            service_group=grp.getgrgid(os.getegid()).gr_name,
            evaluation_protocol=self.protocol, baseline=self.baseline,
            workload_loader=load_workload)
        with patch('open_cake_ir.lab.runtime.run_supervised', side_effect=command):
            attempt = submitter.submit(self.candidate, case_id='tiny', purpose='search', attempt=1)
        self.assertEqual(attempt.receipt.canonical_sha256, retained.canonical_sha256)
        self.assertEqual(len(requests), 1)
        self.assertTrue(all(value.startswith('baseline-') for value in requests[0]['baseline']['artifact_paths'].values()))

    def test_nested_paired_receipt_archives_and_replays_against_raw_evidence(self):
        receipt = self.execute()
        self.assertIsInstance(receipt.timing['pooled_medians_ms'], MappingProxyType)
        evidence = EvidenceStore.create(self.output / 'receipt-evidence')
        references = _archive_evaluation_receipt(evidence, receipt)
        stored = {item['role']: evidence.read_object(item) for item in references}
        document = json.loads(stored.pop('evaluation_receipt'))
        artifact_digests = document.pop('artifact_payload_sha256')
        self.assertEqual(set(stored), {'correctness_output', 'launch_receipt', 'timing_samples'})
        self.assertEqual(stored, receipt.artifact_payloads)
        self.assertEqual(artifact_digests, {role: item['sha256'] for item in references
                                        if (role := item['role']) != 'evaluation_receipt'})
        self.assertEqual(document['timing'], self.result['receipt']['timing'])
        replayed = EvaluationReceipt(**document, artifact_payloads=stored)
        self.assertEqual(replayed.timing, receipt.timing)
        self.assertEqual(replayed.correctness, receipt.correctness)
        validate_receipt_policy(replayed, self.protocol, candidate_identity(self.baseline), self.candidate)
        validate_paired_broker(replayed, 'gpuq-123456789abc', self.result['counters'])
        document['timing']['pooled_medians_ms']['baseline'] *= 2
        with self.assertRaisesRegex(ValueError, 'paired timing summary'):
            EvaluationReceipt(**document, artifact_payloads=stored)

    def test_paired_dispatch_requires_the_strict_task_loader_before_process_start(self):
        document = self.workload.document
        document['semantics']['epsilon'] = 0
        path = self.output / 'invalid-workload.json'
        path.write_bytes(encoded(document))
        arguments = dict(command=('CPU-fixture',), workload_path=path,
            workload_sha256=sha256(encoded(document)).hexdigest(),
            protocol_sha256=sha256(encoded(self.protocol)).hexdigest(), cwd=ROOT,
            executor=SimpleNamespace(reference={}, project_root=ROOT),
            compiler_reference=compiler_reference(ROOT),
            service_user=pwd.getpwuid(os.geteuid()).pw_name,
            service_group=grp.getgrgid(os.getegid()).gr_name,
            evaluation_protocol=self.protocol, baseline=self.baseline)
        with self.assertRaisesRegex(ValueError, 'requires the task Workload loader'):
            CommandBrokerSubmitter(**arguments)
        submitter = CommandBrokerSubmitter(**arguments, workload_loader=load_workload)
        with patch('open_cake_ir.lab.runtime.run_supervised') as process:
            with self.assertRaisesRegex(ValueError, 'RMSNorm epsilon'):
                submitter.submit(self.candidate, case_id='tiny', purpose='search', attempt=1)
            process.assert_not_called()

    def test_correct_slower_candidate_is_a_valid_observation(self):
        self.latencies['candidate'] = 2.0
        receipt = self.execute()
        self.assertTrue(receipt.correctness_passed)
        self.assertTrue(receipt.timing['measurement_quality_passed'])
        self.assertEqual(receipt.timing['classification'], 'second_arm_faster')

    def test_each_partner_preflight_and_every_callback_output_is_checked(self):
        for role, call in [('candidate', 1), ('baseline', 1), ('candidate', 2),
                           ('baseline', 42), ('candidate', 422), ('baseline', 422)]:
            with self.subTest(role=role, call=call):
                # Each assay creates new artifacts, just like a fresh broker job.
                with tempfile.TemporaryDirectory() as directory:
                    self.output = Path(directory).resolve()
                    self.bad_role, self.bad_call = role, call
                    receipt = self.execute()
                    self.assertFalse(receipt.correctness_passed)
                    if call == 1:
                        self.assertIsNone(receipt.timing)

    def test_callback_count_drift_fails_without_reusing_an_output(self):
        self.callback_count = 41
        with self.assertRaisesRegex(RuntimeError, 'invocation count'):
            self.execute()

    def test_raw_identity_order_missing_partner_and_summary_tampering_rejected(self):
        receipt = self.execute()
        payloads = dict(receipt.artifact_payloads)
        original = json.loads(payloads['timing_samples'])
        mutations = [
            lambda raw: raw['participants'].pop('baseline'),
            lambda raw: raw['measurements'].pop(),
            lambda raw: raw['measurements'][0].update(order=['baseline', 'candidate']),
            lambda raw: raw['measurements'][0]['arms']['candidate'].update(position=1),
            lambda raw: raw['measurements'][0]['arms']['baseline'].update(candidate_record_sha256='a' * 64),
            lambda raw: raw['measurements'][0]['arms']['candidate'].update(route_calls=41),
            lambda raw: raw['measurements'][0]['arms']['candidate']['samples_ms'].pop(),
            lambda raw: raw['measurements'][0]['arms']['candidate']['summary'].update(cv=0.02),
            lambda raw: raw.update(gpu_uuid='other-gpu'),
            lambda raw: raw.update(purpose='confirmatory'),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                raw = copy.deepcopy(original)
                mutate(raw)
                with self.assertRaises((ValueError, TypeError, KeyError)):
                    self.receipt(payloads={**payloads, 'timing_samples': encoded(raw)})
        with self.assertRaisesRegex(ValueError, 'fixed baseline'):
            validate_receipt_policy(receipt, self.protocol, candidate_identity(self.candidate), self.candidate)

    def test_stable_single_receipt_cannot_satisfy_new_paired_policy(self):
        receipt = self.execute()
        payloads = dict(receipt.artifact_payloads)
        payloads['timing_samples'] = encoded({'cohorts_ms': [[1.0] * 25] * 5})
        old = self.receipt(payloads=payloads, timing={'measurement_quality_passed': True, 'pooled_median_ms': 1.0})
        self.assertEqual(old.measurement_quality, 'stable')
        with self.assertRaisesRegex(ValueError, 'single-candidate'):
            _validate_receipt_authority(old, candidate=self.candidate,
                workload_sha256=self.workload.canonical_sha256,
                protocol_sha256=old.evaluation_protocol_sha256, case_id='tiny', purpose='search',
                evaluation_protocol=self.protocol, fixed_baseline=candidate_identity(self.baseline))

    def test_wrong_target_case_and_abi_never_enter_worker(self):
        wrong_case, _ = sealed(self.workload, 'baseline', case_id='zeros')
        with self.assertRaisesRegex(ValueError, 'Workload'):
            validate_pair_candidates(self.candidate, wrong_case, self.workload, 'tiny')
        with self.assertRaisesRegex(ValueError, 'manifest'):
            validate_pair_candidates(replace(self.candidate, target='sm_100a'), self.baseline, self.workload, 'tiny')
        payloads = dict(self.baseline.artifact_payloads)
        raw = json.loads(payloads['launch_manifest'])
        raw['tensor_abi'][0]['shape'][1] = 3
        payloads['launch_manifest'] = encoded(raw)
        wrong = replace(self.baseline, artifact_payloads=payloads,
            artifact_roles={key: sha256(value).hexdigest() for key, value in payloads.items()},
            launch_spec_sha256=sha256(payloads['launch_manifest']).hexdigest())
        with self.assertRaisesRegex(ValueError, 'Workload'):
            validate_pair_candidates(self.candidate, wrong, self.workload, 'tiny')

    def test_external_baseline_request_is_read_only_and_rejects_unsafe_paths(self):
        for role, value in self.baseline.artifact_payloads.items():
            (self.output / role).write_bytes(value)
        request = {**candidate_identity(self.baseline), 'artifact_paths':
            {role: role for role in self.baseline.artifact_payloads}, 'executor_revision': {'historical': True}}
        path = self.output / 'request.json'
        path.write_bytes(encoded(request))
        before = path.read_bytes()
        self.assertEqual(load_baseline_bundle(ROOT, path), self.baseline)
        self.assertEqual(path.read_bytes(), before)
        request['artifact_paths']['cubin'] = '../cubin'
        path.write_bytes(encoded(request))
        with self.assertRaisesRegex(ValueError, 'unsafe'):
            load_baseline_bundle(ROOT, path)
        link = self.output / 'inside.json'
        link.symlink_to(TEMPLATE)
        with self.assertRaisesRegex(ValueError, 'custody'):
            external_file(ROOT, str(link), 'qualification')
        with self.assertRaisesRegex(ValueError, 'custody'):
            external_file(ROOT, str(TEMPLATE), 'qualification')

    def test_binding_group_is_explicit_closed_and_cannot_override_treatment(self):
        study = StudyContract.load(TEMPLATE)
        with self.assertRaisesRegex(ValueError, 'complete campaign binding'):
            resolve_execution_bindings(ROOT, study, None)
        for field in ('model', 'reasoning_effort', 'evaluation_protocol'):
            path = self.output / f'{field}.json'
            path.write_bytes(encoded({'schema_version': 1, 'qualification_path': '/a',
                'qualification_anchor_path': '/b', 'runtime_config_path': '/c',
                'fixed_baseline_bundle_path': '/d', field: 'override'}))
            with self.assertRaisesRegex(ValueError, 'fields'):
                resolve_execution_bindings(ROOT, study, path)
        document = copy.deepcopy(study.document)
        document['arms']['open_cake']['provider']['revision'] = 'exact-runtime'
        with self.assertRaisesRegex(ValueError, 'complete campaign binding'):
            resolve_execution_bindings(ROOT, replace(study, document=document), self.output / 'model.json')
        with self.assertRaisesRegex(ValueError, 'complete campaign binding'):
            resolve_execution_bindings(ROOT, replace(study, state='frozen'), self.output / 'model.json')

    def test_external_preflight_binds_stable_study_and_rejects_runtime_or_baseline_drift(self):
        self._external_preflight_contract(TEMPLATE, self.workload, 'native_triton')

    def test_cute_external_preflight_binds_stable_study_and_rejects_runtime_or_baseline_drift(self):
        from tests.contracts.test_native_cute_pairing import baseline
        workload = load_workload(ROOT / 'contracts/workloads/gemm-bias-bf16-fp32-v2.json')
        template = ROOT / 'contracts/studies/matched-search-cute-b300-gemm-optimization-template.json'
        self._external_preflight_contract(template, workload, 'native_cute_dsl', schedule=baseline(workload))

    def _external_preflight_contract(self, template_path, workload, comparison, schedule=None):
        from open_cake_ir.lab.pairing import native_backend, native_block
        policy = native_backend(comparison)
        # Independent CPU fixture: real descriptor bytes, source closure, inventory
        # and resolver. No released project descriptor or host environment is edited.
        project = self.output / 'project'
        project.mkdir()
        for directory in ('contracts', 'corpus', 'compiler', 'docs', 'examples/python'):
            shutil.copytree(ROOT / directory, project / directory)
        template = project / template_path.relative_to(ROOT)
        if schedule is not None:
            # Own the synthetic CPU fixture's Schedule/source reference without editing a release.
            fixture_study = json.loads(template.read_bytes())
            fixture_study['arms']['open_cake']['lowering_route'] = schedule['lowering']
            skeleton = fixture_study['arms']['open_cake']['schedule_skeleton']
            (project / skeleton['path']).write_bytes(encoded(schedule))
            skeleton['canonical_sha256'] = sha256(encoded(schedule)).hexdigest()
            template.write_bytes(encoded(fixture_study))
        fixture_study = json.loads(template.read_bytes())
        fixture_study['evaluation_protocol']['note'] = '共同的候选与基线协议'
        template.write_bytes(encoded(fixture_study))
        study = StudyContract.load(template)
        before = template.read_bytes()
        source = project / 'cpu-executor-source.txt'
        source.write_bytes(b'Independent CPU resolver fixture; not GPU qualification.\n')
        descriptor = {
            'schema_version': 1, 'executor_id': 'cpu-fixture-executor', 'state': 'released',
            'sources': [{'path': source.name, 'sha256': sha256(source.read_bytes()).hexdigest(),
                         'size_bytes': source.stat().st_size}],
            'host_environment': {
                'python': {'invocation_path': '/cpu-fixture/python', 'version': 'fixture', 'resolved_sha256': 'a'*64},
                'packages': {'triton': 'fixture'},
                'cupti_python': {'site_packages_path': '/cpu-fixture/site-packages',
                    'distribution': 'cupti-python', 'version': 'fixture',
                    'files': [{'path': 'cupti/__init__.py', 'sha256': 'b'*64, 'size_bytes': 1}]},
                'flashinfer_helper': {'path': '/cpu-fixture/testing.py', 'distribution': 'flashinfer-python',
                    'version': 'fixture', 'sha256': 'c'*64, 'size_bytes': 1},
            },
        }
        executor_path = project / 'runtime/executors/cpu-fixture.json'
        executor_path.parent.mkdir(parents=True)
        executor_path.write_bytes(encoded(descriptor))
        executor_ref = {'executor_id': descriptor['executor_id'],
            'path': executor_path.relative_to(project).as_posix(),
            'canonical_sha256': sha256(encoded(descriptor)).hexdigest()}
        inventory = project / 'inventory/EXECUTOR_REVISIONS.json'
        inventory.parent.mkdir()
        inventory.write_bytes(encoded({'schema_version': 2,
            'current_by_target': {'sm_103a': executor_ref}}))
        draft = DraftCompilerFixture()
        schedule = bind_baseline(json.loads((project / study.document['arms']['open_cake']['schedule_skeleton']['path']).read_bytes()),
                                 workload, 'primary')
        lowering = draft.lower(draft.assess(schedule))
        requirements = lowering.toolchain_requirements
        manifest = TensorLaunchManifest.for_workload(workload, 'primary', target='sm_103a',
            kernel_name=requirements['kernel_entry_point'], grid=requirements['grid'],
            block=native_block(requirements),
            dynamic_shared_memory_bytes=0, hidden_null_pointer_parameters=policy.hidden_null_pointer_parameters)
        payloads = {'cubin': b'CPU-baseline-fixture', 'lowered_source': lowering.source.encode(),
                    'launch_manifest': encoded(manifest.as_dict())}
        # A Python-authored identity may differ from the JSON skeleton identity.
        baseline = LaunchableCandidate('f' * 64, 'sm_103a', manifest.kernel_name,
            {key: sha256(value).hexdigest() for key, value in payloads.items()}, manifest.canonical_sha256, payloads)
        for role, value in payloads.items():
            (self.output / role).write_bytes(value)
        bundle = self.output / 'baseline.json'
        bundle.write_bytes(encoded({**candidate_identity(baseline), 'artifact_paths': {k:k for k in payloads}}))
        executable = self.output / 'provider'
        executable.write_bytes(b'CPU-provider-fixture')
        executable.chmod(0o700)
        helper = executable.with_name('codex-code-mode-host')
        helper.write_bytes(b'CPU Code Mode host fixture')
        helper.chmod(0o700)
        provider = study.document['arms']['open_cake']['provider']
        configuration = {key: provider[key] for key in ('model', 'reasoning_effort', 'service_tier',
            'removed_environment', 'sandbox', 'cwd_policy', 'reference_visibility', 'disabled_features', 'web_search')}
        configuration['code_mode_host'] = {'path': str(helper.resolve()), 'sha256': sha256(helper.read_bytes()).hexdigest()}
        configuration['output_schema_sha256'] = provider['output_schema']['sha256']
        configuration['submission_contract'] = 'candidate_set_envelope_v1'
        qualification = json.loads((ROOT / 'contracts/providers/fixture-provider-candidate-set-ralph-v1.json').read_bytes())
        qualification.update(scope='live_two_turn_current_provider',
            executable_sha256=sha256(executable.read_bytes()).hexdigest(),
            configuration_sha256=sha256(encoded(configuration)).hexdigest())
        qp = self.output / 'qualification.json'; qp.write_bytes(encoded(qualification))
        anchor = {'schema_version':1, 'kind':'codex_provider_qualification_evidence_anchor',
            'run_id':'CPU-fixture', 'evidence_root':str(self.output / '提供器证据'),
            'authority_sha256':'a'*64, 'qualification_receipt_sha256':sha256(encoded(qualification)).hexdigest(),
            'immediate_audit_integrity':True, 'terminal_seal_sha256':'b'*64}
        ap = self.output / 'anchor.json'; ap.write_bytes(encoded(anchor))
        config = {'schema_version':1,
            'provider':{'executable':str(executable),'workspace_root':str(self.output / 'new-author-workspaces')},
            'toolchain':{'python':'fixture-python','bubblewrap':'fixture-bwrap','runtime_roots':[],
                         'triton_version':'fixture','timeout_seconds':30},
            'broker':{'command':['fixture-broker'],'cwd':str(project),'timeout_seconds':30,
                      'service_user':'fixture','service_group':'fixture'}}
        if comparison == 'native_cute_dsl':
            config['toolchain'].pop('triton_version')
            config['toolchain'].update(cutlass_version='4.5.2', cuobjdump='fixture-cuobjdump')
        rp = self.output / 'runtime.json'; rp.write_bytes(encoded(config))
        bindings = {'schema_version':1, 'qualification_path':str(qp), 'qualification_anchor_path':str(ap),
                    'runtime_config_path':str(rp), 'fixed_baseline_bundle_path':str(bundle)}
        bp = self.output / 'bindings.json'; bp.write_bytes(encoded(bindings))
        gate = SimpleNamespace(compiler_revision_id='fixture',compiler_revision_sha256='a'*64,passed=True)
        compiler_ref = {'revision_id':'fixture','path':'compiler/revision.lock.json','canonical_sha256':'a'*64}
        with ExitStack() as stack:
            stack.enter_context(patch('open_cake_ir.lab.preflight._resolve_compiler_reference',
                return_value=(gate, compiler_ref['path'], compiler_ref)))
            from open_cake_ir.lab.bindings import resolve_executor
            from open_cake_ir.lab.executor import ExecutorRevision
            resolved_executors = []
            def resolve_once(*args, **kwargs):
                value = resolve_executor(*args, **kwargs)
                resolved_executors.append(value)
                return value
            resolver = stack.enter_context(patch('open_cake_ir.lab.bindings.resolve_executor', side_effect=resolve_once))
            stack.enter_context(patch('open_cake_ir.lab.preflight.resolve_executor',
                side_effect=AssertionError('preflight must retain the already-bound Executor')))
            stack.enter_context(patch('open_cake_ir.compiler.Compiler.load', return_value=draft))
            toolchain = stack.enter_context(patch('open_cake_ir.lab.triton_build.IsolatedTritonCompiler' if comparison == 'native_triton' else 'open_cake_ir.lab.cute_build.IsolatedCuTeCompiler'))
            toolchain.return_value.canonical_sha256 = 'b'*64
            stack.enter_context(patch('open_cake_ir.lab.runtime_config.broker_execution_sha256', return_value='c'*64))
            lock = Lab(project).preflight(template, execution_bindings_path=bp)
            resolver.assert_called_once_with(project, {'binding': 'current_release'},
                'study.execution', template=True, target='sm_103a')
            self.assertEqual(lock.document['execution']['executor_revision'], executor_ref)
            bound_executor = toolchain.return_value.check_executor.call_args.args[0]
            self.assertIsInstance(bound_executor, ExecutorRevision)
            self.assertIs(bound_executor, resolved_executors[0])
            self.assertEqual(dict(bound_executor.reference), executor_ref)
            self.assertEqual(bound_executor.project_root, project)
            self.assertEqual(bound_executor.document['sources'][0]['path'], source.name)
            # The resolver still rejects exact references in an original template.
            with self.assertRaisesRegex(ValueError, 'Study template Executor binding differs'):
                resolve_executor(project, executor_ref, 'study.execution', template=True,
                                 target='sm_103a')

            self.assertEqual(lock.document['study']['canonical_sha256'], study.canonical_sha256)
            self.assertEqual(template.read_bytes(), before)
            self.assertEqual(lock.document['execution']['fixed_baseline']['candidate'], candidate_identity(baseline))
            self.assertEqual(lock.document['execution']['runtime_config']['path'], str(rp))
            self.assertEqual(lock.document['resolved_inputs']['arm_environments'][comparison]['provider']['model'], 'gpt-5.6-sol')
            self.assertFalse((self.output / 'new-author-workspaces').exists())
            self.assertEqual(CampaignLock.from_dict(lock.document).canonical_sha256, lock.canonical_sha256)
            for arm in ('open_cake', comparison):
                package = Lab(project).task_package(lock, arm + '-1')
                self.assertIn('"sm_103a"', package.task_markdown)
                self.assertIn('candidate-set.json', package.agents_markdown)
                self.assertNotIn('prompt_template', package.task_markdown)
                def frozen_json(name):
                    section = package.task_markdown.split(f"## Frozen reference: `{name}`\n", 1)[1]
                    return json.loads(section.split("```json\n", 1)[1].split("\n```", 1)[0])
                self.assertEqual(frozen_json("target.json")["target_id"], "sm_103a")
                self.assertEqual(frozen_json("run-authority.json")["execution"]["target"], "sm_103a")
                if arm == "open_cake":
                    self.assertEqual(frozen_json("schedule-skeleton.json")["target"], "sm_103a")
                if arm == comparison:
                    self.assertIn(policy.baseline_file, package.task_markdown)
                else:
                    self.assertIn('restricted Python', package.agents_markdown)

            # Exercise live composition through its existing Lab.execute handoff.
            # The CPU fixture constructs the provider and injects the broker boundary;
            # it requires no service account and invokes neither adapter.
            from open_cake_ir.tasks import compose
            with patch.object(compose, '_admit_executor', return_value=(bound_executor, None)), \
                 patch.object(compose, 'broker_execution_sha256', return_value='c'*64), \
                 patch.object(draft, 'check_corpus', return_value=gate), \
                 patch.object(compose, 'CommandBrokerSubmitter') as broker, \
                 patch.object(compose.TaskLab, 'execute', return_value='CPU-composition-handoff') as execute:
                result = compose.execute_matched_from_config(project, lock, rp, self.output / 'unused-run-evidence')
            self.assertEqual(result, 'CPU-composition-handoff')
            self.assertEqual(broker.call_args.kwargs['baseline'], baseline)
            broker.return_value.submit.assert_not_called()
            environments = execute.call_args.kwargs['environments']
            self.assertEqual(set(environments), {'open_cake', comparison})
            self.assertIs(environments['open_cake']._toolchain, environments[comparison]._toolchain)
            self.assertIs(environments['open_cake']._toolchain._isolated, toolchain.return_value)
            self.assertEqual(environments[comparison].media_type, policy.media_type)
            self.assertEqual(set(execute.call_args.kwargs['provider']._builders), set(lock.run_order))

            if comparison == 'native_triton':
                invalid_workload = workload.document
                invalid_workload['semantics']['epsilon'] = 0
                invalid_workload_path = project / 'contracts/workloads/invalid-rmsnorm.json'
                invalid_workload_path.write_bytes(encoded(invalid_workload))
                invalid_study = copy.deepcopy(study.document)
                invalid_study['workload'] = {'path': invalid_workload_path.relative_to(project).as_posix(),
                    'canonical_sha256': sha256(encoded(invalid_workload)).hexdigest()}
                invalid_study_path = self.output / 'invalid-rmsnorm-study.json'
                invalid_study_path.write_bytes(encoded(invalid_study))
                with self.assertRaisesRegex(ValueError, 'RMSNorm epsilon'):
                    Lab(project).preflight(invalid_study_path, execution_bindings_path=bp)

            # Missing policy cannot mint a new live native Campaign under an old-looking template.
            unpaired = copy.deepcopy(study.document)
            del unpaired['evaluation_protocol']['paired_timing']
            unpaired_path = self.output / 'unpaired-study.json'; unpaired_path.write_bytes(encoded(unpaired))
            with self.assertRaisesRegex(ValueError, 'new live native Campaign'):
                Lab(project).preflight(unpaired_path, execution_bindings_path=bp)
            anchor['qualification_receipt_sha256'] = '0'*64
            ap.write_bytes(encoded(anchor))
            with self.assertRaisesRegex(ValueError, 'anchor evidence'):
                Lab(project).preflight(template, execution_bindings_path=bp)
            anchor['qualification_receipt_sha256'] = sha256(encoded(qualification)).hexdigest()
            ap.write_bytes(encoded(anchor))
            (self.output / 'cubin').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'artifact bytes'):
                Lab(project).preflight(template, execution_bindings_path=bp)



if __name__ == '__main__':
    unittest.main()
