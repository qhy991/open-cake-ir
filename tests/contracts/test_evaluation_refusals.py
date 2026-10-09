"""CPU replay of observed allocation and zero-dispatch MACA resource refusals."""
from copy import deepcopy
import dataclasses
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import grp
import os
import pwd
import sys

from open_cake_ir.evaluation.refusals import (
    EvaluationRefusal, is_local_allocation_refusal, refusal_from_artifacts,
)
from open_cake_ir.evaluation import LogicalEvaluationAttempt
from open_cake_ir.serialization import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[2]
COUNTERS = {'compiler_invocations': 0, 'module_loads': 0, 'preflight_calls': 0,
            'kernel_calls': 0, 'timing_samples': 0, 'fallback_calls': 0}


def busy_result(prefix='maca', suffix='401a2870deb1'):
    return {'admitted': False, 'counters': dict(COUNTERS), 'error': f'{prefix}_broker_busy',
            'failure_class': 'admission', 'job_id': f'{prefix}-{suffix}',
            'mode': 'local_serialized', 'receipt': None, 'schema_version': 1}


class RefusalEvidenceTests(unittest.TestCase):
    def test_retained_busy_result_is_an_unmeasured_allocation_refusal(self):
        result = busy_result()
        self.assertTrue(is_local_allocation_refusal(result, b'', b'', target='xcore1002'))
        for field, value in [('admitted', True), ('receipt', {}), ('failure_class', 'RuntimeError'),
                             ('error', 'device_lost'), ('job_id', 'hip-401a2870deb1')]:
            self.assertFalse(is_local_allocation_refusal({**result, field: value}, b'', b'', target='xcore1002'))
        for value in (1, False, 0.0):
            changed = deepcopy(result); changed['counters']['kernel_calls'] = value
            self.assertFalse(is_local_allocation_refusal(changed, b'', b'', target='xcore1002'))
        self.assertFalse(is_local_allocation_refusal(result, b'unknown output', b'', target='xcore1002'))
        self.assertFalse(is_local_allocation_refusal(result, b'', b'[maca-run] accepted job maca-401a2870deb1\n', target='xcore1002'))
        self.assertFalse(is_local_allocation_refusal(result, b'', b'', target='sm_100a'))

    def resource_artifacts(self):
        result = {'schema_version': 2, 'job_id': 'maca-f60212b5d2de', 'mode': 'local_serialized',
            'admitted': True, 'error': 'evaluator_failed', 'failure_class': 'MetaxLaunchResourceError',
            'receipt': None, 'counters': {**COUNTERS, 'module_loads': 5},
            'failure_artifacts': {'launch_resource': 'failure-launch_resource.bin'}}
        diagnostic = {'kind': 'maca_launch_resource_failure_v1', 'operation': 'mcModuleLaunchKernel',
            'status': 32, 'status_name': 'mcErrorMemoryValueTooLarge', 'completed_target_calls': 0,
            'purpose': 'search', 'arm': 'candidate', 'phase': 'preflight', 'teardown_completed': True,
            'job_id': result['job_id'], 'coverage': 'failed_dispatch_no_correctness_or_timing_receipt',
            'resources': {'dynamic_shared_bytes': 0, 'local_bytes': 16320, 'registers_per_thread': 256},
            'candidate_sha256': 'c'*64, 'target': 'xcore1002', 'input_case_id': 'primary'}
        request = {'candidate_sha256': 'c'*64, 'target': 'xcore1002', 'purpose': 'search', 'case_id': 'primary'}
        raw = {'evaluator_result': canonical_json_bytes(result), 'evaluator_request': canonical_json_bytes(request),
            'failure_launch_resource': canonical_json_bytes(diagnostic),
            'stdout': b'Error in allocating private memory. The system is set to :4 KB/Thread,kernel request:16 KB/thread\n',
            'stderr': b'[maca-run] accepted job maca-f60212b5d2de\nMetaxLaunchResourceError: MACA mcModuleLaunchKernel failed with status 32 (mcErrorMemoryValueTooLarge)\n'}
        return raw, diagnostic

    def test_actual_16320_byte_failure_matches_the_sdk_16kb_display(self):
        raw, _ = self.resource_artifacts()
        candidate = SimpleNamespace(candidate_sha256='c'*64, target='xcore1002')
        refusal = refusal_from_artifacts(raw, candidate=candidate, case_id='primary',
            purpose='search', job_id='maca-f60212b5d2de')
        self.assertIsInstance(refusal, EvaluationRefusal)
        self.assertEqual(refusal.disposition, 'candidate_resource_rejected')
        self.assertIsNone(refusal.correctness_passed)
        self.assertIsNone(refusal.timing)
        self.assertIn('16320', refusal.diagnostic)

    def test_cleanup_partial_dispatch_other_device_and_unknown_errors_remain_faults(self):
        raw, diagnostic = self.resource_artifacts()
        candidate = SimpleNamespace(candidate_sha256='c'*64, target='xcore1002')
        for field, value in [('teardown_completed', False), ('completed_target_calls', 1),
                             ('status', 1009), ('arm', 'baseline'), ('phase', 'postflight'),
                             ('candidate_sha256', 'd'*64), ('target', 'sm_100a'),
                             ('input_case_id', 'another_case')]:
            changed = {**raw, 'failure_launch_resource': canonical_json_bytes({**diagnostic, field: value})}
            with self.subTest(field=field):
                self.assertIsNone(refusal_from_artifacts(changed, candidate=candidate,
                    case_id='primary', purpose='search', job_id='maca-f60212b5d2de'))
        for field, value in [('stdout', raw['stdout'].replace(b'request:16', b'request:160')),
                             ('stderr', raw['stderr']+b'RuntimeError: device lost\n')]:
            self.assertIsNone(refusal_from_artifacts({**raw, field: value}, candidate=candidate,
                case_id='primary', purpose='search', job_id='maca-f60212b5d2de'))
        self.assertIsNone(refusal_from_artifacts(raw, candidate=candidate,
            case_id='primary', purpose='confirmatory', job_id='maca-f60212b5d2de'))


class RefusalRunTests(unittest.TestCase):
    def setUp(self):
        from tests.contracts.test_diagnosis_feedback import DiagnosisRunTests
        DiagnosisRunTests.setUp(self)

    def test_search_and_profile_refusals_reach_next_turn_and_independent_replay(self):
        from open_cake_ir.tasks.runtime import TaskLab
        from tests.contracts.test_lab import FakeEvaluator, FakeProvider, FakeEnvironment, _execute
        from open_cake_ir.evidence import EvidenceStore
        for refused_purpose in ('search', 'attribution'):
            class OnceBusy(FakeEvaluator):
                def evaluate(self, candidate, *, case_id, purpose):
                    outcome = super().evaluate(candidate, case_id=case_id, purpose=purpose)
                    if self.candidate_position(candidate)[1] != 1 or purpose != refused_purpose:
                        return outcome
                    previous = outcome.attempts[0]
                    result = busy_result('cuda', f'{self.calls:012x}')
                    request = json.loads(previous.artifact_payloads['evaluator_request'])
                    request['target'] = candidate.target
                    authority = {k: v for k, v in request.items() if k != 'attempt'}
                    raw = canonical_json_bytes(result)
                    attempt = dataclasses.replace(previous, job_id=result['job_id'], mode='local_serialized',
                        admitted=False, error=result['error'], receipt=None, **COUNTERS,
                        evaluator_arguments_sha256=sha256(canonical_json_bytes(authority)).hexdigest(),
                        artifact_payloads={'broker_record': raw, 'evaluator_result': raw,
                            'evaluator_request': canonical_json_bytes(request), 'stdout': b'', 'stderr': b''})
                    return LogicalEvaluationAttempt(candidate.candidate_sha256, (attempt,), None)

            with self.subTest(purpose=refused_purpose), tempfile.TemporaryDirectory() as directory:
                document = json.loads((ROOT/'contracts/studies/matched-search-infrastructure-template.json').read_text())
                document['budget'].update(maximum_turns=3, limit=240000, checkpoints=[80000, 160000, 240000])
                path = Path(directory)/'study.json'; path.write_text(json.dumps(document))
                lab = TaskLab(ROOT); lock = lab.preflight(path); provider = FakeProvider()
                protocol = lock.document['evaluation_protocol']
                evaluator = OnceBusy(protocol, sha256(canonical_json_bytes(protocol)).hexdigest(), lock.document['workload']['canonical_sha256'])
                campaign = _execute(lab, lock, Path(directory)/'evidence', provider=provider,
                    environments={arm: FakeEnvironment(arm, authority) for arm, authority in lock.document['resolved_inputs']['arm_environments'].items()}, evaluator=evaluator)
                second = [request for request in provider.requests if request.turn == 2]
                self.assertEqual(len(second), len(lock.run_order))
                for request in second:
                    row = request.state_card['optimization_history']['evaluations'][0]
                    self.assertFalse(row['search_qualified'])
                    self.assertIsNone(row['latency_ms'])
                    feedback = request.state_card['previous_feedback']
                    self.assertIn('evaluation_refusal' if refused_purpose == 'search' else 'attribution_refusal', feedback)
                report = lab.audit(campaign)
                self.assertTrue(report.semantic_replay_passed)
                evidence = EvidenceStore.open(Path(directory)/'evidence')
                for run_id in lock.run_order:
                    events = evidence.replay_events(run_id)
                    refusals = [event for event in events if event['kind'] == 'evaluation_refused']
                    self.assertEqual(len(refusals), 1)
                    self.assertEqual(refusals[0]['payload']['purpose'], refused_purpose)
                    self.assertFalse(any(event['kind'] == 'run_fault' for event in events))


from tests.contracts.test_lab import SemanticLabTestCase


class RefusalTransportTests(SemanticLabTestCase):
    def test_original_empty_stderr_busy_envelope_is_retained_without_resubmission(self):
        from open_cake_ir.evaluation import LaunchableCandidate
        from open_cake_ir.lab.runtime import CommandBrokerSubmitter, BoundedBrokerEvaluator
        from open_cake_ir.lab.faults import RunProtocolFault
        from tests.contracts._executor_fixture import compiler_reference

        payloads = {'mcfatbin': b'CPU transport fixture, never loaded', 'launch_manifest': b'CPU manifest fixture'}
        candidate = LaunchableCandidate('c'*64, 'xcore1002', 'kernel',
            {role: sha256(value).hexdigest() for role, value in payloads.items()},
            sha256(payloads['launch_manifest']).hexdigest(), payloads)
        workload_path = ROOT/'contracts/workloads/flash-kmeans-assign-v2.json'
        workload = json.loads(workload_path.read_text())
        protocol = {'kind': 'CPU transport fixture'}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root/'worker.py'
            record = root/'record.json'
            worker.write_text('''import argparse,json,pathlib
p=argparse.ArgumentParser();p.add_argument('--record');p.add_argument('--request');p.add_argument('--output');a=p.parse_args()
pathlib.Path(a.output).write_bytes(pathlib.Path(a.record).read_bytes())
''')
            submitter = CommandBrokerSubmitter(command=(sys.executable, str(worker), '--record', str(record)),
                workload_path=workload_path, workload_sha256=sha256(canonical_json_bytes(workload)).hexdigest(),
                protocol_sha256=sha256(canonical_json_bytes(protocol)).hexdigest(), cwd=root,
                executor=self.executor_fixture.revision(ROOT), compiler_reference=compiler_reference(ROOT),
                service_user=pwd.getpwuid(os.geteuid()).pw_name, service_group=grp.getgrgid(os.getegid()).gr_name)
            original = canonical_json_bytes(busy_result())
            record.write_bytes(original)
            outcome = BoundedBrokerEvaluator(protocol, submitter).evaluate(candidate, case_id='primary', purpose='search')
            self.assertEqual(len(outcome.attempts), 1)
            self.assertIsNone(outcome.final_receipt)
            self.assertFalse(outcome.attempts[0].admitted)
            self.assertEqual(outcome.attempts[0].artifact_payloads['evaluator_result'], original)
            self.assertEqual(outcome.attempts[0].artifact_payloads['stderr'], b'')
            invalid = busy_result(); invalid['counters']['kernel_calls'] = 1
            record.write_bytes(canonical_json_bytes(invalid))
            with self.assertRaisesRegex(RunProtocolFault, 'observation coverage differs'):
                submitter.submit(candidate, case_id='primary', purpose='search', attempt=1)


if __name__ == '__main__':
    unittest.main()
