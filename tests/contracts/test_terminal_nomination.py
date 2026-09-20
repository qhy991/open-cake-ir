"""Terminal confirmation uses one unchanged artifact after all authoring/search."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from open_cake_ir.evidence import EvidenceStore
from open_cake_ir.lab.provider_policy import execution_configuration
from open_cake_ir.lab.replay import replay_matched_run
from open_cake_ir.serialization import canonical_json_bytes as encoded
from tests.contracts import test_run_specification as run_fixture
from tests.contracts.test_lab import SemanticLabTestCase, FakeEnvironment, FakeEvaluator, RalphFakeProvider

ROOT = Path(__file__).resolve().parents[2]


class EarlierSearchEvaluator(FakeEvaluator):
    def latency_ms(self, arm, turn, purpose):
        return (0.4 if turn == 1 else 0.9) if purpose == 'search' else 1.2


class NominationTests(SemanticLabTestCase):
    def execute(self, evaluator_class=EarlierSearchEvaluator, *, token_limit=160000,
                clock=None, budget_updates=None, endpoint_policy=None):
        lab,spec = run_fixture.IndependentRunTests.fixture(self)
        if clock is not None:
            lab = type(lab)(ROOT,clock=clock)
        document = spec.document
        document['budget'].update(limit=token_limit,checkpoints=[80000,token_limit])
        document['budget'].update(budget_updates or {})
        if endpoint_policy is not None:
            document['endpoint_policy'] = endpoint_policy
        spec = run_fixture.RunSpecification.from_dict(document)
        provider = RalphFakeProvider({spec.run_id:lab.task_package(spec,spec.run_id)})
        provider.configuration = execution_configuration(document['authoring']['provider'])
        provider.qualification_sha256 = document['authoring']['provider']['qualification']['canonical_sha256']
        evaluator = evaluator_class(document['evaluation_protocol'],
            sha256(encoded(document['evaluation_protocol'])).hexdigest(),document['workload']['canonical_sha256'])
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        run = lab.execute_run(spec,Path(temporary.name)/'evidence',provider=provider,
            environment=FakeEnvironment('open_cake',document['authoring']),evaluator=evaluator)
        audit,replay = lab.audit_run(run)
        self.assertTrue(replay,replay.refusals)
        evidence = EvidenceStore.open(run.evidence_root)
        events = evidence.replay_events(spec.run_id)
        return lab,spec,run,audit,evidence,events,provider

    def test_earlier_search_winner_is_confirmed_once_after_all_search_at_its_actual_total_cost(self):
        _,spec,_,audit,_,events,provider = self.execute()
        nominee = next(e['payload'] for e in events if e['kind']=='candidate_nominated')
        self.assertEqual(nominee['source_turn'],1)
        confirmations = [e['payload'] for e in events if e['kind']=='candidate_evaluated'
                         and e['payload']['purpose']=='confirmatory']
        self.assertEqual(len(confirmations),1)
        self.assertEqual(confirmations[0]['source_turn'],1)
        self.assertNotIn('turn',confirmations[0])
        self.assertEqual(confirmations[0]['candidate_sha256'],nominee['candidate_sha256'])
        end = next(i for i,e in enumerate(events) if e['kind']=='search_completed')
        self.assertTrue(all(e['kind'] not in {'provider_turn_completed','candidate_set_filtered'} for e in events[end+1:]))
        self.assertEqual(len(provider.requests),2)
        self.assertTrue(all('confirmed_latency_ms' not in request.feedback for request in provider.requests))
        self.assertEqual(audit.endpoint['best_confirmed_latency_ms'],1.2)
        self.assertEqual(audit.endpoint['budget'],160000)
        checkpoints = next(e['payload']['checkpoints'] for e in events if e['kind']=='checkpoints_projected')
        self.assertEqual(checkpoints[0]['best_search_latency_ms'],0.4)
        self.assertNotIn('best_confirmed_latency_ms',checkpoints[0])

    def test_confirmation_failure_does_not_nominate_or_evaluate_a_replacement(self):
        class Failing(EarlierSearchEvaluator):
            def evaluate(self,candidate,*,case_id,purpose):
                if purpose == 'confirmatory':
                    raise RuntimeError('CPU fixture confirmation unavailable')
                return super().evaluate(candidate,case_id=case_id,purpose=purpose)
        _,_,_,audit,_,events,provider = self.execute(Failing)
        self.assertEqual(audit.protocol_adherence,'broker_fault')
        self.assertEqual(audit.endpoint_observation,'missing')
        self.assertEqual(sum(e['kind']=='candidate_nominated' for e in events),1)
        starts = [e['payload'] for e in events if e['kind']=='evaluation_attempt_started'
                  and e['payload']['purpose']=='confirmatory']
        self.assertEqual(len(starts),1)
        self.assertEqual(starts[0]['source_turn'],1)
        self.assertEqual(len(provider.requests),2)

    def test_numerically_incorrect_confirmation_is_observed_without_using_search_latency(self):
        class Incorrect(EarlierSearchEvaluator):
            def evaluate(self,candidate,*,case_id,purpose):
                attempt = super().evaluate(candidate,case_id=case_id,purpose=purpose)
                if purpose != 'confirmatory': return attempt
                correctness = {'tie_aware_distance_match':False}
                receipt = replace(attempt.final_receipt,correctness_passed=False,correctness=correctness,
                    artifact_payloads={**attempt.final_receipt.artifact_payloads,
                                       'correctness_output':encoded(correctness)})
                broker = attempt.attempts[0]
                raw = json.loads(broker.artifact_payloads['broker_record'])
                raw['receipt'].update(correctness_passed=False,correctness=correctness)
                broker = replace(broker,receipt=receipt,artifact_payloads={**broker.artifact_payloads,
                    'broker_record':encoded(raw),'evaluator_result':encoded(raw)})
                return replace(attempt,attempts=(broker,),final_receipt=receipt)
        _,_,_,audit,_,events,_ = self.execute(Incorrect)
        self.assertEqual(audit.protocol_adherence,'adhered')
        self.assertEqual(audit.endpoint_observation,'no_qualified_candidate')
        self.assertNotIn('best_confirmed_latency_ms',audit.endpoint)
        self.assertEqual(sum(e['kind']=='candidate_nominated' for e in events),1)
        self.assertEqual(sum(e['kind']=='evaluation_attempt_started' and e['payload']['purpose']=='confirmatory'
                             for e in events),1)

    def test_replay_refuses_posthoc_nominee_changes_and_search_after_nomination(self):
        lab,spec,_,audit,evidence,events,_ = self.execute()
        for mutation in ('source','artifact','duplicate','resume'):
            changed = deepcopy(list(events))
            index = next(i for i,e in enumerate(changed) if e['kind']=='candidate_nominated')
            if mutation == 'source': changed[index]['payload']['source_turn'] = 2
            elif mutation == 'artifact': changed[index]['payload']['candidate_record_sha256'] = '0'*64
            elif mutation == 'duplicate': changed.insert(index+1,deepcopy(changed[index]))
            else:
                selected = next(e for e in changed if e['kind']=='candidate_selected')
                changed.insert(index+1,deepcopy(selected))
            with self.subTest(mutation=mutation),patch.object(evidence,'replay_events',return_value=changed):
                replay = replay_matched_run(evidence,audit,spec,project_root=ROOT,
                    manifest_parser=lab._parse_manifest,task_package=lab.task_package)
                self.assertFalse(replay)

    def test_overspending_cannot_claim_success_at_the_earlier_nominees_token_cost(self):
        _,_,_,audit,_,events,_ = self.execute(token_limit=150000)
        self.assertEqual(audit.endpoint_observation,'no_qualified_candidate')
        nominee = next(e['payload'] for e in events if e['kind']=='candidate_nominated')
        self.assertEqual(nominee['source_turn'],1)
        self.assertEqual(events[-2]['payload']['ralph']['cumulative_provider_tokens'],160000)

    def test_live_writer_rejects_a_search_job_relabelled_as_fresh_confirmation(self):
        class Cached(EarlierSearchEvaluator):
            def evaluate(self,candidate,*,case_id,purpose):
                if purpose != 'confirmatory':
                    result = super().evaluate(candidate,case_id=case_id,purpose=purpose)
                    if purpose == 'search' and candidate.entry_point.endswith('_turn_1'):
                        self.cached = result
                    return result
                receipt = replace(self.cached.final_receipt,purpose='confirmatory')
                return replace(self.cached,final_receipt=receipt,
                    attempts=(replace(self.cached.attempts[0],receipt=receipt),))
        _,_,_,audit,_,events,_ = self.execute(Cached)
        self.assertEqual(audit.protocol_adherence,'harness_fault')
        self.assertEqual(audit.endpoint_observation,'missing')
        fault = next(e['payload'] for e in events if e['kind']=='run_fault')
        self.assertIn('reused an earlier broker job',fault['exception_message'])
        self.assertIn('rejected_evaluation_receipt',{ref['role'] for ref in fault['objects']})

    def test_raw_confirmation_request_and_job_identity_are_independently_checked(self):
        from open_cake_ir.lab.replay.attempts import _replay_evaluation_attempt_event
        from open_cake_ir.lab.replay.artifacts import _replay_launchable_candidate
        from open_cake_ir.lab.replay.refusals import ReplayRefusal
        from open_cake_ir.lab.replay.artifacts import _replay_evaluation_receipt
        lab,spec,_,_,evidence,events,_ = self.execute()
        nomination = next(e['payload'] for e in events if e['kind']=='candidate_nominated')
        candidate = _replay_launchable_candidate(evidence,
            [e for e in events if e['kind']=='launchable_candidate_sealed'],turn=nomination['source_turn'],
            candidate_sha256=nomination['candidate_sha256'],arm='open_cake',manifest_parser=lab._parse_manifest)
        evaluated = next(e['payload'] for e in events if e['kind']=='candidate_evaluated' and e['payload']['purpose']=='confirmatory')
        protocol_sha = sha256(encoded(spec.document['evaluation_protocol'])).hexdigest()
        case_id = spec.document['evaluation_protocol']['case_id']
        receipt = _replay_evaluation_receipt(evidence,evaluated,launchable=candidate,
            candidate_sha256=candidate.candidate_sha256,workload_sha256=spec.document['workload']['canonical_sha256'],
            protocol_sha256=protocol_sha,case_id=case_id,purpose='confirmatory',
            evaluation_protocol=spec.document['evaluation_protocol'],fixed_baseline=None,location='test')
        payload = next(e['payload'] for e in events if e['kind']=='evaluation_attempt_completed' and e['payload']['purpose']=='confirmatory')
        jobs = set()
        arguments = dict(candidate=candidate,protocol_sha256=protocol_sha,
            compiler_reference=spec.document['compiler_revision'],final_receipt=receipt,case_id=case_id)
        _replay_evaluation_attempt_event(evidence,payload,used_job_ids=jobs,**arguments)
        with self.assertRaisesRegex(ReplayRefusal,'reused across'):
            _replay_evaluation_attempt_event(evidence,payload,used_job_ids=jobs,**arguments)
        for field in ('purpose','case_id'):
            with self.subTest(field=field),self.assertRaisesRegex(ReplayRefusal,'worker request authority'):
                _replay_evaluation_attempt_event(evidence,{**payload,'purpose':'search'} if field=='purpose' else payload,
                    used_job_ids=set(),**{**arguments,**({'case_id':'foreign-case'} if field=='case_id' else {})})
        ledger_ref = next(ref for ref in payload['objects'] if ref['role']=='broker_attempt_ledger')
        ledger = json.loads(evidence.read_object(ledger_ref))
        original_read = evidence.read_object
        for invalid in (None,7):
            with self.subTest(attempts=invalid),patch.object(evidence,'read_object',side_effect=
                    lambda ref:encoded({**ledger,'attempts':invalid}) if ref==ledger_ref else original_read(ref)):
                with self.assertRaisesRegex(ReplayRefusal,'cardinality'):
                    _replay_evaluation_attempt_event(evidence,payload,used_job_ids=set(),**arguments)

    def test_confirmation_timestamp_stays_between_search_end_and_run_end(self):
        lab,spec,_,audit,evidence,events,_ = self.execute()
        end = events[-2]['payload']['ralph']['elapsed_wall_seconds']
        for elapsed in (0,end+1):
            changed = deepcopy(list(events))
            next(e['payload'] for e in changed if e['kind']=='candidate_evaluated'
                 and e['payload']['purpose']=='confirmatory')['elapsed_wall_seconds'] = elapsed
            with self.subTest(elapsed=elapsed),patch.object(evidence,'replay_events',return_value=changed):
                replay = replay_matched_run(evidence,audit,spec,project_root=ROOT,
                    manifest_parser=lab._parse_manifest,task_package=lab.task_package)
            self.assertFalse(replay)
            self.assertIn('confirmation time lies outside',str(replay.refusals[0]))
