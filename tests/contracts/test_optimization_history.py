"""Author history survives lost conversation context and remains evidence-derived."""
import copy
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from hashlib import sha256

from open_cake_ir.evaluation import EvaluationReceipt
from open_cake_ir.lab.optimization_history import (
    evaluated_observation, optimization_history, profile_observation,
)
from open_cake_ir.lab.replay.history import replay_optimization_history
from open_cake_ir.lab.replay.refusals import ReplayRefusal
from open_cake_ir.lab.diagnoses import findings_feedback

ROOT=Path(__file__).resolve().parents[2]


def receipt(letter, latency, *, quality=True, correct=True):
    return EvaluationReceipt(letter*64,'f'*64,'e'*64,'search','primary',correct,
        {'matched':correct},1,0,'d'*64,
        {'measurement_quality_passed':quality,'pooled_median_ms':latency,
         'classification':'first_arm_faster' if quality else 'measurement_quality_failed',
         'speedup':2/latency,'pooled_medians_ms':{'candidate':latency,'baseline':2.0}})


class HistoryProjectionTests(unittest.TestCase):
    def test_long_unicode_best_diagnostics_keep_coverage_and_fit_history(self):
        finding = {key: '字' * 512 for key in ('code', 'path', 'category', 'severity', 'message')}
        finding['blocks_lowering'] = False
        diagnostics = findings_feedback([finding] * 10)
        row = evaluated_observation(1, receipt('a', 1.0), diagnostics=diagnostics)
        view = optimization_history([row], [], [])
        kept = view['best_qualified_search']['diagnostics']
        self.assertEqual(len(kept['findings']), 2)
        self.assertEqual(kept['omitted_findings'], 8)
        self.assertLessEqual(len(json.dumps(view, ensure_ascii=False).encode()), 24576)
        self.assertEqual(len(diagnostics['findings']), 8)

    def test_frozen_nested_receipts_and_unstable_values_remain_distinct(self):
        fast=receipt('a',1.0);slow=receipt('b',1.8);unstable=receipt('c',0.01,quality=False)
        wrong=receipt('d',0.005,correct=False)
        rows=[evaluated_observation(1,r) for r in (fast,slow,unstable,wrong)]
        view=optimization_history(rows,[],[])
        self.assertEqual(len(view['evaluations']),4)
        self.assertEqual(view['best_qualified_search']['candidate_sha256'],'a'*64)
        self.assertIsNone(rows[2]['latency_ms']);self.assertIsNone(rows[2]['baseline_comparison'])
        self.assertIsNone(rows[3]['latency_ms'])
        self.assertEqual(rows[0]['baseline_comparison']['pooled_medians_ms']['baseline'],2.0)
        view['evaluations'][0]['baseline_comparison']['pooled_medians_ms']['candidate']=99
        self.assertEqual(fast.timing['pooled_medians_ms']['candidate'],1.0)
        self.assertEqual(rows[0]['baseline_comparison']['pooled_medians_ms']['candidate'],1.0)

    def test_bounds_keep_old_best_and_expose_omissions(self):
        rows=[evaluated_observation(i+1,receipt('a',1+i)) for i in range(30)]
        view=optimization_history(rows,[],[])
        self.assertEqual(view['best_qualified_search']['turn'],1)
        self.assertEqual([r['turn'] for r in view['evaluations']],list(range(19,31)))
        self.assertEqual(view['omitted']['evaluations'],18)
        profile=profile_observation(SimpleNamespace(attribution_feedback={'kind':'large_program','stages':['x'*5000]*50}))
        self.assertTrue(profile['summary_omitted'])
        rejections=[{'turn':i,'findings':[{'message':'错误'*512}]*8} for i in range(30)]
        bounded=optimization_history(rows,rejections,[])
        self.assertLessEqual(len(json.dumps(bounded,ensure_ascii=False).encode()),24576)
        self.assertEqual(bounded['omitted']['rejections']+len(bounded['rejections']),30)

    def test_replay_refuses_invented_or_future_history_and_skips_terminal_profile(self):
        a,b=receipt('a',1),receipt('b',0.5)
        records={(1,'search','a'*64):a,(2,'search','b'*64):b}
        expected=optimization_history([evaluated_observation(1,a)],[],[])
        bundles={'turn2':json.dumps({'state_card':{'optimization_history':expected}}).encode()}
        evidence=SimpleNamespace(read_object=lambda ref:bundles[ref['key']])
        events=[{'sequence':0,'kind':'candidate_set_filtered','payload':{'turn':1,'order':[
                    {'candidate_sha256':'a'*64,'diagnostics':findings_feedback([])}]}},
                {'sequence':1,'kind':'candidate_evaluated','payload':{'turn':1,'purpose':'search','candidate_sha256':'a'*64}},
                {'sequence':2,'kind':'provider_turn_completed','payload':{'turn':2,'objects':[{'role':'provider_reference_bundle','key':'turn2'}]}},
                {'sequence':3,'kind':'candidate_set_filtered','payload':{'turn':2,'order':[
                    {'candidate_sha256':'b'*64,'diagnostics':findings_feedback([])}]}},
                {'sequence':4,'kind':'candidate_evaluated','payload':{'turn':2,'purpose':'search','candidate_sha256':'b'*64}},
                {'sequence':5,'kind':'candidate_evaluated','payload':{'source_turn':2,'purpose':'attribution','candidate_sha256':'b'*64}}]
        replay_optimization_history(events=events,evidence=evidence,receipts=records,arm='open_cake')
        future=optimization_history([evaluated_observation(1,a),evaluated_observation(2,b)],[],[])
        bundles['turn2']=json.dumps({'state_card':{'optimization_history':future}}).encode()
        with self.assertRaisesRegex(ReplayRefusal,'earlier Run-local'):
            replay_optimization_history(events=events,evidence=evidence,receipts=records,arm='open_cake')
        forged=copy.deepcopy(expected);forged['evaluations'][0]['latency_ms']=0.01
        bundles['turn2']=json.dumps({'state_card':{'optimization_history':forged}}).encode()
        with self.assertRaises(ReplayRefusal):
            replay_optimization_history(events=events,evidence=evidence,receipts=records,arm='open_cake')

    def test_replay_binds_diagnostics_to_preceding_candidate_and_turn(self):
        a = receipt('a', 1.0)
        diagnostics = findings_feedback([{'code': 'PRESSURE', 'path': 'buffers[2]',
            'blocks_lowering': False, 'message': 'Measured allocation is still required.'}])
        expected = optimization_history([evaluated_observation(1, a, diagnostics=diagnostics)], [], [])
        bundle = {'state_card': {'optimization_history': expected}}
        evidence = SimpleNamespace(read_object=lambda ref: json.dumps(bundle).encode())
        events = [
            {'sequence': 0, 'kind': 'candidate_set_filtered', 'payload': {'turn': 1, 'order': [
                {'candidate_sha256': 'b'*64, 'diagnostics': findings_feedback([])},
                {'candidate_sha256': 'a'*64, 'diagnostics': diagnostics}]}},
            {'sequence': 1, 'kind': 'candidate_evaluated', 'payload': {
                'turn': 1, 'purpose': 'search', 'candidate_sha256': 'a'*64}},
            {'sequence': 2, 'kind': 'provider_turn_completed', 'payload': {'turn': 2,
                'objects': [{'role': 'provider_reference_bundle'}]}},
        ]
        records = {(1, 'search', 'a'*64): a}
        replay_optimization_history(events=events, evidence=evidence, receipts=records, arm='open_cake')
        forged = copy.deepcopy(expected)
        forged['evaluations'][0]['diagnostics']['findings'][0]['code'] = 'BORROWED_DIAGNOSIS'
        bundle['state_card']['optimization_history'] = forged
        with self.assertRaisesRegex(ReplayRefusal, 'earlier Run-local'):
            replay_optimization_history(events=events, evidence=evidence, receipts=records, arm='open_cake')
        with self.assertRaisesRegex(ReplayRefusal, 'preceding candidate-bound'):
            replay_optimization_history(events=events[1:], evidence=evidence, receipts=records, arm='open_cake')

    def test_failed_message_request_history_is_checked_without_a_completed_turn(self):
        specification = SimpleNamespace(document={'authoring': {
            'provider': {'event_contract': 'responses_messages_v1'}}})
        a = receipt('a', 1.0)
        prior = [
            {'sequence': 0, 'kind': 'candidate_set_filtered', 'payload': {'turn': 1, 'order': [
                {'candidate_sha256': 'a'*64, 'diagnostics': findings_feedback([])}]}},
            {'sequence': 1, 'kind': 'candidate_evaluated', 'payload': {
                'turn': 1, 'purpose': 'search', 'candidate_sha256': 'a'*64}},
        ]
        for previous in (False, True):
            with self.subTest(previous=previous):
                expected = optimization_history([evaluated_observation(1, a)] if previous else [], [], [])
                state = {'optimization_history': expected}
                evidence = SimpleNamespace(read_object=lambda ref: json.dumps({'request': {
                    'input': [{'role': 'user', 'content': json.dumps({'state_card': state})}]}}).encode())
                events = (prior if previous else []) + [
                    {'sequence': 2 if previous else 0, 'kind': 'run_fault', 'payload': {
                        'turn': 2 if previous else 1, 'stage': 'provider',
                        'objects': [{'role': 'provider_stdout'}]}}]
                records = {(1, 'search', 'a'*64): a} if previous else {}
                replay_optimization_history(events=events, evidence=evidence, receipts=records,
                    arm='open_cake', specification=specification)
                forged = copy.deepcopy(expected)
                forged['total']['evaluations'] = 999
                state['optimization_history'] = forged
                with self.assertRaisesRegex(ReplayRefusal, 'earlier Run-local'):
                    replay_optimization_history(events=events, evidence=evidence, receipts=records,
                        arm='open_cake', specification=specification)


class HistoryRunTests(unittest.TestCase):
    def setUp(self):
        from tests.contracts.test_diagnosis_feedback import DiagnosisRunTests
        DiagnosisRunTests.setUp(self)

    def test_real_loop_delivers_all_survivors_across_turns_and_replays(self):
        from open_cake_ir.tasks.runtime import TaskLab
        from tests.contracts.test_lab import FakeProvider,FakeEnvironment,FakeEvaluator,_execute,_enable_candidate_set,_submission_envelope
        class Pairs(FakeProvider):
            def turn(self,request):
                result=super().turn(request)
                values=tuple(json.dumps({'run_id':request.run_id,'turn':request.turn,'variant':i},sort_keys=True,separators=(',',':')).encode() for i in range(2))
                return dataclasses.replace(result,candidates=values,candidate_sha256s=tuple(sha256(v).hexdigest() for v in values),
                    raw_submission=_submission_envelope(request.arm,values))
        with tempfile.TemporaryDirectory() as directory:
            document=json.loads((ROOT/'contracts/studies/matched-search-infrastructure-template.json').read_text())
            _enable_candidate_set(document,2)
            document['budget']['maximum_turns']=3
            document['budget']['limit']=500000
            document['budget']['checkpoints']=[500000]
            document['evaluation_protocol']['searches_per_turn']=2
            path=Path(directory)/'study.json';path.write_text(json.dumps(document))
            lab=TaskLab(ROOT);lock=lab.preflight(path);provider=Pairs()
            protocol=lock.document['evaluation_protocol']
            evaluator=FakeEvaluator(protocol,sha256(json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()).hexdigest(),lock.document['workload']['canonical_sha256'])
            campaign=_execute(lab,lock,Path(directory)/'evidence',provider=provider,
                environments={arm:FakeEnvironment(arm,authority) for arm,authority in lock.document['resolved_inputs']['arm_environments'].items()},evaluator=evaluator)
            third=[r for r in provider.requests if r.turn==3]
            self.assertEqual(len(third),len(lock.run_order))
            for request in third:
                history=request.state_card['optimization_history']
                self.assertEqual(history['total']['evaluations'],4)
                self.assertEqual([r['turn'] for r in history['evaluations']],[1,1,2,2])
                self.assertEqual(history['best_qualified_search']['turn'],2)
                self.assertEqual(history['scope'],'current_run_observations_only')
            self.assertTrue(lab.audit(campaign).semantic_replay_passed)
