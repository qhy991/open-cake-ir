"""A fixed confirmation reserve and truthful rejection of atomic-call overshoot."""
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

from open_cake_ir.lab import RunSpecification
from open_cake_ir.lab.endpoints import NORMAL_BUDGET_TERMINAL
from open_cake_ir.lab.replay import replay_matched_run
from tests.contracts import test_terminal_nomination as fixture
from tests.contracts import test_run_specification as run_fixture
from tests.contracts.test_lab import SemanticLabTestCase


class ConfirmationTimeBudgetTests(SemanticLabTestCase):
    def execute(self, *, search_seconds, confirmation_seconds, turns=2, reserve_seconds=4.):
        class Clock:
            value = 0.
            def __call__(self): return self.value
        clock = Clock()
        class Evaluator(fixture.EarlierSearchEvaluator):
            def evaluate(self,candidate,*,case_id,purpose):
                result = super().evaluate(candidate,case_id=case_id,purpose=purpose)
                clock.value += search_seconds if purpose=='search' else confirmation_seconds if purpose=='confirmatory' else 0.
                return result
        return fixture.NominationTests.execute(self,Evaluator,clock=clock,
            budget_updates={'wall_time_seconds':10.,'confirmation_wall_time_seconds':reserve_seconds,
                            'active_authoring_time_seconds':8.,'maximum_turns':turns},
            endpoint_policy=NORMAL_BUDGET_TERMINAL)

    def test_search_stops_at_its_allocation_and_preserves_four_seconds_for_confirmation(self):
        _,_,_,audit,_,events,provider = self.execute(search_seconds=6.,confirmation_seconds=3.)
        self.assertEqual(len(provider.requests),1)
        self.assertEqual(audit.endpoint_observation,'qualified')
        self.assertEqual(audit.endpoint['budget_exceeded'],[])
        search = next(e['payload']['state'] for e in events if e['kind']=='search_completed')
        self.assertEqual(search['terminal_reason'],'wall_time_limit')
        self.assertEqual(search['elapsed_wall_seconds'],6.)
        self.assertEqual(search['remaining']['confirmation_wall_time_seconds'],4.)
        self.assertEqual(events[-2]['payload']['ralph']['remaining']['confirmation_wall_time_seconds'],1.)

    def test_search_overrun_is_not_hidden_by_finishing_before_the_total_deadline(self):
        _,_,_,audit,_,events,_ = self.execute(search_seconds=7.,confirmation_seconds=1.)
        self.assertEqual(events[-2]['payload']['ralph']['elapsed_wall_seconds'],8.)
        self.assertEqual(audit.endpoint_observation,'no_qualified_candidate')
        self.assertEqual(audit.endpoint['budget_exceeded'],['search_wall_time'])
        self.assertTrue(any(e['kind']=='candidate_evaluated' and e['payload']['purpose']=='confirmatory' for e in events))

    def test_confirmation_cannot_borrow_unused_search_time(self):
        _,_,_,audit,_,events,_ = self.execute(search_seconds=1.,confirmation_seconds=4.5,turns=1)
        self.assertEqual(events[-2]['payload']['ralph']['elapsed_wall_seconds'],5.5)
        self.assertEqual(audit.endpoint_observation,'no_qualified_candidate')
        self.assertEqual(audit.endpoint['budget_exceeded'],['confirmation_wall_time'])

    def test_fractional_reserve_replays_at_the_writers_clock_precision(self):
        _,_,_,audit,_,_,_ = self.execute(search_seconds=1.,confirmation_seconds=.1,turns=1,reserve_seconds=1/3)
        self.assertEqual(audit.endpoint_observation,'qualified')

    def test_threshold_projection_retains_overrun_cost_without_claiming_success(self):
        from open_cake_ir.lab.reporting import threshold_view
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,
                {'OPEN_CAKE_CUSTODY_DIRECTORY':str(Path(directory)/'registry')}):
            _,spec,run,audit,_,_,_ = self.execute(search_seconds=7.,confirmation_seconds=1.)
            self.assertTrue(audit.archive_integrity)
            self.assertTrue(audit.filesystem_custody_verified)
            campaign = SimpleNamespace(evidence_root=run.evidence_root,lock=SimpleNamespace(
                study_kind='matched_search',run_order=(spec.run_id,),
                document={'resolved_inputs':{'budget':spec.document['budget']}}))
            # The view consumes the real independently audited Run; this adapter
            # carries only the legacy Campaign-shaped report input, no new Study.
            report = SimpleNamespace(run_audits=(audit,),semantic_replay_passed=True)
            view = threshold_view(campaign,2.,audit_campaign=lambda _:report)
        row = view['runs'][0]
        self.assertEqual(row['status'],'budget_exceeded')
        self.assertEqual(row['budget_exceeded'],['search_wall_time'])
        self.assertEqual(row['observed_provider_tokens'],80000)
        self.assertEqual(row['observed_wall_seconds'],8.)
        self.assertIsNone(row['confirmed_latency_ms'])

    def test_replay_refuses_forged_remaining_reserve_and_overrun_success(self):
        lab,spec,_,audit,evidence,events,_ = self.execute(search_seconds=7.,confirmation_seconds=1.)
        for kind in ('search_completed','checkpoints_projected'):
            changed = deepcopy(list(events))
            payload = next(e['payload'] for e in changed if e['kind']==kind)
            state = payload['state'] if kind=='search_completed' else payload['ralph']
            state['remaining']['confirmation_wall_time_seconds'] = 100.
            with self.subTest(kind=kind),patch.object(evidence,'replay_events',return_value=changed):
                replay = replay_matched_run(evidence,audit,spec,project_root=fixture.ROOT,
                    manifest_parser=lab._parse_manifest,task_package=lab.task_package)
                self.assertFalse(replay)
        changed = deepcopy(list(events))
        nominee = next(e['payload'] for e in events if e['kind']=='candidate_nominated')
        forged = {'qualified_by_budget':True,'budget_exceeded':[],
                  **{key:value for key,value in audit.endpoint.items() if key not in {'qualified_by_budget','budget_exceeded'}},
                  'best_candidate_sha256':nominee['candidate_sha256'],'best_confirmed_latency_ms':1.2}
        changed[-1]['payload'].update(endpoint_observation='qualified',endpoint=forged)
        with patch.object(evidence,'replay_events',return_value=changed):
            replay = replay_matched_run(evidence,replace(audit,endpoint_observation='qualified',endpoint=forged),spec,
                project_root=fixture.ROOT,manifest_parser=lab._parse_manifest,task_package=lab.task_package)
        self.assertFalse(replay)

    def test_reserve_must_be_explicit_positive_and_leave_search_time(self):
        _,spec = run_fixture.IndependentRunTests.fixture(self)
        for value in (None,True,0.,float('nan'),spec.document['budget']['wall_time_seconds']):
            document = spec.document
            document['budget']['confirmation_wall_time_seconds'] = value
            with self.subTest(value=value),self.assertRaises(ValueError):
                RunSpecification.from_dict(document)
        document = spec.document
        document['budget'].pop('confirmation_wall_time_seconds')
        with self.assertRaises(ValueError): RunSpecification.from_dict(document)
