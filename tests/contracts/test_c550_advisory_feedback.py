"""C550 advisory wiring; synthetic latencies are software fixtures only."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from open_cake_ir.compiler import Compiler, Program, frontend
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.lab import CandidateSubmission, TritonToolchainBuilder
from open_cake_ir.lab.executor import ExecutorRevision
from open_cake_ir.lab.selection import _EmpiricalSelection, _empirical_context, _empirical_filter
from open_cake_ir.lab.workload_binding import bind_program_workload
from open_cake_ir.tasks.environments import TaskOpenCakeEnvironment
from open_cake_ir.tasks.workloads import create_task
from open_cake_ir.serialization import canonical_json_bytes
from tests.contracts.test_native_program_tensors import McaCompilationFixture

ROOT=Path(__file__).resolve().parents[2]

class C550AdvisoryFeedback(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler=Compiler.load(ROOT)
        document,cls.source=create_task('rmsnorm',backend='triton-metax',rows=8,columns=128)
        cls.workload=WorkloadContract(document)
        cls.executor=ExecutorRevision('fixture',{'host_environment':{'packages':{'triton':'CPU fixture'}}},ROOT,'fixture')
        cls.context=_empirical_context(cls.executor,workload_sha256=cls.workload.canonical_sha256,
                                      case_id='primary',target='xcore1002')
        cls.program=bind_program_workload(Program.from_schedule(frontend.parse(cls.source).document),
                                        cls.workload.canonical_sha256)
        cls.schedule=cls.program.document['stages'][0]['schedule']
        cls.revision=cls.compiler.assess(cls.schedule).compiler_revision_id

    def selection(self,context=None):
        template=deepcopy(self.schedule)
        varying=next(b for b in template['buffers'] if b['space']=='global')
        model={'schema_version':3,'model_id':'synthetic-mcpti-software-fixture',
            'compiler_revision_id':self.revision,'target':'xcore1002','context':self.context,
            'reported_evidence':{'kind':'CPU fixture; not hardware calibration'},
            'curves':[{'template':template,'varying_dimensions':[{'buffer':varying['name'],'dimension':0}],
                'extent_multiple':1,'points':[{'extent':varying['shape'][0],'kernel_us':10.}],
                'relative_error_envelope':.1}]}
        binding={'kind':'external_empirical_advisory_v1','model':model}
        return binding,_EmpiricalSelection(binding,context=context or self.context,
                          compiler_revision_id=self.revision,target='xcore1002')

    def test_context_names_native_timer_and_fixed_executor_not_cupti(self):
        from open_cake_ir.evaluation.metax_benchmark import TIMER,RESET
        self.assertEqual((self.context['timer'],self.context['cache_protocol']),(TIMER,RESET))
        self.assertEqual(self.context['runtime']['executor_revision'],'fixture')
        wrong={**self.context,'timer':'cupti'}
        _,selection=self.selection(wrong)
        estimate=selection.estimate(self.schedule)
        self.assertFalse(estimate['covered'])
        self.assertIn('context differs',estimate['reason'])

    def test_python_schedule_program_and_replay_share_prediction_and_unknown_preserves_order(self):
        _,selection=self.selection()
        expected=selection.estimate(self.schedule)
        self.assertTrue(expected['covered'])
        self.assertEqual(selection.estimate({'python_source':self.source}),expected)
        self.assertEqual(selection.estimate(self.program.document),expected)
        unknown=deepcopy(self.schedule)
        unknown['roles'][0]['execution_groups']=[0,1]
        estimate=selection.estimate(unknown)
        self.assertFalse(estimate['covered'])
        rows=[{'candidate_sha256':'first','disposition':'launchable','empirical_cost':estimate},
              {'candidate_sha256':'second','disposition':'launchable','empirical_cost':expected}]
        ordered,summary=_empirical_filter(rows)
        self.assertEqual([r['candidate_sha256'] for r in ordered],['first','second'])
        self.assertFalse(summary['order_applied'])
        context=deepcopy(self.context);context['runtime']['executor_revision']='other@commit'
        _,selection=self.selection(context)
        self.assertFalse(selection.estimate(self.schedule)['covered'])

    def test_complete_c550_program_build_reports_static_domain_and_empirical_selection(self):
        binding,selection=self.selection()
        builder=TritonToolchainBuilder(workload=self.workload,case_id='primary',isolated_compiler=McaCompilationFixture())
        authority={'lowering_route':self.schedule['lowering'],'input_format':'python_source_v1',
                   'compiler_revision':{'revision_id':self.revision},'candidate_selection':binding}
        environment=TaskOpenCakeEnvironment(self.compiler,builder,authority_document=authority,
                            workload=self.workload,case_id='primary',executor=self.executor)
        result=environment.build(CandidateSubmission.seal(environment.media_type,self.program.document_bytes))
        self.assertEqual(result.disposition,'launchable',result.feedback)
        self.assertEqual(result.empirical_cost,selection.estimate(self.program.document))
        profile=next(iter(result.feedback['static_profiles'].values()))
        self.assertIsNotNone(profile['residency'])
        self.assertEqual(profile['ncu_metrics'],[])
        self.assertTrue(all(row['resource']!='registers' for row in profile['residency']['bounds']))
        self.assertIn('implicit backend allocations are not examined',profile['abstentions'][0])
        # The model does not admit a malformed Schedule or replace the verifier.
        bad=self.program.document
        bad['stages'][0]['schedule']['buffers'][0]['dtype']='invalid'
        rejected=environment.build(CandidateSubmission.seal(environment.media_type,canonical_json_bytes(bad)))
        self.assertEqual(rejected.disposition,'rejected')
        self.assertIsNone(rejected.empirical_cost)
