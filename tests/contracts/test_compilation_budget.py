"""Native invocation quotas through actual builders and independent Run replay."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import tempfile

from open_cake_ir.compiler import Compiler, Program
from open_cake_ir.evidence import EvidenceStore
from open_cake_ir.lab import RunSpecification
from open_cake_ir.lab.build import TritonToolchainBuilder
from open_cake_ir.lab.candidate_filter import _build_filter_candidates
from open_cake_ir.lab.compilation import CompilationRecorder
from open_cake_ir.lab.environments import OpenCakeEnvironment
from open_cake_ir.lab.endpoints import NORMAL_BUDGET_TERMINAL
from open_cake_ir.lab.faults import CandidateCompileRejected, CompilationBudgetExceeded
from open_cake_ir.lab.provider_policy import execution_configuration
from open_cake_ir.lab.ralph import RalphBudget, RalphController
from open_cake_ir.lab.replay.compilations import replay_compilations
from open_cake_ir.lab.replay.refusals import ReplayRefusal
from open_cake_ir.serialization import canonical_json_bytes as encoded
from tests.contracts import test_author_actions as action_fixture
from tests.contracts import test_alignment_variants as alignment_fixture
from tests.contracts.test_lab import SemanticLabTestCase, RalphFakeProvider, FakeEvaluator, _submission_envelope
from tests.contracts.test_native_triton_pairing import CompilationFixture
from tests.contracts.test_program_evaluation import workload_for
from tests.contracts.test_program_rewrites import epilogue_program

ROOT = Path(__file__).resolve().parents[2]


def controller(maximum):
    return RalphController(RalphBudget(160000,2,2,60,30,4,1,4,maximum,6),
        searches_per_turn=1,profile_each_search_survivor=False,clock=lambda:0.)


class CompilationBudgetTests(SemanticLabTestCase):
    def run_fixture(self, *, failed=False, second_stage=False):
        lab,spec,workload,program = action_fixture.AuthorActionTests.fixture(self,[])
        if second_stage:
            # This candidate is stopped during construction, before any semantic
            # Evaluation. Its complete public ABI remains the fixed Workload's.
            output = program['outputs'][0]
            second = deepcopy(program['stages'][0])
            program['tensors']['intermediate'] = deepcopy(program['tensors'][output])
            program['stages'][0]['bindings'][output] = 'intermediate'
            second['name'] = 'second'
            second['schedule']['schedule_id'] = 'second'
            second['schedule']['lowering']['entry_point'] = 'second'
            second['bindings'][program['inputs'][0]] = 'intermediate'
            program['stages'].append(second)
        document = spec.document
        document['budget']['maximum_compilations'] = 1
        document['endpoint_policy'] = NORMAL_BUDGET_TERMINAL
        spec = RunSpecification.from_dict(document)
        class Provider(RalphFakeProvider):
            def turn(self,request):
                result = super().turn(request)
                payload = encoded(program)
                return replace(result,candidates=(payload,),candidate_sha256s=(sha256(payload).hexdigest(),),
                    raw_submission=_submission_envelope(request.arm,(payload,)))
        class NativeCompiler(CompilationFixture):
            def compile(self,source,requirements):
                result = super().compile(source,requirements)
                if failed:
                    raise CandidateCompileRejected('CPU native compiler refusal',artifact_payloads={'compiler_stderr':b'refused'})
                return result
        native = NativeCompiler()
        environment = OpenCakeEnvironment(Compiler.load(ROOT,ROOT/'compiler/revision.json'),
            TritonToolchainBuilder(workload=workload,case_id='primary',isolated_compiler=native),
            authority_document=document['authoring'],workload=workload,case_id='primary')
        provider = Provider({spec.run_id:lab.task_package(spec,spec.run_id)})
        provider.configuration = execution_configuration(document['authoring']['provider'])
        provider.qualification_sha256 = document['authoring']['provider']['qualification']['canonical_sha256']
        evaluator = action_fixture.ProgramEvaluator(document['evaluation_protocol'],sha256(encoded(document['evaluation_protocol'])).hexdigest(),workload.canonical_sha256)
        temporary = tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        run = lab.execute_run(spec,Path(temporary.name)/'evidence',provider=provider,environment=environment,evaluator=evaluator)
        audit,replay = lab.audit_run(run)
        self.assertTrue(replay,replay.refusals)
        return spec,audit,EvidenceStore.open(run.evidence_root).replay_events(spec.run_id),native,provider,evaluator

    def test_native_compilation_consumes_quota_before_the_next_author_turn(self):
        _,audit,events,native,provider,evaluator = self.run_fixture()
        self.assertEqual(len(native.requests),1)
        self.assertEqual(len(provider.requests),1)
        self.assertEqual(evaluator.calls,2)  # Terminal confirmation still evaluates the sealed nominee.
        self.assertEqual(audit.endpoint_observation,'qualified')
        state = events[-2]['payload']['ralph']
        self.assertEqual(state['compilation_count'],1)
        self.assertEqual(state['terminal_reason'],'compilation_budget')
        self.assertEqual(state['remaining']['compilations'],0)

    def test_failed_native_call_counts_and_does_not_become_an_infrastructure_fault(self):
        _,audit,events,native,provider,evaluator = self.run_fixture(failed=True)
        self.assertEqual(len(native.requests),1)
        self.assertEqual(len(provider.requests),1)
        self.assertEqual(evaluator.calls,0)
        self.assertEqual(audit.protocol_adherence,'adhered')
        self.assertEqual(audit.endpoint_observation,'no_qualified_candidate')
        completion = next(e['payload'] for e in events if e['kind']=='compilation_completed')
        self.assertEqual(completion['outcome'],'raised')

    def test_partial_program_at_quota_is_retained_as_budget_refusal_and_replays(self):
        _,audit,events,native,_,evaluator = self.run_fixture(second_stage=True)
        self.assertEqual(len(native.requests),1)
        self.assertEqual(evaluator.calls,0)
        self.assertEqual(audit.endpoint_observation,'no_qualified_candidate')
        refusal = next(e['payload'] for e in events if e['kind']=='candidate_rejected')
        self.assertEqual(refusal['feedback']['stage'],'budget')
        self.assertIsNone(refusal['routed_to'])
        self.assertEqual(sum(e['kind']=='compilation_refused' for e in events),1)

    def test_program_stages_consume_separate_permits_and_stop_before_the_second_native_call(self):
        document = epilogue_program()
        program = Program.from_dict(document)
        workload = workload_for(program)
        native = CompilationFixture()
        environment = OpenCakeEnvironment(Compiler.load(ROOT,ROOT/'compiler/revision.json'),
            TritonToolchainBuilder(workload=workload,case_id='primary',isolated_compiler=native),
            workload=workload,case_id='primary',authority_document={'lowering_route':{'backend':'triton','entry_point':'starter'}})
        events = []
        ledger = SimpleNamespace(append=lambda kind,payload:events.append({'kind':kind,'payload':payload}))
        payload = encoded(document)
        built,*_ = _build_filter_candidates(empirical_enabled=False,environment=environment,ledger=ledger,
            candidate_payloads=(payload,),turn_number=1,ralph=controller(1))
        self.assertEqual(len(native.requests),1)
        self.assertEqual(built[0][1].feedback['stage'],'budget')
        self.assertIsNone(built[0][1].launchable)
        self.assertEqual([e['kind'] for e in events],['compilation_started','compilation_completed','compilation_refused','candidate_set_filtered'])

    def test_alignment_specialization_obtains_a_second_permit(self):
        alignment_fixture.AlignmentVariants.setUpClass()
        fixture = alignment_fixture.AlignmentVariants()
        events = []
        ledger = SimpleNamespace(append=lambda kind,payload:events.append({'kind':kind,'payload':payload}))
        with self.assertRaises(CompilationBudgetExceeded):
            fixture.build(16,compilation=CompilationRecorder(controller(1),ledger,1,'a'*64))
        self.assertEqual([e['payload']['variant'] for e in events if e['kind']=='compilation_started'],['generic'])
        self.assertEqual(events[-1]['kind'],'compilation_refused')
        events.clear()
        _,_,calls = fixture.build(16,compilation=CompilationRecorder(controller(2),ledger,1,'a'*64))
        self.assertEqual(len(calls),2)
        self.assertEqual([e['payload']['variant'] for e in events if e['kind']=='compilation_started'],['generic','aligned'])

    def test_replay_rejects_forged_counts_and_foreign_candidate_calls(self):
        spec,_,events,_,_,_ = self.run_fixture()
        started = next(e['payload'] for e in events if e['kind']=='compilation_started')
        candidates = {started['turn']:(started['candidate_sha256'],)}
        for kind,key,value in [('compilation_started','number',2),('compilation_started','candidate_sha256','f'*64),
                               ('compilation_completed','outcome','accepted')]:
            changed = deepcopy(list(events))
            next(e['payload'] for e in changed if e['kind']==kind)[key] = value
            with self.subTest(key=key),self.assertRaises(ReplayRefusal):
                replay_compilations(changed,candidates_by_turn=candidates,maximum=1,target=spec.document['execution']['target'])
        changed = deepcopy(list(events))
        start = next(i for i,e in enumerate(changed) if e['kind']=='compilation_started')
        pair = changed[start:start+2]
        del changed[start:start+2]
        changed[1:1] = pair
        with self.assertRaisesRegex(ReplayRefusal,'preceded resolution'):
            replay_compilations(changed,candidates_by_turn=candidates,maximum=1,target=spec.document['execution']['target'])
        for order in (None,[{}],[{'candidate_sha256':[],'disposition':'launchable'}]):
            changed = deepcopy(list(events))
            next(e['payload'] for e in changed if e['kind']=='candidate_set_filtered')['order'] = order
            with self.subTest(order=order),self.assertRaisesRegex(ReplayRefusal,'readable filter rows'):
                replay_compilations(changed,candidates_by_turn=candidates,maximum=1,target=spec.document['execution']['target'])
