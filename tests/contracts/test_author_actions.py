"""Explicit transfer actions through the real Run/Compiler path; CPU fixtures only."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

from open_cake_ir.compiler import Compiler, Program, frontend
from open_cake_ir.evidence import EvidenceStore
from open_cake_ir.lab import RunSpecification, OpenCakeEnvironment, TritonToolchainBuilder
from open_cake_ir.lab.actions import resolve_action_set
from open_cake_ir.lab.knowledge import OptimizationKnowledge
from open_cake_ir.lab.provider_policy import execution_configuration
from open_cake_ir.serialization import canonical_json_bytes as encoded
from open_cake_ir.tasks.workloads import create_task, load_workload
from tests.contracts.test_run_specification import IndependentRunTests
from tests.contracts.test_lab import SemanticLabTestCase, RalphFakeProvider, FakeEvaluator, _submission_envelope
from tests.contracts.test_native_triton_pairing import CompilationFixture
from tests.contracts.test_program_rewrites import epilogue_program

ROOT = Path(__file__).resolve().parents[2]
PASS = 'specialize_triton_warps'
KNOWLEDGE = ROOT/'contracts/knowledge/rounded-epilogue-fusion-v1.json'


def rewrite(parent, *, transformation=PASS, stage='seed', width=8):
    return {'action':'transform', 'parent':parent, 'transformation':transformation,
            'parameters':{'stage':stage,'num_warps':width,'schedule_id':'rewritten','entry_point':'rewritten'}}


class ProgramEvaluator(FakeEvaluator):
    """Fixed-cost fixture: physical entry points do not encode author Turn identity."""
    def candidate_position(self,candidate):
        return 'open_cake',1


class AuthorActionTests(SemanticLabTestCase):
    def test_source_file_run_exposes_python_file_without_candidate_set_or_actions(self):
        from open_cake_ir.lab.provider_documents import PYTHON_SOURCE_FILE_V1
        lab, specification, _, _ = self.fixture([])
        document = specification.document
        document['budget']['maximum_candidates_per_turn'] = 1
        document['evaluation_protocol']['searches_per_turn'] = 1
        document['authoring'].update(input_format='python_source_v1',
                                     tool_surface=['submit_python_source'])
        document['authoring']['provider']['submission_contract'] = PYTHON_SOURCE_FILE_V1
        scaffold = 'contracts/scaffolds/python-artifact-optimization-source-file-v1.md'
        document['authoring']['scaffold'] = {'path': scaffold,
            'sha256': sha256((ROOT/scaffold).read_bytes()).hexdigest()}
        successor = RunSpecification.from_dict(document)
        package = lab.task_package(successor, successor.run_id)
        self.assertIn('candidate.py', package.task_markdown)
        self.assertIn('Write only `candidate.py`', package.agents_markdown)
        self.assertNotIn('Write exactly one valid UTF-8 JSON `candidate-set.json`', package.task_markdown)
        self.assertNotIn('Write only `candidate-set.json`', package.agents_markdown)
        self.assertNotIn('Granted Compiler transformations', package.agents_markdown)
        for field in ('maximum_candidates_per_turn', 'searches_per_turn', 'transformations'):
            invalid = json.loads(encoded(document))
            if field == 'maximum_candidates_per_turn':
                invalid['budget'][field] = 2
            elif field == 'searches_per_turn':
                invalid['evaluation_protocol'][field] = 2
            else:
                invalid['knowledge'][field] = [PASS]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'one direct Cake candidate'):
                RunSpecification.from_dict(invalid)

    def test_source_file_run_archives_raw_python_and_replays_its_projection(self):
        from open_cake_ir.lab.provider_documents import PYTHON_SOURCE_FILE_V1
        from open_cake_ir.lab.providers import ProviderQualificationReceipt
        from open_cake_ir.tasks.workloads import load_workload
        lab, specification, workload, _ = self.fixture([])
        document = specification.document
        source = Path(document['authoring']['schedule_skeleton']['path']).read_text()
        raw = source.encode()
        candidate = encoded({'python_source': source})
        document['budget']['maximum_candidates_per_turn'] = 1
        document['evaluation_protocol']['searches_per_turn'] = 1
        document['authoring'].update(input_format='python_source_v1',
                                     tool_surface=['submit_python_source'])
        scaffold = 'contracts/scaffolds/python-artifact-optimization-source-file-v1.md'
        document['authoring']['scaffold'] = {'path': scaffold,
            'sha256': sha256((ROOT/scaffold).read_bytes()).hexdigest()}
        provider_document = document['authoring']['provider']
        provider_document['submission_contract'] = PYTHON_SOURCE_FILE_V1
        from open_cake_ir.lab.provider_policy import execution_configuration
        configuration = execution_configuration(provider_document)
        qualification = ProviderQualificationReceipt.load(ROOT/provider_document['qualification']['path'])
        qualification = replace(qualification, configuration_sha256=sha256(encoded(configuration)).hexdigest())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            receipt = root/'qualification.json'
            receipt.write_bytes(encoded(qualification.document))
            provider_document['qualification'] = {'path': str(receipt),
                'canonical_sha256': qualification.canonical_sha256}
            successor = RunSpecification.from_dict(document)
            class SourceProvider(RalphFakeProvider):
                def turn(self, request):
                    old = super().turn(request)
                    events = old.raw_events.replace(b'candidate-set.json', b'candidate.py')
                    return replace(old, candidates=(candidate,),
                        candidate_sha256s=(sha256(candidate).hexdigest(),), raw_submission=raw,
                        raw_events=events, raw_events_sha256=sha256(events).hexdigest())
            provider = SourceProvider({successor.run_id:lab.task_package(successor, successor.run_id)})
            provider.configuration = configuration
            provider.qualification_sha256 = qualification.canonical_sha256
            compilation = CompilationFixture()
            environment = OpenCakeEnvironment(Compiler.load(ROOT,ROOT/'compiler/revision.json'),
                TritonToolchainBuilder(workload=workload,case_id='primary',isolated_compiler=compilation),
                authority_document=successor.document['authoring'],workload=workload,case_id='primary')
            evaluator = ProgramEvaluator(document['evaluation_protocol'],
                sha256(encoded(document['evaluation_protocol'])).hexdigest(),workload.canonical_sha256)
            run = lab.execute_run(successor,root/'evidence',provider=provider,
                                  environment=environment,evaluator=evaluator)
            audit, replay = lab.audit_run(run)
            self.assertTrue(replay, replay.refusals)
            self.assertEqual(audit.protocol_adherence, 'adhered')
            evidence = EvidenceStore.open(run.evidence_root)
            event = next(row for row in evidence.replay_events(successor.run_id)
                         if row['kind'] == 'provider_turn_completed')
            source_ref = next(item for item in event['payload']['objects']
                              if item['role'] == 'provider_source_file')
            self.assertEqual(evidence.read_object(source_ref), raw)
            self.assertEqual(len(compilation.requests), 2)

    def test_python_only_author_admission_refuses_schedule_json_but_keeps_internal_rewrites(self):
        compiler = Compiler.load(ROOT, ROOT/'compiler/revision.json')
        _, source = create_task('silu', backend='triton-b200', rows=2, columns=8)
        authored = encoded({'python_source': source})
        schedule = frontend.parse(source).document
        context = dict(environment_kind='open_cake', transformations=['fuse_pointwise_epilogue'],
                       candidates={}, baselines={'reference': encoded(epilogue_program())},
                       compiler_factory=lambda: compiler, allow_python=True, python_only=True)
        direct, wrapped, python, submitted_python, transformed = resolve_action_set((
            encoded(schedule),
            encoded({'action': 'submit', 'candidate': schedule}),
            authored,
            encoded({'action': 'submit', 'candidate': {'python_source': source}}),
            encoded({'action': 'transform', 'parent': 'baseline:reference',
                     'transformation': 'fuse_pointwise_epilogue',
                     'parameters': {'producer': 'producer', 'epilogue': 'epilogue',
                                    'schedule_id': 'fused', 'entry_point': 'fused'}}),
        ), **context)
        self.assertEqual((direct.reason, wrapped.reason), ('author_format', 'author_format'))
        self.assertIsNone(direct.candidate)
        self.assertIsNone(wrapped.candidate)
        self.assertEqual(python.candidate, authored)
        self.assertEqual(submitted_python.candidate, authored)
        self.assertEqual(transformed.reason, 'applied')
        self.assertEqual(len(Program.from_dict(json.loads(transformed.candidate)).stages), 1)

    def fixture(self, grants, *, baseline=False, python_only=False):
        lab, template = IndependentRunTests.fixture(self)
        document = template.document
        temporary = tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        workload_document, source = create_task('silu', backend='triton-b200', rows=2, columns=8)
        (root/'workload.json').write_bytes(encoded(workload_document))
        (root/'starter.py').write_text(source)
        schedule = frontend.parse(source).document
        program = Program.from_schedule(schedule).document
        program['program_id'] = 'open_cake_turn_1'
        program['stages'][0]['name'] = 'seed'
        workload = load_workload(root/'workload.json')
        document['workload'] = {'workload_id':workload.workload_id,'path':str(root/'workload.json'),
                                'canonical_sha256':workload.canonical_sha256}
        document['knowledge']['transformations'] = grants
        document['authoring'].update(reference_access='known_kernel_reproduction', input_format='schedule_or_python_v1',
            lowering_route=schedule['lowering'], schedule_skeleton={'path':str(root/'starter.py'),'canonical_sha256':sha256(encoded(schedule)).hexdigest()})
        if python_only:
            document['authoring'].update(input_format='python_source_v1', tool_surface=['submit_python_source'])
        document['evaluation_protocol'] = {'case_id':'primary','search_evaluation':'correctness_then_paired_cupti',
            'confirmatory_evaluation':'fresh_fixed_candidate_correctness_then_paired_cupti'}
        if workload.document['validation'].get('all_cases_required'):
            document['evaluation_protocol']['validation_case_ids'] = list(workload.case_ids)
        document['budget'].update(limit=160000,checkpoints=[80000,160000],maximum_turns=2,maximum_candidates_per_turn=2)
        if baseline: document['reference_inputs']['baseline_programs'] = {'seed':program}
        return lab, RunSpecification.from_dict(document), workload, program

    def execute_fixture(self, grants, *, only_transform=False, parent=None, python_only=False):
        lab, specification, workload, program = self.fixture(grants, python_only=python_only)
        document = specification.document
        source = Path(document['authoring']['schedule_skeleton']['path']).read_text()
        candidate = {'python_source': source} if python_only else program
        parent_id = sha256(encoded(candidate)).hexdigest()
        stage = frontend.parse(source).document['schedule_id'] if python_only else 'seed'
        class Provider(RalphFakeProvider):
            def turn(self, request):
                observed = super().turn(request)
                action = (rewrite(parent or parent_id, stage=stage) if only_transform or request.turn > 1
                          else {'action':'submit','candidate':candidate})
                payload = encoded(action)
                return replace(observed,candidates=(payload,),candidate_sha256s=(sha256(payload).hexdigest(),),
                               raw_submission=_submission_envelope(request.arm,(payload,)))
        provider = Provider({specification.run_id:lab.task_package(specification,specification.run_id)})
        provider.configuration = execution_configuration(document['authoring']['provider'])
        provider.qualification_sha256 = document['authoring']['provider']['qualification']['canonical_sha256']
        compilation = CompilationFixture()
        environment = OpenCakeEnvironment(Compiler.load(ROOT,ROOT/'compiler/revision.json'),
            TritonToolchainBuilder(workload=workload,case_id='primary',isolated_compiler=compilation),
            authority_document=document['authoring'],workload=workload,case_id='primary')
        evaluator = ProgramEvaluator(document['evaluation_protocol'],sha256(encoded(document['evaluation_protocol'])).hexdigest(),workload.canonical_sha256)
        root = tempfile.TemporaryDirectory();self.addCleanup(root.cleanup)
        run = lab.execute_run(specification,Path(root.name)/'evidence',provider=provider,environment=environment,evaluator=evaluator)
        audit, replay = lab.audit_run(run)
        self.assertTrue(replay, replay.refusals)
        self.assertEqual(audit.protocol_adherence,'adhered')
        events = EvidenceStore.open(run.evidence_root).replay_events(specification.run_id)
        return lab, run, audit, events, compilation, evaluator

    def test_explicit_transform_produces_new_program_and_replays_its_parent(self):
        _,_,audit,events,compiled,evaluator = self.execute_fixture([PASS])
        actions = [event['payload']['actions'][0] for event in events if event['kind']=='author_actions_resolved']
        self.assertEqual([row['kind'] for row in actions],['submit','transform'])
        self.assertNotEqual(actions[0]['action_sha256'],actions[0]['candidate_sha256'])
        self.assertEqual(actions[1]['parent'],actions[0]['candidate_sha256'])
        self.assertNotEqual(actions[1]['candidate_sha256'],actions[0]['candidate_sha256'])
        self.assertEqual(actions[1]['reason'],'applied')
        self.assertEqual(len(compiled.requests),2)
        self.assertNotEqual(compiled.requests[0][1]['compile_options']['num_warps'],8)
        self.assertEqual(compiled.requests[1][1]['compile_options']['num_warps'],8)
        self.assertEqual(evaluator.calls,3)
        self.assertEqual(audit.endpoint_observation,'qualified')

    def test_python_only_run_preserves_source_submission_and_compiler_transform_replay(self):
        _,_,audit,events,compiled,_ = self.execute_fixture([PASS], python_only=True)
        actions = [event['payload']['actions'][0] for event in events if event['kind']=='author_actions_resolved']
        self.assertEqual([row['kind'] for row in actions], ['submit', 'transform'])
        self.assertEqual([row['reason'] for row in actions], ['submitted', 'applied'])
        self.assertEqual(len(compiled.requests), 2)
        self.assertEqual(audit.protocol_adherence, 'adhered')

    def test_withheld_pass_refuses_without_a_second_build_or_evaluation(self):
        _,_,audit,events,compiled,evaluator = self.execute_fixture([])
        rows = [event['payload']['actions'][0] for event in events if event['kind']=='author_actions_resolved']
        self.assertEqual(rows[1]['reason'],'transform_not_granted')
        self.assertIsNone(rows[1]['candidate_sha256'])
        self.assertEqual(rows[1]['objects'],[])
        self.assertEqual(len(compiled.requests),1)
        self.assertEqual(evaluator.calls,2)
        self.assertEqual(audit.endpoint_observation,'qualified')  # nominates and confirms the earlier submit after the refused action

    def test_all_denied_actions_form_an_observed_nonqualifying_run(self):
        _,_,audit,events,compiled,evaluator = self.execute_fixture([],only_transform=True)
        self.assertFalse(compiled.requests)
        self.assertEqual(evaluator.calls,0)
        self.assertEqual(audit.endpoint_observation,'no_qualified_candidate')
        self.assertTrue(all(event['payload']['candidate_sha256'] is None for event in events if event['kind']=='candidate_selected'))

    def test_foreign_run_parent_is_not_resolved(self):
        _,_,_,events,compiled,evaluator = self.execute_fixture([PASS],parent='f'*64)
        row = [event for event in events if event['kind']=='author_actions_resolved'][-1]['payload']['actions'][0]
        self.assertEqual(row['reason'],'parent_not_authorized')
        self.assertEqual(len(compiled.requests),1)
        self.assertEqual(evaluator.calls,2)

    def test_authorized_complete_baseline_fuses_without_caller_private_tensor_promise(self):
        compiler = Compiler.load(ROOT, ROOT/'compiler/revision.json')
        action = {'action':'transform','parent':'baseline:reference','transformation':'fuse_pointwise_epilogue',
            'parameters':{'producer':'producer','epilogue':'epilogue','schedule_id':'fused','entry_point':'fused'}}
        result, = resolve_action_set((encoded(action),), environment_kind='open_cake',
            transformations=['fuse_pointwise_epilogue'],candidates={},
            baselines={'reference':encoded(epilogue_program())},compiler_factory=lambda:compiler)
        self.assertEqual(result.reason,'applied')
        self.assertEqual(len(Program.from_dict(json.loads(result.candidate)).stages),1)
        self.assertEqual(result.parent,'baseline:reference')

    def test_malformed_python_parent_is_an_action_refusal_not_a_batch_fault(self):
        payload = encoded({'python_source':42})
        parent = sha256(payload).hexdigest()
        compiler = Mock(side_effect=AssertionError('malformed parent reached Compiler'))
        results = resolve_action_set((encoded(rewrite(parent)),encoded({'action':'submit','candidate':{'turn':2}})),
            environment_kind='open_cake',transformations=[PASS],candidates={parent:payload},baselines={},
            compiler_factory=compiler,allow_python=True)
        self.assertEqual(results[0].reason,'parent_not_program')
        self.assertIsNone(results[0].candidate)
        self.assertEqual(json.loads(results[1].candidate),{'turn':2})
        compiler.assert_not_called()

    def test_replay_rejects_changed_parent_or_result(self):
        lab,run,audit,events,_,_ = self.execute_fixture([PASS])
        from copy import deepcopy
        from open_cake_ir.lab import replay as reader
        store = EvidenceStore.open(run.evidence_root)
        for field,value in [('parent','e'*64),('candidate_sha256','e'*64),('reason','unchanged')]:
            changed = deepcopy(events)
            [event for event in changed if event['kind']=='author_actions_resolved'][-1]['payload']['actions'][0][field] = value
            proxy = Mock(wraps=store)
            proxy.replay_events.return_value = changed
            result = reader.replay_matched_run(proxy,audit,run.specification,project_root=ROOT,
                manifest_parser=lab._parse_manifest,task_package=lab.task_package)
            self.assertFalse(result)
            self.assertIn('action result differs',str(result.refusals[0]))

    def test_material_and_pass_selection_are_orthogonal_and_frozen(self):
        lab,specification,_,program = self.fixture([])
        unit = OptimizationKnowledge.load(KNOWLEDGE).document
        base = specification.document
        for explain, grant in ((False,False),(True,False),(False,True),(True,True)):
            document = json.loads(encoded(base))
            document['knowledge'] = {'materials':[unit] if explain else [],
                                     'transformations':['fuse_pointwise_epilogue'] if grant else []}
            frozen = RunSpecification.from_dict(document)
            package = lab.task_package(frozen,frozen.run_id)
            self.assertEqual(unit['mechanism'] in package.task_markdown,explain)
            self.assertEqual('## Frozen reference: `transformation-api.json`' in package.task_markdown,grant)
            document['knowledge']['materials'].clear()
            self.assertEqual(bool(frozen.document['knowledge']['materials']),explain)
        bad = specification.document;bad['reference_inputs']['baseline_programs']={'p':program}
        bad['authoring']['reference_access']='clean_start'
        with self.assertRaisesRegex(ValueError,'known-kernel'):
            RunSpecification.from_dict(bad)
