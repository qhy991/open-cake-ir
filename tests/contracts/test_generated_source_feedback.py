"""Own-candidate source transport and independent replay, with CPU-only doubles.

No native compiler, provider or GPU is invoked. Real Compiler emission and existing
Program artifact owners remain in the seam; fake binaries make no device claim.
"""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.evaluation import LaunchableCandidate
from open_cake_ir.evaluation.metal_manifest import MetalTensorLaunchManifest
from open_cake_ir.lab.feedback import derive_turn_feedback
from open_cake_ir.lab.generated_source import (
    GENERATED_SOURCE_V1, MAX_CANDIDATE_SOURCE_BYTES, MAX_TURN_SOURCE_BYTES,
    MAX_SOURCE_STAGES, MAX_CANDIDATE_VIEW_BYTES, candidate_source_feedback, generated_source_permission,
    sealed_sources, source_views,
)
from open_cake_ir.lab.replay.artifacts import _replay_launchable_candidate
from open_cake_ir.lab.replay.feedback import replay_feedback
from open_cake_ir.lab.replay.refusals import ReplayRefusal
from open_cake_ir.lab.workload_binding import bind_schedule_workload
from open_cake_ir.serialization import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[2]


def authority(route=None):
    return {'environment_kind': 'open_cake', 'reference_access': 'known_kernel_reproduction',
            'feedback': ['findings', 'correctness', GENERATED_SOURCE_V1], 'provider': {},
            'lowering_route': route or {'backend': 'metal', 'entry_point': 'rmsnorm'}}


def seal(payloads, identity, target, entry_point, manifest):
    return LaunchableCandidate(identity, target, entry_point,
        {role: sha256(raw).hexdigest() for role, raw in payloads.items()},
        manifest.canonical_sha256, payloads)


class GeneratedSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)

    def metal_candidate(self, name='own-candidate', *, python_envelope=False):
        from tools.metal import rmsnorm
        from tests.contracts.test_metal_artifacts import abi_fixture
        workload = abi_fixture()
        source = rmsnorm.source(3, 7, target=workload.target)
        document = frontend.parse(source).document
        if not python_envelope:
            document['schedule_id'] = name
        authored = canonical_json_bytes({'python_source': source} if python_envelope else document)
        lowering = self.compiler.lower(self.compiler.assess(
            bind_schedule_workload(document, workload.canonical_sha256)))
        req = lowering.toolchain_requirements
        manifest = MetalTensorLaunchManifest.for_workload(workload, 'odd', target=workload.target,
            kernel_name=lowering.route.entry_point, grid=req['threadgroups_per_grid'],
            block=req['threads_per_threadgroup'], threadgroup_memory_bytes=req['threadgroup_memory_bytes'])
        payloads = {'lowered_source': lowering.source.encode(),
                    'metal_binary_archive': b'CPU artifact fixture; never load',
                    'metal_build_report': b'{"CPU_fixture":true}',
                    'launch_manifest': canonical_json_bytes(manifest.as_dict())}
        candidate = seal(payloads, sha256(authored).hexdigest(), workload.target,
                         lowering.route.entry_point, manifest)
        return authored, candidate, workload, authority(document['lowering'])

    def replay_artifact(self, authored, candidate, workload, policy):
        from open_cake_ir.tasks.launch import parse_launch_manifest
        event = {'kind': 'launchable_candidate_sealed', 'payload': {'turn': 1,
            'candidate_sha256': candidate.candidate_sha256,
            'candidate_record_sha256': candidate.canonical_sha256,
            'objects': [{'role': role, 'sha256': value} for role, value in candidate.artifact_roles.items()]}}
        evidence = SimpleNamespace(read_object=lambda ref: candidate.artifact_payloads[ref['role']])
        return _replay_launchable_candidate(evidence, [event], turn=1,
            candidate_sha256=candidate.candidate_sha256, arm='open_cake',
            manifest_parser=parse_launch_manifest, compiler_factory=lambda: self.compiler,
            authored_bytes=authored, workload_sha256=workload.canonical_sha256,
            source_authority=policy)

    def test_permission_is_explicit_and_never_grants_other_arms_or_references(self):
        self.assertFalse(generated_source_permission({}))
        for change in ({'reference_access': 'clean_start'}, {'reference_access': 'direct_low_level'},
                       {'environment_kind': 'direct_cuda'}, {'environment_kind': 'native_triton'},
                       {'feedback': GENERATED_SOURCE_V1}, {'feedback': [GENERATED_SOURCE_V1]*2},
                       {'feedback': 3}, {'feedback': False}, {'feedback': {GENERATED_SOURCE_V1: True}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                generated_source_permission({**authority(), **change})
        self.assertTrue(generated_source_permission(authority()))

    def test_msl_matches_frozen_schedule_and_resealed_substitution_is_refused(self):
        authored, candidate, workload, policy = self.metal_candidate()
        verified = self.replay_artifact(authored, candidate, workload, policy)
        view = sealed_sources(verified, authored)
        rendered, _ = source_views(view, MAX_TURN_SOURCE_BYTES)
        stage = rendered['stages'][0]
        self.assertEqual(stage['language'], 'metal')
        self.assertIsNone(stage['stage_id'])
        self.assertEqual(stage['route'], policy['lowering_route'])
        self.assertIn('CAKE_OP:', stage['source'])
        self.assertEqual(stage['line_count'], len(stage['source'].splitlines()))
        # Change the source and all surrounding artifact seals consistently. The
        # refusal must name the Compiler relation, not borrow an outer hash check.
        _, other, _, _ = self.metal_candidate('other-candidate')
        payloads = {**candidate.artifact_payloads, 'lowered_source': other.artifact_payloads['lowered_source']}
        manifest = MetalTensorLaunchManifest.from_dict(json.loads(payloads['launch_manifest']))
        forged = seal(payloads, candidate.candidate_sha256, candidate.target, candidate.entry_point, manifest)
        with self.assertRaisesRegex(ReplayRefusal, 'authored Schedule and frozen Compiler lowering'):
            self.replay_artifact(authored, forged, workload, policy)
        # Existing Runs without the opt-in retain their previous replay domain.
        self.replay_artifact(authored, forged, workload, {**policy, 'feedback': ['findings', 'correctness']})

        # Source inspection binds the authored route, not a native binary symbol.
        # The latter is outside this ordinary Schedule source-replay boundary.
        manifest = replace(manifest, kernel_name='different_native_symbol')
        payloads = {**candidate.artifact_payloads, 'launch_manifest': canonical_json_bytes(manifest.as_dict())}
        changed = seal(payloads, candidate.candidate_sha256, candidate.target, manifest.kernel_name, manifest)
        identity, _ = sealed_sources(changed, authored)[0]
        self.assertEqual(identity['route'], policy['lowering_route'])
        self.assertNotIn('entry_point', identity)

    def test_python_envelope_and_missing_source_cannot_borrow_another_candidate(self):
        authored, candidate, workload, policy = self.metal_candidate(python_envelope=True)
        self.replay_artifact(authored, candidate, workload, policy)
        specification = SimpleNamespace(document={'authoring': policy, 'execution': {'target': candidate.target}})
        identity = candidate.candidate_sha256
        other = 'f'*64
        with self.assertRaisesRegex(ValueError, 'resolved candidate'):
            candidate_source_feedback(candidates=(other,), launchables={other: candidate},
                authored={other: authored}, specification=specification)
        unavailable = candidate_source_feedback(candidates=(identity,), launchables={}, authored={},
                                               specification=specification)
        self.assertEqual(unavailable[identity]['reason'], 'no_sealed_search_candidate')
        missing, _ = source_views([(sealed_sources(candidate, authored)[0][0], None)], MAX_TURN_SOURCE_BYTES)
        self.assertEqual(missing['stages'][0]['reason'], 'sealed_source_missing')

    def test_complete_stage_text_is_bounded_without_truncation_or_history_copy(self):
        identity = {'stage_id': 'first', 'target': 'sm_100a', 'route': {'backend': 'triton', 'entry_point': 'first'},
                    'language': 'python', 'artifact_role': 'lowered_source'}
        exact = ('字' * (MAX_CANDIDATE_SOURCE_BYTES // 3)).encode()
        view, used = source_views([(identity, exact), ({**identity, 'stage_id': 'second'}, b'next')],
                                 MAX_TURN_SOURCE_BYTES)
        self.assertEqual(view['stages'][0]['source'].encode(), exact)
        self.assertEqual(view['stages'][1]['reason'], 'candidate_source_byte_limit')
        self.assertEqual(used, len(exact))
        view, used = source_views([(identity, b'1234')], 3)
        self.assertEqual(view['stages'][0]['reason'], 'turn_source_byte_limit')
        self.assertNotIn('source', view['stages'][0])
        self.assertEqual(used, 0)
        view, _ = source_views([(identity, b'')]*(MAX_SOURCE_STAGES+3), MAX_TURN_SOURCE_BYTES)
        self.assertEqual(len(view['stages']), MAX_SOURCE_STAGES)
        self.assertEqual(view['omitted_stage_count'], 3)
        view, used = source_views([({**identity, 'stage_id': 'x' * MAX_CANDIDATE_VIEW_BYTES}, b'')],
                                 MAX_TURN_SOURCE_BYTES)
        self.assertEqual(view['stages'], [])
        self.assertEqual(view['omitted_stage_count'], 1)
        self.assertEqual(view['stage_omission_reason'], 'view_byte_limit')
        self.assertLessEqual(len(json.dumps(view).encode()), MAX_CANDIDATE_VIEW_BYTES)
        with self.assertRaises(UnicodeError):
            source_views([(identity, b'\xff')], MAX_TURN_SOURCE_BYTES)

    def program_candidate(self, name=None):
        from tests.contracts.test_program_evaluation import ProgramEvaluationTests, workload_for
        from tests.contracts.test_program_rewrites import epilogue_program
        from open_cake_ir.compiler import Program
        document = epilogue_program()
        if name is not None:
            document['program_id'] = name
            for stage in document['stages']:
                stage['schedule']['schedule_id'] += '-' + name
        candidate, workload, _ = ProgramEvaluationTests.build(self, document)
        return canonical_json_bytes(document), candidate, workload_for(Program.from_dict(document))

    def test_multi_stage_program_sources_replay_through_existing_artifact_owner(self):
        authored, candidate, workload = self.program_candidate()
        policy = authority({'backend': 'triton', 'entry_point': 'starter'})
        verified = self.replay_artifact(authored, candidate, workload, policy)
        sources = sealed_sources(verified, authored)
        view, used = source_views(sources, MAX_TURN_SOURCE_BYTES)
        self.assertEqual([row['stage_id'] for row in view['stages']], ['producer', 'epilogue'])
        self.assertTrue(all(row['language'] == 'python' for row in view['stages']))
        self.assertTrue(all(row['status'] == 'available' for row in view['stages']))
        self.assertEqual(used, sum(len(raw) for _, raw in sources))
        from open_cake_ir.evaluation.kernel_bundle import pack_candidates
        from open_cake_ir.evaluation.program import program_components
        manifest, children, _ = program_components(candidate)
        child = children['producer']
        replaced = {**child.artifact_payloads, 'lowered_source': children['epilogue'].artifact_payloads['lowered_source']}
        source_identity = sha256(replaced['lowered_source']).hexdigest()
        report = json.loads(replaced['stage_compilation'])
        report['source_sha256'] = source_identity
        replaced['stage_compilation'] = canonical_json_bytes(report)
        changed_child = replace(child, artifact_payloads=replaced,
            artifact_roles={role: sha256(value).hexdigest() for role, value in replaced.items()})
        changed_manifest = replace(manifest,
            lowered_sources={**manifest.lowered_sources, 'producer': source_identity})
        payloads = {**candidate.artifact_payloads,
                    'launch_manifest': canonical_json_bytes(changed_manifest.as_dict()),
                    'program_bundle': pack_candidates({**children, 'producer': changed_child})}
        forged = seal(payloads, candidate.candidate_sha256, candidate.target, candidate.entry_point, changed_manifest)
        # All child, compiler-report and manifest seals agree with the wrong
        # source. Artifact custody alone accepts; frozen relowering rejects.
        program_components(forged)
        with self.assertRaisesRegex(ValueError, 'stage source differs from its pinned Compiler lowering'):
            self.replay_artifact(authored, forged, workload, policy)

    def test_single_stage_program_keeps_authored_stage_identity(self):
        from open_cake_ir.compiler import Program
        from tests.contracts.test_program_evaluation import ProgramEvaluationTests, workload_for
        from tests.contracts.test_program_rewrites import epilogue_program
        rewritten = self.compiler.rewrite_program(Program.from_dict(epilogue_program()),
            'fuse_pointwise_epilogue', {'producer': 'producer', 'epilogue': 'epilogue',
                                       'schedule_id': 'fused', 'entry_point': 'fused'})
        program = rewritten.program
        candidate, _, _ = ProgramEvaluationTests.build(self, program.document)
        self.assertFalse(candidate.is_program)
        authored = canonical_json_bytes(program.document)
        verified = self.replay_artifact(authored, candidate, workload_for(program),
                                       authority({'backend': 'triton', 'entry_point': 'starter'}))
        view, _ = source_views(sealed_sources(verified, authored), MAX_TURN_SOURCE_BYTES)
        self.assertEqual([row['stage_id'] for row in view['stages']], [program.stages[0].name])
        self.assertEqual(view['stages'][0]['source'].encode(), candidate.artifact_payloads['lowered_source'])

    def feedback_fixture(self):
        from tests.contracts.test_optimization_history import receipt
        authored, candidate, _ = self.program_candidate()
        program_authored, program, _ = self.program_candidate('alternate')
        policy = authority({'backend': 'triton', 'entry_point': 'starter'})
        candidates = (candidate.candidate_sha256, program.candidate_sha256, 'f'*64)
        launchables = dict(zip(candidates[:2], (candidate, program)))
        payloads = dict(zip(candidates[:2], (authored, program_authored)))
        specification = SimpleNamespace(environment_kind='open_cake', document={
            'authoring': policy, 'evaluation_protocol': {'searches_per_turn': 2},
            'execution': {'target': candidate.target}})
        actions = [{'ordinal': i, 'kind': 'submit', 'candidate_sha256': identity, 'action_sha256': identity}
                   for i, identity in enumerate(candidates)]
        rows = [{'candidate_sha256': identity, 'disposition': 'launchable', 'semantic_sha256': None,
                 'diagnostics': {'findings': [], 'omitted_findings': 0, 'text_truncated': False}, 'cost': None}
                for identity in candidates]
        receipts = {identity: replace(receipt('a', i+1), candidate_sha256=identity)
                    for i, identity in enumerate(candidates[:2])}
        kwargs = dict(turn=1, candidates=candidates, filter_rows=rows, selected=candidates[0],
            receipts=receipts, attributions={}, rejected_feedback={}, actions=actions, arm='open_cake',
            specification=specification, launchables=launchables, authored=payloads)
        return kwargs

    def test_next_request_replay_reconstructs_each_candidate_and_stage(self):
        from open_cake_ir.lab.task_package import TaskPackage, render_task_request
        kwargs = self.feedback_fixture()
        feedback = derive_turn_feedback(**kwargs)
        self.assertNotIn('generated_source', feedback)
        self.assertEqual(len(feedback['candidate_results'][1]['generated_source']['stages']), 2)
        self.assertEqual(feedback['candidate_results'][2]['generated_source']['reason'], 'no_sealed_search_candidate')
        package = TaskPackage('source-fixture', 'open_cake', 'Frozen task', 'Frozen permissions')
        prompt, raw = render_task_request(package, {'previous_feedback': feedback})
        self.assertIn('CAKE_OP:', prompt)
        records = {'next': raw}
        evidence = SimpleNamespace(read_object=lambda ref: records[ref['key']])
        events = [
            {'kind': 'author_actions_resolved', 'payload': {'turn': 1, 'actions': kwargs['actions']}},
            {'kind': 'candidate_set_filtered', 'payload': {'turn': 1, 'order': kwargs['filter_rows']}},
        ]
        for identity in kwargs['receipts']:
            events.extend([
                {'kind': 'launchable_candidate_sealed', 'payload': {'turn': 1, 'candidate_sha256': identity}},
                {'kind': 'candidate_evaluated', 'payload': {'turn': 1, 'candidate_sha256': identity, 'purpose': 'search'}},
            ])
        events += [
            {'kind': 'candidate_selected', 'payload': {'turn': 1, 'candidate_sha256': kwargs['selected']}},
            {'kind': 'provider_turn_completed', 'payload': {'turn': 2, 'objects': [
                {'role': 'provider_reference_bundle', 'key': 'next'}]}},
        ]
        replay = dict(events=events, evidence=evidence, specification=kwargs['specification'],
            provider_candidates_by_turn={1: kwargs['candidates']},
            receipts={(1, 'search', k): v for k, v in kwargs['receipts'].items()},
            launchables={(1, k): v for k, v in kwargs['launchables'].items()},
            authored={(1, k): v for k, v in kwargs['authored'].items()})
        replay_feedback(**replay)
        # The received bundle is self-consistent; independently reconstructed
        # source provenance, not its own rendering, rejects the substituted view.
        forged = deepcopy(feedback)
        forged['candidate_results'][0]['generated_source'] = forged['candidate_results'][1]['generated_source']
        records['next'] = package.evidence_bundle({'previous_feedback': forged})
        with self.assertRaisesRegex(ReplayRefusal, 'reconstructed from completed earlier'):
            replay_feedback(**replay)
        records['next'] = raw
        late = [event for event in events if event['kind'] != 'launchable_candidate_sealed']
        late += [event for event in events if event['kind'] == 'launchable_candidate_sealed']
        with self.assertRaises(ReplayRefusal):
            replay_feedback(**{**replay, 'events': late})
        with self.assertRaises(ReplayRefusal):
            replay_feedback(**{**replay, 'launchables': {(2, k): v for k, v in kwargs['launchables'].items()}})
        # Neither a terminal state nor a provider fault fabricates a subsequent
        # author handoff. They can retain the same feedback without a Turn 2 bundle.
        for last in ({'kind': 'search_completed', 'payload': {'state': {'previous_feedback': feedback}}},
                     {'kind': 'run_fault', 'payload': {'turn': 2, 'stage': 'provider'}}):
            replay_feedback(**{**replay, 'events': events[:-1] + [last]})
        old = deepcopy(kwargs['specification'].document)
        old['authoring']['feedback'].remove(GENERATED_SOURCE_V1)
        old_feedback = derive_turn_feedback(**{**kwargs, 'specification': SimpleNamespace(document=old)})
        self.assertTrue(all('generated_source' not in row for row in old_feedback['candidate_results']))

    def test_codex_initial_and_resume_deliver_the_retained_source_bundle(self):
        from open_cake_ir.lab.provider_events import ProviderTurn
        from open_cake_ir.lab.providers import CodexRunProvider, ProviderQualificationReceipt
        from open_cake_ir.lab.task_package import TaskPackage, materialize_task_package
        from tests.contracts.test_provider_runtime import ProviderRuntimeContractTests
        fixture = ProviderRuntimeContractTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        workspace = fixture.root / 'task'
        workspace.mkdir()
        package = TaskPackage('own-source', 'open_cake', 'Frozen CPU task', 'Frozen CPU rules')
        materialize_task_package(workspace, package)
        builder = fixture.builder(workspace=workspace)
        # In-memory CPU qualification double only: the adapter below records
        # argv and writes bytes; it never starts a provider or issues a receipt.
        qualification = ProviderQualificationReceipt(provider_revision=builder.provider_revision,
            executable_sha256=sha256(fixture.executable.read_bytes()).hexdigest(),
            configuration_sha256=builder.configuration_sha256, initial_and_resume_equivalent=True,
            file_lifecycle_observed=True, usage_observed=True, qualified=True,
            scope='live_two_turn_current_provider')
        invocations = []
        class Adapter:
            def execute(self, invocation, *, candidate_path, expected_terminal_message, **kwargs):
                invocations.append(invocation)
                payload = b'{"CPU_fixture":true}'
                candidate_path.write_bytes(payload)
                return ProviderTurn(thread_id='01234567-89ab-cdef-0123-456789abcdef',
                    provider_tokens=len(invocations)*100, candidates=(payload,), raw_submission=payload,
                    candidate_sha256s=(sha256(payload).hexdigest(),), raw_events=b'{}\n',
                    raw_events_sha256=sha256(b'{}\n').hexdigest(),
                    terminal_message=expected_terminal_message, terminal_message_count=1,
                    normalization='single_exact')
        provider = CodexRunProvider(qualification=qualification, builders={'own-source': builder},
                                   task_packages={'own-source': package}, adapter=Adapter())
        feedback = derive_turn_feedback(**self.feedback_fixture())
        previous = None
        for turn, state in ((1, {'previous_feedback': {}}), (2, {'previous_feedback': feedback})):
            result = provider.turn(SimpleNamespace(run_id='own-source', arm='open_cake',
                environment_kind='open_cake', turn=turn, thread_id=previous,
                maximum_candidates_per_turn=2, cumulative_provider_tokens=100*(turn-1), state_card=state))
            prompt = invocations[-1].argv[-1]
            self.assertTrue(prompt.endswith(result.reference_bundle.decode('utf-8')))
            delivered = json.loads(prompt.split('\n\n', 1)[1])
            self.assertEqual(delivered, json.loads(result.reference_bundle))
            self.assertEqual(delivered['state_card'], state)
            if turn == 2:
                self.assertIn('resume', invocations[-1].argv)
                self.assertEqual(delivered['state_card']['previous_feedback']['candidate_results'],
                                 feedback['candidate_results'])
            previous = result.thread_id

    def test_task_entry_flag_binds_permission_before_any_execution(self):
        from tools import launch_task
        with tempfile.TemporaryDirectory() as directory:
            captured = []
            original = launch_task.task_run_inputs
            def inputs(*args, **kwargs):
                value = original(*args, **kwargs)
                captured.append(value)
                return value
            arguments = ['--task', 'rmsnorm', '--backend', 'metal-m1-pro', '--model', 'gpt-6.1-sol',
                '--harness', 'codex', '--effort', 'xhigh', '--workspace', str(Path(directory).resolve()/'task'),
                '--rows', '2', '--columns', '7', '--generated-source-feedback']
            with patch.object(launch_task, '_provider_executable', return_value=Path('/fixture/provider')), \
                 patch.object(launch_task, 'task_run_inputs', side_effect=inputs), \
                 patch.object(launch_task, '_admit_stack', side_effect=RuntimeError('stop before admission')):
                with self.assertRaisesRegex(RuntimeError, 'stop before admission'):
                    launch_task.main(arguments)
            self.assertIn(GENERATED_SOURCE_V1, captured[0]['authoring']['feedback'])
            self.assertEqual(captured[0]['authoring']['provider']['model'], 'gpt-6.1-sol')
            self.assertEqual(captured[0]['authoring']['provider']['reasoning_effort'], 'xhigh')

    def test_real_loop_delivers_sealed_msl_next_turn_but_not_after_terminal_or_fault(self):
        from open_cake_ir.evidence import EvidenceStore
        from open_cake_ir.lab.execution import _execute_run
        from open_cake_ir.lab.environments import EnvironmentResult
        from open_cake_ir.lab.faults import RunProtocolFault
        from open_cake_ir.lab.task_package import TaskPackage
        from tests.contracts.test_lab import FakeProvider, FakeEvaluator, _submission_envelope
        values = [self.metal_candidate('first'), self.metal_candidate('second')]
        payloads = tuple(item[0] for item in values)
        candidates = {item[1].candidate_sha256: item[1] for item in values}
        workload, policy = values[0][2:]
        protocol = {'case_id': 'odd', 'searches_per_turn': 2}
        document = {'sequence': 1, 'authoring': policy, 'evaluation_protocol': protocol,
            'compiler_revision': {'path': 'compiler/revision.json', 'revision_id': 'CPU-fixture'},
            'knowledge': {'transformations': []}, 'reference_inputs': {},
            'workload': {'canonical_sha256': workload.canonical_sha256},
            'execution': {'target': workload.target},
            'budget': {'limit': None, 'checkpoints': [], 'maximum_turns': 2,
                'maximum_candidates_per_turn': 2, 'maximum_compilations': 8,
                'wall_time_seconds': 100, 'active_authoring_time_seconds': 80,
                'confirmation_wall_time_seconds': 10,
                'evaluation_limits': {'search': 4, 'confirmatory': 1, 'attribution': 1}}}
        specification = SimpleNamespace(document=document, run_id='source-loop', condition_id='open_cake',
            environment_kind='open_cake', canonical_sha256=sha256(canonical_json_bytes(document)).hexdigest(),
            terminal_policy={'endpoint_policy': 'normal_budget_terminal_v1'})
        class Provider(FakeProvider):
            fail = False
            def turn(self, request):
                if self.fail and request.turn == 2:
                    raise RunProtocolFault('provider_fault', 'CPU fixture stops before completion')
                observed = super().turn(request)
                return replace(observed, candidates=payloads, candidate_sha256s=tuple(candidates),
                               raw_submission=_submission_envelope(request.arm, payloads))
        class Evaluator(FakeEvaluator):
            def candidate_position(self, candidate):
                return 'open_cake', tuple(candidates).index(candidate.candidate_sha256) + 1
            def evaluate(self, candidate, **kwargs):
                observed = super().evaluate(candidate, **kwargs)
                self.observed[(kwargs['purpose'], candidate.candidate_sha256)] = observed.final_receipt
                return observed
        def build(submission, *, compilation=None):
            return EnvironmentResult('launchable', submission.sha256, candidates[submission.sha256], {})
        environment = SimpleNamespace(media_type='application/vnd.open-cake.schedule+json', build=build)
        for failure in (False, True):
            with self.subTest(provider_fault=failure), tempfile.TemporaryDirectory() as temporary:
                provider = Provider()
                provider.fail = failure
                provider.packages = {'source-loop': TaskPackage('source-loop', 'open_cake', 'CPU task', 'CPU rules')}
                evaluator = Evaluator(protocol, sha256(canonical_json_bytes(protocol)).hexdigest(),
                    workload.canonical_sha256, compiler_revision_reference=document['compiler_revision'])
                evaluator.observed = {}
                evidence = EvidenceStore.create(Path(temporary)/'evidence')
                _execute_run(specification, project_root=ROOT, evidence=evidence, clock=lambda: 0.0,
                    provider=provider, environment=environment, evaluator=evaluator)
                events = evidence.replay_events('source-loop')
                completions = [event for event in events if event['kind'] == 'provider_turn_completed']
                self.assertEqual(len(completions), 1 if failure else 2)
                faults = [event['payload'] for event in events if event['kind'] == 'run_fault']
                self.assertEqual(len(faults), 1 if failure else 0, faults)
                if failure:
                    self.assertEqual(faults[0]['stage'], 'provider')
                else:
                    received = provider.requests[1].state_card['previous_feedback']
                    self.assertEqual(received['source_turn'], 1)
                    self.assertTrue(all(row['generated_source']['stages'][0]['status'] == 'available'
                                        for row in received['candidate_results']))
                    self.assertNotIn('generated_source', provider.requests[1].state_card['optimization_history'])
                by_turn = {event['payload']['turn']: tuple(candidates) for event in completions}
                launchables = {(event['payload']['turn'], event['payload']['candidate_sha256']):
                    self.replay_artifact(payloads[tuple(candidates).index(event['payload']['candidate_sha256'])],
                        candidates[event['payload']['candidate_sha256']], workload, policy)
                    for event in events if event['kind'] == 'launchable_candidate_sealed'}
                receipts = {(event['payload']['turn'], event['payload']['purpose'], event['payload']['candidate_sha256']):
                    evaluator.observed[(event['payload']['purpose'], event['payload']['candidate_sha256'])]
                    for event in events if event['kind'] == 'candidate_evaluated' and 'turn' in event['payload']}
                replay_feedback(events=events, evidence=evidence, specification=specification,
                    provider_candidates_by_turn=by_turn, receipts=receipts, launchables=launchables,
                    authored={(turn, identity): payload for turn in by_turn
                              for identity, payload in zip(candidates, payloads)}, fault_turn=2 if failure else None)

    def test_management_cell_flag_is_explicit_and_forwarded_without_changing_v1(self):
        from tools import kernel_experiment
        from tests.contracts.test_gpu_infra import ExperimentInputTests
        import io
        fixture = ExperimentInputTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        config = fixture._per_cell_config()
        for value in ('yes', 1, None):
            config['cells'][0]['generated_source_feedback'] = value
            with self.assertRaisesRegex(ValueError, 'explicit boolean'):
                kernel_experiment.validate(config)
        config['cells'][0]['generated_source_feedback'] = True
        kernel_experiment.validate(config)
        old = deepcopy(fixture.config)
        old['cells'][0]['generated_source_feedback'] = True
        with self.assertRaises(ValueError):
            kernel_experiment.validate(old)
        payload = {'cell': config['cells'][0], 'source_commit': '0'*40, 'scaffold': 'CPU instructions',
                   'provider': config['provider'], 'budget': config['budget']}
        with patch('sys.stdin', io.StringIO(json.dumps(payload))), \
             patch('subprocess.run', return_value=SimpleNamespace(returncode=0)) as execute:
            with self.assertRaises(SystemExit) as result:
                exec(kernel_experiment._NODE, {})
        self.assertEqual(result.exception.code, 0)
        self.assertIn('--generated-source-feedback', execute.call_args.args[0])
