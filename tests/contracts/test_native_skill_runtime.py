"""Runtime boundary mechanics against synthetic sealed archives, never a live model.

Entrance checks stop at the next owned boundary; separate zero-turn/first-fault
cases execute and replay the real Run engine with CPU fixtures. Neither establishes
real model qualification or device execution. The pre-existing refusal matrix
still tests missing qualifications before files, callbacks and early returns.
"""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from open_cake_ir.lab import execution, execution_admission, bindings
from open_cake_ir.lab.admission import admit_native_skill_authoring
from open_cake_ir.lab.native_skill_qualification import verify_qualification_evidence
from open_cake_ir.lab.native_skills import NativeSkillPackage
from open_cake_ir.lab.provider_policy import execution_configuration
from open_cake_ir.lab.providers import CodexRunProvider, QualifiedRunProvider
from open_cake_ir.lab.replay import _replay_matched_run
from open_cake_ir.lab.task_package import TaskPackage, materialize_task_package
from open_cake_ir.serialization import canonical_json_bytes
from open_cake_ir.tasks import compose
from tests.contracts import test_native_qualification_admission as admission_fixtures
from tests.contracts.test_provider_qualification import ROOT


class ReachedNextBoundary(Exception):
    pass


class NativeSkillRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = admission_fixtures.NativeQualificationAdmissionTests(methodName='runTest')
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.root = self.fixture.root
        self.receipt, self.anchor, self.provider = self.fixture.synthetic_archive()
        self.authoring = {'environment_kind': 'open_cake', 'provider': self.provider}

    def test_admission_matches_configuration_and_environment_not_condition_name(self):
        for kind in ('open_cake', 'direct_cuda'):
            self.assertEqual(admit_native_skill_authoring(
                authoring=dict(self.authoring, environment_kind=kind), project_root=ROOT), self.receipt)
        with self.assertRaisesRegex(ValueError, 'no retained turns for environment: native_triton'):
            admit_native_skill_authoring(authoring=dict(self.authoring, environment_kind='native_triton'),
                                         project_root=ROOT)
        for field, value in (('model', 'different-model'), ('reasoning_effort', 'low')):
            author = deepcopy(self.authoring)
            author['provider'][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'qualification bytes or capability'):
                admit_native_skill_authoring(authoring=author, project_root=ROOT)
        author = deepcopy(self.authoring)
        author['provider']['native_skill_package']['sha256'] = 'a'*64
        with self.assertRaises(ValueError):
            admit_native_skill_authoring(authoring=author, project_root=ROOT)

    def test_direct_provider_requires_archive_and_covered_environment(self):
        package = NativeSkillPackage.read(ROOT, self.root/'skills.tar')
        task = TaskPackage('independent-run', 'condition-a', 'task', 'agents',
                           environment_kind='open_cake', native_skill_package=package)
        workspace = self.root/'runtime-task'
        workspace.mkdir()
        materialize_task_package(workspace, task)
        builder = SimpleNamespace(configuration=execution_configuration(self.provider),
            provider_revision=self.receipt.provider_revision, executable=self.root/'codex',
            workspace=workspace, native_skill_package=package,
            qualified_system_skills_sha256=self.receipt.system_skills_sha256)
        for provider_type in (QualifiedRunProvider, CodexRunProvider):
            def build(anchor=self.anchor, selected=task, receipt=self.receipt):
                return provider_type(qualification=receipt, builders={task.run_id: builder},
                    task_packages={task.run_id: selected}, adapter=Mock(), qualification_anchor=anchor)
            with self.subTest(provider=provider_type.__name__):
                actual = build()
                self.assertEqual(actual.qualification_sha256, self.receipt.canonical_sha256)
                with self.assertRaisesRegex(ValueError, 'not qualified'):
                    build(anchor=None)
                with self.assertRaisesRegex(ValueError, 'not qualified'):
                    build(receipt=self.fixture.receipt)
                with self.assertRaisesRegex(ValueError, 'unverified'):
                    build(anchor=dict(self.anchor, terminal_seal_sha256='a'*64))
                for change in ({'schema_version': 2}, {'immediate_audit_integrity': False}, {'unexpected': True}):
                    with self.assertRaisesRegex(ValueError, 'anchor differs'):
                        build(anchor=dict(self.anchor, **change))
                with self.assertRaisesRegex(ValueError, 'no retained turns for environment'):
                    build(selected=replace(task, environment_kind='native_triton'))

    def test_positive_public_entries_admit_before_the_next_owned_boundary(self):
        specification = SimpleNamespace(document={'authoring': self.authoring})
        lock = SimpleNamespace(document={'resolved_inputs': {'arm_environments': {
            'condition-a': self.authoring}}})
        common = dict(project_root=ROOT, workload_loader=Mock(), clock=Mock())
        for entry in ('run', 'campaign', 'factory'):
            owner, boundary = ((execution.RunSpecification, 'from_dict') if entry == 'run' else
                (execution, 'require_qualified_clean_start_execution') if entry == 'campaign' else
                (execution.CampaignLock, 'from_dict'))
            with self.subTest(entry=entry), patch.object(execution, 'admit_new_campaign_path'), \
                 patch.object(owner, boundary, side_effect=ReachedNextBoundary) as reached, \
                 patch.object(execution.EvidenceStore, 'create') as evidence:
                with self.assertRaises(ReachedNextBoundary):
                    if entry == 'run':
                        execution.execute_run(specification, '/unwritten', **common,
                            provider=Mock(), environment=Mock(), evaluator=Mock(), task_package=Mock())
                    elif entry == 'campaign':
                        execution.execute_campaign(lock, '/unwritten', **common,
                            provider=Mock(), environments={}, evaluator=Mock(), validate_authoring=Mock())
                    else:
                        execution.execute_campaign_with_factory(lock, '/unwritten', **common,
                            runtime_factory=Mock(), task_package=Mock(), validate_run=Mock(), validate_authoring=Mock())
                reached.assert_called_once()
                evidence.assert_not_called()

        with patch.object(bindings, 'load_compiler_reference', side_effect=ReachedNextBoundary) as reached:
            # This point precedes event reads and all zero-turn/first-fault exits.
            replay_spec = SimpleNamespace(document={'authoring': dict(self.authoring,
                reference_access='known_kernel_reproduction'), 'compiler_revision': {
                'revision_id': 'fixture', 'path': 'compiler/fixture.json', 'sha256': 'a'*64}})
            # The compiler reference parser is a separate owner; reaching it suffices
            # to distinguish native admission from the former unconditional refusal.
            with patch('open_cake_ir.lab.replay._identity_reference', return_value={}), \
                 self.assertRaises(ReachedNextBoundary):
                _replay_matched_run(Mock(), Mock(), replay_spec, project_root=ROOT,
                                    manifest_parser=Mock(), task_package=Mock())
            reached.assert_called_once()

    def test_compose_and_binding_entries_check_before_runtime_work(self):
        specification = SimpleNamespace(document={'authoring': self.authoring})
        with patch('open_cake_ir.lab.custody.admit_new_campaign_path',
                   side_effect=ReachedNextBoundary) as reached, \
             patch.object(compose, 'run_runtime_factory') as factory:
            with self.assertRaises(ReachedNextBoundary):
                compose.execute_run_from_config(ROOT, specification, '/unread/runtime', '/unwritten')
            reached.assert_called_once()
            factory.assert_not_called()
        # Patch the actual next callable after admission, not shared pathlib APIs.
        with patch.object(compose, 'TaskLab') as lab, \
             patch.object(compose, 'load_runtime_config') as runtime:
            lab.return_value.preflight_run.side_effect = ReachedNextBoundary
            factory = compose.run_runtime_factory(ROOT, self.root/'codex')
            with self.assertRaises(ReachedNextBoundary):
                factory(specification, self.root/'unwritten')
            lab.return_value.preflight_run.assert_called_once()
            runtime.assert_not_called()
        protocol = {}
        evaluator = SimpleNamespace(protocol=protocol,
            protocol_sha256=sha256(canonical_json_bytes(protocol)).hexdigest())
        specification.document['evaluation_protocol'] = protocol
        with patch('open_cake_ir.lab.admission.admit_run_inputs', side_effect=ReachedNextBoundary) as reached:
            with self.assertRaises(ReachedNextBoundary):
                execution_admission.validate_run_bindings(specification, project_root=ROOT,
                    workload_loader=Mock(), provider=Mock(), environment=Mock(),
                    evaluator=evaluator, task_package=Mock())
            reached.assert_called_once()

    def test_preparation_checks_configuration_and_archive_before_credentials(self):
        config = {'provider': {'executable': str(self.root/'codex'),
                              'auth_source': str(self.root/'auth.json')}}
        row = SimpleNamespace(runtime_kind='fixture')
        def bind(provider=self.provider, receipt_path=None, anchor_path=None, kinds=('open_cake',)):
            return bindings.bind_cli_provider(ROOT, provider, row,
                runtime_path=self.root/'unread-runtime.json',
                receipt_path=receipt_path or self.provider['qualification']['path'],
                anchor_path=anchor_path or self.provider['qualification_anchor']['path'],
                environment_kinds=kinds)
        with patch.object(bindings, 'load_runtime_config', return_value=config), \
             patch('open_cake_ir.lab.author_home.verify_auth_source') as auth:
            bound, observed = bind()
            self.assertEqual(observed, config)
            self.assertEqual(bound['qualification'], self.provider['qualification'])
            auth.assert_called_once()
            auth.reset_mock()
            with self.assertRaisesRegex(ValueError, 'qualification bytes or capability'):
                bind(provider=dict(self.provider, model='unqualified-model'))
            auth.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'no retained turns for environment'):
                bind(kinds=('open_cake', 'native_triton'))
            auth.assert_not_called()
            stale = self.root/'stale-anchor.json'
            stale.write_bytes(canonical_json_bytes(dict(self.anchor, terminal_seal_sha256='a'*64)))
            with self.assertRaisesRegex(ValueError, 'unverified'):
                bind(anchor_path=stale)
            auth.assert_not_called()
        _, _, old_provider = self.fixture.synthetic_archive(capability=None)
        with patch.object(bindings, 'load_runtime_config') as runtime, \
             patch('open_cake_ir.lab.author_home.verify_auth_source') as auth:
            with self.assertRaisesRegex(ValueError, 'not qualified'):
                bind(receipt_path=old_provider['qualification']['path'],
                     anchor_path=old_provider['qualification_anchor']['path'])
            runtime.assert_not_called()
            auth.assert_not_called()

    def test_zero_turn_and_first_fault_runs_seal_and_replay_without_evaluation(self):
        from open_cake_ir.evidence import EvidenceStore
        from open_cake_ir.lab import RunSpecification
        from open_cake_ir.lab.faults import RunProtocolFault
        from open_cake_ir.tasks.runtime import TaskLab
        from tests.contracts import test_run_specification as run_fixtures
        from tests.contracts import test_native_skill_run as native_fixtures
        from tests.contracts import test_lab as lab_fixtures

        template = run_fixtures.IndependentRunTests(methodName='runTest')
        self.addCleanup(template.doCleanups)
        _, original = template.fixture(condition='open_cake')
        for mode in ('zero_turn', 'call_fault', 'returned_native_rejection'):
            document = deepcopy(original.document)
            document['authoring']['provider'] = deepcopy(self.provider)
            specification = RunSpecification.from_dict(document)
            readings = [0.0]
            def clock():
                if mode != 'zero_turn':
                    return 0.0
                value = readings[0]
                readings[0] = document['budget']['wall_time_seconds'] + 1.0
                return value
            lab = TaskLab(ROOT, clock=clock)
            provider = SimpleNamespace(provider_revision=self.receipt.provider_revision,
                qualification_sha256=self.receipt.canonical_sha256,
                executable_sha256=self.receipt.executable_sha256,
                configuration=execution_configuration(self.provider), turn=Mock())
            if mode == 'returned_native_rejection':
                # Deliberately return a Turn from a different Run/configuration.
                # This exercises the real archive refusal and its first-fault replay.
                foreign = native_fixtures.NativeSkillRunEvidenceTests(methodName='runTest')
                self.addCleanup(foreign.doCleanups)
                foreign.setUp()
                provider.turn.return_value = foreign.make_turn(1)
            else:
                provider.turn.side_effect = RunProtocolFault('provider_fault', 'CPU fixture call failed before input capture')
            evaluator = lab_fixtures.FakeEvaluator(document['evaluation_protocol'],
                sha256(canonical_json_bytes(document['evaluation_protocol'])).hexdigest(),
                document['workload']['canonical_sha256'])
            environment = lab_fixtures.FakeEnvironment('open_cake', document['authoring'])
            with self.subTest(mode=mode):
                run = lab.execute_run(specification, self.root/mode, provider=provider,
                                      environment=environment, evaluator=evaluator)
                audit, replay = lab.audit_run(run)
                self.assertTrue(audit.archive_integrity)
                self.assertTrue(audit.filesystem_custody_verified)
                self.assertTrue(replay, replay.refusals)
                self.assertEqual(evaluator.calls, 0)
                events = EvidenceStore.open(run.evidence_root).replay_events(specification.run_id)
                self.assertNotIn('provider_turn_completed', [event['kind'] for event in events])
                if mode == 'zero_turn':
                    provider.turn.assert_not_called()
                    self.assertEqual(audit.protocol_adherence, 'adhered')
                else:
                    provider.turn.assert_called_once()
                    self.assertEqual(audit.protocol_adherence, 'provider_fault')
                    if mode == 'returned_native_rejection':
                        from open_cake_ir.lab.native_skill_fault import has_native_rejection
                        fault = next(event['payload'] for event in events if event['kind'] == 'run_fault')
                        self.assertTrue(has_native_rejection(fault))

    def test_request_environment_shape_is_checked_before_archive_open(self):
        for kinds in (None, 'open_cake', [None], ['']):
            with self.subTest(kinds=kinds), patch('open_cake_ir.evidence.EvidenceStore.open') as opened:
                with self.assertRaisesRegex(ValueError, 'requested environment kinds differ'):
                    verify_qualification_evidence(qualification=None, anchor=None,
                                                 required_environment_kinds=kinds)
                opened.assert_not_called()


if __name__ == '__main__':
    unittest.main()
