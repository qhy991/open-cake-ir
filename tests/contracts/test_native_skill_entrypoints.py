"""CPU entrypoint checks; these do not observe native Codex skill delivery."""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from open_cake_ir.lab.author_home import ISOLATED_AUTH_ONLY_V1, ISOLATED_SKILL_PACKAGE_V1
from open_cake_ir.lab import execution
from open_cake_ir.tasks import compose
from tools import launch_task, qualify_codex_provider as qualifier


class NativeSkillEntrypointTests(unittest.TestCase):
    def qualification_args(self):
        return ['--harness', 'codex', '--model', 'fixture-model', '--reasoning-effort', 'xhigh',
                '--executable', '/unread-skill-fixture/codex', '--provider-revision', 'fixture-provider',
                '--output-schema', '/unread-skill-fixture/schema.json',
                '--workspace', '/unread-skill-fixture/workspace',
                '--receipt-output', '/unread-skill-fixture/receipt.json',
                '--anchor-output', '/unread-skill-fixture/anchor.json',
                '--evidence-root', '/unread-skill-fixture/evidence', '--run-id', 'fixture',
                '--auth-source', '/unread-skill-fixture/auth.json']

    def launch_args(self):
        return ['--task', 'rmsnorm', '--backend', 'metal-m1-pro', '--harness', 'codex',
                '--model', 'fixture-model', '--effort', 'xhigh',
                '--workspace', '/unread-skill-fixture/workspace',
                '--author-skill-package', '/unread-skill-fixture/skills.tar']

    def test_live_qualification_refuses_before_reading_provider_auth_or_package(self):
        for policy in ([], ['--author-home-policy', ISOLATED_SKILL_PACKAGE_V1]):
            arguments = self.qualification_args() + policy + [
                '--author-skill-package', '/unread-skill-fixture/skills.tar']
            with (self.subTest(policy=policy), mock.patch.object(sys, 'argv', ['qualify'] + arguments),
                  mock.patch.object(qualifier.NativeSkillPackage, 'read') as read,
                  mock.patch.object(qualifier, 'external_file') as external,
                  mock.patch.object(qualifier, '_new_path') as output,
                  contextlib.redirect_stderr(io.StringIO()) as stderr):
                with self.assertRaises(SystemExit) as error:
                    qualifier.main()
                self.assertEqual(error.exception.code, 2)
                self.assertIn('native skill discovery and delivery', stderr.getvalue())
                for operation in (read, external, output):
                    operation.assert_not_called()

    def test_fixture_qualification_requires_matching_package_policy(self):
        for options in (
            ['--author-home-policy', ISOLATED_SKILL_PACKAGE_V1],
            ['--author-home-policy', ISOLATED_AUTH_ONLY_V1,
             '--author-skill-package', '/unread-skill-fixture/skills.tar'],
        ):
            arguments = self.qualification_args() + ['--fixture-only'] + options
            with (self.subTest(options=options), mock.patch.object(sys, 'argv', ['qualify'] + arguments),
                  mock.patch.object(qualifier.NativeSkillPackage, 'read') as read,
                  contextlib.redirect_stderr(io.StringIO()) as stderr):
                with self.assertRaises(SystemExit):
                    qualifier.main()
                self.assertIn('requires --author-skill-package', stderr.getvalue())
                read.assert_not_called()

    def test_fixture_skill_policy_is_codex_only(self):
        arguments = self.qualification_args()
        arguments[arguments.index('codex')] = 'claude-code'
        arguments += ['--fixture-only', '--author-skill-package', '/unread-skill-fixture/skills.tar']
        with (mock.patch.object(sys, 'argv', ['qualify'] + arguments),
              mock.patch.object(qualifier.NativeSkillPackage, 'read') as read,
              contextlib.redirect_stderr(io.StringIO()) as stderr):
            with self.assertRaises(SystemExit):
                qualifier.main()
            self.assertIn('Codex harness', stderr.getvalue())
            read.assert_not_called()

    def test_launch_refuses_live_skill_policy_before_preparation_or_existing_receipt(self):
        for options in ([], ['--preflight-only'],
                        ['--qualification', '/unread-skill-fixture/receipt.json',
                         '--qualification-anchor', '/unread-skill-fixture/anchor.json']):
            with (self.subTest(options=options),
                  mock.patch.object(launch_task, '_new_workspace') as workspace,
                  mock.patch.object(launch_task, '_provider_executable') as executable,
                  mock.patch.object(launch_task, '_codex_auth_source') as auth,
                  mock.patch.object(launch_task, '_qualify') as qualify,
                  contextlib.redirect_stderr(io.StringIO()) as stderr):
                with self.assertRaises(SystemExit):
                    launch_task.main(self.launch_args() + options)
                self.assertIn('native skill discovery and delivery', stderr.getvalue())
                for operation in (workspace, executable, auth, qualify):
                    operation.assert_not_called()

    def test_launch_skill_package_requires_known_kernel_codex_authoring(self):
        for option, value in (('--harness', 'claude-code'), ('--reference-access', 'clean_start')):
            arguments = self.launch_args() + ['--baseline-only']
            if option in arguments:
                arguments[arguments.index(option) + 1] = value
            else:
                arguments += [option, value]
            with (self.subTest(option=option), mock.patch.object(launch_task, '_new_workspace') as workspace,
                  contextlib.redirect_stderr(io.StringIO()) as stderr):
                with self.assertRaises(SystemExit):
                    launch_task.main(arguments)
                self.assertIn('Codex known-kernel authoring', stderr.getvalue())
                workspace.assert_not_called()

    def test_qualification_command_carries_the_explicit_package_policy(self):
        # Only command construction is exercised. Main's live gate stays in force,
        # and neither the provider nor the qualifier subprocess is executed here.
        arguments = SimpleNamespace(qualification=None, provider_revision='fixture-provider',
            harness='codex', model='fixture-model', effort='xhigh', max_candidates=1,
            source_file=True, auth_source=Path('/unread-skill-fixture/auth.json'),
            author_skill_package=Path('/unread-skill-fixture/skills.tar'),
            response_model_alias=[], wall_seconds=30)
        responses = [subprocess.CompletedProcess([], 0, stdout='fixture-provider', stderr=''),
                     subprocess.CompletedProcess([], 0, stdout='', stderr='')]
        with (mock.patch.object(launch_task.subprocess, 'run', side_effect=responses) as run,
              mock.patch.object(launch_task, '_write'),
              mock.patch.object(launch_task, '_report_provider_limitations')):
            launch_task._qualify(ROOT, Path('/unread-skill-fixture/workspace'), arguments,
                                 Path('/unread-skill-fixture/codex'), Path('/unread-skill-fixture/starter.py'))
        command = run.call_args_list[1].args[0]
        self.assertEqual(command[command.index('--author-home-policy') + 1], ISOLATED_SKILL_PACKAGE_V1)
        self.assertEqual(command[command.index('--author-skill-package') + 1], str(arguments.author_skill_package))

    def test_retained_invocation_and_resume_bind_both_home_and_original_package(self):
        package = SimpleNamespace(reference={'path': '/fixture/skills.tar', 'sha256': '0' * 64})
        shared = dict(cwd=Path('/fixture/task'), sandbox='workspace-write', provider_revision='fixture',
                      removed_environment=(), codex_home=Path('/fixture/codex-home'),
                      user_home=Path('/fixture/user-home'), native_skill_package=package)
        initial = SimpleNamespace(argv=('codex', 'exec', 'prompt-1'), thread_id=None, **shared)
        resumed = SimpleNamespace(argv=('codex', 'exec', 'resume', 'thread-1', 'prompt-2'),
                                  thread_id='thread-1', **shared)
        qualifier._validate_invocation_pair(initial, resumed, thread_id='thread-1')
        retained = qualifier._invocation_document(initial)
        self.assertEqual(retained['user_home'], '/fixture/user-home')
        self.assertEqual(retained['codex_home'], '/fixture/codex-home')
        self.assertEqual(retained['native_skill_package'], package.reference)
        for field, value in (
            ('user_home', Path('/fixture/another-user-home')),
            ('native_skill_package', SimpleNamespace(reference={'path': '/fixture/other.tar', 'sha256': '1' * 64})),
        ):
            with self.subTest(field=field):
                changed = SimpleNamespace(**{**vars(resumed), field: value})
                with self.assertRaisesRegex(ValueError, 'initial and resume environments differ'):
                    qualifier._validate_invocation_pair(initial, changed, thread_id='thread-1')

    def test_all_public_execution_entries_refuse_fixture_receipts_and_custom_providers(self):
        authoring = {'provider': {'author_home_policy': ISOLATED_SKILL_PACKAGE_V1}}
        specification = SimpleNamespace(document={'authoring': authoring})
        # A new-policy arm must also be found after an otherwise unaffected arm.
        lock = SimpleNamespace(document={'resolved_inputs': {'arm_environments': {
            'native_triton': {'provider': {'author_home_policy': ISOLATED_AUTH_ONLY_V1}},
            'open_cake': authoring,
        }}})
        for entry in ('run', 'campaign', 'campaign_with_factory'):
            with self.subTest(entry=entry), contextlib.ExitStack() as stack:
                boundaries = [stack.enter_context(mock.patch.object(owner, name)) for owner, name in (
                    (execution, 'admit_new_campaign_path'),
                    (execution.EvidenceStore, 'create'),
                    (execution.RunSpecification, 'from_dict'),
                    (execution.CampaignLock, 'from_dict'),
                    (execution, 'validate_run_bindings'),
                    (execution, 'validate_execution_bindings'),
                    (execution, '_execute_run'),
                )]
                callbacks = {name: mock.Mock(name=name) for name in (
                    'workload_loader', 'clock', 'validate_authoring', 'validate_run',
                    'runtime_factory', 'task_package', 'environment', 'evaluator',
                )}
                provider = mock.Mock(name='custom_provider')
                provider.qualification = SimpleNamespace(
                    qualified=True, scope='zero_gpu_contract_fixture_only',
                    initial_and_resume_equivalent=True,
                    file_lifecycle_observed=True, usage_observed=True)
                callbacks['runtime_factory'].return_value = {
                    'provider': provider, 'environment': callbacks['environment'],
                    'evaluator': callbacks['evaluator'],
                }
                common = dict(project_root=ROOT, workload_loader=callbacks['workload_loader'],
                              clock=callbacks['clock'])
                with self.assertRaisesRegex(ValueError, 'native skill discovery.*not qualified'):
                    if entry == 'run':
                        execution.execute_run(specification, '/unwritten-skill-fixture/evidence',
                            **common, provider=provider, environment=callbacks['environment'],
                            evaluator=callbacks['evaluator'], task_package=callbacks['task_package'],
                            validate_run=callbacks['validate_run'])
                    elif entry == 'campaign':
                        execution.execute_campaign(lock, '/unwritten-skill-fixture/evidence',
                            **common, provider=provider, evaluator=callbacks['evaluator'],
                            environments={'open_cake': callbacks['environment'],
                                          'native_triton': callbacks['environment']},
                            validate_authoring=callbacks['validate_authoring'])
                    else:
                        execution.execute_campaign_with_factory(lock, '/unwritten-skill-fixture/evidence',
                            **common, runtime_factory=callbacks['runtime_factory'],
                            task_package=callbacks['task_package'], validate_run=callbacks['validate_run'],
                            validate_authoring=callbacks['validate_authoring'])
                for boundary in boundaries:
                    boundary.assert_not_called()
                for callback in (*callbacks.values(), provider):
                    self.assertEqual(callback.mock_calls, [])

    def test_compose_run_refuses_before_paths_evidence_or_runtime_preparation(self):
        specification = SimpleNamespace(document={'authoring': {
            'provider': {'author_home_policy': ISOLATED_SKILL_PACKAGE_V1}}})
        with (mock.patch.object(compose.Path, 'resolve') as resolve,
              mock.patch('open_cake_ir.lab.custody.admit_new_campaign_path') as admit,
              mock.patch.object(compose, 'run_runtime_factory') as runtime,
              mock.patch.object(compose, 'TaskLab') as lab,
              mock.patch.object(execution.EvidenceStore, 'create') as evidence):
            with self.assertRaisesRegex(ValueError, 'native skill discovery.*not qualified'):
                compose.execute_run_from_config('/unread-skill-fixture/project', specification,
                    '/unread-skill-fixture/runtime.json', '/unwritten-skill-fixture/evidence')
            for boundary in (resolve, admit, runtime, lab, evidence):
                boundary.assert_not_called()

    def test_compose_runtime_builder_refuses_before_preflight_or_provider_loading(self):
        specification = SimpleNamespace(document={'authoring': {
            'provider': {'author_home_policy': ISOLATED_SKILL_PACKAGE_V1}}})
        with (mock.patch.object(compose.Path, 'resolve', side_effect=(ROOT, Path('/fixture/runtime.json'))),
              mock.patch.object(compose, 'TaskLab') as lab,
              mock.patch.object(compose, 'load_runtime_config') as runtime,
              mock.patch.object(compose.ProviderQualificationReceipt, 'load') as receipt,
              mock.patch.object(compose, 'materialize_task_package') as materialize,
              mock.patch.object(compose, 'CodexInvocationBuilder') as provider):
            receipt.return_value = SimpleNamespace(qualified=True, scope='zero_gpu_contract_fixture_only')
            build = compose.run_runtime_factory('/fixture/project', '/fixture/runtime.json')
            with self.assertRaisesRegex(ValueError, 'native skill discovery.*not qualified'):
                build(specification, Path('/unwritten-skill-fixture/runtime'))
            for boundary in (lab.return_value.preflight_run, runtime, receipt, materialize, provider):
                boundary.assert_not_called()


if __name__ == '__main__':
    unittest.main()
