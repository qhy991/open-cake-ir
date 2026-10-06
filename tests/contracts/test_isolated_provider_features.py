"""Author-home isolation constrains extensions without disabling native task skills."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from open_cake_ir.evidence import EvidenceStore
from open_cake_ir.lab.author_home import ISOLATED_SKILL_PACKAGE_V1, ISOLATED_AUTH_ONLY_V1
from open_cake_ir.lab.provider_documents import ProviderQualificationReceipt, PYTHON_CANDIDATE_BUNDLE_V1
from open_cake_ir.lab.provider_invocation import CodexInvocationBuilder
from open_cake_ir.lab.provider_policy import execution_configuration
from open_cake_ir.lab.native_skill_qualification import verify_qualification_evidence
from tests.contracts import test_native_skill_provider as provider_fixtures
from tests.contracts import test_provider_qualification as qualification_fixtures

ROOT, THREAD = provider_fixtures.ROOT, provider_fixtures.THREAD

RESTRICTIONS = ('apps', 'plugins', 'remote_plugin')


class IsolatedProviderFeatures(unittest.TestCase):
    def setUp(self):
        self.fixture = provider_fixtures.NativeSkillProviderTests(methodName='runTest')
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()

    def builder(self, policy=ISOLATED_SKILL_PACKAGE_V1, features=RESTRICTIONS):
        base, package, home, codex = self.fixture.builder('author')
        builder = CodexInvocationBuilder(executable=self.fixture.executable, provider_revision='fixture',
            model='gpt-6.1-sol', reasoning_effort='xhigh', service_tier='default',
            workspace=base.workspace, output_schema=ROOT/'contracts/providers/open-cake-optimization-output-schema-v1.json',
            removed_environment=('OPENAI_API_KEY', 'ANTHROPIC_API_KEY'),
            event_contract='tool_rich_candidate_v1', disabled_features=features,
            author_home_policy=policy, codex_home=codex,
            user_home=home if policy == ISOLATED_SKILL_PACKAGE_V1 else None,
            native_skill_package=package if policy == ISOLATED_SKILL_PACKAGE_V1 else None)
        return builder, codex

    def test_initial_resume_and_frozen_configuration_bind_the_same_restrictions(self):
        builder, codex = self.builder()
        first = builder.build('first', thread_id=None)
        builder.remember_system_skills()
        second = builder.build('second', thread_id=THREAD)
        for invocation in (first, second):
            disabled = tuple(invocation.argv[i+1] for i, arg in enumerate(invocation.argv[:-1])
                             if arg == '--disable')
            self.assertEqual(disabled, RESTRICTIONS)
            self.assertNotIn('shell_tool', disabled)
            self.assertNotIn('web_search="disabled"', invocation.argv)
            self.assertIn('approval_policy="never"', invocation.argv)
            self.assertIn('sandbox_mode="workspace-write"', invocation.argv)
        self.assertEqual(builder.configuration['disabled_features'], list(RESTRICTIONS))
        configuration = dict(builder.configuration)
        provider = {**configuration, 'revision': 'fixture', 'qualification': None,
            'qualification_anchor': None, 'executable_sha256': 'a'*64,
            'output_schema': {'path': str(ROOT/'contracts/providers/open-cake-optimization-output-schema-v1.json'),
                              'sha256': configuration['output_schema_sha256']},
            'native_skill_package': builder.native_skill_package.reference}
        provider.pop('output_schema_sha256')
        provider.pop('native_skill_package_sha256')
        self.assertEqual(execution_configuration(provider), configuration)
        for features in ([], ['plugins'], [*RESTRICTIONS, 'shell_tool'], list(reversed(RESTRICTIONS))):
            with self.subTest(features=features), self.assertRaisesRegex(ValueError, 'configuration differs'):
                execution_configuration(dict(provider, disabled_features=features))
        # A CLI that ignores the flags still cannot progress to resume.
        (codex/'plugins').mkdir()
        with self.assertRaisesRegex(ValueError, 'contains plugins'):
            builder.build('third', thread_id=THREAD)

    def test_auth_only_isolation_uses_the_same_extension_boundary(self):
        builder, _ = self.builder(policy=ISOLATED_AUTH_ONLY_V1)
        self.assertEqual(builder.configuration['disabled_features'], list(RESTRICTIONS))

    def test_unrestricted_isolated_request_refuses_before_any_home_or_executable_read(self):
        for policy in (ISOLATED_SKILL_PACKAGE_V1, ISOLATED_AUTH_ONLY_V1):
            with self.subTest(policy=policy), patch('open_cake_ir.lab.provider_invocation.verify_codex_home') as verify:
                with self.assertRaisesRegex(ValueError, 'feature and event'):
                    CodexInvocationBuilder(executable=Path('/unread/codex'), provider_revision='fixture',
                        model='gpt-6.1-sol', reasoning_effort='xhigh', service_tier='default',
                        workspace=Path('/unread/workspace'), output_schema=Path('/unread/schema'),
                        removed_environment=('OPENAI_API_KEY', 'ANTHROPIC_API_KEY'),
                        event_contract='tool_rich_candidate_v1', disabled_features=(),
                        author_home_policy=policy, codex_home=Path('/unread/home'))
                verify.assert_not_called()

    def test_real_fixture_cli_retains_reconstructable_initial_resume_flags_and_skill_input(self):
        helper = qualification_fixtures.ProviderQualificationContractTests(methodName='runTest')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            executable = root/'codex'
            helper._write_provider(executable, tool_rich=True, expected_tool_rich_disabled=RESTRICTIONS)
            package = root/'skills.tar'
            package.write_bytes(provider_fixtures.archive_bytes())
            auth = root/'auth.json'
            auth.write_bytes(b'fixture credential')
            auth.chmod(0o600)
            completed, receipt_path, anchor_path, evidence_root = helper._run_qualification(
                root, executable, provider_revision='isolated-tool-rich-fixture', run_id='qualified-fixture',
                feature_policy='provider_defaults_optimization', maximum_candidates_per_turn=1,
                reasoning_effort='xhigh',
                output_schema=ROOT/'contracts/providers/open-cake-optimization-output-schema-v1.json',
                submission_contract=PYTHON_CANDIDATE_BUNDLE_V1,
                author_skill_package=package, auth_source=auth)
            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            receipt = ProviderQualificationReceipt.load(receipt_path)
            self.assertEqual(receipt.scope, 'zero_gpu_contract_fixture_only')
            store = EvidenceStore.open(evidence_root)
            self.assertEqual(store.replay_authority('qualified-fixture')['disabled_features'], list(RESTRICTIONS))
            turns = verify_qualification_evidence(qualification=receipt, anchor=json.loads(anchor_path.read_bytes()),
                                                 requested_names=('cake',))
            self.assertEqual(len(turns['open_cake']), 2)
            event = next(event for event in store.replay_events('qualified-fixture')
                         if event['kind'] == 'provider_qualification_observed')
            refs = {item['role']: item for item in event['payload']['objects']}
            for phase in ('initial', 'resumed'):
                invocation = json.loads(store.read_object(refs[f'open_cake_{phase}_invocation']))
                argv = invocation['argv']
                self.assertEqual([argv[i+1] for i, arg in enumerate(argv[:-1]) if arg == '--disable'],
                                 list(RESTRICTIONS))
