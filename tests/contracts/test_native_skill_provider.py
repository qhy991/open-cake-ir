"""CPU author-state preparation and invocation checks, never native qualification."""
from dataclasses import replace
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from open_cake_ir.lab.author_home import (ISOLATED_SKILL_PACKAGE_V1, provision_codex_home,
                                        provision_user_home)
from open_cake_ir.lab.native_skills import NativeSkillPackage, author_skill_reference
from open_cake_ir.lab.provider_invocation import CodexInvocationBuilder
from open_cake_ir.lab.providers import CodexProviderAdapter, QualifiedRunProvider
from open_cake_ir.lab.process import SupervisedProcessTimeout
from open_cake_ir.lab.faults import RunProtocolFault
from open_cake_ir.lab.provider_policy import execution_configuration
from open_cake_ir.lab.task_package import TaskPackage, render_task_request

ROOT = Path(__file__).resolve().parents[2]
THREAD = '01234567-89ab-cdef-0123-456789abcdef'


def archive_bytes(body=b'Inspect Cake operations.\n'):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w') as archive:
        for name, payload in (
            ('skills/cake/SKILL.md', b'---\nname: cake\ndescription: fixture\n---\n' + body),
            ('skills/cake/scripts/check.py', b'print("fixture")\n'),
            ('skills/cake/assets/data.bin', bytes(range(256))),
        ):
            item = tarfile.TarInfo(name)
            item.size = len(payload)
            item.mode = 0o644
            archive.addfile(item, io.BytesIO(payload))
    return output.getvalue()


class NativeSkillProviderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.executable = self.root/'codex'
        self.executable.write_text('#!/bin/sh\nexit 99\n')
        self.executable.chmod(0o755)
        helper = self.root/'codex-code-mode-host'
        helper.write_text('#!/bin/sh\nexit 99\n')
        helper.chmod(0o755)
        self.auth = self.root/'fixture-auth.json'
        self.auth.write_bytes(b'fixture credential')
        self.auth.chmod(0o600)

    def builder(self, name, raw=None):
        root = self.root/name
        root.mkdir()
        source = root/'native-skills.tar'
        source.write_bytes(archive_bytes() if raw is None else raw)
        package = NativeSkillPackage.read(ROOT, source)
        home = provision_user_home(root/'user-home', package)
        codex = provision_codex_home(self.auth, root/'codex-home')
        workspace = root/'actor'
        workspace.mkdir()
        builder = CodexInvocationBuilder(executable=self.executable, provider_revision='fixture',
            model='gpt-6.1-sol', reasoning_effort='xhigh', service_tier='default',
            workspace=workspace, output_schema=ROOT/'contracts/providers/codex-turn-output-schema-v1.json',
            removed_environment=('OPENAI_API_KEY', 'ANTHROPIC_API_KEY'),
            author_home_policy=ISOLATED_SKILL_PACKAGE_V1, codex_home=codex,
            user_home=home, native_skill_package=package)
        return builder, package, home, codex

    def test_same_material_at_different_run_paths_reuses_configuration_not_mutable_state(self):
        first, first_package, first_home, first_codex = self.builder('first')
        second, second_package, second_home, second_codex = self.builder('second')
        self.assertNotEqual(first_package.reference['path'], second_package.reference['path'])
        self.assertEqual(first.configuration, second.configuration)
        self.assertNotEqual(first_home, second_home)
        self.assertNotEqual(first_codex, second_codex)
        changed, changed_package, _, _ = self.builder('changed', archive_bytes(b'Different material.\n'))
        self.assertNotEqual(first.configuration, changed.configuration)
        invocation = replace(first.build('fixture', thread_id=None), native_skill_package=changed_package)
        with patch('open_cake_ir.lab.providers.run_supervised') as execute:
            with self.assertRaisesRegex(ValueError, 'projection'):
                CodexProviderAdapter().execute(invocation, candidate_path=first.workspace/'candidate-set.json',
                    expected_change='add', expected_terminal_message='{}')
            execute.assert_not_called()

    def test_initial_resume_adapter_uses_both_bound_homes_and_refuses_changed_files(self):
        builder, package, home, codex = self.builder('run')
        first = builder.build('first', thread_id=None)
        builder.remember_system_skills()
        second = builder.build('second', thread_id=THREAD)
        for invocation in (first, second):
            self.assertEqual(invocation.user_home, home)
            self.assertEqual(invocation.codex_home, codex)
            self.assertIs(invocation.native_skill_package, package)
            with patch('open_cake_ir.lab.providers.run_supervised',
                       side_effect=SupervisedProcessTimeout(b'', b'')) as execute:
                with self.assertRaises(RunProtocolFault):
                    CodexProviderAdapter().execute(invocation,
                        candidate_path=builder.workspace/'candidate-set.json',
                        expected_change='add', expected_terminal_message='{}')
                environment = execute.call_args.kwargs['environment']
                self.assertEqual(environment['HOME'], str(home))
                self.assertEqual(environment['CODEX_HOME'], str(codex))
        changed = home/'.agents/skills/cake/scripts/check.py'
        changed.chmod(0o600)
        changed.write_bytes(b'changed after Turn\n')
        with self.assertRaisesRegex(ValueError, 'projection'):
            builder.build('third', thread_id=THREAD)
        with self.assertRaisesRegex(ValueError, 'projection'):
            builder.remember_system_skills()

    def test_frozen_configuration_projection_matches_builder_and_excludes_paths(self):
        builder, package, home, codex = self.builder('run')
        schema = ROOT/'contracts/providers/codex-turn-output-schema-v1.json'
        configuration = dict(builder.configuration)
        provider = {**configuration, 'revision': 'fixture', 'qualification': None,
                    'qualification_anchor': None, 'executable_sha256': 'a'*64,
                    'output_schema': {'path': str(schema), 'sha256': configuration['output_schema_sha256']},
                    'native_skill_package': package.reference}
        provider.pop('output_schema_sha256')
        provider.pop('native_skill_package_sha256', None)
        self.assertEqual(execution_configuration(provider), configuration)
        encoded = json.dumps(configuration)
        for path in (home, codex, Path(package.reference['path'])):
            self.assertNotIn(str(path), encoded)

    def test_task_projection_contains_metadata_without_binary_or_script_body(self):
        _, package, _, _ = self.builder('run')
        task = TaskPackage('run', 'arm', 'task', 'agents', native_skill_package=package)
        prompt, bundle = render_task_request(task, {'turn': 1})
        metadata = json.loads(bundle)['native_skill_package']
        self.assertEqual(metadata['reference'], package.reference)
        self.assertEqual(metadata['package_entries'], ['cake'])
        self.assertIn('unverified', metadata['observation'])
        self.assertNotIn('print("fixture")', prompt)
        self.assertNotIn('Inspect Cake operations.', prompt)
        retained = NativeSkillPackage.from_bytes(package.raw_bytes, package.reference)
        self.assertEqual(retained.files, package.files)

    def test_formal_provider_and_binding_cannot_borrow_even_a_fixture_receipt(self):
        builder, _, _, _ = self.builder('run')
        with self.assertRaisesRegex(ValueError, 'not qualified'):
            QualifiedRunProvider(qualification=None, builders={'run': builder}, task_packages={}, adapter=None)
        from open_cake_ir.lab.bindings import bind_cli_provider
        with patch('open_cake_ir.lab.bindings.load_runtime_config') as runtime:
            with self.assertRaisesRegex(ValueError, 'not qualified'):
                bind_cli_provider(ROOT, {'author_home_policy': ISOLATED_SKILL_PACKAGE_V1}, None,
                    runtime_path=Path('/unread'), receipt_path=Path('/unread'), anchor_path=Path('/unread'))
            runtime.assert_not_called()

    def test_common_declaration_gate_rejects_mismatched_policy_even_without_turns(self):
        for author in (
            {'reference_access': 'known_kernel_reproduction', 'provider': {
                'native_skill_package': {'path': 'x', 'sha256': 'a'*64}}},
            {'reference_access': 'clean_start', 'provider': {
                'author_home_policy': ISOLATED_SKILL_PACKAGE_V1,
                'native_skill_package': {'path': 'x', 'sha256': 'a'*64}}},
            {'reference_access': 'known_kernel_reproduction', 'provider': {
                'author_home_policy': ISOLATED_SKILL_PACKAGE_V1}},
        ):
            with self.subTest(author=author), self.assertRaises(ValueError):
                author_skill_reference(author)
