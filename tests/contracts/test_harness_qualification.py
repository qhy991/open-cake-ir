"""Executable fixture tests for the existing qualification tool; never live evidence."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from open_cake_ir.evidence import EvidenceStore
from open_cake_ir.lab.claude import ClaudeInvocationBuilder, ClaudeProviderAdapter
from open_cake_ir.lab.faults import RunProtocolFault
from open_cake_ir.lab.providers import ProviderQualificationReceipt, QualifiedRunProvider
from open_cake_ir.lab.task_package import TaskPackage
from tools import qualify_codex_provider as qualifier

ROOT = Path(__file__).resolve().parents[2]
MODEL = "claude-opus-test-exact"
SESSION = "01234567-89ab-cdef-0123-456789abcdef"


class HarnessQualificationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.executable = self.root / "claude-fixture"
        self.schema = self.root / "output-schema.json"
        schema = json.loads((ROOT / "contracts/providers/codex-turn-output-schema-v2.json").read_text())
        schema["properties"]["arm"]["enum"] = ["open_cake"]
        self.schema.write_text(json.dumps(schema))
        self.source = self.root / "starter.py"
        self.source.write_text('from open_cake_ir.compiler import frontend as cake\n'
                               '@cake.schedule(name="qualification", target="apple_gpu_family7", backend="metal", entry_point="copy")\n'
                               'def candidate(lm, x: cake.Tensor((1, 7), "fp32"), out: cake.Tensor((1, 7), "fp32", mode="output")):\n'
                               '    compute = lm.role(warps=[0])\n'
                               '    row = lm.program(x, axis=0, dimension=0, tile=1)\n'
                               '    with compute:\n'
                               '        values = lm.load(x[row, :])\n'
                               '        lm.store(out[row, :], values, coalesced=False)\n')

    def provider(self, failure=None):
        self.executable.write_text(textwrap.dedent(f'''\
            #!{sys.executable}
            import json
            import sys
            from pathlib import Path
            args = sys.argv[1:]
            assert args[0] == '-p' and '--safe-mode' in args
            assert args[args.index('--model')+1] == {MODEL!r}
            assert args[args.index('--effort')+1] == 'high'
            assert args[args.index('--permission-mode')+1] == 'acceptEdits'
            assert args[args.index('--tools')+1] == 'Read,Write,Edit,Glob,Grep'
            projection = json.loads(args[-1].split('\\n\\n', 1)[1])
            assert projection['task_markdown'] == Path('TASK.md').read_text()
            assert projection['agents_markdown'] == Path('AGENTS.md').read_text()
            plan = json.loads(next(line.split('=',1)[1] for line in projection['task_markdown'].splitlines()
                                   if line.startswith('QUALIFICATION_PLAN_JSON=')))
            resumed = '--resume' in args
            turn = 2 if resumed else 1
            assert projection['state_card']['turn'] == turn
            session = {SESSION!r}
            if resumed:
                assert args[args.index('--resume')+1] == session
            if resumed and {failure!r} == 'session':
                session = 'fedcba98-7654-3210-fedc-ba9876543210'
            path = Path(plan['candidate_path'])
            assert path.parent == Path.cwd() and path.exists() == resumed
            entry = plan['turns'][turn-1]
            path.write_text(json.dumps(entry['submission']))
            if {failure!r} == 'task':
                Path('TASK.md').chmod(0o644)
                Path('TASK.md').write_text('tampered task')
            model = 'wrong-model' if {failure!r} == 'model' else {MODEL!r}
            events = [
              {{'type':'system','subtype':'init','session_id':session,'model':model}},
              {{'type':'assistant','session_id':session,'message':{{'model':model,'content':[
                 {{'type':'tool_use','id':'write1','name':'Edit' if resumed else 'Write',
                   'input':{{'file_path':str(path)}}}}]}}}},
              {{'type':'user','session_id':session,'message':{{'content':[
                 {{'type':'tool_result','tool_use_id':'write1','is_error':False,'content':'written'}}]}}}},
              {{'type':'result','subtype':'success','session_id':session,'is_error':False,
                'result':json.dumps(entry['terminal_message']),
                'usage':{{'input_tokens':0 if {failure!r} == 'usage' else 100+turn,
                         'output_tokens':0 if {failure!r} == 'usage' else 10,
                         'cache_creation_input_tokens':0,'cache_read_input_tokens':0}}}}
            ]
            for event in events: print(json.dumps(event))
            '''))
        self.executable.chmod(0o700)

    def argv(self):
        return ['qualify_codex_provider.py', '--harness', 'claude-code', '--model', MODEL,
                '--reasoning-effort', 'high', '--fixture-only', '--executable', str(self.executable),
                '--provider-revision', 'claude-executable-fixture', '--output-schema', str(self.schema),
                '--python-source', str(self.source), '--feature-policy', 'provider_defaults_optimization',
                '--workspace', str(self.root / 'workspace'), '--receipt-output', str(self.root / 'receipt.json'),
                '--anchor-output', str(self.root / 'anchor.json'), '--evidence-root', str(self.root / 'evidence'),
                '--run-id', 'claude-qualification-fixture']

    def run_qualification(self, arguments=None):
        with patch.object(sys, 'argv', arguments or self.argv()), contextlib.redirect_stdout(io.StringIO()):
            return qualifier.main()

    def test_claude_qualifies_native_source_transport_and_archives_fixture_scope(self):
        self.provider()
        self.assertEqual(self.run_qualification(), 0)
        receipt = ProviderQualificationReceipt.load(self.root / 'receipt.json')
        self.assertEqual(receipt.scope, 'zero_gpu_contract_fixture_only')
        evidence = EvidenceStore.open(self.root / 'evidence')
        events = list(evidence.replay_events('claude-qualification-fixture'))
        observed = next(event['payload'] for event in events if event['kind'] == 'provider_qualification_observed')
        self.assertEqual(set(observed['arms']), {'open_cake'})
        arm = observed['arms']['open_cake']
        self.assertEqual(arm['thread_id'], SESSION)
        self.assertEqual((arm['initial_provider_tokens'], arm['resumed_provider_tokens']), (111,112))
        self.assertEqual(arm['reported_models'], [[MODEL],[MODEL]])
        objects = {item['role']:item for item in observed['objects']}
        for phase, turn in (('initial',1),('resumed',2)):
            source_member = json.loads(evidence.read_object(objects[f'open_cake_{phase}_candidate_0000']))
            self.assertEqual(set(source_member), {'python_source'})
            self.assertTrue(source_member['python_source'].startswith(self.source.read_text()))
            self.assertIn(f'qualification turn {turn}', source_member['python_source'])
            invocation = json.loads(evidence.read_object(objects[f'open_cake_{phase}_invocation']))
            self.assertEqual(invocation['sandbox'], 'none')
            self.assertEqual(invocation['thread_id'], SESSION if turn == 2 else None)
            self.assertEqual(invocation['cwd'], str(self.root/'workspace/open_cake'))
        self.assertEqual(json.loads((self.root/'anchor.json').read_text())['kind'],
                         'provider_qualification_evidence_anchor')
        workspace = self.root/'workspace/open_cake'
        builder = ClaudeInvocationBuilder(executable=self.executable, provider_revision=receipt.provider_revision,
            model=MODEL, reasoning_effort='high', workspace=workspace,
            removed_environment=('OPENAI_API_KEY','ANTHROPIC_API_KEY'))
        run_id = 'claude-qualification-fixture-open_cake'
        package = TaskPackage(run_id, 'open_cake', (workspace/'TASK.md').read_text(), (workspace/'AGENTS.md').read_text())
        with self.assertRaisesRegex(ValueError, 'authority differs'):
            QualifiedRunProvider(qualification=receipt, builders={run_id:builder}, task_packages={run_id:package},
                                 adapter=ClaudeProviderAdapter())

    def test_codex_single_arm_transports_the_same_python_member_contract(self):
        from tests.contracts.test_provider_qualification import ProviderQualificationContractTests
        fixture = ProviderQualificationContractTests("runTest")
        fixture._write_provider(self.executable, tool_rich=True)
        args = self.argv()
        args[args.index('--harness')+1] = 'codex'
        args[args.index('--model')+1] = 'gpt-5.6-sol'
        self.assertEqual(self.run_qualification(args), 0)
        receipt = ProviderQualificationReceipt.load(self.root/'receipt.json')
        self.assertEqual(receipt.scope, 'zero_gpu_contract_fixture_only')
        envelope = json.loads((self.root/'workspace/open_cake/candidate-set.json').read_text())
        self.assertEqual(envelope['arm'], 'open_cake')
        self.assertEqual(set(envelope['candidates'][0]), {'python_source'})
        self.assertTrue(envelope['candidates'][0]['python_source'].startswith(self.source.read_text()))
        self.assertIn('qualification turn 2', envelope['candidates'][0]['python_source'])
        self.assertEqual({path.name for path in (self.root/'workspace').iterdir()}, {'open_cake'})

    def test_native_failures_never_issue_qualification(self):
        for failure in ('session','usage','task','model'):
            with self.subTest(failure=failure):
                # A separate fixture root per attempted qualification preserves failure evidence.
                self.setUp()
                self.provider(failure)
                with self.assertRaises((ValueError, RunProtocolFault)):
                    self.run_qualification()
                self.assertFalse((self.root/'receipt.json').exists())
                if failure == 'model':
                    envelope = json.loads((self.root/'workspace/open_cake/candidate-set.json').read_text())
                    self.assertIn('qualification turn 1', envelope['candidates'][0]['python_source'])
                audit = EvidenceStore.open(self.root/'evidence').audit_run('claude-qualification-fixture')
                self.assertEqual(audit.endpoint_observation, 'missing')
                self.assertEqual(audit.protocol_adherence, 'provider_fault')

    def test_output_admission_rejects_enclosing_checkout_and_symlink_alias(self):
        checkout = self.root/'checkout'
        checkout.mkdir()
        (checkout/'.git').write_text('gitdir: external-linked-worktree')
        nested = checkout/'nested'
        nested.mkdir()
        alias = self.root/'alias'
        alias.symlink_to(nested, target_is_directory=True)
        for path in (nested/'receipt.json', alias/'receipt.json', Path('relative-output.json')):
            with self.subTest(path=path), self.assertRaises(ValueError):
                qualifier._new_path(path)
            self.assertFalse(path.exists())
        self.provider()
        args = self.argv()
        args[args.index('--evidence-root')+1] = str(nested/'evidence')
        with self.assertRaisesRegex(ValueError, 'outside Git checkouts'):
            self.run_qualification(args)
        self.assertFalse((self.root/'workspace').exists())

    def test_required_harness_model_effort_refuse_before_process_execution(self):
        self.provider()
        for flag in ('--harness','--model','--reasoning-effort'):
            args = self.argv()
            index = args.index(flag)
            del args[index:index+2]
            with self.subTest(flag=flag), patch.object(qualifier.ClaudeProviderAdapter, 'execute') as invoke:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    self.run_qualification(args)
                invoke.assert_not_called()
                self.assertFalse((self.root/'workspace').exists())

    def test_claude_refuses_paired_or_missing_python_source_before_workspace_creation(self):
        self.provider()
        variants = []
        args = self.argv()
        index = args.index('--python-source')
        del args[index:index+2]
        variants.append(args)
        args = self.argv()
        args[args.index('--output-schema')+1] = str(ROOT/'contracts/providers/codex-turn-output-schema-v2.json')
        variants.append(args)
        for args in variants:
            with self.assertRaises(ValueError): self.run_qualification(args)
            self.assertFalse((self.root/'workspace').exists())


if __name__ == '__main__':
    unittest.main()
