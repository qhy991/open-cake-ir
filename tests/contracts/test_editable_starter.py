"""Editable reference copies change authoring cost, not reference permissions."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from open_cake_ir.lab.task_package import TaskPackage, materialize_task_package, verify_task_package, render_task_request
from open_cake_ir.lab.provider_documents import PYTHON_CANDIDATE_BUNDLE_V1, ProviderQualificationReceipt
from open_cake_ir.lab.python_candidate_bundle import project_python_candidate_bundle
from open_cake_ir.serialization import canonical_json_bytes

PROGRAM = '''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="first", target="xcore1002", backend="triton", entry_point="first")
def first(lm, x: cake.Tensor((1, 7), "fp32"), tmp: cake.Tensor((1, 7), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0])
    row = lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
        value = lm.load(x[row, :])
        lm.store(tmp[row, :], value)
@cake.schedule(name="second", target="xcore1002", backend="triton", entry_point="second")
def second(lm, tmp: cake.Tensor((1, 7), "fp32"), out: cake.Tensor((1, 7), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0])
    row = lm.program(tmp, axis=0, dimension=0, tile=1)
    with compute:
        value = lm.load(tmp[row, :])
        lm.store(out[row, :], value)
cake.program(program_id="seed_program", inputs=("x",), outputs=("out",), stages=(
    cake.stage(name="first", schedule=first, bindings={"x":"x", "tmp":"tmp"}),
    cake.stage(name="second", schedule=second, bindings={"tmp":"tmp", "out":"out"}),
))
'''


class EditableMaterial(unittest.TestCase):
    def test_only_the_declared_known_kernel_file_author_can_receive_a_seed(self):
        authoring = {'reference_access': 'known_kernel_reproduction', 'environment_kind': 'open_cake',
            'input_format': 'python_source_v1', 'provider': {'harness': 'claude-code',
            'submission_contract': PYTHON_CANDIDATE_BUNDLE_V1}}
        self.assertTrue(TaskPackage.permits_editable_starter(authoring))
        for key, value in [('reference_access', 'clean_start'), ('reference_access', 'direct_low_level'),
                           ('environment_kind', 'native_triton'), ('input_format', 'schedule_or_python_v1')]:
            self.assertFalse(TaskPackage.permits_editable_starter({**authoring, key: value}))
        for provider in ({'harness': 'responses'}, {'harness': 'codex'}, {'submission_contract': 'python_source_file_v1'}):
            self.assertFalse(TaskPackage.permits_editable_starter({**authoring, 'provider': {**authoring['provider'], **provider}}))

    def test_exclusive_copy_preserves_original_and_resumed_bytes(self):
        source = PROGRAM
        package = TaskPackage('seed', 'open_cake', '# Original\n' + source, '# Rules', initial_candidate_source=source)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_task_package(root, package)
            candidate = root / 'candidate-set.py'
            self.assertEqual(candidate.read_text(), source)
            self.assertEqual(set(p.name for p in root.iterdir()), {'TASK.md', 'AGENTS.md', 'candidate-set.py'})
            self.assertEqual((root / 'TASK.md').stat().st_mode & 0o222, 0)
            self.assertTrue(candidate.stat().st_mode & 0o200)
            self.assertEqual(package.candidate_change(1), 'update')
            self.assertEqual(package.candidate_change(2), 'update')
            self.assertEqual(len(project_python_candidate_bundle(candidate.read_bytes(), maximum_candidates_per_turn=1)), 1)
            candidate.write_text(source + '\n# Author edit retained\n')
            with self.assertRaisesRegex(ValueError, 'empty'):
                materialize_task_package(root, package)
            self.assertTrue(candidate.read_text().endswith('# Author edit retained\n'))
            verify_task_package(root, package)
            for turn in (1, 2):
                prompt, _ = render_task_request(package, {'turn': turn})
                self.assertIn(json.dumps(source, ensure_ascii=False)[1:-1], prompt)
        self.assertEqual(replace(package, initial_candidate_source=None).candidate_change(1), 'add')
        with self.assertRaisesRegex(ValueError, 'immutable task material'):
            TaskPackage('bad', 'open_cake', 'No authorized implementation', 'Rules', initial_candidate_source=source)

    def test_provider_checks_initial_bytes_once_and_never_resets_on_resume(self):
        from open_cake_ir.lab.providers import QualifiedRunProvider
        from open_cake_ir.lab.provider_documents import ProviderTurn
        from open_cake_ir.lab.claude import CLAUDE_EVENT_CONTRACT
        package = TaskPackage('seed', 'open_cake', PROGRAM, '# Rules', initial_candidate_source=PROGRAM)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_task_package(root, package)
            calls = []
            class Adapter:
                def execute(self, invocation, *, candidate_path, expected_change, expected_terminal_message, **kwargs):
                    self_before = candidate_path.read_text()
                    calls.append((expected_change, self_before))
                    source = self_before.replace('seed_program', 'first_edit') if len(calls) == 1 else self_before.replace('first_edit', 'second_edit')
                    candidate_path.write_text(source)
                    raw = source.encode()
                    candidates = project_python_candidate_bundle(raw, maximum_candidates_per_turn=1)
                    return ProviderTurn(thread_id='01234567-89ab-cdef-0123-456789abcdef', provider_tokens=10,
                        candidates=candidates, candidate_sha256s=tuple(sha256(c).hexdigest() for c in candidates),
                        raw_submission=raw, raw_events=b'CPU component double', raw_events_sha256='1' * 64,
                        terminal_message=expected_terminal_message, terminal_message_count=1, normalization='CPU fixture')
            # This unit exercises turn materialization/custody only. The separate
            # qualification fixture runs the actual Claude adapter and Replay.
            provider = QualifiedRunProvider.__new__(QualifiedRunProvider)
            provider._builders = {'seed': SimpleNamespace(workspace=root, build=lambda *a, **k: SimpleNamespace())}
            provider._task_packages = {'seed': package}
            provider._submission_contract = PYTHON_CANDIDATE_BUNDLE_V1
            provider._event_contract = CLAUDE_EVENT_CONTRACT
            provider.configuration = {'harness': 'claude-code', 'event_contract': CLAUDE_EVENT_CONTRACT}
            provider._adapter = Adapter()
            request = dict(run_id='seed', arm='open_cake', environment_kind='open_cake',
                           maximum_candidates_per_turn=1, state_card={})
            path = root / 'candidate-set.py'
            path.write_text(PROGRAM + '# unexpected mutation\n')
            with self.assertRaisesRegex(ValueError, 'initial provider material'):
                provider.turn(SimpleNamespace(**request, turn=1, thread_id=None, cumulative_provider_tokens=0))
            self.assertEqual(calls, [])
            path.write_text(PROGRAM)
            first = provider.turn(SimpleNamespace(**request, turn=1, thread_id=None, cumulative_provider_tokens=0))
            saved = root / 'saved-author-source.py'
            path.rename(saved)
            path.symlink_to(saved)
            with self.assertRaisesRegex(ValueError, 'candidate lifecycle'):
                provider.turn(SimpleNamespace(**request, turn=2, thread_id=first.thread_id, cumulative_provider_tokens=10))
            self.assertEqual(len(calls), 1)
            path.unlink()
            saved.rename(path)
            provider.turn(SimpleNamespace(**request, turn=2, thread_id=first.thread_id, cumulative_provider_tokens=10))
            self.assertEqual([call[0] for call in calls], ['update', 'update'])
            self.assertIn('first_edit', calls[1][1])
            self.assertIn('second_edit', path.read_text())
            self.assertIn('seed_program', (root / 'TASK.md').read_text())


class EditableQualification(unittest.TestCase):
    def setUp(self):
        from tests.contracts.test_harness_qualification import HarnessQualificationTests
        self.fixture = HarnessQualificationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.provider()
        arguments = self.fixture.argv() + ['--submission-contract', PYTHON_CANDIDATE_BUNDLE_V1,
            '--maximum-candidates-per-turn', '1', '--editable-starter']
        self.assertEqual(self.fixture.run_qualification(arguments), 0)
        from open_cake_ir.evidence import EvidenceStore
        self.evidence = EvidenceStore.open(self.fixture.root / 'evidence')
        self.run_id = 'claude-qualification-fixture'
        self.authority = self.evidence.replay_authority(self.run_id)
        self.observed = next(event['payload'] for event in self.evidence.replay_events(self.run_id)
                             if event['kind'] == 'provider_qualification_observed')
        self.receipt = ProviderQualificationReceipt.load(self.fixture.root / 'receipt.json')
        self.provider = {key: self.authority[key] for key in ('model', 'reasoning_effort', 'event_contract')}
        self.refs = {item['role']: item for item in self.observed['objects']}

    def validate(self, authority=None, read_object=None):
        from open_cake_ir.lab.admission import validate_editable_starter_observation
        validate_editable_starter_observation(authority=authority or self.authority, payload=self.observed,
            read_object=read_object or self.evidence.read_object, provider=self.provider, qualification=self.receipt)

    def test_real_tool_adapter_normalizes_initial_edit_and_preserved_resume(self):
        self.validate()
        audit = self.evidence.audit_run(self.run_id)
        self.assertFalse(audit.endpoint['add_observed'])
        self.assertTrue(audit.endpoint['update_observed'])
        self.assertEqual(self.receipt.scope, 'zero_gpu_contract_fixture_only')
        from open_cake_ir.lab.admission import require_editable_starter_qualification
        with self.assertRaisesRegex(ValueError, 'live anchored'):
            require_editable_starter_qualification(self.receipt,
                json.loads((self.fixture.root / 'anchor.json').read_text()), self.provider)
        first = self.evidence.read_object(self.refs['open_cake_initial_source_file'])
        resumed = self.evidence.read_object(self.refs['open_cake_resumed_source_file'])
        self.assertNotEqual(first, resumed)
        self.assertEqual(first.replace(b'qualification_edit_1', b'qualification_edit_2'), resumed)

    def test_old_add_plan_and_unwitnessed_edit_cannot_grant_seeded_authoring(self):
        old = {**self.authority, 'turns': ['initial_add', 'same_thread_resume_update']}
        with self.assertRaisesRegex(ValueError, 'initial-Edit'):
            self.validate(authority=old)
        reference = self.refs['open_cake_initial_provider_events']
        raw = self.evidence.read_object(reference)
        altered = raw.replace(b'"name": "Edit"', b'"name": "Write"')
        self.assertNotEqual(altered, raw)
        def read(item):
            return altered if item['sha256'] == reference['sha256'] else self.evidence.read_object(item)
        with self.assertRaisesRegex(ValueError, 'native Edits'):
            self.validate(read_object=read)

    def test_actual_run_replay_uses_the_same_initial_update_rule(self):
        from open_cake_ir.lab.claude import parse_claude_turn_events
        from open_cake_ir.lab.replay.provider import _replay_provider_turns
        from open_cake_ir.lab.replay.refusals import ReplayRefusal
        documents = json.loads(self.evidence.read_object(self.refs['qualification_reference']))['open_cake']
        initial = self.evidence.read_object(self.refs['open_cake_initial_source_file']).decode()
        seed = initial.replace('qualification_edit_1', 'qualification_edit_0')
        package = TaskPackage('seed-replay', 'open_cake', documents['task_markdown'], documents['agents_markdown'],
                              initial_candidate_source=seed)
        memory, events, cumulative = {}, [], 0
        def store(role, data):
            identity = sha256(data).hexdigest()
            memory[identity] = data
            return {'role': role, 'sha256': identity}
        for number, phase in ((1, 'initial'), (2, 'resumed')):
            raw = self.evidence.read_object(self.refs[f'open_cake_{phase}_provider_events'])
            source = self.evidence.read_object(self.refs[f'open_cake_{phase}_source_file'])
            terminal = canonical_json_bytes({'arm': 'open_cake', 'candidate_written': True,
                'kind': 'open_cake_ir_turn', 'turn': number}).decode()
            parsed = parse_claude_turn_events(raw, expected_terminal_message=terminal,
                event_contract=self.provider['event_contract'], candidate_filename='candidate-set.py')
            state = {'kind': 'ralph_state_v1', 'iteration': number,
                     'cumulative_provider_tokens': cumulative, 'terminal_reason': None}
            objects = [store('provider_events', raw), store('provider_source_file', source),
                       store('provider_reference_bundle', package.evidence_bundle(state))]
            candidates = project_python_candidate_bundle(source, maximum_candidates_per_turn=1)
            objects += [store(f'candidate_submission_{index:04d}', candidate) for index, candidate in enumerate(candidates)]
            cumulative += parsed.provider_tokens
            events.append({'kind': 'provider_turn_completed', 'payload': {'turn': number,
                'thread_id': parsed.thread_id, 'turn_provider_tokens': parsed.provider_tokens,
                'cumulative_provider_tokens': cumulative, 'normalization': parsed.normalization,
                'candidate_count': len(candidates), 'objects': objects,
                'auxiliary_activity': [dict(item.document) for item in parsed.tool_activity]}})
        kwargs = dict(arm='open_cake', audit=SimpleNamespace(run_id='seed-replay'),
            event_contract=self.provider['event_contract'], evidence=SimpleNamespace(read_object=lambda ref: memory[ref['sha256']]),
            expected_task_package=package, maximum_candidates_per_turn=1,
            provider_authority={**self.provider, 'submission_contract': PYTHON_CANDIDATE_BUNDLE_V1}, provider_events=events)
        result = _replay_provider_turns(**kwargs)
        self.assertEqual(set(result[0]), {1, 2})
        with self.assertRaisesRegex(ReplayRefusal, "not a Write"):
            _replay_provider_turns(**{**kwargs, 'expected_task_package': replace(package, initial_candidate_source=None)})


if __name__ == '__main__':
    unittest.main()
