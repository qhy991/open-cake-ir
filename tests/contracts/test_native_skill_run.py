"""CPU archive/replay integration. Fixture records grant no live qualification."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.evidence import EvidenceStore
from open_cake_ir.lab.archive import _archive_provider_turn
from open_cake_ir.lab.author_home import system_skills_identity, require_live_skill_qualification
from open_cake_ir.lab.native_skill_observation import project_rollout
from open_cake_ir.lab.native_skill_run import bind_invocation
from open_cake_ir.lab.providers import normalize_codex_turn
from open_cake_ir.lab.replay.provider import _replay_provider_turns, _expected_terminal_message
from open_cake_ir.lab.replay.refusals import ReplayRefusal
from open_cake_ir.lab.task_package import TaskPackage
from open_cake_ir.serialization import canonical_json_bytes
from tests.contracts import test_native_skill_observation as native
from tests.contracts import test_native_skill_provider as fixtures
from tests.contracts import test_provider as provider_fixtures


class NativeSkillRunEvidenceTests(unittest.TestCase):
    def setUp(self):
        fixtures.NativeSkillProviderTests.setUp(self)
        self.builder, package, self.home, self.codex = fixtures.NativeSkillProviderTests.builder(self, 'run')
        self.package = TaskPackage('fixture-run', 'open_cake', 'task', 'agents', native_skill_package=package)
        configuration = dict(self.builder.configuration)
        self.provider = {**configuration, 'revision': 'fixture', 'qualification': None,
            'qualification_anchor': None, 'executable_sha256': 'a'*64,
            'output_schema': {'path': str(fixtures.ROOT/'contracts/providers/codex-turn-output-schema-v1.json'),
                              'sha256': configuration.pop('output_schema_sha256')},
            'native_skill_package': package.reference, 'system_skills_sha256': system_skills_identity(())}
        self.provider.pop('output_schema_sha256')
        self.provider.pop('native_skill_package_sha256')
        self.events = []
        self.store = EvidenceStore.create(self.root/'evidence')
        authority = {'kind': 'cpu-native-skill-fixture'}
        real_ledger = self.store.start_run(self.package.run_id,
            authority_sha256=sha256(canonical_json_bytes(authority)).hexdigest(), authority=authority)
        def append(kind, payload):
            real_ledger.append(kind, payload)
            self.events.append({'kind': kind, 'payload': deepcopy(payload)})
        self.ledger = SimpleNamespace(run_id=self.package.run_id, append=append)
        self.previous = None
        self.native_rows = []

    def make_turn(self, number):
        candidate = self.builder.workspace/'candidate-set.json'
        candidate.write_bytes(canonical_json_bytes({'schema_version': 1, 'arm': 'open_cake',
                                                  'candidates': [{'fixture_candidate': number}]}))
        terminal = _expected_terminal_message('open_cake', number, 'closed_file_change_v1')
        raw = provider_fixtures.ProviderContractTests._events(self, candidate, duplicate=False,
                                                           second_text=terminal)
        if number > 1:
            raw = raw.replace(b'"kind":"add"', b'"kind":"update"')
        events = [json.loads(line) for line in raw.splitlines()]
        # Codex reports a session cumulative counter; the Run archives a delta.
        events[-1]['usage'] = {'input_tokens': 100*number, 'output_tokens': 20*number}
        raw = native.encode(events)
        turn = normalize_codex_turn(raw, candidate_path=candidate,
            expected_change='add' if number == 1 else 'update', expected_terminal_message=terminal,
            arm='open_cake', maximum_candidates_per_turn=1)
        invocation = self.builder.build('private fixture prompt', thread_id=None if number == 1 else native.THREAD)
        self.builder.remember_system_skills()
        skill_path = str(self.home/'.agents/skills/cake/SKILL.md')
        body = next(item.payload.decode() for item in self.package.native_skill_package.files
                    if item.path == 'skills/cake/SKILL.md')
        prior_rows = list(self.native_rows)
        self.native_rows += native.rows(native.TURN1 if number == 1 else native.TURN2,
            initial=number == 1, path=skill_path, cwd=str(self.builder.workspace), body=body)
        observation = project_rollout(native.encode(self.native_rows),
            previous=native.encode(prior_rows) if prior_rows else None,
            thread_id=native.THREAD, cwd=str(self.builder.workspace), model='gpt-6.1-sol', effort='xhigh',
            resumed=number > 1, package_paths=(skill_path,), system_paths=())
        binding = bind_invocation(invocation=invocation,
            request=SimpleNamespace(run_id=self.package.run_id, arm='open_cake', turn=number),
            configuration=self.builder.configuration, system_skills_snapshot=self.builder.remembered_system_skills)
        state = {'kind': 'ralph_state_v1', 'iteration': number,
                 'cumulative_provider_tokens': 120*(number-1), 'terminal_reason': None}
        from open_cake_ir.lab.provider_events import provider_token_delta
        return replace(turn, reference_bundle=self.package.evidence_bundle(state),
            provider_tokens=provider_token_delta(turn.provider_tokens, provider=self.provider,
                                                 previous_tokens=120*(number-1)),
            native_skill_input=canonical_json_bytes(observation), native_skill_binding=binding)

    def archive(self, turn, number=1, **overrides):
        args = dict(arm='open_cake', candidate_media_type='application/json', cumulative_tokens=120*number,
            evidence=self.store, ledger=self.ledger, maximum_candidates_per_turn=1,
            provider_document=self.provider, provider_turn=turn, thread_id=native.THREAD,
            turn_number=number, task_package=self.package,
            previous_native_input=self.previous.native_skill_input if self.previous else None,
            previous_native_binding=self.previous.native_skill_binding if self.previous else None)
        args.update(overrides)
        _archive_provider_turn(**args)
        self.previous = turn

    def replay(self, **overrides):
        args = dict(arm='open_cake', audit=SimpleNamespace(run_id=self.package.run_id),
            event_contract='closed_file_change_v1', evidence=self.store, expected_task_package=self.package,
            maximum_candidates_per_turn=1, provider_authority=self.provider, provider_events=self.events)
        args.update(overrides)
        return _replay_provider_turns(**args)

    def replace_object(self, role, raw, event=0):
        refs = self.events[event]['payload']['objects']
        index = next(i for i, ref in enumerate(refs) if ref['role'] == role)
        refs[index] = self.store.put(raw, media_type='application/json').reference(role)

    def test_completed_two_turns_replay_from_evidence_without_original_homes(self):
        self.archive(self.make_turn(1))
        self.archive(self.make_turn(2), 2)
        with patch('pathlib.Path.open', side_effect=AssertionError('replay reopened live source')):
            self.assertEqual(self.replay()[0], {1: 120, 2: 240})
        roles = [ref['role'] for ref in self.events[0]['payload']['objects']]
        self.assertEqual(roles.count('provider_native_skill_input'), 1)
        self.assertEqual(roles.count('provider_native_skill_binding'), 1)
        self.assertNotIn(b'private fixture prompt', self.previous.native_skill_binding)

    def test_archive_refuses_missing_or_foreign_evidence_before_completion(self):
        turn = self.make_turn(1)
        for field, value in (('native_skill_input', None), ('native_skill_binding', None),
                             ('native_skill_binding', b'{}')):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.archive(replace(turn, **{field: value}))
            self.assertEqual(self.events, [])
        with self.assertRaisesRegex(ValueError, 'TaskPackage'):
            self.archive(turn, task_package=replace(self.package, run_id='foreign'))

    def test_run_arm_turn_and_invocation_policy_are_bound_before_archive(self):
        turn = self.make_turn(1)
        original = json.loads(turn.native_skill_binding)
        mutations = [('run_id', 'foreign'), ('arm', 'direct_cuda'), ('turn', 2), ('turn', True)]
        for field, value in mutations:
            document = deepcopy(original)
            document[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'binding'):
                self.archive(replace(turn, native_skill_binding=canonical_json_bytes(document)))
        for field, value in [('cwd', '/foreign'), ('user_home', original['invocation']['cwd']),
                              ('provider_revision', 'foreign'), ('thread_id', native.THREAD)]:
            document = deepcopy(original)
            document['invocation'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.archive(replace(turn, native_skill_binding=canonical_json_bytes(document)))
        self.assertEqual(self.events, [])

    def test_model_flags_schema_and_installed_tree_cannot_be_changed(self):
        turn = self.make_turn(1)
        original = json.loads(turn.native_skill_binding)
        for variant in ('model', 'effort', 'configuration', 'schema', 'snapshot', 'extra'):
            document = deepcopy(original)
            argv = document['invocation']['argv_without_prompt']
            if variant == 'model': argv[argv.index('--model')+1] = 'other'
            elif variant == 'effort': argv[argv.index('model_reasoning_effort="xhigh"')] = 'model_reasoning_effort="low"'
            elif variant == 'configuration': document['configuration']['model'] = 'other'
            elif variant == 'schema': argv[argv.index('--output-schema')+1] = '/foreign/schema.json'
            elif variant == 'snapshot': document['system_skills_snapshot'] = [['foreign/', 0o755, '']]
            else: document['unowned'] = True
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                self.archive(replace(turn, native_skill_binding=canonical_json_bytes(document)))

    def test_replay_requires_exactly_one_native_role_each(self):
        self.archive(self.make_turn(1))
        original = deepcopy(self.events)
        for role in ('provider_native_skill_input', 'provider_native_skill_binding'):
            for mutation in ('missing', 'duplicate', 'renamed'):
                self.events = deepcopy(original)
                refs = self.events[0]['payload']['objects']
                ref = next(item for item in refs if item['role'] == role)
                if mutation == 'missing': refs.remove(ref)
                elif mutation == 'duplicate': refs.append(deepcopy(ref))
                else: ref['role'] = 'unowned'
                with self.subTest(role=role, mutation=mutation), self.assertRaisesRegex(ReplayRefusal, role):
                    self.replay()

    def test_replay_reconstructs_native_facts_instead_of_trusting_projection(self):
        turn = self.make_turn(1)
        self.archive(turn)
        changed = json.loads(turn.native_skill_input)
        changed['loaded_this_turn'] = []
        self.replace_object('provider_native_skill_input', canonical_json_bytes(changed))
        with self.assertRaisesRegex(ReplayRefusal, 'native_skill_input'):
            self.replay()

    def test_replay_rejects_different_run_even_with_valid_native_frames(self):
        turn = self.make_turn(1)
        self.archive(turn)
        changed = json.loads(turn.native_skill_binding)
        changed['run_id'] = 'foreign'
        self.replace_object('provider_native_skill_binding', canonical_json_bytes(changed))
        with self.assertRaisesRegex(ReplayRefusal, 'Run/arm/turn'):
            self.replay()

    def test_resume_refuses_reused_initial_input_and_changed_allocations(self):
        first = self.make_turn(1)
        self.archive(first)
        second = self.make_turn(2)
        with self.assertRaises(ValueError):
            self.archive(replace(second, native_skill_input=first.native_skill_input), 2)
        changed = json.loads(second.native_skill_binding)
        changed['invocation']['codex_home'] = '/new/codex-home'
        with self.assertRaisesRegex(ValueError, 'paths changed'):
            self.archive(replace(second, native_skill_binding=canonical_json_bytes(changed)), 2)
        self.archive(second, 2)
        self.replace_object('provider_native_skill_input', first.native_skill_input, event=1)
        with self.assertRaisesRegex(ReplayRefusal, 'native_skill_input'):
            self.replay()

    def test_frozen_material_change_cannot_borrow_original_delivery(self):
        turn = self.make_turn(1)
        changed = fixtures.NativeSkillProviderTests.builder(self, 'changed', fixtures.archive_bytes(b'other'))[1]
        package = replace(self.package, native_skill_package=changed)
        with self.assertRaises(ValueError):
            self.archive(turn, task_package=package)
        self.archive(turn)
        with self.assertRaises(ReplayRefusal):
            self.replay(expected_task_package=package)

    def test_non_skill_policy_keeps_old_roles_and_rejects_native_evidence(self):
        turn = self.make_turn(1)
        provider = {key: value for key, value in self.provider.items()
                    if key not in {'author_home_policy', 'native_skill_package', 'system_skills_sha256'}}
        package = replace(self.package, native_skill_package=None)
        with self.assertRaisesRegex(ValueError, 'undeclared'):
            self.archive(turn, provider_document=provider, task_package=package)
        state = {'kind': 'ralph_state_v1', 'iteration': 1, 'cumulative_provider_tokens': 0, 'terminal_reason': None}
        clean = replace(turn, native_skill_input=None, native_skill_binding=None,
                        reference_bundle=package.evidence_bundle(state))
        self.archive(clean, provider_document=provider, task_package=package)
        self.assertEqual(len(self.events[0]['payload']['objects']), 4)
        self.assertEqual(self.replay(provider_authority=provider, expected_task_package=package)[0], {1: 120})
        self.events[0]['payload']['objects'].append(
            self.store.put(turn.native_skill_input, media_type='application/json').reference('provider_native_skill_input'))
        with self.assertRaisesRegex(ReplayRefusal, 'provider_native_skill_input'):
            self.replay(provider_authority=provider, expected_task_package=package)

    def test_archive_evidence_does_not_open_live_admission(self):
        self.archive(self.make_turn(1))
        self.replay()
        with self.assertRaisesRegex(ValueError, 'not qualified'):
            require_live_skill_qualification(self.provider['author_home_policy'])
