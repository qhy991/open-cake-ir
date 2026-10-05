"""Native-record semantics and adapter capture; no model or qualification claims."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.lab.native_skill_observation import (
    after_invocation, before_invocation, project_rollout,
)
from open_cake_ir.lab.providers import CodexProviderAdapter
from open_cake_ir.lab.provider_documents import ProviderTurn
from open_cake_ir.lab.faults import RunProtocolFault
from tests.contracts import test_native_skill_provider as provider_fixtures

THREAD = '01234567-89ab-cdef-0123-456789abcdef'
TURN1 = '11234567-89ab-cdef-0123-456789abcdef'
TURN2 = '21234567-89ab-cdef-0123-456789abcdef'
SKILL = '/fixture/home/.agents/skills/cake/SKILL.md'
CWD = '/fixture/task'
BODY = '---\nname: cake\ndescription: fixture\n---\nExplore Cake.\n'


def record(kind, **payload):
    return {'type': kind, 'payload': payload}


def catalog(path=SKILL):
    root = str(Path(path).parent.parent)
    relative = str(Path(path).relative_to(root))
    return ('<skills_instructions>\n## Skills\n### Skill roots\n'
            f'- `r0` = `{root}`\n### Available skills\n'
            f'- cake: fixture (file: r0/{relative})\n</skills_instructions>')


def frame(text, turn, kind, role):
    return record('response_item', type='message', role=role,
        content=[{'type': 'input_text', 'text': text}],
        internal_chat_message_metadata_passthrough={
            'turn_id': turn, 'content_item_kinds': [kind]})


def rows(turn=TURN1, *, initial=True, path=SKILL, cwd=CWD, body=BODY, load=True):
    result = [record('session_meta', id=THREAD, cwd=cwd, cli_version='0.159.2',
                     base_instructions='not retained: private unrelated instructions')] if initial else []
    result.append(record('event_msg', type='task_started', turn_id=turn))
    if initial:
        result.append(frame(catalog(path), turn, 'host_skills.instructions', 'developer'))
        result.append(record('world_state', full=True, state={'host_skills': {
            'body': catalog(path).removeprefix('<skills_instructions>').removesuffix('</skills_instructions>'),
            'includeInstructions': True}}))
    result.append(record('turn_context', turn_id=turn, cwd=cwd, model='gpt-6.1-sol', effort='xhigh'))
    result.append(record('response_item', type='message', role='user',
                         content=[{'type': 'input_text', 'text': 'private task prompt not retained'}]))
    if load:
        result.append(frame(f'<skill>\n<name>cake</name>\n<path>{path}</path>\n{body}\n</skill>',
                            turn, 'skills.selected_skill_instructions', 'user'))
    result.append(record('event_msg', type='task_complete', turn_id=turn))
    return result


def encode(records):
    return b''.join(json.dumps(row).encode() + b'\n' for row in records)


class NativeSkillObservationTests(unittest.TestCase):
    def project(self, records=None, **kwargs):
        args = dict(previous=None, thread_id=THREAD, cwd=CWD,
            model='gpt-6.1-sol', effort='xhigh', resumed=False,
            package_paths=(SKILL,), system_paths=())
        args.update(kwargs)
        return project_rollout(encode(rows() if records is None else records), **args)

    def test_current_body_is_separate_from_resume_history(self):
        first = rows()
        second = rows(TURN2, initial=False)
        observed = self.project(first + second, previous=encode(first), resumed=True)
        self.assertEqual(observed['turn_id'], TURN2)
        self.assertEqual(observed['catalog_turn_id'], TURN1)
        self.assertEqual([item['turn_id'] for item in observed['loaded_this_turn']], [TURN2])
        self.assertEqual(observed['loaded_this_turn'][0]['body'], BODY)
        without_new_body = self.project(first + rows(TURN2, initial=False, load=False),
                                        previous=encode(first), resumed=True)
        self.assertEqual(without_new_body['loaded_this_turn'], [])
        self.assertNotIn('private task prompt', json.dumps(observed))
        self.assertNotIn('private unrelated instructions', json.dumps(observed))

    def test_exact_identity_and_version_are_required(self):
        mutations = [('session_meta', 'id', TURN2), ('session_meta', 'cli_version', '0.159.3'),
                     ('session_meta', 'cwd', '/other'), ('turn_context', 'cwd', '/other'),
                     ('turn_context', 'model', 'other'), ('turn_context', 'effort', 'low'),
                     ('turn_context', 'turn_id', TURN2)]
        for kind, key, value in mutations:
            data = rows()
            next(r for r in data if r['type'] == kind)['payload'][key] = value
            with self.subTest(kind=kind, key=key), self.assertRaises(ValueError):
                self.project(data)

    def test_unknown_missing_duplicate_and_aliased_sources_are_refused(self):
        for text in (catalog('/outside/skills/cake/SKILL.md'),
                     catalog().replace('r0/cake', 'unknown/cake'),
                     catalog().replace('r0/cake', 'r0/../cake'),
                     catalog().replace('### Available skills', '- `r0` = `/other`\n### Available skills'),
                     catalog().replace('</skills_instructions>', '- cake: fixture (file: r0/cake/SKILL.md)\n</skills_instructions>')):
            data = rows()
            data[2]['payload']['content'][0]['text'] = text
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.project(data)
        with self.assertRaisesRegex(ValueError, 'missing sources'):
            self.project(package_paths=(SKILL, '/fixture/home/.agents/skills/other/SKILL.md'))
        with self.assertRaisesRegex(ValueError, 'missing sources'):
            self.project(system_paths=('/fixture/codex/skills/.system/example/SKILL.md',))

    def test_selected_frame_requires_native_current_turn_and_catalog_membership(self):
        for field, value in (('role', 'assistant'), ('internal_chat_message_metadata_passthrough', {})):
            data = rows()
            data[-2]['payload'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.project(data)
        data = rows()
        data[-2]['payload']['internal_chat_message_metadata_passthrough']['turn_id'] = TURN2
        with self.assertRaisesRegex(ValueError, 'turn binding'):
            self.project(data)
        data = rows()
        data[-2]['payload']['content'][0]['text'] = data[-2]['payload']['content'][0]['text'].replace('<name>cake', '<name>other')
        with self.assertRaisesRegex(ValueError, 'bound catalog'):
            self.project(data)

    def test_resume_needs_exact_prior_prefix_and_one_new_complete_turn(self):
        first, second = rows(), rows(TURN2, initial=False)
        for previous, data in ((None, first+second), (encode(first), first),
                               (encode(first), first+second[:-1]),
                               (encode(first), first+second+second)):
            with self.subTest(previous=previous is None), self.assertRaises(ValueError):
                self.project(data, previous=previous, resumed=True)
        changed = deepcopy(first)
        changed[0]['payload']['base_instructions'] = 'changed history'
        with self.assertRaisesRegex(ValueError, 'prior prefix'):
            self.project(changed + second, previous=encode(first), resumed=True)

    def test_world_state_and_description_drift_are_not_hidden_by_same_paths(self):
        for mutate in ('description', 'disabled', 'missing'):
            data = rows()
            state = next(r['payload'] for r in data if r['type'] == 'world_state')
            if mutate == 'description':
                state['state']['host_skills']['body'] = state['state']['host_skills']['body'].replace('fixture', 'different')
            elif mutate == 'disabled':
                state['state']['host_skills']['includeInstructions'] = False
            else:
                state['state'] = {}
            with self.subTest(mutate=mutate), self.assertRaisesRegex(ValueError, 'world state'):
                self.project(data)
        first = rows()
        second = rows(TURN2, initial=False)
        second.insert(1, frame(catalog().replace('fixture (file:', 'changed (file:'),
                               TURN2, 'host_skills.instructions', 'developer'))
        with self.assertRaisesRegex(ValueError, 'drifted'):
            self.project(first+second, previous=encode(first), resumed=True)

    def test_compaction_rollback_and_partial_json_do_not_claim_coverage(self):
        for kind in ('compacted', 'thread_rolled_back'):
            data = rows() + [record(kind)]
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'coverage'):
                self.project(data)
        args = dict(previous=None, thread_id=THREAD, cwd=CWD, model='gpt-6.1-sol',
                    effort='xhigh', resumed=False, package_paths=(SKILL,), system_paths=())
        for raw in (encode(rows())[:-1], b'{"type":1,"type":2,"payload":{}}\n'):
            with self.assertRaises(ValueError):
                project_rollout(raw, **args)


class NativeSkillAdapterObservationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = provider_fixtures.NativeSkillProviderTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.builder, self.package, self.home, self.codex = self.fixture.builder('observed')
        self.path = self.codex/'sessions/2026/10/05'/f'rollout-fixture-{THREAD}.jsonl'
        self.skill_path = self.home/'.agents/skills/cake/SKILL.md'
        self.body = self.skill_path.read_text()
        self.turn = ProviderTurn(thread_id=THREAD, provider_tokens=12, candidates=(b'candidate',),
            candidate_sha256s=('a'*64,), raw_submission=b'candidate', raw_events=b'{}\n',
            raw_events_sha256='b'*64, terminal_message='done', terminal_message_count=1,
            normalization='fixture')

    def data(self, turn=TURN1, initial=True, body=None):
        return rows(turn, initial=initial, path=str(self.skill_path), cwd=str(self.builder.workspace),
                    body=self.body if body is None else body)

    def write(self, data):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(encode(data))

    def execute(self, invocation, data):
        def native(*args, **kwargs):
            self.write(data)
            return SimpleNamespace(returncode=0, stdout=b'{}\n', stderr=b'diagnostic notice')
        with patch('open_cake_ir.lab.providers.run_supervised', side_effect=native), patch(
            'open_cake_ir.lab.providers.normalize_codex_turn', return_value=self.turn):
            return CodexProviderAdapter().execute(invocation, candidate_path=self.builder.workspace/'candidate-set.json',
                expected_change='add', expected_terminal_message='done')

    def test_adapter_collects_actual_selected_policy_and_preserves_resume_binding(self):
        first = self.data()
        result = self.execute(self.builder.build('first', thread_id=None), first)
        self.assertEqual(json.loads(result.native_skill_input)['loaded_this_turn'][0]['body'], self.body)
        self.builder.remember_system_skills()
        second = self.execute(self.builder.build('second', thread_id=THREAD), first+self.data(TURN2, False))
        observed = json.loads(second.native_skill_input)
        self.assertEqual(observed['turn_id'], TURN2)
        self.assertEqual(len(observed['loaded_this_turn']), 1)

    def test_body_substitution_fails_before_accepting_candidate(self):
        with self.assertRaisesRegex(RunProtocolFault, 'frozen package'):
            self.execute(self.builder.build('first', thread_id=None), self.data(body='changed body\n'))

    def test_missing_rollout_links_and_duplicate_thread_files_refuse(self):
        invocation = self.builder.build('first', thread_id=None)
        with self.assertRaisesRegex(ValueError, 'no matching rollout'):
            after_invocation(invocation, thread_id=THREAD, previous=None)
        self.write(self.data())
        duplicate = self.path.with_name('other-' + THREAD + '.jsonl')
        duplicate.write_bytes(self.path.read_bytes())
        with self.assertRaisesRegex(ValueError, 'multiple rollouts'):
            after_invocation(invocation, thread_id=THREAD, previous=None)
        duplicate.unlink()
        link = self.path.parent/'foreign'
        link.symlink_to(self.skill_path)
        with self.assertRaisesRegex(ValueError, 'link or special'):
            before_invocation(replace(invocation, thread_id=THREAD))

    def test_other_policy_does_not_claim_or_require_native_skill_observation(self):
        invocation = replace(self.builder.build('first', thread_id=None), user_home=None, native_skill_package=None)
        with patch('open_cake_ir.lab.native_skill_observation.before_invocation') as before:
            observed = self.execute(invocation, self.data())
        self.assertIsNone(observed.native_skill_input)
        before.assert_not_called()
