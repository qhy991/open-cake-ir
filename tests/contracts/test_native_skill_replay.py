"""Independent reconstruction of retained native skill inputs; no model needed."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from open_cake_ir.lab.native_skill_observation import project_rollout, replay_observation
from open_cake_ir.serialization import canonical_json_bytes
from tests.contracts import test_native_skill_observation as fixtures


class NativeSkillReplayTests(unittest.TestCase):
    def capture(self, rows=None, previous=None, system_paths=()):
        return canonical_json_bytes(project_rollout(
            fixtures.encode(fixtures.rows() if rows is None else rows),
            previous=fixtures.encode(previous) if previous is not None else None,
            thread_id=fixtures.THREAD, cwd=fixtures.CWD, model='gpt-6.1-sol', effort='xhigh',
            resumed=previous is not None, package_paths=(fixtures.SKILL,), system_paths=system_paths))

    def replay(self, raw, **overrides):
        args = dict(previous=None, thread_id=fixtures.THREAD, cwd=fixtures.CWD,
            model='gpt-6.1-sol', effort='xhigh', turn=1,
            package_files={fixtures.SKILL: fixtures.BODY.encode()}, system_paths=())
        args.update(overrides)
        return replay_observation(raw, **args)

    def test_two_turns_reconstruct_without_a_live_filesystem(self):
        first_rows = fixtures.rows()
        first = self.capture(first_rows)
        second = self.capture(first_rows + fixtures.rows(fixtures.TURN2, initial=False), first_rows)
        with patch('pathlib.Path.open', side_effect=AssertionError('replay touched live disk')):
            self.assertEqual(self.replay(first), json.loads(first))
            result = self.replay(second, previous=first, turn=2)
        self.assertEqual(result['turn_id'], fixtures.TURN2)
        self.assertEqual([item['turn_id'] for item in result['loaded_this_turn']], [fixtures.TURN2])
        # Both bodies remain source evidence, but only the new one is current input.
        texts = [str(fact) for fact in result['native_records']]
        self.assertEqual(sum('selected_skill_instructions' in text for text in texts), 2)

    def test_capture_omits_unrelated_fields_even_in_mixed_native_records(self):
        rows = fixtures.rows()
        rows[0]['payload']['base_instructions'] = 'PRIVATE-BASE'
        next(row for row in rows if row['type'] == 'turn_context')['payload']['developer_instructions'] = 'PRIVATE-CONTEXT'
        next(row for row in rows if row['type'] == 'world_state')['payload']['state']['other'] = 'PRIVATE-WORLD'
        rows[-1]['payload']['last_agent_message'] = 'PRIVATE-LAST'
        selected = rows[-2]['payload']
        selected['content'].insert(0, {'type': 'input_text', 'text': 'PRIVATE-MIXED'})
        selected['internal_chat_message_metadata_passthrough']['content_item_kinds'].insert(0, 'unrelated')
        raw = self.capture(rows)
        self.assertNotIn(b'PRIVATE-', raw)
        self.assertNotIn(b'private task prompt', raw)
        self.replay(raw)

    def test_projection_cannot_override_retained_source_facts(self):
        original = json.loads(self.capture())
        for key, value in [('cwd', '/foreign'), ('model', 'other'), ('reasoning_effort', 'low'),
                           ('thread_id', fixtures.TURN2), ('turn_id', fixtures.TURN2),
                           ('resumed', True), ('prior_turn_count', 1), ('prior_record_count', 1),
                           ('loaded_this_turn', []), ('catalog', []), ('record_count', True),
                           ('unowned', 'extra')]:
            document = deepcopy(original)
            document[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.replay(canonical_json_bytes(document))

    def test_missing_or_changed_native_source_facts_refuse(self):
        original = json.loads(self.capture())
        for kind in ('session_meta', 'turn_context', 'world_state'):
            for action in ('missing', 'changed'):
                document = deepcopy(original)
                index = next(i for i, fact in enumerate(document['native_records'])
                             if fact['record']['type'] == kind)
                if action == 'missing':
                    document['native_records'].pop(index)
                else:
                    payload = document['native_records'][index]['record']['payload']
                    if kind == 'session_meta': payload['cli_version'] = 'other'
                    elif kind == 'turn_context': payload['effort'] = 'low'
                    else: payload['state']['host_skills']['includeInstructions'] = False
                with self.subTest(kind=kind, action=action), self.assertRaises(ValueError):
                    self.replay(canonical_json_bytes(document))
        document = deepcopy(original)
        document['native_records'][-1]['record']['payload']['turn_id'] = fixtures.TURN2
        with self.assertRaises(ValueError): self.replay(canonical_json_bytes(document))

    def test_external_invocation_and_material_authority_cannot_be_swapped(self):
        raw = self.capture()
        for overrides in ({'thread_id': fixtures.TURN2}, {'cwd': '/foreign'}, {'model': 'other'},
                          {'effort': 'low'}, {'turn': 2},
                          {'package_files': {fixtures.SKILL: b'changed'}},
                          {'package_files': {'/foreign/SKILL.md': fixtures.BODY.encode()}},
                          {'system_paths': ('/foreign/system/SKILL.md',)}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.replay(raw, **overrides)

    def test_replayed_resume_requires_the_preceding_retained_prefix(self):
        rows = fixtures.rows()
        first = self.capture(rows)
        second = self.capture(rows + fixtures.rows(fixtures.TURN2, initial=False), rows)
        for raw, overrides in ((second, {}), (first, {'previous': first, 'turn': 2}),
                               (second, {'previous': first, 'turn': 3})):
            with self.assertRaises(ValueError): self.replay(raw, **overrides)
        changed = json.loads(second)
        fact = next(f for f in changed['native_records'] if f['record']['type'] == 'turn_context')
        fact['record']['payload']['model'] = 'changed history'
        with self.assertRaisesRegex(ValueError, 'prior prefix'):
            self.replay(canonical_json_bytes(changed), previous=first, turn=2)

    def test_record_order_bounds_duplicates_and_extra_fields_are_closed(self):
        original = json.loads(self.capture())
        for mutation in ('order', 'duplicate', 'outside', 'extra', 'private'):
            document = deepcopy(original)
            records = document['native_records']
            if mutation == 'order': records.reverse()
            elif mutation == 'duplicate': records.insert(1, records[0])
            elif mutation == 'outside': records[-1]['line'] = 100001
            elif mutation == 'extra': records[0]['unowned'] = 1
            else: records[0]['record']['payload']['private'] = 'should never be retained'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.replay(canonical_json_bytes(document))
        with self.assertRaises(ValueError): self.replay(b'{"schema_version":2,"schema_version":2}')
        with self.assertRaises(ValueError): self.replay(b'{}')

    def test_malformed_native_frame_metadata_is_a_semantic_refusal(self):
        for kinds in ([{}], ['host_skills.instructions', 'skills.selected_skill_instructions']):
            document = json.loads(self.capture())
            frame = next(f['record']['payload'] for f in document['native_records']
                         if f['record']['type'] == 'response_item')
            frame['internal_chat_message_metadata_passthrough']['content_item_kinds'] = kinds
            with self.subTest(kinds=kinds), self.assertRaises(ValueError):
                self.replay(canonical_json_bytes(document))

    def test_old_projection_and_whole_transcript_are_not_replay_certificates(self):
        old = json.loads(self.capture())
        old['schema_version'] = 1
        old['kind'] = 'codex_skill_input_observation_v1'
        with self.assertRaisesRegex(ValueError, 'v2'): self.replay(canonical_json_bytes(old))
        with self.assertRaises(ValueError): self.replay(fixtures.encode(fixtures.rows()))

    def test_installed_but_undelivered_system_entry_is_retained_as_absent(self):
        hidden = ('/fixture/codex/skills/.system/explicit/SKILL.md',)
        raw = self.capture(system_paths=hidden)
        result = self.replay(raw, system_paths=hidden)
        self.assertEqual(result['system_entrypoints_not_in_catalog'], list(hidden))
