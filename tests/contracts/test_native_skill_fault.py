"""Returned-Turn native rejection through real fault retention and replay owners."""
from copy import deepcopy
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

from open_cake_ir.lab.native_skill_fault import (
    BINDING_ROLE, CONTEXT_ROLE, INPUT_ROLE, NativeSkillRunInputFault,
)
from open_cake_ir.lab.provider_events import reported_provider_usage
from open_cake_ir.lab.replay.provider import replay_fault_usage
from open_cake_ir.lab.replay.refusals import ReplayRefusal
from open_cake_ir.lab.run_completion import record_run_fault
from open_cake_ir.serialization import canonical_json_bytes
from tests.contracts import test_native_skill_run as fixtures


class NativeSkillFaultEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.NativeSkillRunEvidenceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def changed_projection(self, turn):
        document = json.loads(turn.native_skill_input)
        document['loaded_this_turn'] = []
        return replace(turn, native_skill_input=canonical_json_bytes(document))

    def record_rejection(self, turn, number=1):
        f = self.fixture
        completed = len(f.events)
        with self.assertRaises(NativeSkillRunInputFault) as raised:
            f.archive(turn, number)
        self.assertEqual(len(f.events), completed)
        error = raised.exception
        usage = reported_provider_usage(turn.raw_events, provider=f.provider,
            previous_tokens=120*(number-1), expected_thread_id=turn.thread_id if number > 1 else None)
        record_run_fault(error=error, live_stage='provider', turn_number=number,
            cumulative_tokens=120*number, evidence=f.store, ledger=f.ledger,
            pending_provider_usage=True, observed_usage=usage, declared_usage=usage,
            artifact_payloads={**error.artifact_payloads, 'provider_stdout': turn.raw_events})
        return deepcopy(f.events[-1]['payload'])

    def replay(self, payload, **overrides):
        f = self.fixture
        completed = [event for event in f.events if event['kind'] == 'provider_turn_completed']
        args = dict(payload=payload, evidence=f.store, provider=f.provider,
            expected_thread_id=completed[-1]['payload']['thread_id'] if completed else None,
            previous_tokens=120*len(completed), run_id=f.package.run_id,
            expected_task_package=f.package, provider_events=completed)
        args.update(overrides)
        return replay_fault_usage(**args)

    def replace_object(self, payload, role, raw):
        refs = payload['objects']
        index = next(i for i, ref in enumerate(refs) if ref['role'] == role)
        refs[index] = self.fixture.store.put(raw, media_type='application/json').reference(role)

    def test_initial_rejection_replays_reason_and_consumed_usage_without_completion(self):
        turn = self.changed_projection(self.fixture.make_turn(1))
        payload = self.record_rejection(turn)
        self.assertEqual([event['kind'] for event in self.fixture.events], ['run_fault'])
        refs = {ref['role']: ref for ref in payload['objects']}
        self.assertEqual(self.fixture.store.read_object(refs[INPUT_ROLE]), turn.native_skill_input)
        self.assertEqual(self.fixture.store.read_object(refs[BINDING_ROLE]), turn.native_skill_binding)
        with patch('pathlib.Path.open', side_effect=AssertionError('read live author state')):
            self.assertEqual(self.replay(payload), 120)
        self.assertEqual(payload['provider_usage']['provider_tokens'], 120)
        self.assertEqual(payload['terminal_provider_tokens'], 120)

    def test_resumed_rejection_uses_validated_previous_turn_and_native_usage_delta(self):
        f = self.fixture
        f.archive(f.make_turn(1))
        f.replay()
        payload = self.record_rejection(self.changed_projection(f.make_turn(2)), 2)
        self.assertEqual(self.replay(payload), 120)
        self.assertEqual(payload['terminal_provider_tokens'], 240)
        self.assertEqual([event['kind'] for event in f.events], ['provider_turn_completed', 'run_fault'])
        with self.assertRaisesRegex(ReplayRefusal, 'native_skill_input'):
            self.replay(payload, provider_events=[])

    def test_missing_input_is_distinct_from_lost_retained_input(self):
        turn = replace(self.fixture.make_turn(1), native_skill_input=None)
        payload = self.record_rejection(turn)
        refs = {ref['role']: ref for ref in payload['objects']}
        context = json.loads(self.fixture.store.read_object(refs[CONTEXT_ROLE]))
        self.assertEqual(context['inputs'][INPUT_ROLE], 'missing')
        self.assertNotIn(INPUT_ROLE, refs)
        self.assertEqual(self.replay(payload), 120)
        payload['objects'] = [ref for ref in payload['objects'] if ref['role'] != BINDING_ROLE]
        with self.assertRaisesRegex(ReplayRefusal, 'retention differs'):
            self.replay(payload)

    def test_missing_binding_and_malformed_observation_reproduce_their_own_refusals(self):
        original = self.fixture.make_turn(1)
        for overrides in ({'native_skill_binding': None}, {'native_skill_input': b'{'},
                          {'native_skill_input': b''}):
            with self.subTest(overrides=overrides):
                payload = self.record_rejection(replace(original, **overrides))
                self.assertEqual(self.replay(payload), 120)

    def test_missing_duplicate_or_damaged_context_is_refused(self):
        payload = self.record_rejection(self.changed_projection(self.fixture.make_turn(1)))
        for role in (CONTEXT_ROLE, INPUT_ROLE, BINDING_ROLE):
            for variant in ('missing', 'duplicate'):
                changed = deepcopy(payload)
                reference = next(ref for ref in changed['objects'] if ref['role'] == role)
                if variant == 'missing': changed['objects'].remove(reference)
                else: changed['objects'].append(deepcopy(reference))
                with self.subTest(role=role, variant=variant), self.assertRaisesRegex(ReplayRefusal, 'native_skill_input'):
                    self.replay(changed)
        for raw in (b'{}', b'null', b'[]', b'{"schema_version":1,"schema_version":1}'):
            changed = deepcopy(payload)
            self.replace_object(changed, CONTEXT_ROLE, raw)
            with self.subTest(raw=raw), self.assertRaisesRegex(ReplayRefusal, 'native_skill_input'):
                self.replay(changed)

    def test_rejection_cannot_borrow_another_run_arm_turn_or_usage_thread(self):
        payload = self.record_rejection(self.changed_projection(self.fixture.make_turn(1)))
        ref = next(ref for ref in payload['objects'] if ref['role'] == CONTEXT_ROLE)
        original = json.loads(self.fixture.store.read_object(ref))
        for key, value in (('run_id', 'foreign'), ('arm', 'direct_cuda'), ('turn', 2), ('turn', True),
                           ('thread_id', '11111111-2222-3333-4444-555555555555')):
            context, changed = deepcopy(original), deepcopy(payload)
            context[key] = value
            self.replace_object(changed, CONTEXT_ROLE, canonical_json_bytes(context))
            with self.subTest(key=key), self.assertRaisesRegex(ReplayRefusal, 'context differs'):
                self.replay(changed)

    def test_changed_reason_or_now_valid_inputs_cannot_validate_a_failure(self):
        valid = self.fixture.make_turn(1)
        payload = self.record_rejection(self.changed_projection(valid))
        changed = deepcopy(payload)
        changed['exception_message'] = 'another reason'
        with self.assertRaisesRegex(ReplayRefusal, 'reason differs'):
            self.replay(changed)
        self.replace_object(payload, INPUT_ROLE, valid.native_skill_input)
        with self.assertRaisesRegex(ReplayRefusal, 'do not reproduce'):
            self.replay(payload)

    def test_unretained_values_are_bounded_and_never_claim_replayed_refusal(self):
        original = self.fixture.make_turn(1)
        from open_cake_ir.lab.native_skill_run import MAX_BINDING_BYTES
        for overrides, role in (({'native_skill_input': 'not bytes'}, INPUT_ROLE),
                                ({'native_skill_binding': b'x'*(MAX_BINDING_BYTES+1)}, BINDING_ROLE)):
            with self.subTest(role=role):
                payload = self.record_rejection(replace(original, **overrides))
                self.assertNotIn(role, [ref['role'] for ref in payload['objects']])
                with self.assertRaisesRegex(ReplayRefusal, 'unverified'):
                    self.replay(payload)

    def test_scope_exception_type_and_missing_native_usage_cannot_be_substituted(self):
        payload = self.record_rejection(self.changed_projection(self.fixture.make_turn(1)))
        for key, value in (('exception_type', 'ValueError'), ('stage', 'environment'), ('fault', 'harness_fault')):
            changed = deepcopy(payload)
            changed[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ReplayRefusal, 'native_skill_input'):
                self.replay(changed)
        with self.assertRaisesRegex(ReplayRefusal, 'boundary differs'):
            self.replay(payload, provider={**self.fixture.provider, 'author_home_policy': 'isolated_auth_only_v1'})
        changed = deepcopy(payload)
        changed['objects'] = [ref for ref in changed['objects'] if ref['role'] != 'provider_stdout']
        changed['provider_usage'] = {'status': 'unavailable', 'provider_tokens': None}
        changed['terminal_provider_tokens_scope'] = 'known_subtotal'
        with self.assertRaisesRegex(ReplayRefusal, 'native usage witness'):
            self.replay(changed)
