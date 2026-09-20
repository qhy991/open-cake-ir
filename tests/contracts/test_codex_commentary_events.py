"""A real tool-rich turn may report progress before its structured final marker."""
import json
from pathlib import Path
import tempfile
import unittest

from open_cake_ir.lab.providers import normalize_codex_turn, parse_codex_turn_events

TERMINAL = '{"arm":"open_cake","candidate_written":true,"kind":"open_cake_ir_turn","turn":1}'


def message(identity, text):
    return {"type": "item.completed", "item": {"id": identity, "type": "agent_message", "text": text}}


def events():
    return [
        {"type": "thread.started", "thread_id": "00000000-0000-4000-8000-000000000001"},
        {"type": "turn.started"},
        message("progress", "I'll read the task files and write the requested candidate."),
        {"type": "item.started", "item": {"id": "command", "type": "command_execution", "command": "pwd", "status": "in_progress"}},
        {"type": "item.completed", "item": {"id": "command", "type": "command_execution", "command": "pwd", "status": "completed"}},
        message("final", TERMINAL),
        {"type": "turn.completed", "usage": {"input_tokens": 100, "cached_input_tokens": 80, "cache_write_input_tokens": 0, "output_tokens": 20, "reasoning_output_tokens": 0}},
    ]


def encode(value):
    return b"\n".join(json.dumps(event).encode() for event in value) + b"\n"


class CodexCommentaryEventsTests(unittest.TestCase):
    def parse(self, value):
        return parse_codex_turn_events(encode(value), expected_terminal_message=TERMINAL,
                                      event_contract="tool_rich_candidate_v1")

    def test_progress_is_retained_and_only_final_marker_counts_as_terminal(self):
        parsed = self.parse(events())
        self.assertEqual(parsed.terminal_message_count, 1)
        self.assertEqual(parsed.normalization, "single_exact")
        self.assertEqual(parsed.provider_tokens, 120)
        self.assertEqual([(a.item_id, a.item_type) for a in parsed.tool_activity],
                         [("progress", "agent_message"), ("command", "command_execution")])

    def test_progress_between_tools_and_terminal_is_retained(self):
        value = events()
        value.insert(-2, message("progress_after_tools", "The candidate file is ready."))
        parsed = self.parse(value)
        self.assertEqual(parsed.terminal_message_count, 1)
        self.assertEqual(parsed.tool_activity[-1].item_id, "progress_after_tools")

    def test_progress_after_terminal_or_in_place_of_terminal_is_refused(self):
        for value in (events()[:-2] + [message("prose", "Done."), events()[-1]],
                      events()[:-1] + [message("prose", "Done."), events()[-1]]):
            with self.assertRaisesRegex(ValueError, "final agent message"):
                self.parse(value)

    def test_wrong_or_malformed_structured_terminal_is_not_hidden_as_progress(self):
        for text in (TERMINAL.replace('"turn":1', '"turn":2'), '{"kind":', '[1]'):
            value = events()
            value.insert(-2, message("wrong_terminal", text))
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "normalization"):
                self.parse(value)

    def test_progress_does_not_supply_functional_tool_activity(self):
        value = events()
        del value[3:5]
        value.insert(3, message("more_progress", "Still preparing."))
        with self.assertRaisesRegex(ValueError, "functional tool"):
            self.parse(value)

    def test_closed_contract_still_refuses_unstructured_commentary(self):
        with self.assertRaises(ValueError):
            parse_codex_turn_events(encode(events()), expected_terminal_message=TERMINAL,
                                    event_contract="closed_file_change_v1")

    def test_candidate_postconditions_still_apply_after_valid_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate-set.json"
            kwargs = dict(candidate_path=candidate, expected_change="add",
                          expected_terminal_message=TERMINAL, event_contract="tool_rich_candidate_v1",
                          arm="open_cake", maximum_candidates_per_turn=2)
            with self.assertRaises(FileNotFoundError):
                normalize_codex_turn(encode(events()), **kwargs)
            candidate.write_text('{"schema_version":1,"arm":"open_cake","candidates":[{"schedule":1}]}')
            turn = normalize_codex_turn(encode(events()), **kwargs)
            self.assertEqual(len(turn.candidates), 1)
            candidate.write_text('{"schema_version":1,"arm":"wrong","candidates":[{"schedule":1}]}')
            with self.assertRaises(ValueError):
                normalize_codex_turn(encode(events()), **kwargs)
