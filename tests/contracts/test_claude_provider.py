"""Claude-native Lab provider contracts; fixtures only, never a live qualification."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from open_cake_ir.compiler.frontend import parse as parse_python_schedule
from open_cake_ir.lab.claude import (
    CLAUDE_EVENT_CONTRACT, ClaudeInvocationBuilder, ClaudeProviderAdapter,
    ClaudeRunProvider, normalize_claude_turn, parse_claude_turn_events,
)
from open_cake_ir.lab.faults import RunProtocolFault
from open_cake_ir.lab.process import SupervisedProcessTimeout
from open_cake_ir.lab.providers import CodexRunProvider, QualifiedRunProvider

SESSION = "01234567-89ab-cdef-0123-456789abcdef"
OTHER_SESSION = "11111111-2222-3333-4444-555555555555"
TERMINAL = '{"candidate_written":true}'


class ClaudeProviderContracts(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.executable = self.root / "claude"
        self.executable.write_bytes(b"not a CLI; never execute this fixture")
        self.executable.chmod(0o700)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.candidate = self.workspace / "candidate-set.json"
        self.source = (Path(__file__).resolve().parents[2] / "examples/python/metal_row_sum.py").read_text()
        self.submission = json.dumps({"schema_version": 1, "arm": "open_cake", "candidates": [
            {"python_source": self.source}]}).encode()
        self.candidate.write_bytes(self.submission)

    def builder(self, **changes):
        args = dict(executable=self.executable, provider_revision="claude-native-contract-fixture",
                    model="exact-requested-model", reasoning_effort="high", workspace=self.workspace,
                    removed_environment=("OPENAI_API_KEY", "ANTHROPIC_API_KEY"))
        args.update(changes)
        return ClaudeInvocationBuilder(**args)

    def events(self):
        return [
            {"type": "system", "subtype": "init", "session_id": SESSION, "model": "opus"},
            {"type": "assistant", "session_id": SESSION, "parent_tool_use_id": None, "message": {
                "model": "reported-response-model", "content": [
                    {"type": "tool_use", "id": "toolu_write", "name": "Write", "input": {
                        "file_path": str(self.candidate), "content": self.submission.decode()}}]}},
            {"type": "user", "session_id": SESSION, "parent_tool_use_id": None, "message": {"content": [
                {"type": "tool_result", "tool_use_id": "toolu_write", "content": "File created successfully"}]}},
            {"type": "assistant", "session_id": SESSION, "message": {"model": "reported-response-model", "content": [
                {"type": "text", "text": TERMINAL}]}},
            {"type": "result", "subtype": "success", "is_error": False, "session_id": SESSION, "result": TERMINAL,
             "usage": {"input_tokens": 10, "output_tokens": 5, "cache_creation_input_tokens": 90,
                       "cache_read_input_tokens": 100, "cache_creation": {
                           "ephemeral_5m_input_tokens": 70, "ephemeral_1h_input_tokens": 20}}},
        ]

    def raw(self, events=None):
        return b"".join(json.dumps(event).encode() + b"\n"
                        for event in (self.events() if events is None else events))

    def normalize(self, raw=None, **kwargs):
        args = dict(candidate_path=self.candidate, expected_change="add", expected_terminal_message=TERMINAL,
                    arm="open_cake")
        args.update(kwargs)
        return normalize_claude_turn(self.raw() if raw is None else raw, **args)

    def test_native_stream_projects_existing_python_envelope_and_raw_evidence(self):
        raw = self.raw()
        parsed = parse_claude_turn_events(raw, expected_terminal_message=TERMINAL)
        turn = self.normalize(raw)
        self.assertEqual(parsed.reported_models, ("opus", "reported-response-model"))
        self.assertEqual(turn.thread_id, SESSION)
        self.assertEqual(turn.provider_tokens, 205)
        self.assertEqual(turn.raw_events, raw)
        self.assertEqual(turn.raw_submission, self.submission)
        self.assertEqual(json.loads(turn.candidates[0]), {"python_source": self.source})
        authored = parse_python_schedule(json.loads(turn.candidates[0])["python_source"])
        self.assertEqual(authored.document["lowering"]["backend"], "metal")
        self.assertEqual(turn.tool_activity[0].tool, "Write")
        self.assertEqual(turn.tool_activity[0].item_type, "tool_use")
        self.assertIsNone(turn.reference_bundle)  # The shared Run lifecycle supplies it.
        self.assertNotIn(b'"thread.started"', turn.raw_events)

    def test_initial_and_resumed_commands_keep_exact_model_effort_and_workspace(self):
        builder = self.builder(model="opaque-model;$(whoami)")
        first = builder.build("first prompt", thread_id=None)
        resumed = builder.build("second prompt", thread_id=SESSION)
        for invocation in (first, resumed):
            argv = invocation.argv
            self.assertEqual(invocation.cwd, self.workspace)
            self.assertEqual(invocation.sandbox, "none")
            self.assertEqual(argv[argv.index("--model") + 1], "opaque-model;$(whoami)")
            self.assertEqual(argv[argv.index("--effort") + 1], "high")
            self.assertEqual(argv[argv.index("--permission-mode") + 1], "acceptEdits")
            self.assertIn("--safe-mode", argv)
            self.assertEqual(argv[-2], "--")
            self.assertFalse(any("bypass" in value or "skip-permissions" in value for value in argv))
            self.assertNotIn("--fallback-model", argv)
            self.assertNotIn("Bash", argv[argv.index("--tools") + 1])
        self.assertNotIn("--resume", first.argv)
        self.assertEqual(resumed.argv[resumed.argv.index("--resume") + 1], SESSION)
        self.assertEqual(builder.configuration["sandbox"], "none")
        with self.assertRaises(ValueError):
            self.builder(reasoning_effort="")
        with self.assertRaises(ValueError):
            builder.build("task", thread_id="--last")

    def test_both_harnesses_reuse_the_same_qualified_run_lifecycle(self):
        self.assertIs(ClaudeRunProvider.turn, QualifiedRunProvider.turn)
        self.assertIs(CodexRunProvider.turn, QualifiedRunProvider.turn)

    def test_successful_edit_is_update_only_and_preserves_session_identity(self):
        events = self.events()
        events[1]["message"]["content"][0]["name"] = "Edit"
        raw = self.raw(events)
        self.assertEqual(self.normalize(raw, expected_change="update", expected_thread_id=SESSION).thread_id, SESSION)
        with self.assertRaisesRegex(ValueError, "initial write"):
            self.normalize(raw)
        with self.assertRaisesRegex(ValueError, "resumed session"):
            self.normalize(raw, expected_change="update", expected_thread_id=OTHER_SESSION)

    def test_incomplete_failure_foreign_session_and_tool_faults_refuse(self):
        variants = []
        missing = self.events(); missing.pop(2); variants.append(missing)
        failed = self.events(); failed[-1]["is_error"] = True; variants.append(failed)
        wrong = self.events(); wrong[-1]["session_id"] = OTHER_SESSION; variants.append(wrong)
        duplicate = self.events(); duplicate.insert(3, copy.deepcopy(duplicate[2])); variants.append(duplicate)
        failed_tool = self.events(); failed_tool[2]["message"]["content"][0]["is_error"] = True; variants.append(failed_tool)
        forbidden = self.events(); forbidden[1]["message"]["content"][0]["name"] = "Bash"; variants.append(forbidden)
        outside = self.events(); outside[1]["message"]["content"][0]["input"]["file_path"] = str(self.workspace / "AGENTS.md"); variants.append(outside)
        child = self.events(); child[1]["parent_tool_use_id"] = "parent"; variants.append(child)
        malformed = self.events(); malformed[1]["message"]["content"][0]["type"] = []; variants.append(malformed)
        for index, events in enumerate(variants):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.normalize(self.raw(events))

    def test_missing_invalid_and_zero_usage_refuse_without_accounting_guess(self):
        for usage in (None, {}, {"input_tokens": 10, "output_tokens": 3},
                      {"input_tokens": True, "output_tokens": 3, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
                      {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}):
            events = self.events(); events[-1]["usage"] = usage
            with self.subTest(usage=usage), self.assertRaisesRegex(ValueError, "usage"):
                self.normalize(self.raw(events))

    def test_terminal_semantics_are_typed_and_candidate_custody_stays_canonical(self):
        events = self.events(); events[-1]["result"] = '{ "candidate_written" : true }'
        self.assertEqual(self.normalize(self.raw(events)).normalization, "claude_result_semantic")
        events[-1]["result"] = '{"candidate_written":1}'
        with self.assertRaisesRegex(ValueError, "terminal message"):
            self.normalize(self.raw(events))
        with self.assertRaisesRegex(ValueError, "maximum candidates"):
            self.normalize(maximum_candidates_per_turn=0)

    def test_candidate_path_custody_and_unsupported_invocation_refuse(self):
        stored = self.workspace / "retained.json"
        self.candidate.rename(stored)
        self.candidate.symlink_to(stored)
        with self.assertRaises(OSError):
            self.normalize()
        invocation = self.builder().build("task", thread_id=SESSION)
        with patch("open_cake_ir.lab.claude.run_supervised") as process:
            with self.assertRaises(ValueError):
                ClaudeProviderAdapter().execute(invocation, candidate_path=self.candidate,
                    expected_change="update", expected_terminal_message=TERMINAL,
                    event_contract="closed_file_change_v1", arm="open_cake")
        process.assert_not_called()

    def test_adapter_uses_existing_supervisor_and_retains_failed_streams(self):
        invocation = self.builder().build("task", thread_id=SESSION)
        adapter = ClaudeProviderAdapter(timeout_seconds=17)
        completed = subprocess.CompletedProcess(invocation.argv, 0, self.raw(), b"")
        with patch("open_cake_ir.lab.claude.run_supervised", return_value=completed) as process:
            turn = adapter.execute(invocation, candidate_path=self.candidate, expected_change="update",
                                   expected_terminal_message=TERMINAL, arm="open_cake")
        self.assertEqual(turn.provider_tokens, 205)
        self.assertEqual(process.call_args.kwargs["timeout_seconds"], 17)
        self.assertEqual(process.call_args.kwargs["cwd"], self.workspace)
        failure = SupervisedProcessTimeout(b"partial Claude JSONL", b"stderr retained")
        with patch("open_cake_ir.lab.claude.run_supervised", side_effect=failure):
            with self.assertRaises(RunProtocolFault) as captured:
                adapter.execute(invocation, candidate_path=self.candidate, expected_change="update",
                                expected_terminal_message=TERMINAL, arm="open_cake")
        self.assertEqual(captured.exception.artifact_payloads["provider_stdout"], failure.stdout)
        self.assertEqual(captured.exception.protocol_adherence, "provider_fault")
        self.assertEqual(CLAUDE_EVENT_CONTRACT, self.builder().configuration["event_contract"])


if __name__ == "__main__":
    unittest.main()
