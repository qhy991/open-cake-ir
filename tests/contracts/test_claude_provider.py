"""Claude-native Lab provider contracts; fixtures only, never a live qualification."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from open_cake_ir.compiler.frontend import parse as parse_python_schedule
from open_cake_ir.lab.claude import (
    CLAUDE_EVENT_CONTRACT, ClaudeInvocationBuilder, ClaudeProviderAdapter,
    ClaudeRunProvider, normalize_claude_turn, parse_claude_turn_events, claude_model_usage, terminal_schema, reported_claude_usage,
)
from open_cake_ir.lab.faults import RunProtocolFault
from open_cake_ir.lab.process import SupervisedProcessTimeout
from open_cake_ir.lab.providers import CodexRunProvider, QualifiedRunProvider, ProviderAuxiliaryActivity

SESSION = "01234567-89ab-cdef-0123-456789abcdef"
OTHER_SESSION = "11111111-2222-3333-4444-555555555555"
TERMINAL = '{"arm":"open_cake","candidate_written":true,"kind":"open_cake_ir_turn","turn":1}'


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
            {"type": "system", "subtype": "init", "session_id": SESSION, "model": "exact-requested-model"},
            {"type": "assistant", "session_id": SESSION, "parent_tool_use_id": None, "message": {
                "model": "exact-requested-model", "content": [
                    {"type": "tool_use", "id": "toolu_write", "name": "Write", "input": {
                        "file_path": str(self.candidate), "content": self.submission.decode()}}]}},
            {"type": "user", "session_id": SESSION, "parent_tool_use_id": None, "message": {"content": [
                {"type": "tool_result", "tool_use_id": "toolu_write", "content": "File created successfully"}]}},
            {"type": "assistant", "session_id": SESSION, "message": {"model": "exact-requested-model", "content": [
                {"type": "text", "text": TERMINAL}]}},
            {"type": "result", "subtype": "success", "is_error": False, "session_id": SESSION, "result": "", "structured_output": json.loads(TERMINAL),
             "modelUsage": {"exact-requested-model": {"inputTokens": 10, "outputTokens": 5, "cacheCreationInputTokens": 90, "cacheReadInputTokens": 100}},
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
        self.assertEqual(parsed.reported_models, ("exact-requested-model",))
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
            # F-2026-09-10-008: the compaction boundary is declared at launch.
            self.assertEqual(argv[argv.index("--autocompact") + 1], "1M")
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
        events = self.events(); events[-1]["structured_output"] = json.loads(TERMINAL)
        self.assertEqual(self.normalize(self.raw(events)).normalization, "claude_native_structured_output_exact")
        events[-1]["structured_output"]["candidate_written"] = 1
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
        events = self.events()
        events[0]["model"] = "exact-requested-model"
        for event in events:
            if event.get("type") == "assistant":
                event["message"]["model"] = "exact-requested-model"
        completed = subprocess.CompletedProcess(invocation.argv, 0, self.raw(events), b"")
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

    def test_live_adapter_refuses_model_substitution_before_returning_a_candidate(self):
        invocation = self.builder().build("task", thread_id=SESSION)
        events = self.events()
        events[0]["model"] = "different-requested-model"
        for event in events:
            if event.get("type") == "assistant":
                event["message"]["model"] = "different-requested-model"
        events[-1]["modelUsage"]["different-requested-model"] = events[-1]["modelUsage"].pop("exact-requested-model")
        completed = subprocess.CompletedProcess(invocation.argv, 0, self.raw(events), b"")
        with patch("open_cake_ir.lab.claude.run_supervised", return_value=completed):
            with self.assertRaisesRegex(RunProtocolFault, "reported model differs") as captured:
                ClaudeProviderAdapter().execute(invocation, candidate_path=self.candidate,
                    expected_change="update", expected_terminal_message=TERMINAL, arm="open_cake")
        self.assertEqual(captured.exception.artifact_payloads["provider_stdout"], completed.stdout)


    def metadata(self):
        return [
            {"type": "rate_limit_event", "uuid": OTHER_SESSION, "session_id": SESSION,
             "rate_limit_info": {"status": "allowed", "resetsAt": 1788892200, "rateLimitType": "five_hour",
                "overageStatus": "rejected", "overageDisabledReason": "org_level_disabled", "isUsingOverage": False}},
            {"type": "system", "subtype": "thinking_tokens", "uuid": OTHER_SESSION, "session_id": SESSION,
             "estimated_tokens": 600, "estimated_tokens_delta": 100},
        ]

    def test_observed_metadata_is_typed_and_does_not_charge_estimated_thinking(self):
        events = self.events(); events[1:1] = self.metadata()
        self.assertEqual(self.normalize(self.raw(events)).provider_tokens, 205)
        mutations = (
            lambda rows: rows[1]["rate_limit_info"].update(status="rejected"),
            lambda rows: rows[1]["rate_limit_info"].update(isUsingOverage=True),
            lambda rows: rows[1]["rate_limit_info"].update(status=["allowed"]),
            # A status this contract does not name still fails closed.
            lambda rows: rows[1]["rate_limit_info"].update(status="allowed_soon"),
            lambda rows: rows[1]["rate_limit_info"].update(utilization=1.5),
            lambda rows: rows[1]["rate_limit_info"].update(utilization=True),
            lambda rows: rows[1]["rate_limit_info"].update(utilization="0.55"),
            # And so does a field the CLI has not been observed to emit.
            lambda rows: rows[1]["rate_limit_info"].update(unobserved=1),
            lambda rows: rows[2].update(estimated_tokens=True),
            lambda rows: rows[2].update(session_id=OTHER_SESSION),
            lambda rows: rows[2].update(subtype="api_retry"),
            lambda rows: rows[2].update(command="unexpected control"),
        )
        for mutation in mutations:
            changed = copy.deepcopy(events); mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.normalize(self.raw(changed))

    def test_a_quota_warning_is_served_and_carries_its_observed_utilization(self):
        """The live CLI reports `allowed_warning` with a `utilization` fraction.

        Refusing it would strand a campaign on an account that is merely partway through
        its window, because the turn that carries the warning is served normally.
        """
        for served in ("allowed", "allowed_warning"):
            events = self.events(); events[1:1] = self.metadata()
            events[1]["rate_limit_info"].update(status=served, utilization=0.55,
                                                rateLimitType="seven_day")
            with self.subTest(status=served):
                self.assertEqual(self.normalize(self.raw(events)).provider_tokens, 205)
        # The boundaries of the observed fraction are admitted; nothing outside them is.
        for utilization in (0, 0.0, 1, 1.0):
            events = self.events(); events[1:1] = self.metadata()
            events[1]["rate_limit_info"].update(status="allowed_warning", utilization=utilization)
            with self.subTest(utilization=utilization):
                self.assertEqual(self.normalize(self.raw(events)).provider_tokens, 205)

    def test_all_reported_model_usage_is_charged_once_and_auxiliary_models_stay_separate(self):
        events = self.events(); events[1:1] = self.metadata()
        usage = {"input_tokens": 4, "output_tokens": 3297, "cache_creation_input_tokens": 12266,
            "cache_read_input_tokens": 15736, "output_tokens_details": {"thinking_tokens": 624},
            "cache_creation": {"ephemeral_1h_input_tokens": 12266, "ephemeral_5m_input_tokens": 0},
            "iterations": [{"input_tokens": 2, "output_tokens": 112, "cache_creation_input_tokens": 3306,
                            "cache_read_input_tokens": 12348}]}
        events[-1]["usage"] = usage
        events[-1]["modelUsage"] = {
            "exact-requested-model": {"inputTokens": 4, "outputTokens": 3297,
                "cacheCreationInputTokens": 12266, "cacheReadInputTokens": 15736},
            "claude-haiku-4-5-20251001": {"inputTokens": 5528, "outputTokens": 13,
                "cacheCreationInputTokens": 0, "cacheReadInputTokens": 0, "costUSD": 0.005593,
                "webSearchRequests": 0, "contextWindow": 200000, "maxOutputTokens": 32000,
                "canonicalModel": "claude-haiku-4-5", "provider": "firstParty"}}
        parsed = parse_claude_turn_events(self.raw(events), expected_terminal_message=TERMINAL)
        turn = self.normalize(self.raw(events))
        self.assertEqual(parsed.reported_models, ("exact-requested-model",))
        self.assertEqual(turn.provider_tokens, 36844)
        counts = {activity.model: activity.provider_tokens for activity in turn.tool_activity if activity.model is not None}
        self.assertEqual(counts, {"exact-requested-model": 31303, "claude-haiku-4-5-20251001": 5541})
        self.assertEqual(sum(counts.values()), turn.provider_tokens)
        self.assertEqual(set(turn.tool_activity[0].document), {"item_id", "item_type", "status", "server", "tool"})
        self.assertEqual([dict(activity.document) for activity in parsed.tool_activity], [dict(activity.document) for activity in turn.tool_activity])
        for mutate in (
            lambda rows: rows[-1]["modelUsage"]["exact-requested-model"].update(inputTokens=5),
            lambda rows: rows[-1]["modelUsage"].pop("exact-requested-model"),
            lambda rows: rows[-1]["modelUsage"]["claude-haiku-4-5-20251001"].update(outputTokens=True),
            lambda rows: rows[3]["message"].update(model="claude-haiku-4-5-20251001"),
        ):
            changed = copy.deepcopy(events); mutate(changed)
            with self.assertRaises(ValueError):
                self.normalize(self.raw(changed))

    def test_native_structured_terminal_is_full_typed_object_not_prose_or_fence_extraction(self):
        events = self.events()
        events[-1]["result"] = "Display prose is not the terminal authority."
        self.assertEqual(self.normalize(self.raw(events)).normalization, "claude_native_structured_output_exact")
        for result in (TERMINAL, "```json\n" + TERMINAL + "\n```", "Wrote candidate. " + TERMINAL):
            changed = copy.deepcopy(events); changed[-1].pop("structured_output"); changed[-1]["result"] = result
            with self.assertRaisesRegex(ValueError, "structured terminal"):
                self.normalize(self.raw(changed))
        for value in ({"candidate_written": True}, {**json.loads(TERMINAL), "turn": 2},
                      {**json.loads(TERMINAL), "arm": "another"}, {**json.loads(TERMINAL), "candidate_written": 1}):
            changed = copy.deepcopy(events); changed[-1]["structured_output"] = value
            with self.assertRaises(ValueError): self.normalize(self.raw(changed))
        with self.assertRaises(ValueError): self.normalize(event_contract="claude_stream_candidate_v1")

    def test_a_recovered_tool_error_is_turn_local_and_the_terminal_tool_is_not(self):
        """F-2026-09-10-003: an authoring probe the provider survived is not fatal.

        The turn's own lifecycle still decides: every invocation must complete and the
        candidate must be written. What changed is that a tool reporting an error, which
        the author reads and works around inside the same turn, no longer kills the
        campaign after the fact. Refusing it spent two campaigns and roughly 7M provider
        tokens. The fact is retained rather than dropped, so a run stays auditable.
        """
        events = self.events(); events[2]["message"]["content"][0]["is_error"] = True
        parsed = parse_claude_turn_events(self.raw(events), expected_terminal_message=TERMINAL)
        recovered = [a for a in parsed.tool_activity if a.status == "error_recovered"]
        self.assertEqual([(a.item_id, a.item_type, a.tool) for a in recovered],
                         [("toolu_write", "tool_use", "Write")])
        # Exactly one record per invocation; the error restates it, never duplicates it.
        self.assertEqual(len([a for a in parsed.tool_activity if a.item_id == "toolu_write"]), 1)
        # And the turn still normalizes end to end.
        self.normalize(self.raw(events))
        # Pairing stays zero-tolerance, and so does a non-boolean flag.
        for mutation in (lambda rows: rows[2]["message"]["content"][0].update(tool_use_id="ghost"),
                         lambda rows: rows[2]["message"]["content"][0].update(is_error="yes"),
                         lambda rows: rows[2]["message"]["content"][0].update(is_error=1)):
            changed = copy.deepcopy(events); mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "tool completion"):
                parse_claude_turn_events(self.raw(changed), expected_terminal_message=TERMINAL)

    def test_a_relative_write_path_is_judged_where_the_cli_resolves_it(self):
        """F-2026-09-10-007: the envelope check is containment, not spelling.

        A provider that writes the cwd-relative name of the very file it is supposed to
        write was campaign-fatal, even though the CLI resolves it to exactly the sealed
        target. The path is now resolved against the turn's reported cwd before the same
        containment and identity rules are applied.
        """
        events = self.events()
        absolute = events[1]["message"]["content"][0]["input"]["file_path"]
        events[0]["cwd"] = str(Path(absolute).parent)
        events[1]["message"]["content"][0]["input"]["file_path"] = "candidate-set.json"
        parsed = parse_claude_turn_events(self.raw(events), expected_terminal_message=TERMINAL)
        self.assertEqual(parsed.candidate_path, absolute)
        # Containment still refuses anything that resolves elsewhere or renames the file.
        for spelling in ("../candidate-set.json", "sub/../candidate-set.json",
                         "notes.json", "sub/notes.json"):
            changed = copy.deepcopy(events)
            changed[1]["message"]["content"][0]["input"]["file_path"] = spelling
            with self.subTest(spelling=spelling), self.assertRaisesRegex(ValueError, "candidate envelope"):
                parse_claude_turn_events(self.raw(changed), expected_terminal_message=TERMINAL)
        # Without a reported cwd a relative spelling has nothing to resolve against.
        changed = copy.deepcopy(events); changed[0].pop("cwd")
        with self.assertRaisesRegex(ValueError, "candidate envelope"):
            parse_claude_turn_events(self.raw(changed), expected_terminal_message=TERMINAL)
        # A reported cwd that is not an absolute path is itself a contract breach.
        changed = copy.deepcopy(events); changed[0]["cwd"] = "relative/dir"
        with self.assertRaisesRegex(ValueError, "working directory"):
            parse_claude_turn_events(self.raw(changed), expected_terminal_message=TERMINAL)

    def test_a_compacted_turn_is_refused_by_name_rather_than_absorbed(self):
        """F-2026-09-10-008: compaction rewrites the context the author was working in.

        The launch pins the window at the largest value this CLI admits -- it has no off
        switch -- though for a model the CLI does not recognize that window is clamped
        to the CLI's assumed model context, so it is not unreachable (F-2026-09-10-013).
        When it fires the Turn ran on a context the Lab cannot reconstruct, and it is
        reported as that rather than as an unclassified event or, worse, silently accepted.
        """
        for subtype in ("compact_boundary", "compacting"):
            events = self.events()
            events[1:1] = [{"type": "system", "subtype": subtype, "session_id": SESSION,
                            "uuid": OTHER_SESSION}]
            with self.subTest(subtype=subtype), self.assertRaisesRegex(ValueError, "compacted"):
                parse_claude_turn_events(self.raw(events), expected_terminal_message=TERMINAL)
        # F-2026-09-10-013: this CLI also announces compaction as a bare status --
        # `status: "compacting"` when it starts, `status: null` with `compact_result`
        # when it lands (gemm and pairwise_sqdist turn 2 on Executor v98) -- and the
        # subtype rule alone left those to the unclassified-event refusal. Both are
        # refused by the same name, and an unrelated status still fails closed.
        for status_fields in ({"status": "compacting"},
                              {"status": None, "compact_result": "success"}):
            events = self.events()
            events[1:1] = [{"type": "system", "subtype": "status", **status_fields,
                            "session_id": SESSION, "uuid": OTHER_SESSION}]
            with self.subTest(status_fields=status_fields), self.assertRaisesRegex(ValueError, "compacted"):
                parse_claude_turn_events(self.raw(events), expected_terminal_message=TERMINAL)
        events = self.events()
        events[1:1] = [{"type": "system", "subtype": "status", "status": "idle",
                        "session_id": SESSION, "uuid": OTHER_SESSION}]
        with self.assertRaisesRegex(ValueError, "outside the declared native contract"):
            parse_claude_turn_events(self.raw(events), expected_terminal_message=TERMINAL)

    def test_a_tool_progress_heartbeat_is_admitted_and_any_other_shape_fails_closed(self):
        """F-2026-09-11-015: the CLI emits a heartbeat while one tool call runs long.

        Observed as two `tool_progress` events (30 s, then 60 s) around a Read of a
        multi-megabyte single-line JSON evidence object, on gemm turn 2 under Executor
        v99. A heartbeat rewrites nothing and carries no author-visible content, so it
        is admitted under exactly the observed shape instead of refused: the refusal
        killed the turn as an unclassified event before the compaction notices that
        followed, hiding the named diagnosis three lines later. `parent_tool_use_id`
        here names the owning tool call, not a subagent, and the heartbeat records no
        activity of its own -- the stream retains it.
        """
        heartbeat = {"type": "tool_progress", "tool_use_id": "call_x-heartbeat-0",
                     "tool_name": "Read", "parent_tool_use_id": "call_x",
                     "elapsed_time_seconds": 30, "heartbeat": True,
                     "session_id": SESSION, "uuid": OTHER_SESSION}
        events = self.events(); events.insert(2, heartbeat)
        raw = self.raw(events)
        parsed = parse_claude_turn_events(raw, expected_terminal_message=TERMINAL)
        self.assertEqual(parsed.provider_tokens, 205)
        self.assertNotIn("tool_progress", {activity.item_type for activity in parsed.tool_activity})
        self.normalize(raw)
        for mutation in (lambda row: row.update(heartbeat=False),
                         lambda row: row.update(heartbeat=None),
                         lambda row: row.update(elapsed_time_seconds=True),
                         lambda row: row.update(elapsed_time_seconds=-1),
                         lambda row: row.update(elapsed_time_seconds=30.0),
                         lambda row: row.update(tool_name=""),
                         lambda row: row.update(tool_use_id=""),
                         lambda row: row.pop("parent_tool_use_id"),
                         lambda row: row.update(unobserved=1)):
            changed = copy.deepcopy(events); mutation(changed[2])
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "heartbeat"):
                self.normalize(self.raw(changed))

    def test_a_cli_synthetic_continuation_is_recorded_and_an_unmarked_one_is_not(self):
        """The CLI writes its own user turn when a response had no visible output.

        Providers that answer with thinking alone trigger it routinely, so refusing it
        would strand those campaigns; but it is content the Lab did not author and it
        changes what the author saw, so it is retained as observed activity rather than
        passed over. Only the CLI's own marked turn is admitted -- an unmarked user text
        block is someone injecting into the conversation and stays fatal.
        """
        nudge = {"type": "user", "session_id": SESSION, "uuid": OTHER_SESSION,
                 "isSynthetic": True, "parent_tool_use_id": None,
                 "message": {"content": [{"type": "text", "text":
                     "[Your previous response had no visible output. Please continue.]"}]}}
        events = self.events(); events[1:1] = [nudge]
        parsed = parse_claude_turn_events(self.raw(events), expected_terminal_message=TERMINAL)
        self.assertEqual([(a.item_type, a.status) for a in parsed.tool_activity
                          if a.item_type == "synthetic_continuation"],
                         [("synthetic_continuation", "observed")])
        for mutation in (lambda row: row.pop("isSynthetic"),
                         lambda row: row.update(isSynthetic=False),
                         lambda row: row.update(isSynthetic="yes")):
            changed = copy.deepcopy(events); mutation(changed[1])
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "event contract"):
                parse_claude_turn_events(self.raw(changed), expected_terminal_message=TERMINAL)

    def test_native_schema_tool_has_only_exact_terminal_arguments_and_closed_success(self):
        events = self.events()
        events[-1:-1] = [
            {"type": "assistant", "session_id": SESSION, "message": {"model": "exact-requested-model", "content": [
                {"type": "tool_use", "id": "terminal-tool", "name": "StructuredOutput", "input": json.loads(TERMINAL)}]}},
            {"type": "user", "session_id": SESSION, "message": {"content": [
                {"type": "tool_result", "tool_use_id": "terminal-tool", "is_error": False, "content": "Structured output provided successfully"}]}},
        ]
        self.normalize(self.raw(events))
        wrong = copy.deepcopy(events); wrong[-3]["message"]["content"][0]["input"]["turn"] = 9
        with self.assertRaisesRegex(ValueError, "schema terminal tool"): self.normalize(self.raw(wrong))
        wrong = copy.deepcopy(events); wrong[-2]["message"]["content"][0]["is_error"] = True
        with self.assertRaisesRegex(ValueError, "tool completion"): self.normalize(self.raw(wrong))
        wrong = copy.deepcopy(events); wrong.pop(-2)
        with self.assertRaisesRegex(ValueError, "lifecycle"): self.normalize(self.raw(wrong))

    def test_schema_is_bound_and_identical_on_initial_and_resume(self):
        builder = self.builder()
        first = builder.build("first", thread_id=None); resumed = builder.build("next", thread_id=SESSION)
        self.assertEqual(builder.configuration["terminal_schema"], terminal_schema())
        for invocation in (first, resumed):
            self.assertEqual(json.loads(invocation.argv[invocation.argv.index("--json-schema") + 1]), terminal_schema())
        self.assertEqual(first.argv[:-2], resumed.argv[:-4])
        for argv in ((str(self.executable), "--json-schema"), (str(self.executable), "--json-schema", "not-json")):
            with patch("open_cake_ir.lab.claude.run_supervised") as process:
                with self.assertRaisesRegex(ValueError, "native terminal schema"):
                    ClaudeProviderAdapter().execute(replace(first, argv=argv), candidate_path=self.candidate,
                        expected_change="add", expected_terminal_message=TERMINAL, arm="open_cake")
                process.assert_not_called()

    def test_optional_usage_activity_fields_preserve_old_tool_projection(self):
        old = ProviderAuxiliaryActivity("tool", "tool_use", "completed", tool="Read")
        self.assertNotIn("model", old.document); self.assertNotIn("provider_tokens", old.document)
        with self.assertRaises(ValueError): ProviderAuxiliaryActivity("usage", "model_usage", "reported", model="helper")


    def retry_event(self):
        return {"type": "system", "subtype": "api_retry", "attempt": 1, "max_retries": 10,
            "retry_delay_ms": 509, "error_status": None, "error": "unknown", "uuid": OTHER_SESSION, "session_id": SESSION}

    def test_native_transport_retry_is_observed_inside_one_invocation_without_extra_tokens(self):
        events = self.events(); events.insert(3, self.retry_event())
        raw = self.raw(events)
        invocation = self.builder().build("one resumed turn", thread_id=SESSION)
        completed = subprocess.CompletedProcess(invocation.argv, 0, raw, b"")
        with patch("open_cake_ir.lab.claude.run_supervised", return_value=completed) as process:
            turn = ClaudeProviderAdapter().execute(invocation, candidate_path=self.candidate,
                expected_change="update", expected_terminal_message=TERMINAL, arm="open_cake")
        process.assert_called_once()
        self.assertEqual(turn.provider_tokens, 205)
        self.assertEqual(turn.thread_id, SESSION)
        self.assertEqual(turn.terminal_message_count, 1)
        self.assertEqual(turn.raw_events, raw)
        retry, = [entry for entry in turn.tool_activity if entry.item_type == "api_retry"]
        self.assertEqual(retry.item_id, OTHER_SESSION)
        self.assertEqual(retry.status, "observed")
        self.assertIsNone(retry.provider_tokens)
        self.assertEqual(CLAUDE_EVENT_CONTRACT, "claude_stream_candidate_v3")
        with self.assertRaises(ValueError): self.normalize(raw, event_contract="claude_stream_candidate_v2")

    def summary_event(self):
        return {"type": "system", "subtype": "post_turn_summary", "summarizes_uuid": OTHER_SESSION,
            "status_category": "review_ready", "status_detail": "turn 1: created candidate-set.json",
            "needs_action": "", "uuid": OTHER_SESSION, "session_id": SESSION}

    def test_native_post_turn_summary_is_retained_without_deciding_the_turn(self):
        events = self.events(); events.insert(-1, self.summary_event())
        raw = self.raw(events)
        invocation = self.builder().build("one resumed turn", thread_id=SESSION)
        completed = subprocess.CompletedProcess(invocation.argv, 0, raw, b"")
        with patch("open_cake_ir.lab.claude.run_supervised", return_value=completed):
            turn = ClaudeProviderAdapter().execute(invocation, candidate_path=self.candidate,
                expected_change="update", expected_terminal_message=TERMINAL, arm="open_cake")
        summary, = [entry for entry in turn.tool_activity if entry.item_type == "post_turn_summary"]
        self.assertEqual(summary.item_id, OTHER_SESSION)
        # The category is retained as observed, not read as an outcome.
        self.assertEqual(summary.status, "review_ready")
        self.assertIsNone(summary.provider_tokens)
        self.assertEqual(turn.provider_tokens, 205)
        # An unfamiliar category and a populated action note stay admissible.
        for change in ({"status_category": "later_unmodeled_category"}, {"needs_action": "look at the diff"}):
            events = self.events(); events.insert(-1, {**self.summary_event(), **change})
            with self.subTest(change=change):
                self.normalize(self.raw(events))

    def test_native_post_turn_summary_schema_fails_closed(self):
        changes = ({"status_category": ""}, {"status_category": 3}, {"status_detail": None},
            {"needs_action": None}, {"summarizes_uuid": "not-a-uuid"}, {"summarizes_uuid": 7},
            {"uuid": ""}, {"session_id": OTHER_SESSION}, {"extra": 1})
        for change in changes:
            events = self.events(); events.insert(-1, {**self.summary_event(), **change})
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.normalize(self.raw(events))

    def test_native_retry_counter_and_control_schema_fail_closed(self):
        changes = ({"attempt": 0}, {"attempt": True}, {"attempt": 11}, {"max_retries": 0},
            {"max_retries": "10"}, {"retry_delay_ms": -1}, {"retry_delay_ms": 509.0},
            {"error_status": False}, {"error_status": 600}, {"error": "new-unmodeled-error"},
            {"no_response": {}}, {"subtype": "other_retry"}, {"session_id": OTHER_SESSION}, {"uuid": ""})
        for change in changes:
            events = self.events(); events.insert(3, {**self.retry_event(), **change})
            with self.subTest(change=change), self.assertRaises(ValueError): self.normalize(self.raw(events))
        events = self.events(); retry = self.retry_event(); retry.pop("attempt"); events.insert(3, retry)
        with self.assertRaises(ValueError): self.normalize(self.raw(events))

    def test_fault_usage_is_reported_independently_of_candidate_acceptance(self):
        events = self.events(); events[-1].pop("structured_output")
        events.insert(2, {"type": "system", "subtype": "unknown_control", "session_id": SESSION})
        raw = self.raw(events)
        with self.assertRaises(ValueError): self.normalize(raw)
        usage = reported_claude_usage(raw, expected_model="exact-requested-model", expected_thread_id=SESSION)
        self.assertIsNotNone(usage)
        self.assertEqual(usage.provider_tokens, 205)
        self.assertEqual(usage.event_contract, CLAUDE_EVENT_CONTRACT)
        self.assertEqual(usage.thread_id, SESSION)
        self.assertIsNone(reported_claude_usage(raw, expected_model="another-model"))
        self.assertIsNone(reported_claude_usage(raw, expected_model="exact-requested-model", expected_thread_id=OTHER_SESSION))
        self.assertIsNone(reported_claude_usage(raw[:-8], expected_model="exact-requested-model"))
        self.assertIsNone(reported_claude_usage(None, expected_model="exact-requested-model"))
        self.assertIsNone(reported_claude_usage(b"[" * 2000 + b"]" * 2000, expected_model="exact-requested-model"))
        duplicate = self.events(); duplicate.insert(-1, copy.deepcopy(duplicate[-1]))
        self.assertIsNone(reported_claude_usage(self.raw(duplicate), expected_model="exact-requested-model"))

    def test_observed_zero_fault_usage_is_not_missing_or_successful_turn_usage(self):
        events = self.events()
        for key in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
            events[-1]["usage"][key] = 0
        for key in events[-1]["modelUsage"]["exact-requested-model"]:
            events[-1]["modelUsage"]["exact-requested-model"][key] = 0
        events[-1]["usage"]["cache_creation"] = {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0}
        raw = self.raw(events)
        zero = reported_claude_usage(raw, expected_model="exact-requested-model")
        self.assertIsNotNone(zero); self.assertEqual(zero.provider_tokens, 0)
        with self.assertRaises(ValueError): self.normalize(raw)
        events[-1].pop("modelUsage")
        self.assertIsNone(reported_claude_usage(self.raw(events), expected_model="exact-requested-model"))

    def test_adapter_attaches_validated_usage_to_stdout_bearing_faults_without_retry(self):
        invocation = self.builder().build("task", thread_id=SESSION)
        events = self.events(); events.insert(3, {"type":"system", "subtype":"unknown_control", "session_id":SESSION})
        raw = self.raw(events)
        completed = subprocess.CompletedProcess(invocation.argv, 0, raw, b"retained stderr")
        with patch("open_cake_ir.lab.claude.run_supervised", return_value=completed) as process:
            with self.assertRaises(RunProtocolFault) as captured:
                ClaudeProviderAdapter().execute(invocation, candidate_path=self.candidate,
                    expected_change="update", expected_terminal_message=TERMINAL, arm="open_cake")
        process.assert_called_once()
        self.assertEqual(captured.exception.reported_usage.provider_tokens, 205)
        self.assertEqual(captured.exception.artifact_payloads["provider_stdout"], raw)
        timeout = SupervisedProcessTimeout(raw, b"timeout")
        with patch("open_cake_ir.lab.claude.run_supervised", side_effect=timeout):
            with self.assertRaises(RunProtocolFault) as captured:
                ClaudeProviderAdapter().execute(invocation, candidate_path=self.candidate,
                    expected_change="update", expected_terminal_message=TERMINAL, arm="open_cake")
        self.assertEqual(captured.exception.reported_usage.provider_tokens, 205)


if __name__ == "__main__":
    unittest.main()
