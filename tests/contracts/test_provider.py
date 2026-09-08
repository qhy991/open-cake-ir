from __future__ import annotations

import json
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.lab.providers import (  # noqa: E402
    CANDIDATE_SET_ENVELOPE_V1,
    CODEX_DISABLED_FEATURES,
    CodexInvocationBuilder,
    CodexRunProvider,
    ProviderQualificationReceipt,
    ProviderTurn,
    normalize_codex_turn,
    parse_codex_turn_events,
    required_live_provider_qualification_scope,
)
from open_cake_ir.lab.faults import RunProtocolFault  # noqa: E402
from open_cake_ir.lab.task_package import (  # noqa: E402
    TaskPackage,
    materialize_task_package,
    render_task_request,
)


class ProviderContractTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.executable = Path(directory.name) / "codex"
        self.executable.write_bytes(b"CPU CLI fixture")
        self.executable.chmod(0o700)
        helper = self.executable.with_name("codex-code-mode-host")
        helper.write_bytes(b"CPU Code Mode host fixture")
        helper.chmod(0o700)

    def test_claim_scope_owns_the_live_provider_capability(self) -> None:
        self.assertEqual(
            required_live_provider_qualification_scope("artifact_optimization_only"),
            "live_two_turn_tool_rich_provider",
        )
        for scope in ("scientific_matched_search", "system_qualification_only"):
            self.assertEqual(
                required_live_provider_qualification_scope(scope),
                "live_two_turn_current_provider",
            )
        with self.assertRaisesRegex(ValueError, "no live provider qualification"):
            required_live_provider_qualification_scope("bounded_local_b200_reconstruction")

    def _write_schedule_set(self, path: Path, value: int) -> None:
        path.write_bytes(json.dumps({
            "arm": "open_cake", "candidates": [{"schedule": value}], "schema_version": 1,
        }, sort_keys=True, separators=(",", ":")).encode() + b"\n")

    def _events(
        self,
        path: Path,
        *,
        duplicate: bool,
        first_text: str | None = None,
        second_text: str | None = None,
    ) -> bytes:
        message = '{"candidate_written":true}'
        file_started = {
            "type": "item.started",
            "item": {
                "id": "item_0",
                "type": "file_change",
                "changes": [{"path": str(path.absolute()), "kind": "add"}],
                "status": "in_progress",
            },
        }
        file_completed = json.loads(json.dumps(file_started))
        file_completed["type"] = "item.completed"
        file_completed["item"]["status"] = "completed"
        terminal = {
            "type": "item.completed",
            "item": {"id": "item_1", "type": "agent_message", "text": message},
        }
        events = [
            {"type": "thread.started", "thread_id": "01234567-89ab-cdef-0123-456789abcdef"},
            {"type": "turn.started"},
        ]
        if duplicate:
            first = json.loads(json.dumps(terminal))
            if first_text is not None:
                first["item"]["text"] = first_text
            events.append(first)
        events.extend([file_started, file_completed, terminal])
        if second_text is not None:
            events[-1] = json.loads(json.dumps(terminal))
            events[-1]["item"]["text"] = second_text
        events.append(
            {"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 20}}
        )
        return b"".join(
            json.dumps(event, separators=(",", ":")).encode() + b"\n" for event in events
        )

    def test_initial_and_resume_share_the_complete_authoring_environment(self) -> None:
        builder = CodexInvocationBuilder(
            executable=self.executable,
            provider_revision="codex-fixture-v1",
            model="gpt-5.6-sol",
            reasoning_effort="xhigh",
            service_tier="default",
            workspace=ROOT,
            output_schema=ROOT / "contracts/providers/codex-turn-output-schema-v1.json",
            removed_environment=("OPENAI_API_KEY", "CUDA_VISIBLE_DEVICES"),
        )

        initial = builder.build("initial prompt", thread_id=None)
        resumed = builder.build("resume prompt", thread_id="01234567-89ab-cdef-0123-456789abcdef")

        self.assertEqual(initial.cwd, resumed.cwd)
        self.assertEqual(initial.sandbox, resumed.sandbox, "workspace-write")
        self.assertEqual(initial.provider_revision, resumed.provider_revision)
        self.assertEqual(initial.removed_environment, resumed.removed_environment)
        for invocation in (initial, resumed):
            self.assertEqual(invocation.argv.count('sandbox_mode="workspace-write"'), 1)
            self.assertEqual(invocation.argv.count('approval_policy="never"'), 1)
            self.assertNotIn("--sandbox", invocation.argv)
            disabled = builder.configuration["disabled_features"]
            observed_disabled = [
                invocation.argv[index + 1]
                for index, value in enumerate(invocation.argv[:-1])
                if value == "--disable"
            ]
            self.assertEqual(observed_disabled, disabled)
            self.assertIn("apps", disabled)
            self.assertEqual(disabled.count('shell_tool'), 1)
            self.assertIn('web_search="disabled"', invocation.argv)
            self.assertEqual(invocation.argv.count('code_mode_host'), 1)
            for feature in ('code_mode', 'code_mode_only', 'code_mode_host'):
                self.assertNotIn(feature, disabled)
        self.assertNotIn("resume", initial.argv)
        self.assertIn("resume", resumed.argv)
        self.assertIn('model_reasoning_effort="xhigh"', initial.argv)
        self.assertEqual(builder.configuration["reasoning_effort"], "xhigh")

    def test_current_closed_builder_rejects_missing_tool_exclusions(self):
        for feature in ('shell_tool', 'browser_use'):
            with self.subTest(feature=feature), self.assertRaisesRegex(ValueError, 'feature and event'):
                CodexInvocationBuilder(executable=self.executable, provider_revision='fixture',
                    model='gpt-5.6-sol', reasoning_effort='max', service_tier='default', workspace=ROOT,
                    output_schema=ROOT / 'contracts/providers/codex-turn-output-schema-v1.json',
                    removed_environment=('OPENAI_API_KEY', 'ANTHROPIC_API_KEY'),
                    disabled_features=tuple(v for v in CODEX_DISABLED_FEATURES if v != feature))

    def test_code_mode_startup_error_is_rejected_before_and_after_turn_started(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / 'candidate.json'
            candidate.write_text('{"schedule":2}')
            events = [json.loads(line) for line in self._events(candidate, duplicate=False).splitlines()]
            error = {'type':'item.completed', 'item':{'id':'startup-error', 'type':'error',
                'message':'Unable to start Code Mode without its host'}}
            for position in (1, 2):
                with self.subTest(position=position), self.assertRaises(ValueError):
                    changed = events[:position] + [error] + events[position:]
                    normalize_codex_turn(b''.join(json.dumps(v).encode() + b'\n' for v in changed),
                        candidate_path=candidate, expected_change='add',
                        expected_terminal_message='{"candidate_written":true}')

    def test_provider_default_features_emit_no_forced_disable_flags(self) -> None:
        builder = CodexInvocationBuilder(
            executable=self.executable,
            provider_revision="codex-fixture-full-v1",
            model="gpt-5.6-sol",
            reasoning_effort="max",
            service_tier="default",
            workspace=ROOT,
            output_schema=ROOT / "contracts/providers/codex-turn-output-schema-v1.json",
            removed_environment=("OPENAI_API_KEY", "CUDA_VISIBLE_DEVICES"),
            disabled_features=(),
            event_contract="tool_rich_candidate_v1",
        )

        invocation = builder.build("optimization prompt", thread_id=None)

        self.assertEqual(builder.configuration["disabled_features"], [])
        self.assertEqual(
            builder.configuration["event_contract"],
            "tool_rich_candidate_v1",
        )
        self.assertNotIn("--disable", invocation.argv)
        self.assertTrue(set(CODEX_DISABLED_FEATURES))


    def test_single_and_bracketed_duplicate_terminal_forms_normalize_equally(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.json"
            self._write_schedule_set(candidate, 1)
            single = normalize_codex_turn(
                self._events(candidate, duplicate=False),
                candidate_path=candidate,
                expected_change="add",
                expected_terminal_message='{"candidate_written":true}',
                         arm="open_cake",
            )
            duplicate = normalize_codex_turn(
                self._events(candidate, duplicate=True),
                candidate_path=candidate,
                expected_change="add",
                expected_terminal_message='{"candidate_written":true}',
                            arm="open_cake",
            )

        self.assertEqual(single.provider_tokens, duplicate.provider_tokens, 120)
        self.assertEqual(single.candidate_sha256s, duplicate.candidate_sha256s)
        self.assertEqual(single.terminal_message_count, 1)
        self.assertEqual(duplicate.terminal_message_count, 2)
        self.assertEqual(duplicate.normalization, "duplicate_exact_bracketed")

    def test_codex_0149_cache_write_usage_is_an_input_token_detail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.json"
            events = [
                json.loads(line)
                for line in self._events(candidate, duplicate=False).splitlines()
            ]
            events[-1]["usage"] = {
                "input_tokens": 131565,
                "cached_input_tokens": 106240,
                "cache_write_input_tokens": 0,
                "output_tokens": 2709,
                "reasoning_output_tokens": 1284,
            }

            parsed = parse_codex_turn_events(
                b"".join(
                    json.dumps(event, separators=(",", ":")).encode() + b"\n"
                    for event in events
                ),
                expected_terminal_message='{"candidate_written":true}',
            )

        self.assertEqual(parsed.provider_tokens, 134274)

    def test_provider_usage_rejects_unknown_and_excessive_cache_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.json"
            events = [
                json.loads(line)
                for line in self._events(candidate, duplicate=False).splitlines()
            ]
            events[-1]["usage"] = {
                "input_tokens": 100,
                "cached_input_tokens": 60,
                "cache_write_input_tokens": 101,
                "output_tokens": 20,
                "reasoning_output_tokens": 10,
            }

            with self.assertRaisesRegex(ValueError, "provider usage differs"):
                parse_codex_turn_events(
                    b"".join(
                        json.dumps(event, separators=(",", ":")).encode() + b"\n"
                        for event in events
                    ),
                    expected_terminal_message='{"candidate_written":true}',
                )

            events[-1]["usage"] = {
                "input_tokens": 100,
                "output_tokens": 20,
                "future_usage_tokens": 1,
            }
            with self.assertRaisesRegex(ValueError, "provider usage fields differ"):
                parse_codex_turn_events(
                    b"".join(
                        json.dumps(event, separators=(",", ":")).encode() + b"\n"
                        for event in events
                    ),
                    expected_terminal_message='{"candidate_written":true}',
                )

    def test_bracketed_terminal_whitespace_normalizes_by_json_meaning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.json"
            self._write_schedule_set(candidate, 1)
            turn = normalize_codex_turn(
                self._events(
                    candidate,
                    duplicate=True,
                    first_text='{ "candidate_written": true }',
                ),
                candidate_path=candidate,
                expected_change="add",
                expected_terminal_message='{"candidate_written":true}',
                       arm="open_cake",
            )

        self.assertEqual(turn.terminal_message_count, 2)
        self.assertEqual(turn.normalization, "duplicate_semantic_bracketed")

    def test_delete_then_add_of_the_same_candidate_is_one_resume_update(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.json"
            self._write_schedule_set(candidate, 2)
            terminal = {
                "type": "item.completed",
                "item": {
                    "id": "message",
                    "type": "agent_message",
                    "text": '{"candidate_written":true}',
                },
            }
            events = [
                {
                    "type": "thread.started",
                    "thread_id": "01234567-89ab-cdef-0123-456789abcdef",
                },
                {"type": "turn.started"},
                terminal,
            ]
            for item_id, kind in (("remove", "delete"), ("replace", "add")):
                started = {
                    "type": "item.started",
                    "item": {
                        "id": item_id,
                        "type": "file_change",
                        "changes": [
                            {"path": str(candidate.absolute()), "kind": kind}
                        ],
                        "status": "in_progress",
                    },
                }
                completed = json.loads(json.dumps(started))
                completed["type"] = "item.completed"
                completed["item"]["status"] = "completed"
                events.extend((started, completed))
            events.extend(
                (
                    terminal,
                    {
                        "type": "turn.completed",
                        "usage": {"input_tokens": 193014, "output_tokens": 44243},
                    },
                )
            )
            raw = b"".join(
                json.dumps(event, separators=(",", ":")).encode() + b"\n"
                for event in events
            )

            turn = normalize_codex_turn(
                raw,
                candidate_path=candidate,
                expected_change="update",
                expected_terminal_message='{"candidate_written":true}',
                       arm="open_cake",
            )
            different_paths = json.loads(json.dumps(events))
            for index in (5, 6):
                different_paths[index]["item"]["changes"][0]["path"] = str(
                    (Path(directory) / "different.json").absolute()
                )
            with self.assertRaisesRegex(ValueError, "replacement lifecycle"):
                parse_codex_turn_events(
                    b"".join(
                        json.dumps(event, separators=(",", ":")).encode() + b"\n"
                        for event in different_paths
                    ),
                    expected_terminal_message='{"candidate_written":true}',
                )

        self.assertEqual(turn.provider_tokens, 237257)
        self.assertEqual(turn.candidates, (b'{"schedule":2}',))
        self.assertEqual(turn.normalization, "duplicate_exact_bracketed")

    def test_resume_bare_add_label_is_the_authoritative_update(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.json"
            self._write_schedule_set(candidate, 2)

            turn = normalize_codex_turn(
                self._events(candidate, duplicate=False),
                candidate_path=candidate,
                expected_change="update",
                expected_terminal_message='{"candidate_written":true}',
                       arm="open_cake",
            )
            initial_events = [
                json.loads(line)
                for line in self._events(candidate, duplicate=False).splitlines()
            ]
            for event in initial_events:
                item = event.get("item", {})
                if item.get("type") == "file_change":
                    item["changes"][0]["kind"] = "update"
            with self.assertRaisesRegex(ValueError, "change kind"):
                normalize_codex_turn(
                    b"".join(
                        json.dumps(event, separators=(",", ":")).encode() + b"\n"
                        for event in initial_events
                    ),
                    candidate_path=candidate,
                    expected_change="add",
                    expected_terminal_message='{"candidate_written":true}',
                    arm="open_cake",
                )

        self.assertEqual(turn.candidates, (b'{"schedule":2}',))
        self.assertEqual(turn.provider_tokens, 120)

    def test_canonical_candidate_set_projects_ordered_members_for_each_arm(self) -> None:
        cases = (
            (
                "open_cake",
                [{"variant": 2}, {"variant": 0}, {"variant": 1}],
                (b'{"variant":2}', b'{"variant":0}', b'{"variant":1}'),
            ),
            (
                "direct_cuda",
                ["// variant 2\n", "// variant 0\n", "// variant 1\n"],
                (b"// variant 2\n", b"// variant 0\n", b"// variant 1\n"),
            ),
        )
        for arm, members, expected in cases:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                candidate = Path(directory) / "candidate-set.json"
                envelope = {
                    "schema_version": 1,
                    "arm": arm,
                    "candidates": members,
                }
                candidate.write_bytes(
                    json.dumps(
                        envelope,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                    + b"\n"
                )
                turn = normalize_codex_turn(
                    self._events(candidate, duplicate=False),
                    candidate_path=candidate,
                    expected_change="add",
                    expected_terminal_message='{"candidate_written":true}',
                    submission_contract=CANDIDATE_SET_ENVELOPE_V1,
                    arm=arm,
                    maximum_candidates_per_turn=3,
                )

            self.assertEqual(turn.candidates, expected)
            self.assertEqual(
                turn.candidate_sha256s,
                tuple(sha256(member).hexdigest() for member in expected),
            )

    def test_candidate_set_rejects_wrong_arm_and_over_bound_envelopes(self) -> None:
        valid = {
            "schema_version": 1,
            "arm": "open_cake",
            "candidates": [{"variant": 0}, {"variant": 1}],
        }
        cases = (
            (
                json.dumps(valid, sort_keys=True, separators=(",", ":")).encode()
                + b"\n",
                "fields",
                "direct_cuda",
                2,
            ),
            (
                json.dumps(valid, sort_keys=True, separators=(",", ":")).encode()
                + b"\n",
                "count",
                "open_cake",
                1,
            ),
        )
        for payload, message, arm, maximum in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                candidate = Path(directory) / "candidate-set.json"
                candidate.write_bytes(payload)
                with self.assertRaisesRegex(ValueError, message):
                    normalize_codex_turn(
                        self._events(candidate, duplicate=False),
                        candidate_path=candidate,
                        expected_change="add",
                        expected_terminal_message='{"candidate_written":true}',
                        submission_contract=CANDIDATE_SET_ENVELOPE_V1,
                        arm=arm,
                        maximum_candidates_per_turn=maximum,
                    )

    def test_json_presentation_preserves_members_order_and_captured_raw_bytes(self) -> None:
        for arm in ("open_cake", "native_triton", "direct_cuda"):
            objects = [
                {"z": [{"text": "海岩 / quoted \"text\"", "value": 1.0}], "a": -0.0},
                {"z": [{"text": "second", "value": 2.0}], "a": 0.5},
            ]
            sources = ["// 海岩 / quoted \"text\"\nfloat x = 1.0;\n", "// second\r\n"]
            members = sources if arm == "direct_cuda" else objects
            expected = tuple(
                member.encode("utf-8") if arm == "direct_cuda"
                else json.dumps(member, sort_keys=True, separators=(",", ":"),
                                ensure_ascii=False).encode("utf-8")
                for member in members
            )
            document = {"schema_version": 1, "candidates": members, "arm": arm}
            presentations = [
                json.dumps(document, indent=2, ensure_ascii=False),
                json.dumps(document, sort_keys=True, separators=(",", ":")),
                json.dumps(document, indent=4).replace("/", "\\/"),
            ]
            if arm != "direct_cuda":
                presentations.append(presentations[0].replace("1.0", "1e0")
                                     .replace("-0.0", "-0.000").replace("0.5", "5e-1"))
            for presentation in presentations:
                with self.subTest(arm=arm, presentation=presentation), tempfile.TemporaryDirectory() as directory:
                    candidate = Path(directory) / "candidate-set.json"
                    raw = (" \n" + presentation + "\t\n").encode("utf-8")
                    candidate.write_bytes(raw)
                    turn = normalize_codex_turn(
                        self._events(candidate, duplicate=False), candidate_path=candidate,
                        expected_change="add", expected_terminal_message='{"candidate_written":true}',
                        arm=arm, maximum_candidates_per_turn=2,
                    )
                    candidate.write_bytes(b"overwritten by a later turn")
                    self.assertEqual(turn.raw_submission, raw)
                    self.assertEqual(turn.candidates, expected)
                    self.assertEqual(turn.candidate_sha256s,
                                     tuple(sha256(member).hexdigest() for member in expected))

    def test_candidate_set_refuses_ambiguous_or_invalid_json(self) -> None:
        prefix = b'{"schema_version":1,"arm":"open_cake","candidates":'
        invalid = [
            b'{"schema_version":1,"schema_version":1,"arm":"open_cake","candidates":[{}]}',
            prefix + b'[{"nested":[{"x":1,"x":2}]}]}',
            prefix + b'[{"x":1,"\\u0078":2}]}',
            *(prefix + b'[{"nested":[' + number + b']}]}'
              for number in (b"NaN", b"Infinity", b"-Infinity", b"1e400", b"-1e400")),
            prefix.replace(b'"schema_version":1', b'"schema_version":true') + b'[{}]}',
            prefix.replace(b'"schema_version":1', b'"schema_version":1.0') + b'[{}]}',
            prefix + b'[{"text":"\xff"}]}',
            (prefix + b'[{}]}').decode().encode("utf-16"),
            prefix + b'[{"text":"\\ud800"}]}',
            prefix + b'[{},{}]}',
            prefix + b'[{"x":1.0},{"x":1e0}]}',
            prefix + b'[]}', prefix + b'{} }', prefix + b'[null]}',
            prefix + b'[true]}', prefix + b'["source"]}',
            prefix + b'[{}],"extra":1}', prefix + b'[{}]} trailing',
            b'[]', b'{"arm":"open_cake","candidates":[{}]}',
            b'{"schema_version":1,"arm":"direct_cuda","candidates":[""]}',
            b'{"schema_version":1,"arm":"direct_cuda","candidates":[{},"source"]}',
            b'{"schema_version":1,"arm":"direct_cuda","candidates":["same","same"]}',
        ]
        for raw in invalid:
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as directory:
                candidate = Path(directory) / "candidate-set.json"
                candidate.write_bytes(raw)
                arm = "direct_cuda" if b'"arm":"direct_cuda"' in raw else "open_cake"
                with self.assertRaises(ValueError):
                    normalize_codex_turn(
                        self._events(candidate, duplicate=False), candidate_path=candidate,
                        expected_change="add", expected_terminal_message='{"candidate_written":true}',
                        arm=arm, maximum_candidates_per_turn=2,
                    )

    def test_nonidentical_duplicate_terminal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.json"
            self._write_schedule_set(candidate, 1)
            with self.assertRaisesRegex(ValueError, "normalization"):
                normalize_codex_turn(
                    self._events(candidate, duplicate=True, second_text="different"),
                    candidate_path=candidate,
                    expected_change="add",
                    expected_terminal_message='{"candidate_written":true}',
                    arm="open_cake",
                )

    def _notice_bracketed_events(self, candidate):
        terminal = '{"arm":"open_cake","candidate_written":true,"kind":"open_cake_ir_turn","turn":1}'
        acknowledgement = lambda identity: {"type": "item.completed", "item": {
            "id": identity, "type": "agent_message", "text": terminal}}
        command = {"id": "write-command", "type": "command_execution", "command": "write candidate"}
        change = {"id": "candidate-file", "type": "file_change", "changes": [
            {"path": str(candidate), "kind": "add"}]}
        failed = {"id": "optional-status", "type": "command_execution", "command": "git status"}
        return terminal, [
            {"type": "thread.started", "thread_id": "01234567-89ab-cdef-0123-456789abcdef"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"id": "notice", "type": "error",
                "message": "Skill descriptions were shortened to fit the context budget"}},
            acknowledgement("ack-before"),
            {"type": "item.started", "item": {**command, "status": "in_progress"}},
            {"type": "item.completed", "item": {**command, "status": "completed", "exit_code": 0}},
            {"type": "item.started", "item": {**change, "status": "in_progress"}},
            {"type": "item.completed", "item": {**change, "status": "completed"}},
            {"type": "item.started", "item": {**failed, "status": "in_progress"}},
            {"type": "item.completed", "item": {**failed, "status": "failed", "exit_code": 128}},
            acknowledgement("ack-after"),
            {"type": "item.completed", "item": {"id": "thought", "type": "reasoning", "text": "finished"}},
            {"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 20,
                "cached_input_tokens": 80, "reasoning_output_tokens": 10}},
        ]

    def test_passive_notices_do_not_change_tool_rich_functional_brackets(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate-set.json"
            self._write_schedule_set(candidate, 1)
            terminal, events = self._notice_bracketed_events(candidate)
            raw = b"\n".join(json.dumps(event).encode() for event in events)
            turn = normalize_codex_turn(raw, candidate_path=candidate, expected_change="add",
                expected_terminal_message=terminal, event_contract="tool_rich_candidate_v1", arm="open_cake")
            self.assertEqual(turn.normalization, "duplicate_exact_bracketed")
            self.assertEqual(turn.candidates, (b'{"schedule":1}',))
            self.assertEqual(turn.provider_tokens, 120)
            self.assertEqual(turn.raw_events, raw)
            self.assertEqual([(a.item_type, a.status) for a in turn.tool_activity], [
                ("error", "completed"), ("command_execution", "completed"),
                ("file_change", "completed"), ("command_execution", "failed"), ("reasoning", "completed")])
            with self.assertRaises(ValueError):
                parse_codex_turn_events(raw, expected_terminal_message=terminal,
                    event_contract="closed_file_change_v1")

    def test_passive_notices_cannot_replace_or_hide_functional_lifecycles(self):
        from copy import deepcopy
        terminal, original = self._notice_bracketed_events(Path("/cpu-fixture/candidate-set.json"))
        mutations = {}
        mutations["no functional activity"] = original[:4] + original[10:]
        mutations["truncated command"] = original[:9] + original[10:]
        mutations["truncated file"] = original[:7] + original[8:]
        before = deepcopy(original); before[3], before[4] = before[4], before[3]
        mutations["initial ack after command starts"] = before
        after = deepcopy(original); after[9], after[10] = after[10], after[9]
        mutations["final ack before failed command completes"] = after
        unknown = deepcopy(original); unknown[2]["type"] = "item.updated"
        mutations["malformed passive lifecycle"] = unknown
        wrong = deepcopy(original); wrong[10]["item"]["text"] = '{"candidate_written":false}'
        mutations["wrong acknowledgement"] = wrong
        for name, events in mutations.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                parse_codex_turn_events(b"\n".join(json.dumps(event).encode() for event in events),
                    expected_terminal_message=terminal, event_contract="tool_rich_candidate_v1")

    def test_passive_notice_admission_preserves_usage_counter_checks(self):
        from copy import deepcopy
        terminal, original = self._notice_bracketed_events(Path("/cpu-fixture/candidate-set.json"))
        for key, value in (("input_tokens", True), ("output_tokens", -1),
                           ("cached_input_tokens", 101), ("reasoning_output_tokens", 21)):
            events = deepcopy(original); events[-1]["usage"][key] = value
            with self.subTest(counter=key), self.assertRaisesRegex(ValueError, "usage"):
                parse_codex_turn_events(b"\n".join(json.dumps(event).encode() for event in events),
                    expected_terminal_message=terminal, event_contract="tool_rich_candidate_v1")

    def test_tool_rich_turn_accepts_and_projects_an_mcp_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.json"
            self._write_schedule_set(candidate, 1)
            events = [json.loads(line) for line in self._events(candidate, duplicate=False).splitlines()]
            events[2:2] = [
                {
                    "type": "item.started",
                    "item": {
                        "id": "tool_0",
                        "type": "mcp_tool_call",
                        "server": "fixture",
                        "tool": "read_reference",
                        "arguments": {},
                        "status": "in_progress",
                    },
                },
                {
                    "type": "item.completed",
                    "item": {
                        "id": "tool_0",
                        "type": "mcp_tool_call",
                        "server": "fixture",
                        "tool": "read_reference",
                        "arguments": {},
                        "result": {"content": []},
                        "status": "completed",
                    },
                },
            ]
            raw_events = b"".join(
                json.dumps(event, separators=(",", ":")).encode() + b"\n"
                for event in events
            )

            turn = normalize_codex_turn(
                raw_events,
                candidate_path=candidate,
                expected_change="add",
                expected_terminal_message='{"candidate_written":true}',
                event_contract="tool_rich_candidate_v1",
                       arm="open_cake",
            )

        self.assertEqual(
            [activity.item_type for activity in turn.tool_activity],
            ["mcp_tool_call", "file_change"],
        )
        self.assertEqual(turn.tool_activity[0].item_type, "mcp_tool_call")
        self.assertEqual(turn.tool_activity[0].server, "fixture")
        self.assertEqual(turn.tool_activity[0].tool, "read_reference")

    def test_tool_rich_shell_write_uses_the_candidate_postcondition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.json"
            self._write_schedule_set(candidate, 1)
            events = [
                {
                    "type": "thread.started",
                    "thread_id": "01234567-89ab-cdef-0123-456789abcdef",
                },
                {"type": "turn.started"},
                {
                    "type": "item.started",
                    "item": {
                        "id": "tool_0",
                        "type": "command_execution",
                        "command": "write candidate",
                        "status": "in_progress",
                    },
                },
                {
                    "type": "item.completed",
                    "item": {
                        "id": "tool_0",
                        "type": "command_execution",
                        "command": "write candidate",
                        "status": "completed",
                    },
                },
                {
                    "type": "item.completed",
                    "item": {
                        "id": "message_0",
                        "type": "agent_message",
                        "text": '{"candidate_written":true}',
                    },
                },
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                },
            ]
            raw_events = b"".join(
                json.dumps(event, separators=(",", ":")).encode() + b"\n"
                for event in events
            )

            turn = normalize_codex_turn(
                raw_events,
                candidate_path=candidate,
                expected_change="add",
                expected_terminal_message='{"candidate_written":true}',
                event_contract="tool_rich_candidate_v1",
                       arm="open_cake",
            )

        self.assertEqual(turn.candidates, (b'{"schedule":1}',))
        self.assertEqual(turn.tool_activity[0].item_type, "command_execution")

    def test_tool_rich_file_changes_leave_candidate_custody_to_postcondition(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate-set.json"
            candidate.write_text(
                '{"arm":"direct_cuda","candidates":["// candidate\\n"],"schema_version":1}\n'
            )
            terminal = '{"candidate_written":true}'
            events = [
                {
                    "type": "thread.started",
                    "thread_id": "01234567-89ab-cdef-0123-456789abcdef",
                },
                {"type": "turn.started"},
                *(
                    {
                        "type": "item.completed",
                        "item": {
                            "id": f"message_{index}",
                            "type": "agent_message",
                            "text": terminal,
                        },
                    }
                    for index in range(2)
                ),
            ]
            changes = (
                (
                    "scratch_add",
                    [
                        {"path": "/tmp/candidate-1.cu", "kind": "add"},
                        {"path": "/tmp/candidate-2.cu", "kind": "add"},
                    ],
                ),
                (
                    "candidate_add",
                    [{"path": str(candidate.absolute()), "kind": "add"}],
                ),
                (
                    "candidate_update",
                    [{"path": str(candidate.absolute()), "kind": "update"}],
                ),
                (
                    "scratch_delete",
                    [
                        {"path": "/tmp/candidate-1.cu", "kind": "delete"},
                        {"path": "/tmp/candidate-2.cu", "kind": "delete"},
                    ],
                ),
            )
            for item_id, item_changes in changes:
                events.extend(
                    [
                        {
                            "type": "item.started",
                            "item": {
                                "id": item_id,
                                "type": "file_change",
                                "changes": item_changes,
                                "status": "in_progress",
                            },
                        },
                        {
                            "type": "item.completed",
                            "item": {
                                "id": item_id,
                                "type": "file_change",
                                "changes": item_changes,
                                "status": "completed",
                            },
                        },
                    ]
                )
            events.extend(
                [
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "message_2",
                            "type": "agent_message",
                            "text": terminal,
                        },
                    },
                    {
                        "type": "turn.completed",
                        "usage": {"input_tokens": 100, "output_tokens": 20},
                    },
                ]
            )
            raw_events = b"".join(
                json.dumps(event, separators=(",", ":")).encode() + b"\n"
                for event in events
            )

            turn = normalize_codex_turn(
                raw_events,
                candidate_path=candidate,
                expected_change="add",
                expected_terminal_message=terminal,
                event_contract="tool_rich_candidate_v1",
                submission_contract=CANDIDATE_SET_ENVELOPE_V1,
                arm="direct_cuda",
                maximum_candidates_per_turn=3,
            )

        self.assertEqual(turn.candidates, (b"// candidate\n",))
        self.assertEqual(turn.terminal_message_count, 3)
        self.assertEqual(turn.normalization, "duplicate_exact_bracketed")
        self.assertEqual(
            [activity.item_type for activity in turn.tool_activity],
            ["file_change", "file_change", "file_change", "file_change"],
        )


    def test_tool_rich_scratch_file_change_requires_a_complete_stable_lifecycle(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.json"
            self._write_schedule_set(candidate, 1)
            events = [
                {
                    "type": "thread.started",
                    "thread_id": "01234567-89ab-cdef-0123-456789abcdef",
                },
                {"type": "turn.started"},
                {
                    "type": "item.started",
                    "item": {
                        "id": "scratch",
                        "type": "file_change",
                        "changes": [{"path": "/tmp/a", "kind": "add"}],
                        "status": "in_progress",
                    },
                },
                {
                    "type": "item.completed",
                    "item": {
                        "id": "scratch",
                        "type": "file_change",
                        "changes": [{"path": "/tmp/b", "kind": "add"}],
                        "status": "completed",
                    },
                },
                {
                    "type": "item.completed",
                    "item": {
                        "id": "message",
                        "type": "agent_message",
                        "text": '{"candidate_written":true}',
                    },
                },
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                },
            ]
            raw_events = b"".join(
                json.dumps(event, separators=(",", ":")).encode() + b"\n"
                for event in events
            )

            with self.assertRaisesRegex(ValueError, "file-change lifecycle"):
                normalize_codex_turn(
                    raw_events,
                    candidate_path=candidate,
                    expected_change="add",
                    expected_terminal_message='{"candidate_written":true}',
                    event_contract="tool_rich_candidate_v1",
                    arm="open_cake",
                )

    def test_tool_rich_turn_rejects_an_incomplete_auxiliary_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.json"
            self._write_schedule_set(candidate, 1)
            events = [json.loads(line) for line in self._events(candidate, duplicate=False).splitlines()]
            events.insert(
                2,
                {
                    "type": "item.started",
                    "item": {
                        "id": "tool_0",
                        "type": "mcp_tool_call",
                        "server": "fixture",
                        "tool": "read_reference",
                        "arguments": {},
                        "status": "in_progress",
                    },
                },
            )
            raw_events = b"".join(
                json.dumps(event, separators=(",", ":")).encode() + b"\n"
                for event in events
            )

            with self.assertRaisesRegex(ValueError, "auxiliary item lifecycle"):
                normalize_codex_turn(
                    raw_events,
                    candidate_path=candidate,
                    expected_change="add",
                    expected_terminal_message='{"candidate_written":true}',
                    event_contract="tool_rich_candidate_v1",
                    arm="open_cake",
                )

    def test_unadmitted_item_lifecycle_event_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.json"
            self._write_schedule_set(candidate, 1)
            events = [json.loads(line) for line in self._events(candidate, duplicate=False).splitlines()]
            events.insert(
                -1,
                {
                    "type": "item.updated",
                    "item": {"id": "item_extra", "type": "agent_message"},
                },
            )
            raw_events = b"".join(
                json.dumps(event, separators=(",", ":")).encode() + b"\n"
                for event in events
            )

            with self.assertRaisesRegex(ValueError, "item lifecycle"):
                normalize_codex_turn(
                    raw_events,
                    candidate_path=candidate,
                    expected_change="add",
                    expected_terminal_message='{"candidate_written":true}',
                    arm="open_cake",
                )

    def test_candidate_symlink_cannot_cross_the_workspace_custody_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside.json"
            outside.write_text('{"secret":"not-read"}')
            candidate = root / "candidate.json"
            candidate.symlink_to(outside)
            with self.assertRaises(OSError):
                normalize_codex_turn(
                    self._events(candidate, duplicate=False),
                    candidate_path=candidate,
                    expected_change="add",
                    expected_terminal_message='{"candidate_written":true}',
                    arm="open_cake",
                )



    def test_task_request_projects_unicode_newlines_and_state_without_mutating_task(self):
        package = TaskPackage('native_triton-1', 'native_triton',
                              '# TASK.md\n中文 "quoted" \\ path\n', '# AGENTS.md\nRules\n')
        original = (package.task_markdown, package.agents_markdown, package.canonical_sha256)
        for iteration in (1, 2):
            prompt, bundle = render_task_request(package, {'iteration':iteration})
            self.assertTrue(prompt.endswith(bundle.decode('utf-8')))
            delivered = json.loads(bundle)
            self.assertEqual(delivered['task_markdown'], package.task_markdown)
            self.assertEqual(delivered['agents_markdown'], package.agents_markdown)
            self.assertEqual(delivered['state_card'], {'iteration':iteration})
        self.assertEqual((package.task_markdown, package.agents_markdown, package.canonical_sha256), original)
        with self.assertRaisesRegex(ValueError, 'StateCard'):
            render_task_request(package, None)

    def test_ralph_provider_exposes_only_task_agents_and_candidate_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "codex"
            executable.write_bytes(b"fixture executable")
            executable.chmod(0o700)
            helper = executable.with_name("codex-code-mode-host")
            helper.write_bytes(b"CPU Code Mode host fixture")
            helper.chmod(0o700)
            workspace = root / "workspace"
            workspace.mkdir()
            schema = root / "schema.json"
            schema.write_text("{}")
            package = TaskPackage(
                "open_cake-1",
                "open_cake",
                "# TASK.md\n\nImplement the frozen task.\n",
                "# AGENTS.md\n\nWrite only candidate-set.json.\n",
            )
            materialize_task_package(workspace, package)
            builder = CodexInvocationBuilder(
                executable=executable,
                provider_revision="codex-ralph-fixture",
                model="gpt-5.6-sol",
                reasoning_effort="max",
                service_tier="default",
                workspace=workspace,
                output_schema=schema,
                removed_environment=("OPENAI_API_KEY",),
                submission_contract=CANDIDATE_SET_ENVELOPE_V1,
                cwd_policy="independent_task_workspace",
                reference_visibility="workspace_task_files",
            )
            qualification = ProviderQualificationReceipt(
                provider_revision=builder.provider_revision,
                executable_sha256=sha256(executable.read_bytes()).hexdigest(),
                configuration_sha256=builder.configuration_sha256,
                initial_and_resume_equivalent=True,
                file_lifecycle_observed=True,
                usage_observed=True,
                qualified=True,
                scope="live_two_turn_current_provider",
            )

            class Adapter:
                def __init__(self):
                    self.invocations = []

                def execute(
                    self,
                    invocation,
                    *,
                    candidate_path,
                    expected_change,
                    expected_terminal_message,
                    event_contract,
                    submission_contract,
                    arm,
                    maximum_candidates_per_turn,
                ):
                    self.invocations.append(invocation)
                    member = {"turn": len(self.invocations)}
                    candidate_path.write_bytes(
                        json.dumps(
                            {
                                "schema_version": 1,
                                "arm": arm,
                                "candidates": [member],
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode()
                        + b"\n"
                    )
                    payload = json.dumps(
                        member, sort_keys=True, separators=(",", ":")
                    ).encode()
                    return ProviderTurn(
                        thread_id="01234567-89ab-cdef-0123-456789abcdef",
                        provider_tokens=100,
                        candidates=(payload,),
                        raw_submission=candidate_path.read_bytes(),
                        candidate_sha256s=(sha256(payload).hexdigest(),),
                        raw_events=b"{}\n",
                        raw_events_sha256=sha256(b"{}\n").hexdigest(),
                        terminal_message=expected_terminal_message,
                        terminal_message_count=1,
                        normalization="single_exact",
                    )

            adapter = Adapter()
            provider = CodexRunProvider(
                qualification=qualification,
                builders={"open_cake-1": builder},
                task_packages={"open_cake-1": package},
                adapter=adapter,
            )
            state = MappingProxyType(
                {
                    "schema_version": 1,
                    "kind": "ralph_state_v1",
                    "iteration": 1,
                }
            )
            first = provider.turn(
                SimpleNamespace(
                    run_id="open_cake-1",
                    arm="open_cake",
                    turn=1,
                    cumulative_provider_tokens=0,
                    thread_id=None,
                    feedback={"kind": "initial"},
                    maximum_candidates_per_turn=2,
                    state_card=state,
                )
            )
            second = provider.turn(
                SimpleNamespace(
                    run_id="open_cake-1",
                    arm="open_cake",
                    turn=2,
                    cumulative_provider_tokens=100,
                    thread_id=first.thread_id,
                    feedback={"kind": "evaluation"},
                    maximum_candidates_per_turn=2,
                    state_card=MappingProxyType(
                        {
                            "schema_version": 1,
                            "kind": "ralph_state_v1",
                            "iteration": 2,
                        }
                    ),
                )
            )

            self.assertEqual(
                {path.name for path in workspace.iterdir()},
                {"TASK.md", "AGENTS.md", "candidate-set.json"},
            )
            for invocation, result in zip(adapter.invocations, (first, second), strict=True):
                self.assertTrue(invocation.argv[-1].endswith(result.reference_bundle.decode()))
                delivered = json.loads(invocation.argv[-1].split('\n\n', 1)[1])
                self.assertEqual(delivered['task_markdown'].encode(), (workspace / 'TASK.md').read_bytes())
                self.assertEqual(delivered['agents_markdown'].encode(), (workspace / 'AGENTS.md').read_bytes())
            first_bundle, second_bundle = json.loads(first.reference_bundle), json.loads(second.reference_bundle)
            self.assertEqual(first_bundle['task_markdown'], second_bundle['task_markdown'])
            self.assertEqual(first_bundle['agents_markdown'], second_bundle['agents_markdown'])
            self.assertNotEqual(first_bundle['state_card'], second_bundle['state_card'])
            self.assertIn("resume", adapter.invocations[1].argv)
            self.assertNotEqual(first.candidate_sha256s, second.candidate_sha256s)
            bundle = json.loads(second.reference_bundle)
            self.assertEqual(bundle["task_markdown"], package.task_markdown)
            self.assertEqual(bundle["agents_markdown"], package.agents_markdown)
            self.assertEqual(bundle["state_card"]["iteration"], 2)
            # A modified immutable file is refused before another provider call.
            (workspace / 'TASK.md').chmod(0o644)
            (workspace / 'TASK.md').write_text('modified task')
            with self.assertRaisesRegex(ValueError, 'task file TASK.md custody'):
                provider.turn(SimpleNamespace(run_id='open_cake-1', arm='open_cake', turn=3,
                    cumulative_provider_tokens=200, thread_id=first.thread_id, feedback={},
                    maximum_candidates_per_turn=2, state_card=state))
            self.assertEqual(len(adapter.invocations), 2)



if __name__ == "__main__":
    unittest.main()
