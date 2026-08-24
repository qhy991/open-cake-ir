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
    required_live_provider_qualification_scope,
)
from open_cake_ir.lab.faults import RunProtocolFault  # noqa: E402


class ProviderContractTests(unittest.TestCase):
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
            executable=ROOT / "pyproject.toml",
            provider_revision="codex-fixture-v1",
            model="gpt-5.6-sol",
            reasoning_effort="max",
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
            self.assertIn("shell_tool", disabled)
        self.assertNotIn("resume", initial.argv)
        self.assertIn("resume", resumed.argv)

    def test_provider_default_features_emit_no_forced_disable_flags(self) -> None:
        builder = CodexInvocationBuilder(
            executable=ROOT / "pyproject.toml",
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

    def test_candidate_set_capability_is_bound_without_changing_legacy_configuration(self) -> None:
        arguments = {
            "executable": ROOT / "pyproject.toml",
            "provider_revision": "codex-fixture-v1",
            "model": "gpt-5.6-sol",
            "reasoning_effort": "max",
            "service_tier": "default",
            "workspace": ROOT,
            "output_schema": ROOT
            / "contracts/providers/codex-turn-output-schema-v1.json",
            "removed_environment": ("OPENAI_API_KEY",),
        }
        legacy = CodexInvocationBuilder(**arguments)
        candidate_set = CodexInvocationBuilder(
            **arguments,
            submission_contract=CANDIDATE_SET_ENVELOPE_V1,
        )

        self.assertNotIn("submission_contract", legacy.configuration)
        self.assertEqual(
            candidate_set.configuration["submission_contract"],
            CANDIDATE_SET_ENVELOPE_V1,
        )
        self.assertNotEqual(
            legacy.configuration_sha256,
            candidate_set.configuration_sha256,
        )

    def test_single_and_bracketed_duplicate_terminal_forms_normalize_equally(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.json"
            candidate.write_text('{"schedule":1}')
            single = normalize_codex_turn(
                self._events(candidate, duplicate=False),
                candidate_path=candidate,
                expected_change="add",
                expected_terminal_message='{"candidate_written":true}',
            )
            duplicate = normalize_codex_turn(
                self._events(candidate, duplicate=True),
                candidate_path=candidate,
                expected_change="add",
                expected_terminal_message='{"candidate_written":true}',
            )

        self.assertEqual(single.provider_tokens, duplicate.provider_tokens, 120)
        self.assertEqual(single.candidate_sha256s, duplicate.candidate_sha256s)
        self.assertEqual(single.terminal_message_count, 1)
        self.assertEqual(duplicate.terminal_message_count, 2)
        self.assertEqual(duplicate.normalization, "duplicate_exact_bracketed")

    def test_bracketed_terminal_whitespace_normalizes_by_json_meaning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.json"
            candidate.write_text('{"schedule":1}')
            turn = normalize_codex_turn(
                self._events(
                    candidate,
                    duplicate=True,
                    first_text='{ "candidate_written": true }',
                ),
                candidate_path=candidate,
                expected_change="add",
                expected_terminal_message='{"candidate_written":true}',
            )

        self.assertEqual(turn.terminal_message_count, 2)
        self.assertEqual(turn.normalization, "duplicate_semantic_bracketed")

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

    def test_candidate_set_rejects_noncanonical_wrong_arm_and_over_bound_envelopes(self) -> None:
        valid = {
            "schema_version": 1,
            "arm": "open_cake",
            "candidates": [{"variant": 0}, {"variant": 1}],
        }
        cases = (
            (json.dumps(valid).encode() + b"\n", "canonical bytes", "open_cake", 2),
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

    def test_nonidentical_duplicate_terminal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.json"
            candidate.write_text('{"schedule":1}')
            with self.assertRaisesRegex(ValueError, "normalization"):
                normalize_codex_turn(
                    self._events(candidate, duplicate=True, second_text="different"),
                    candidate_path=candidate,
                    expected_change="add",
                    expected_terminal_message='{"candidate_written":true}',
                )

    def test_tool_rich_turn_accepts_and_projects_an_mcp_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.json"
            candidate.write_text('{"schedule":1}')
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
            )

        self.assertEqual(len(turn.tool_activity), 1)
        self.assertEqual(turn.tool_activity[0].item_type, "mcp_tool_call")
        self.assertEqual(turn.tool_activity[0].server, "fixture")
        self.assertEqual(turn.tool_activity[0].tool, "read_reference")

    def test_tool_rich_shell_write_uses_the_candidate_postcondition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.json"
            candidate.write_text('{"schedule":1}')
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
            )

        self.assertEqual(turn.candidates, (b'{"schedule":1}',))
        self.assertEqual(turn.tool_activity[0].item_type, "command_execution")

    def test_tool_rich_turn_rejects_an_incomplete_auxiliary_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.json"
            candidate.write_text('{"schedule":1}')
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
                )

    def test_unadmitted_item_lifecycle_event_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.json"
            candidate.write_text('{"schedule":1}')
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
                )

    def test_concrete_run_provider_owns_add_then_same_thread_update(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "codex"
            executable.write_bytes(b"fixture executable")
            executable.chmod(0o700)
            workspace = root / "workspace"
            workspace.mkdir()
            references = root / "references"
            references.mkdir()
            reference = references / "workload.json"
            reference.write_text('{"outer":{"inner":true}}')
            reference.chmod(0o444)
            references.chmod(0o555)
            schema = root / "schema.json"
            schema.write_text("{}")
            builder = CodexInvocationBuilder(
                executable=executable,
                provider_revision="codex-live-fixture",
                model="gpt-5.6-sol",
                reasoning_effort="max",
                service_tier="default",
                workspace=workspace,
                output_schema=schema,
                removed_environment=("OPENAI_API_KEY",),
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

            test_case = self

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
                    event_contract="closed_file_change_v1",
                    submission_contract="single_candidate_v1",
                    arm=None,
                    maximum_candidates_per_turn=1,
                ):
                    test_case.assertEqual(event_contract, "closed_file_change_v1")
                    test_case.assertEqual(submission_contract, "single_candidate_v1")
                    test_case.assertIsNone(arm)
                    test_case.assertEqual(maximum_candidates_per_turn, 1)
                    self.invocations.append(invocation)
                    candidate_path.write_text(
                        json.dumps({"turn": len(self.invocations)})
                    )
                    payload = candidate_path.read_bytes()
                    return ProviderTurn(
                        thread_id="01234567-89ab-cdef-0123-456789abcdef",
                        provider_tokens=100,
                        candidates=(payload,),
                        candidate_sha256s=(sha256(payload).hexdigest(),),
                        raw_events=b'{"type":"fixture"}\n',
                        raw_events_sha256=sha256(b'{"type":"fixture"}\n').hexdigest(),
                        terminal_message=expected_terminal_message,
                        terminal_message_count=1,
                        normalization="single_exact",
                    )

            adapter = Adapter()
            provider = CodexRunProvider(
                qualification=qualification,
                builders={"open_cake-1": builder},
                reference_roots={"open_cake-1": references},
                prompt_templates={
                    "open_cake": ROOT
                    / "src/open_cake_ir/lab/prompts/open_cake_turn_v2.md",
                    "direct_cuda": ROOT
                    / "src/open_cake_ir/lab/prompts/direct_cuda_turn_v2.md",
                },
                adapter=adapter,
            )
            first = provider.turn(
                SimpleNamespace(
                    run_id="open_cake-1",
                    arm="open_cake",
                    turn=1,
                    cumulative_provider_tokens=0,
                    thread_id=None,
                    feedback=MappingProxyType({"kind": "initial"}),
                    maximum_candidates_per_turn=1,
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
                    maximum_candidates_per_turn=1,
                )
            )

        self.assertEqual(len(adapter.invocations), 2)
        self.assertNotIn("resume", adapter.invocations[0].argv)
        self.assertIn("resume", adapter.invocations[1].argv)
        self.assertNotEqual(first.candidate_sha256s, second.candidate_sha256s)

    def test_concrete_candidate_set_provider_owns_one_envelope_and_no_other_file(self) -> None:
        for extra_file in (False, True):
            with self.subTest(extra_file=extra_file), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                executable = root / "codex"
                executable.write_bytes(b"fixture executable")
                executable.chmod(0o700)
                workspace = root / "workspace"
                workspace.mkdir()
                references = root / "references"
                references.mkdir()
                reference = references / "authority.json"
                reference.write_text("{}")
                reference.chmod(0o444)
                references.chmod(0o555)
                schema = root / "schema.json"
                schema.write_text("{}")
                builder = CodexInvocationBuilder(
                    executable=executable,
                    provider_revision="codex-candidate-set-fixture",
                    model="gpt-5.6-sol",
                    reasoning_effort="max",
                    service_tier="default",
                    workspace=workspace,
                    output_schema=schema,
                    removed_environment=("OPENAI_API_KEY",),
                    submission_contract=CANDIDATE_SET_ENVELOPE_V1,
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
                        self.invocation = invocation
                        self.path = candidate_path
                        self.maximum = maximum_candidates_per_turn
                        members = ({"variant": 0}, {"variant": 1})
                        envelope = {
                            "schema_version": 1,
                            "arm": arm,
                            "candidates": members,
                        }
                        candidate_path.write_bytes(
                            json.dumps(
                                envelope,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode()
                            + b"\n"
                        )
                        if extra_file:
                            (candidate_path.parent / "scratch.txt").write_text("leak")
                        payloads = tuple(
                            json.dumps(
                                member,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode()
                            for member in members
                        )
                        return ProviderTurn(
                            thread_id="01234567-89ab-cdef-0123-456789abcdef",
                            provider_tokens=100,
                            candidates=payloads,
                            candidate_sha256s=tuple(
                                sha256(payload).hexdigest() for payload in payloads
                            ),
                            raw_events=b'{}\n',
                            raw_events_sha256=sha256(b'{}\n').hexdigest(),
                            terminal_message=expected_terminal_message,
                            terminal_message_count=1,
                            normalization="single_exact",
                        )

                adapter = Adapter()
                provider = CodexRunProvider(
                    qualification=qualification,
                    builders={"open_cake-1": builder},
                    reference_roots={"open_cake-1": references},
                    prompt_templates={
                        "open_cake": ROOT
                        / "src/open_cake_ir/lab/prompts/open_cake_candidate_set_turn_v1.md",
                        "direct_cuda": ROOT
                        / "src/open_cake_ir/lab/prompts/direct_cuda_candidate_set_turn_v1.md",
                    },
                    adapter=adapter,
                )
                request = SimpleNamespace(
                    run_id="open_cake-1",
                    arm="open_cake",
                    turn=1,
                    cumulative_provider_tokens=0,
                    thread_id=None,
                    feedback={"kind": "initial"},
                    maximum_candidates_per_turn=2,
                )

                if extra_file:
                    with self.assertRaisesRegex(
                        RunProtocolFault, "workspace custody"
                    ):
                        provider.turn(request)
                else:
                    turn = provider.turn(request)
                    self.assertEqual(len(turn.candidates), 2)
                    self.assertEqual(adapter.path.name, "candidate-set.json")
                    self.assertEqual(adapter.maximum, 2)
                    self.assertIn(
                        "one to\n2 Schedule objects",
                        adapter.invocation.argv[-1],
                    )


if __name__ == "__main__":
    unittest.main()
