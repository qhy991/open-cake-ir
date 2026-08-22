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
    CodexInvocationBuilder,
    CodexRunProvider,
    ProviderQualificationReceipt,
    ProviderTurn,
    normalize_codex_turn,
)


class ProviderContractTests(unittest.TestCase):
    def _events(self, path: Path, *, duplicate: bool, second_text: str | None = None) -> bytes:
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
            events.append(terminal)
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
        self.assertEqual(single.candidate_sha256, duplicate.candidate_sha256)
        self.assertEqual(single.terminal_message_count, 1)
        self.assertEqual(duplicate.terminal_message_count, 2)
        self.assertEqual(duplicate.normalization, "duplicate_exact_bracketed")

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
                ):
                    self.invocations.append(invocation)
                    candidate_path.write_text(
                        json.dumps({"turn": len(self.invocations)})
                    )
                    payload = candidate_path.read_bytes()
                    return ProviderTurn(
                        thread_id="01234567-89ab-cdef-0123-456789abcdef",
                        provider_tokens=100,
                        candidate=payload,
                        candidate_sha256=sha256(payload).hexdigest(),
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
                )
            )

        self.assertEqual(len(adapter.invocations), 2)
        self.assertNotIn("resume", adapter.invocations[0].argv)
        self.assertIn("resume", adapter.invocations[1].argv)
        self.assertNotEqual(first.candidate_sha256, second.candidate_sha256)


if __name__ == "__main__":
    unittest.main()
