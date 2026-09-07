from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.evidence import EvidenceStore  # noqa: E402
from open_cake_ir.lab import ProviderQualificationReceipt  # noqa: E402


class ProviderQualificationContractTests(unittest.TestCase):
    def _write_provider(
        self,
        path: Path,
        *,
        break_resumed_thread: bool = False,
        zero_usage: bool = False,
        exit_nonzero: bool = False,
        tool_rich: bool = False,
    ) -> None:
        path.write_text(
            textwrap.dedent(
                f"""\
                #!{sys.executable}
                import json
                import sys
                from pathlib import Path

                arguments = sys.argv[1:]
                if {exit_nonzero!r}:
                    print("qualification provider failure", file=sys.stderr)
                    raise SystemExit(9)
                resumed = arguments[:2] == ["exec", "resume"]
                prompt = arguments[-1]
                candidate_line = next(
                    line for line in prompt.splitlines()
                    if line.startswith("CANDIDATE_PATH_JSON=")
                )
                candidate = Path(json.loads(candidate_line.split("=", 1)[1]))
                arm_lines = [
                    line for line in prompt.splitlines() if line.startswith("ARM=")
                ]
                arm = arm_lines[0].split("=", 1)[1] if arm_lines else "open_cake"
                thread_id = (
                    "11234567-89ab-cdef-0123-456789abcdef"
                    if arm == "direct_cuda"
                    else "01234567-89ab-cdef-0123-456789abcdef"
                )
                reported_thread_id = (
                    "fedcba98-7654-3210-fedc-ba9876543210"
                    if resumed and {break_resumed_thread!r}
                    else thread_id
                )
                if Path.cwd() != candidate.parent:
                    raise SystemExit(31)
                if 'sandbox_mode="workspace-write"' not in arguments:
                    raise SystemExit(32)
                if 'approval_policy="never"' not in arguments:
                    raise SystemExit(33)
                if {tool_rich!r} and "--disable" in arguments:
                    raise SystemExit(36)
                if resumed and arguments[-2] != thread_id:
                    raise SystemExit(34)
                turn = 2 if resumed else 1
                change = "update" if resumed else "add"
                if candidate.exists() is not resumed:
                    raise SystemExit(35)
                candidate_set_lines = [
                    line for line in prompt.splitlines()
                    if line.startswith("EXPECTED_CANDIDATE_SET_JSON=")
                ]
                if candidate_set_lines:
                    expected = json.loads(candidate_set_lines[0].split("=", 1)[1])
                    candidate.write_text(
                        json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\\n",
                        encoding="utf-8",
                    )
                else:
                    expected_line = next(
                        line for line in prompt.splitlines()
                        if line.startswith("Write exactly this JSON object: ")
                    )
                    candidate.write_text(
                        json.dumps(
                            json.loads(expected_line.split(": ", 1)[1]),
                            sort_keys=True,
                        ),
                        encoding="utf-8",
                    )
                terminal_document = {{
                    "arm": arm,
                    "candidate_written": True,
                    "kind": "open_cake_ir_turn",
                    "turn": turn,
                }}
                if not {tool_rich!r}:
                    terminal_document["tool_calls"] = 1
                terminal = json.dumps(
                    terminal_document,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                change_item = {{
                    "id": "item_0",
                    "type": "file_change",
                    "changes": [{{"path": str(candidate.absolute()), "kind": change}}],
                }}
                events = [
                    {{"type": "thread.started", "thread_id": reported_thread_id}},
                    {{"type": "turn.started"}},
                ]
                if {tool_rich!r}:
                    command_item = {{
                        "id": "command_0",
                        "type": "command_execution",
                        "command": "pwd",
                    }}
                    events.extend([
                        {{
                            "type": "item.started",
                            "item": {{**command_item, "status": "in_progress"}},
                        }},
                        {{
                            "type": "item.completed",
                            "item": {{**command_item, "status": "completed"}},
                        }},
                    ])
                events.extend([
                    {{
                        "type": "item.started",
                        "item": {{**change_item, "status": "in_progress"}},
                    }},
                    {{
                        "type": "item.completed",
                        "item": {{**change_item, "status": "completed"}},
                    }},
                    {{
                        "type": "item.completed",
                        "item": {{
                            "id": "item_1",
                            "type": "agent_message",
                            "text": terminal,
                        }},
                    }},
                    {{
                        "type": "turn.completed",
                        "usage": {{
                            "input_tokens": 0 if {zero_usage!r} else 100 + turn,
                            "output_tokens": 0 if {zero_usage!r} else 20,
                        }},
                    }},
                ])
                for event in events:
                    print(json.dumps(event, separators=(",", ":")))
                """
            ),
            encoding="utf-8",
        )
        path.chmod(0o700)

    def _run_qualification(
        self,
        root: Path,
        executable: Path,
        *,
        provider_revision: str,
        run_id: str,
        feature_policy: str = "closed_research",
        maximum_candidates_per_turn: int | None = None,
        reasoning_effort: str = "max",
    ) -> tuple[subprocess.CompletedProcess[bytes], Path, Path, Path]:
        receipt_path = root / "provider-qualification.json"
        anchor_path = root / "provider-qualification-anchor.json"
        evidence_root = root / "evidence"
        command = [
                sys.executable,
                str(ROOT / "tools/qualify_codex_provider.py"),
                "--executable",
                str(executable),
                "--provider-revision",
                provider_revision,
                "--output-schema",
                str(
                    ROOT
                    / (
                        "contracts/providers/codex-turn-output-schema-v2.json"
                        if feature_policy == "provider_defaults_optimization"
                        else "contracts/providers/codex-turn-output-schema-v1.json"
                    )
                ),
                "--workspace",
                str(root / "workspace"),
                "--receipt-output",
                str(receipt_path),
                "--anchor-output",
                str(anchor_path),
                "--evidence-root",
                str(evidence_root),
                "--run-id",
                run_id,
                "--reasoning-effort",
                reasoning_effort,
                "--feature-policy",
                feature_policy,
            ]
        if maximum_candidates_per_turn is not None:
            command.extend(
                [
                    "--maximum-candidates-per-turn",
                    str(maximum_candidates_per_turn),
                ]
            )
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
        return completed, receipt_path, anchor_path, evidence_root

    def test_tool_rich_qualification_binds_provider_defaults_and_auxiliary_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "codex"
            self._write_provider(executable, tool_rich=True)
            completed, receipt_path, _, evidence_root = self._run_qualification(
                root,
                executable,
                provider_revision="codex-tool-rich-fixture-v1",
                run_id="codex-provider-tool-rich",
                feature_policy="provider_defaults_optimization",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            receipt = ProviderQualificationReceipt.load(receipt_path)
            self.assertEqual(receipt.scope, "live_two_turn_tool_rich_provider")
            observed = next(
                event
                for event in EvidenceStore.open(evidence_root).replay_events(
                    "codex-provider-tool-rich"
                )
                if event["kind"] == "provider_qualification_observed"
            )
            self.assertEqual(
                observed["payload"]["arms"]["open_cake"]["initial_auxiliary_activity"][0]["item_type"],
                "command_execution",
            )

    def test_two_real_process_turns_issue_and_archive_the_live_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "codex"
            self._write_provider(executable)
            workspace = root / "workspace"
            completed, receipt_path, anchor_path, evidence_root = self._run_qualification(
                root,
                executable,
                provider_revision="codex-contract-fixture-v1",
                run_id="codex-provider-two-turn-contract",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            receipt = ProviderQualificationReceipt.load(receipt_path)
            self.assertEqual(receipt_path.stat().st_mode & 0o444, 0o444)
            self.assertEqual(anchor_path.stat().st_mode & 0o444, 0o444)
            self.assertEqual(
                receipt.executable_sha256,
                sha256(executable.read_bytes()).hexdigest(),
            )
            self.assertTrue(receipt.initial_and_resume_equivalent)
            self.assertTrue(receipt.file_lifecycle_observed)
            self.assertTrue(receipt.usage_observed)
            self.assertTrue(receipt.qualified)
            self.assertEqual(receipt.scope, "live_two_turn_current_provider")
            candidate = json.loads((workspace / "open_cake" / "candidate-set.json").read_text())["candidates"][0]
            self.assertEqual(candidate["qualification_turn"], 2)
            self.assertEqual(len(candidate["reference_nonce"]), 64)
            self.assertEqual({path.name for path in workspace.iterdir()}, {"open_cake", "direct_cuda"})

            evidence = EvidenceStore.open(evidence_root)
            audit = evidence.audit_run("codex-provider-two-turn-contract")
            anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
            self.assertTrue(audit.archive_integrity)
            self.assertEqual(audit.protocol_adherence, "adhered")
            self.assertEqual(audit.endpoint_observation, "qualified")
            self.assertEqual(
                audit.endpoint["qualification_receipt_sha256"],
                receipt.canonical_sha256,
            )
            self.assertEqual(anchor["terminal_seal_sha256"], audit.terminal_seal_sha256)
            self.assertEqual(
                anchor["qualification_receipt_sha256"], receipt.canonical_sha256
            )
            self.assertEqual(anchor["authority_sha256"], audit.authority_sha256)
            observed = [
                event
                for event in evidence.replay_events(audit.run_id)
                if event["kind"] == "provider_qualification_observed"
            ]
            self.assertEqual(len(observed), 1)
            roles = {item["role"] for item in observed[0]["payload"]["objects"]}
            self.assertTrue({"qualification_receipt", "qualification_reference"} <= roles)
            for arm in ("open_cake", "direct_cuda"):
                for turn in ("initial", "resumed"):
                    self.assertIn(f"{arm}_{turn}_provider_events", roles)
                    self.assertIn(f"{arm}_{turn}_invocation", roles)

    def test_candidate_set_qualification_covers_both_arm_projections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "codex"
            self._write_provider(executable)
            completed, receipt_path, _, evidence_root = self._run_qualification(
                root,
                executable,
                provider_revision="codex-candidate-set-fixture-v1",
                run_id="codex-provider-candidate-set",
                maximum_candidates_per_turn=3,
                reasoning_effort="xhigh",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            receipt = ProviderQualificationReceipt.load(receipt_path)
            evidence = EvidenceStore.open(evidence_root)
            audit = evidence.audit_run("codex-provider-candidate-set")
            observed = next(
                event
                for event in evidence.replay_events(audit.run_id)
                if event["kind"] == "provider_qualification_observed"
            )["payload"]

            self.assertTrue(audit.archive_integrity)
            self.assertEqual(
                audit.endpoint["submission_contract"],
                "candidate_set_envelope_v1",
            )
            self.assertEqual(
                audit.endpoint["arms_qualified"],
                ["open_cake", "direct_cuda"],
            )
            self.assertEqual(set(observed["arms"]), {"open_cake", "direct_cuda"})
            invocation_ref = next(
                item
                for item in observed["objects"]
                if item["role"] == "open_cake_initial_invocation"
            )
            invocation = json.loads(evidence.read_object(invocation_ref))
            self.assertIn('model_reasoning_effort="xhigh"', invocation["argv"])
            self.assertNotEqual(
                observed["arms"]["open_cake"]["thread_id"],
                observed["arms"]["direct_cuda"]["thread_id"],
            )
            for arm in ("open_cake", "direct_cuda"):
                arm_observation = observed["arms"][arm]
                self.assertEqual(len(arm_observation["initial_candidate_sha256s"]), 3)
                self.assertEqual(len(arm_observation["resumed_candidate_sha256s"]), 3)
                envelope = json.loads(
                    (root / "workspace" / arm / "candidate-set.json").read_text()
                )
                self.assertEqual(envelope["arm"], arm)
                self.assertEqual(len(envelope["candidates"]), 3)
            self.assertEqual(receipt.scope, "live_two_turn_current_provider")

    def test_ralph_qualification_exposes_only_task_agents_and_candidate_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "codex"
            self._write_provider(executable)
            completed, receipt_path, _, evidence_root = self._run_qualification(
                root,
                executable,
                provider_revision="codex-ralph-fixture-v1",
                run_id="codex-provider-ralph",
                maximum_candidates_per_turn=2,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            receipt = ProviderQualificationReceipt.load(receipt_path)
            self.assertTrue(receipt.qualified)
            workspace = root / "workspace"
            for arm in ("open_cake", "direct_cuda"):
                self.assertEqual(
                    {path.name for path in (workspace / arm).iterdir()},
                    {"TASK.md", "AGENTS.md", "candidate-set.json"},
                )
                self.assertEqual(
                    (workspace / arm / "TASK.md").stat().st_mode & 0o222,
                    0,
                )
            evidence = EvidenceStore.open(evidence_root)
            audit = evidence.audit_run("codex-provider-ralph")
            self.assertTrue(audit.archive_integrity)
            self.assertEqual(audit.endpoint_observation, "qualified")

    def test_incomplete_two_turn_observation_cannot_issue_a_live_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "codex"
            self._write_provider(executable, break_resumed_thread=True)
            completed, receipt_path, anchor_path, evidence_root = self._run_qualification(
                root,
                executable,
                provider_revision="codex-broken-fixture-v1",
                run_id="codex-provider-broken-thread",
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(receipt_path.exists())
            audit = EvidenceStore.open(evidence_root).audit_run("codex-provider-broken-thread")
            anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
            self.assertTrue(audit.archive_integrity)
            self.assertEqual(audit.protocol_adherence, "provider_fault")
            self.assertEqual(audit.endpoint_observation, "missing")
            self.assertIsNone(anchor["qualification_receipt_sha256"])
            self.assertEqual(anchor["terminal_seal_sha256"], audit.terminal_seal_sha256)

    def test_zero_token_turns_cannot_issue_a_usage_qualified_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "codex"
            self._write_provider(executable, zero_usage=True)
            completed, receipt_path, anchor_path, evidence_root = self._run_qualification(
                root,
                executable,
                provider_revision="codex-zero-usage-fixture-v1",
                run_id="codex-provider-zero-usage",
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(receipt_path.exists())
            audit = EvidenceStore.open(evidence_root).audit_run("codex-provider-zero-usage")
            anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
            self.assertTrue(audit.archive_integrity)
            self.assertEqual(audit.protocol_adherence, "provider_fault")
            self.assertIsNone(anchor["qualification_receipt_sha256"])
            self.assertEqual(anchor["terminal_seal_sha256"], audit.terminal_seal_sha256)

    def test_provider_process_failure_retains_stderr_in_the_terminal_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "codex"
            self._write_provider(executable, exit_nonzero=True)
            completed, receipt_path, anchor_path, evidence_root = self._run_qualification(
                root,
                executable,
                provider_revision="codex-process-failure-fixture-v1",
                run_id="codex-provider-process-failure",
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(receipt_path.exists())
            evidence = EvidenceStore.open(evidence_root)
            audit = evidence.audit_run("codex-provider-process-failure")
            anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
            self.assertTrue(audit.archive_integrity)
            self.assertIsNone(anchor["qualification_receipt_sha256"])
            self.assertEqual(anchor["terminal_seal_sha256"], audit.terminal_seal_sha256)
            failure = next(
                event
                for event in evidence.replay_events(audit.run_id)
                if event["kind"] == "provider_qualification_failed"
            )
            references = {
                item["role"]: item for item in failure["payload"]["objects"]
            }
            self.assertIn(
                b"qualification provider failure",
                evidence.read_object(references["provider_stderr"]),
            )

    def test_existing_anchor_blocks_qualification_before_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "codex"
            self._write_provider(executable)
            anchor_path = root / "provider-qualification-anchor.json"
            anchor_path.write_bytes(b"preexisting anchor\n")

            completed, receipt_path, observed_anchor, evidence_root = (
                self._run_qualification(
                    root,
                    executable,
                    provider_revision="codex-create-only-fixture-v1",
                    run_id="codex-provider-create-only",
                )
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(observed_anchor, anchor_path)
            self.assertEqual(anchor_path.read_bytes(), b"preexisting anchor\n")
            self.assertFalse(receipt_path.exists())
            self.assertFalse(evidence_root.exists())
            self.assertFalse((root / "workspace").exists())


if __name__ == "__main__":
    unittest.main()
