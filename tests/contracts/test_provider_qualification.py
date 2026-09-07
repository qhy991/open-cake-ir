from __future__ import annotations

import json
import contextlib
import io
import subprocess
import sys
import tempfile
import textwrap
import unittest
from hashlib import sha256
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.evidence import EvidenceStore  # noqa: E402
from open_cake_ir.lab import ProviderQualificationReceipt  # noqa: E402
from tools import qualify_codex_provider as qualifier  # noqa: E402


class ProviderQualificationContractTests(unittest.TestCase):
    def _write_provider(
        self,
        path: Path,
        *,
        break_resumed_thread: bool = False,
        zero_usage: bool = False,
        exit_nonzero: bool = False,
        tool_rich: bool = False,
        startup_error: bool = False,
        wrong_resumed_turn: bool = False,
        mutate_task: bool = False,
        mutate_helper: bool = False,
        pretty_submission: bool = False,
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
                projection = json.loads(prompt.split("\\n\\n", 1)[1])
                if projection["kind"] != "task_agents_ralph_v1":
                    raise SystemExit(38)
                for field, name in (("task_markdown", "TASK.md"), ("agents_markdown", "AGENTS.md")):
                    if projection[field].encode() != Path(name).read_bytes():
                        raise SystemExit(39)
                plan_line = next(line for line in projection["task_markdown"].splitlines()
                                 if line.startswith("QUALIFICATION_PLAN_JSON="))
                plan = json.loads(plan_line.split("=", 1)[1])
                candidate = Path(plan["candidate_path"])
                arm = projection["arm"]
                thread_id = (
                    "11234567-89ab-cdef-0123-456789abcdef"
                    if arm != "open_cake"
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
                schema_path = Path(arguments[arguments.index("--output-schema") + 1])
                allowed_arms = json.loads(schema_path.read_text())["properties"]["arm"]["enum"]
                if arm not in allowed_arms:
                    raise SystemExit(37)
                turn = 2 if resumed else 1
                if projection["state_card"] != {{"turn": turn}}:
                    raise SystemExit(40)
                change = "update" if resumed else "add"
                if candidate.exists() is not resumed:
                    raise SystemExit(35)
                selected_turn = 1 if resumed and {wrong_resumed_turn!r} else turn
                declared = next(item for item in plan["turns"] if item["turn"] == selected_turn)
                expected = declared["submission"]
                candidate.write_text(
                    (json.dumps(expected, indent=2, ensure_ascii=False)
                     if {pretty_submission!r} else json.dumps(expected, sort_keys=True, separators=(",", ":"))) + "\\n",
                    encoding="utf-8",
                )
                if {mutate_helper!r} and not resumed:
                    Path(__file__).with_name("codex-code-mode-host").write_bytes(b"changed host")
                if {mutate_task!r} and not resumed:
                    task = Path("TASK.md")
                    task.chmod(0o644)
                    task.write_text(task.read_text() + "changed by provider\\n")
                    task.chmod(0o444)
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
                if {startup_error!r}:
                    events.insert(1, {{'type':'item.completed', 'item':{{'id':'startup-error',
                        'type':'error', 'message':'Unable to start Code Mode without its host'}}}})
                for event in events:
                    print(json.dumps(event, separators=(",", ":")))
                """
            ),
            encoding="utf-8",
        )
        path.chmod(0o700)
        helper = path.with_name("codex-code-mode-host")
        helper.write_bytes(b"CPU Code Mode host fixture")
        helper.chmod(0o700)

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
        output_schema: Path | None = None,
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
                    output_schema or ROOT
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
            authority = json.loads((evidence_root / "runs" / audit.run_id / "authority.json").read_bytes())["authority"]
            helper = executable.with_name("codex-code-mode-host").resolve()
            self.assertEqual(authority["code_mode_host"], {
                "path": str(helper), "sha256": sha256(helper.read_bytes()).hexdigest(),
            })
            self.assertEqual(authority["web_search"], "disabled")
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

    def test_helper_drift_between_turns_preserves_failure_and_issues_no_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "codex"
            self._write_provider(executable, mutate_helper=True)
            completed, receipt_path, anchor_path, evidence_root = self._run_qualification(
                root, executable, provider_revision="cpu-helper-drift", run_id="cpu-helper-drift",
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(receipt_path.exists())
            self.assertIn("Code Mode host differs from the bound runtime", completed.stderr.decode())
            audit = EvidenceStore.open(evidence_root).audit_run("cpu-helper-drift")
            self.assertTrue(audit.archive_integrity)
            self.assertEqual(audit.protocol_adherence, "provider_fault")
            self.assertIsNone(json.loads(anchor_path.read_bytes())["qualification_receipt_sha256"])

    def test_candidate_set_qualification_covers_both_arm_projections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "codex"
            self._write_provider(executable, pretty_submission=True)
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

    def test_qualification_archives_the_captured_submission_after_path_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "codex"
            self._write_provider(executable, pretty_submission=True)
            captured = {}
            original_execute = qualifier.CodexProviderAdapter.execute

            def overwrite_after_capture(adapter, invocation, **kwargs):
                turn = original_execute(adapter, invocation, **kwargs)
                phase = "initial" if kwargs["expected_change"] == "add" else "resumed"
                captured[(kwargs["arm"], phase)] = turn.raw_submission
                kwargs["candidate_path"].write_bytes(b"later file contents")
                return turn

            arguments = [
                "qualify_codex_provider.py", "--executable", str(executable),
                "--provider-revision", "captured-submission-fixture",
                "--output-schema", str(ROOT / "contracts/providers/codex-turn-output-schema-v1.json"),
                "--workspace", str(root / "workspace"),
                "--receipt-output", str(root / "receipt.json"),
                "--anchor-output", str(root / "anchor.json"),
                "--evidence-root", str(root / "evidence"),
                "--run-id", "captured-submission", "--reasoning-effort", "max",
            ]
            with mock.patch.object(sys, "argv", arguments), mock.patch.object(
                qualifier.CodexProviderAdapter, "execute", overwrite_after_capture
            ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(qualifier.main(), 0)
            evidence = EvidenceStore.open(root / "evidence")
            observed = next(event["payload"] for event in evidence.replay_events("captured-submission")
                            if event["kind"] == "provider_qualification_observed")
            objects = {item["role"]: item for item in observed["objects"]}
            self.assertEqual(len(captured), 4)
            for (arm, phase), raw in captured.items():
                with self.subTest(arm=arm, phase=phase):
                    self.assertIn(b"\n  ", raw)
                    self.assertEqual(evidence.read_object(objects[f"{arm}_{phase}_submission_envelope"]), raw)
                    self.assertEqual((root / "workspace" / arm / "candidate-set.json").read_bytes(),
                                     b"later file contents")
                self.assertNotEqual(captured[(arm, "initial")], captured[(arm, "resumed")])

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
            observed = next(event["payload"] for event in evidence.replay_events(audit.run_id)
                            if event["kind"] == "provider_qualification_observed")
            objects = {item["role"]: item for item in observed["objects"]}
            for arm in ("open_cake", "direct_cuda"):
                projections = []
                for phase, turn in (("initial", 1), ("resumed", 2)):
                    raw = evidence.read_object(objects[f"{arm}_{phase}_task_projection"])
                    invocation = json.loads(evidence.read_object(objects[f"{arm}_{phase}_invocation"]))
                    delivered = invocation["argv"][-1].split("\n\n", 1)[1].encode()
                    self.assertEqual(delivered, raw)
                    projection = json.loads(raw)
                    self.assertEqual(projection["state_card"], {"turn": turn})
                    self.assertEqual(projection["task_markdown"].encode(),
                                     (workspace / arm / "TASK.md").read_bytes())
                    self.assertEqual(projection["agents_markdown"].encode(),
                                     (workspace / arm / "AGENTS.md").read_bytes())
                    plan_line = next(line for line in projection["task_markdown"].splitlines()
                                     if line.startswith("QUALIFICATION_PLAN_JSON="))
                    plan = json.loads(plan_line.split("=", 1)[1])
                    self.assertEqual([item["turn"] for item in plan["turns"]], [1, 2])
                    envelope = json.loads(evidence.read_object(
                        objects[f"{arm}_{phase}_submission_envelope"]))
                    self.assertEqual(envelope, plan["turns"][turn - 1]["submission"])
                    projections.append(projection)
                self.assertEqual(
                    {key: value for key, value in projections[0].items() if key != "state_card"},
                    {key: value for key, value in projections[1].items() if key != "state_card"},
                )

    def test_qualification_rejects_the_wrong_task_plan_turn(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "codex"
            self._write_provider(executable, wrong_resumed_turn=True)
            completed, receipt_path, _, evidence_root = self._run_qualification(
                root, executable, provider_revision="wrong-task-turn-fixture",
                run_id="wrong-task-turn",
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(receipt_path.exists())
            audit = EvidenceStore.open(evidence_root).audit_run("wrong-task-turn")
            self.assertEqual(audit.endpoint_observation, "missing")
            self.assertEqual(audit.protocol_adherence, "provider_fault")

    def test_qualification_rejects_task_mutation_before_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "codex"
            self._write_provider(executable, mutate_task=True)
            completed, receipt_path, _, evidence_root = self._run_qualification(
                root, executable, provider_revision="task-mutation-fixture",
                run_id="task-mutation",
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(b"Ralph task file TASK.md custody differs", completed.stderr)
            self.assertFalse(receipt_path.exists())
            candidate = json.loads((root / "workspace/open_cake/candidate-set.json").read_text())
            self.assertEqual(candidate["candidates"][0]["qualification_turn"], 1)
            audit = EvidenceStore.open(evidence_root).audit_run("task-mutation")
            self.assertEqual(audit.endpoint_observation, "missing")
            self.assertEqual(audit.protocol_adherence, "provider_fault")

    def test_native_triton_schema_qualifies_its_actual_pair_through_ralph(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "codex"
            self._write_provider(executable)
            completed, receipt_path, _, evidence_root = self._run_qualification(
                root, executable, provider_revision="native-ralph-fixture",
                run_id="native-ralph", maximum_candidates_per_turn=2,
                output_schema=ROOT / "contracts/providers/codex-triton-optimization-output-schema-v1.json",
            )
            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            self.assertTrue(ProviderQualificationReceipt.load(receipt_path).qualified)
            evidence = EvidenceStore.open(evidence_root)
            observed = next(event["payload"] for event in evidence.replay_events("native-ralph")
                            if event["kind"] == "provider_qualification_observed")
            self.assertEqual(set(observed["arms"]), {"open_cake", "native_triton"})
            self.assertNotEqual(observed["arms"]["open_cake"]["thread_id"],
                                observed["arms"]["native_triton"]["thread_id"])
            envelope = json.loads((root / "workspace/native_triton/candidate-set.json").read_text())
            self.assertEqual(envelope["arm"], "native_triton")
            self.assertEqual(len(envelope["candidates"]), 2)
            for member in envelope["candidates"]:
                self.assertEqual(set(member), {"kernel_source", "compile_constants", "compile_options", "grid"})
                self.assertIn("def qualification_2_", member["kernel_source"])
            audit = evidence.audit_run("native-ralph")
            self.assertTrue(audit.archive_integrity)
            self.assertEqual(audit.endpoint["arms_qualified"], ["open_cake", "native_triton"])
            for phase, turn in (("initial", 1), ("resumed", 2)):
                refs = [item for item in observed["objects"]
                        if item["role"].startswith(f"native_triton_{phase}_candidate_")]
                self.assertEqual(len(refs), 2)
                for index, reference in enumerate(refs):
                    self.assertEqual(reference["media_type"], "application/json")
                    member = json.loads(evidence.read_object(reference))
                    self.assertEqual(set(member), {"kernel_source", "compile_constants", "compile_options", "grid"})
                    self.assertIn(f"def qualification_{turn}_{index}", member["kernel_source"])

    def test_unsupported_schema_pair_is_rejected_before_creating_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "codex"
            self._write_provider(executable)
            schema = root / "unsupported-schema.json"
            schema.write_text(json.dumps({"properties": {"arm": {"enum": ["open_cake", "unknown"]}}}))
            completed, receipt_path, anchor_path, evidence_root = self._run_qualification(
                root, executable, provider_revision="unsupported-pair-fixture",
                run_id="unsupported-pair", output_schema=schema,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(b"supported arm pair", completed.stderr)
            for path in (receipt_path, anchor_path, evidence_root, root / "workspace"):
                self.assertFalse(path.exists())

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

    def test_startup_error_with_success_tail_cannot_issue_live_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / 'codex'
            self._write_provider(executable, startup_error=True)
            completed, receipt_path, anchor_path, evidence_root = self._run_qualification(
                root, executable, provider_revision='codex-startup-error-fixture',
                run_id='codex-provider-startup-error')
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(receipt_path.exists())
            audit = EvidenceStore.open(evidence_root).audit_run('codex-provider-startup-error')
            self.assertTrue(audit.archive_integrity)
            self.assertEqual(audit.protocol_adherence, 'provider_fault')
            self.assertEqual(audit.endpoint_observation, 'missing')
            self.assertIsNone(json.loads(anchor_path.read_bytes())['qualification_receipt_sha256'])

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
