"""CPU failed-provider accounting through real execution, retention and replay owners."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from open_cake_ir.evidence import EvidenceStore
from open_cake_ir.lab.faults import ReportedProviderUsage, RunProtocolFault
from open_cake_ir.lab.provider_events import reported_codex_usage
from open_cake_ir.tasks.runtime import TaskLab
from tests.contracts import test_lab as consumers

THREAD = "01234567-89ab-cdef-0123-456789abcdef"
CONTRACT = "closed_file_change_v1"


def codex_report(tokens, thread=THREAD):
    return b"\n".join(json.dumps(event).encode() for event in (
        {"type": "thread.started", "thread_id": thread}, {"type": "turn.started"},
        # A complete usage envelope does not imply an accepted terminal/candidate.
        {"type": "item.completed", "item": {"id": "bad-terminal", "type": "agent_message", "text": "invalid candidate"}},
        {"type": "turn.completed", "usage": {"input_tokens": tokens, "output_tokens": 0}},
    ))


class ReportedProviderUsageTests(unittest.TestCase):
    def test_usage_requires_complete_typed_native_envelope_but_not_candidate_acceptance(self):
        for tokens in (0, 191499):
            witness = reported_codex_usage(codex_report(tokens), event_contract=CONTRACT)
            self.assertEqual(witness, ReportedProviderUsage(CONTRACT, THREAD, tokens))
        for raw in (None, b"[" * 2000 + b"]" * 2000, b"", b"partial native output", codex_report(12).rsplit(b"\n", 1)[0],
                    codex_report(-1), codex_report(True)):
            with self.subTest(raw=raw):
                self.assertIsNone(reported_codex_usage(raw, event_contract=CONTRACT))
        self.assertIsNone(reported_codex_usage(codex_report(12), event_contract=CONTRACT,
                          expected_thread_id="00000000-0000-0000-0000-000000000001"))
        for bad in (True, -1, 1.5):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                ReportedProviderUsage(CONTRACT, THREAD, bad)


    def test_claude_fault_usage_flows_through_common_native_dispatch_and_replay(self):
        from open_cake_ir.lab.provider_events import reported_provider_usage
        from open_cake_ir.lab.replay_provider import replay_fault_usage
        from tests.contracts.test_claude_provider import ClaudeProviderContracts, CLAUDE_EVENT_CONTRACT
        fixture = ClaudeProviderContracts()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        events = fixture.events()
        events[-1].pop("structured_output")  # Failed Turn; valid reported usage remains observable.
        raw = fixture.raw(events)
        provider = {"event_contract": CLAUDE_EVENT_CONTRACT, "model": "exact-requested-model"}
        usage = reported_provider_usage(raw, provider=provider)
        self.assertEqual(usage, ReportedProviderUsage(CLAUDE_EVENT_CONTRACT, THREAD, 205))
        payload = {"stage": "provider", "provider_usage": {"status": "observed", **usage.document},
                   "objects": [{"role": "provider_stdout"}]}
        evidence = SimpleNamespace(read_object=lambda _: raw)
        self.assertEqual(replay_fault_usage(payload=payload, evidence=evidence, provider=provider), 205)
        self.assertIsNone(replay_fault_usage(payload=payload, evidence=evidence,
                          provider={**provider, "model": "another-model"}))
        self.assertIsNone(replay_fault_usage(payload=payload, evidence=evidence, provider=provider,
                          expected_thread_id="00000000-0000-0000-0000-000000000001"))

    def test_replayed_usage_refuses_boolean_and_float_token_witnesses(self):
        from open_cake_ir.lab.replay_provider import replay_fault_usage
        for native_tokens, claimed_tokens in ((0, False), (1, True), (191499, 191499.0)):
            raw = codex_report(native_tokens)
            payload = {"stage": "provider", "provider_usage": {"status": "observed",
                "event_contract": CONTRACT, "thread_id": THREAD, "provider_tokens": claimed_tokens},
                "objects": [{"role": "provider_stdout"}]}
            with self.subTest(native=native_tokens, claimed=claimed_tokens):
                self.assertIsNone(replay_fault_usage(payload=payload,
                    evidence=SimpleNamespace(read_object=lambda _: raw),
                    provider={"event_contract": CONTRACT}))

    def test_codex_adapter_retains_known_usage_on_format_and_process_failures(self):
        from open_cake_ir.lab.providers import CodexProviderAdapter, ProviderInvocation
        for exit_code in (0, 7):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary).resolve()
                raw = codex_report(150119)
                script = directory / "native-event-fixture.py"
                script.write_text("import sys\nsys.stdout.buffer.write(" + repr(raw) + ")\nraise SystemExit(" + str(exit_code) + ")\n")
                invocation = ProviderInvocation((sys.executable, str(script)), directory,
                    "workspace-write", "CPU-event-fixture", ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"), None)
                with self.assertRaises(RunProtocolFault) as raised:
                    CodexProviderAdapter().execute(invocation, candidate_path=directory / "candidate-set.json",
                        expected_change="add", expected_terminal_message='{"candidate_written":true}',
                        event_contract=CONTRACT, arm="open_cake")
                self.assertEqual(raised.exception.reported_usage,
                                 ReportedProviderUsage(CONTRACT, THREAD, 150119))
                self.assertEqual(raised.exception.artifact_payloads["provider_stdout"], raw)


    def test_post_adapter_workspace_refusal_keeps_reported_native_usage(self):
        from open_cake_ir.lab.providers import QualifiedRunProvider, ProviderQualificationReceipt
        from open_cake_ir.lab.task_package import TaskPackage, materialize_task_package
        from tests.contracts.test_provider_runtime import ProviderRuntimeContractTests
        for kind in ("extra file", "task contamination"):
            with self.subTest(kind=kind):
                fixture = ProviderRuntimeContractTests()
                fixture.setUp()
                self.addCleanup(fixture.doCleanups)
                workspace = fixture.root / "actor"
                workspace.mkdir()
                package = TaskPackage("open_cake-1", "open_cake", "# CPU test\n", "# CPU test\n")
                materialize_task_package(workspace, package)
                builder = fixture.builder(workspace=workspace)
                # A CPU authority double only; this receipt is never published.
                receipt = ProviderQualificationReceipt(builder.provider_revision,
                    sha256(builder.executable.read_bytes()).hexdigest(),
                    sha256(json.dumps(builder.configuration, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                    True, True, True, True, "live_two_turn_current_provider")
                raw = codex_report(150119)
                def execute(invocation, *, candidate_path, **kwargs):
                    candidate_path.write_text("{}")
                    if kind == "extra file":
                        (workspace / "unexpected.txt").write_text("CPU fixture")
                    else:
                        target = workspace / "TASK.md"
                        target.chmod(0o644)
                        target.write_text("deliberate CPU fixture contamination")
                    return SimpleNamespace(raw_events=raw)
                provider = QualifiedRunProvider(qualification=receipt, builders={"open_cake-1": builder},
                    task_packages={"open_cake-1": package}, adapter=SimpleNamespace(execute=execute))
                with self.assertRaises(RunProtocolFault) as raised:
                    provider.turn(SimpleNamespace(run_id="open_cake-1", arm="open_cake", turn=1,
                        maximum_candidates_per_turn=1, thread_id=None,
                        state_card={"schema_version": 1, "kind": "ralph_state_v1", "iteration": 1}))
                self.assertEqual(raised.exception.reported_usage, ReportedProviderUsage(CONTRACT, THREAD, 150119))
                self.assertEqual(raised.exception.artifact_payloads["provider_stdout"], raw)


class FailedProviderConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the existing prospective source closure and explicit synthetic CUDA
        # host. No released descriptor or real host fact is altered or admitted.
        class RuntimeFixture(consumers.EmpiricalSelectionContractTests):
            pass
        cls.fixture_type = RuntimeFixture
        RuntimeFixture.setUpClass()
        cls.root, cls.parent = RuntimeFixture.root, RuntimeFixture.parent
        cls.lab = TaskLab(cls.root, clock=lambda: 0.0)
        cls.lock = cls.lab.preflight(cls.root / "contracts/studies/matched-search-infrastructure-template.json")

    @classmethod
    def tearDownClass(cls):
        cls.fixture_type.tearDownClass()

    def campaign(self, *, fault_turn=2, tokens=191499, unknown=False, mismatch=False, stage="provider"):
        class Provider(consumers.FakeProvider):
            def turn(inner, request):
                if request.turn == fault_turn and stage == "provider":
                    thread = request.thread_id or THREAD
                    raw = b"truncated stdout" if unknown else codex_report(tokens, thread)
                    witness = reported_codex_usage(raw, event_contract=CONTRACT,
                                                   expected_thread_id=request.thread_id)
                    if mismatch:
                        witness = ReportedProviderUsage(CONTRACT, thread, tokens + 1)
                    raise RunProtocolFault("provider_fault", "synthetic provider format fault",
                        artifact_payloads={"provider_stdout": raw}, reported_usage=witness)
                result = super().turn(request)
                events = [json.loads(line) for line in result.raw_events.splitlines()]
                events[-1]["usage"] = {"input_tokens": 94700, "output_tokens": 58}
                raw = b"\n".join(json.dumps(event).encode() for event in events)
                return replace(result, provider_tokens=94758, raw_events=raw,
                               raw_events_sha256=sha256(raw).hexdigest())

        class Environment(consumers.FakeEnvironment):
            def build(inner, submission):
                if stage == "environment":
                    raise RunProtocolFault("harness_fault", "synthetic environment fault",
                        artifact_payloads={"provider_stdout": codex_report(tokens)},
                        reported_usage=ReportedProviderUsage(CONTRACT, THREAD, tokens))
                return super().build(submission)

        directory = Path(tempfile.mkdtemp(dir=self.parent))
        arms = self.lock.document["resolved_inputs"]["arm_environments"]
        protocol = self.lock.document["evaluation_protocol"]
        provider = Provider()
        campaign = consumers._execute(self.lab, self.lock, directory / "evidence", provider=provider,
            environments={name: Environment(name, arm) for name, arm in arms.items()},
            evaluator=consumers.FakeEvaluator(protocol,
                sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                self.lock.document["workload"]["canonical_sha256"]))
        store = EvidenceStore.open(campaign.evidence_root)
        events = store.replay_events("open_cake-1")
        return campaign, store, events

    def test_failed_resumed_invocation_is_counted_once_without_completed_turn_or_candidate(self):
        campaign, store, events = self.campaign()
        fault = next(event["payload"] for event in events if event["kind"] == "run_fault")
        self.assertEqual(fault["terminal_provider_tokens"], 94758 + 191499)
        self.assertEqual(fault["provider_usage"]["provider_tokens"], 191499)
        self.assertEqual(fault["provider_usage"]["status"], "observed")
        self.assertNotIn("terminal_provider_tokens_scope", fault)
        self.assertEqual([event["payload"]["turn"] for event in events if event["kind"] == "provider_turn_completed"], [1])
        self.assertFalse(any(event["payload"].get("turn") == 2 and event["kind"] != "run_fault" for event in events))
        state = events[-2]["payload"]["ralph"]
        self.assertEqual(state["cumulative_provider_tokens"], 286257)
        self.assertEqual(state["remaining"]["provider_tokens"], 0)
        self.assertEqual(state["terminal_reason"], "provider_fault")
        self.assertTrue(self.lab.audit(campaign).semantic_replay_passed)

    def test_zero_completed_turn_fault_uses_native_usage_and_independent_replay(self):
        campaign, store, events = self.campaign(fault_turn=1)
        self.assertEqual(events[1]["payload"]["terminal_provider_tokens"], 191499)
        self.assertEqual([event["kind"] for event in events], ["run_started", "run_fault", "checkpoints_projected", "run_terminal"])
        self.assertTrue(self.lab.audit(campaign).semantic_replay_passed)

    def test_observed_zero_is_distinct_from_unavailable_known_subtotal(self):
        for unknown in (False, True):
            campaign, store, events = self.campaign(fault_turn=1, tokens=0, unknown=unknown)
            fault = events[1]["payload"]
            self.assertEqual(fault["terminal_provider_tokens"], 0)
            self.assertEqual(fault["provider_usage"]["status"], "unavailable" if unknown else "observed")
            self.assertEqual(fault["provider_usage"]["provider_tokens"], None if unknown else 0)
            self.assertEqual(fault.get("terminal_provider_tokens_scope"), "known_subtotal" if unknown else None)
            self.assertTrue(self.lab.audit(campaign).semantic_replay_passed)

    def test_nonprovider_fault_cannot_add_usage_already_accounted_for_a_completed_turn(self):
        campaign, store, events = self.campaign(stage="environment")
        fault = next(event["payload"] for event in events if event["kind"] == "run_fault")
        self.assertEqual(fault["terminal_provider_tokens"], 94758)
        self.assertNotIn("provider_usage", fault)
        self.assertTrue(self.lab.audit(campaign).semantic_replay_passed)

    def test_inconsistent_adapter_witness_is_retained_but_reliable_native_usage_is_not_lost(self):
        campaign, store, events = self.campaign(mismatch=True)
        fault = next(event["payload"] for event in events if event["kind"] == "run_fault")
        self.assertEqual(fault["terminal_provider_tokens"], 286257)
        self.assertEqual(fault["provider_usage_witness_mismatch"]["provider_tokens"], 191500)
        self.assertTrue(self.lab.audit(campaign).semantic_replay_passed)

    def test_independent_replay_rejects_usage_and_checkpoint_tampering(self):
        campaign, store, events = self.campaign()
        audit = store.audit_run("open_cake-1")
        original = list(events)
        def replay(changed, read_object=None):
            proxy = SimpleNamespace(replay_events=lambda _: changed,
                                    read_object=read_object or store.read_object)
            return self.lab._replay_matched_run(proxy, audit, self.lock)
        mutations = {}
        for name in ("wrong usage", "wrong cumulative", "unavailable downgrade", "missing stdout", "wrong StateCard", "wrong remaining", "wrong checkpoint", "wrong stop"):
            changed = deepcopy(original)
            fault = next(e["payload"] for e in changed if e["kind"] == "run_fault")
            state = changed[-2]["payload"]["ralph"]
            if name == "wrong usage": fault["provider_usage"]["provider_tokens"] += 1
            elif name == "wrong cumulative": fault["terminal_provider_tokens"] -= 1
            elif name == "unavailable downgrade":
                fault["provider_usage"] = {"status": "unavailable", "provider_tokens": None}
                fault["terminal_provider_tokens_scope"] = "known_subtotal"
            elif name == "missing stdout": fault.pop("objects")
            elif name == "wrong StateCard": state["cumulative_provider_tokens"] = 94758
            elif name == "wrong remaining": state["remaining"]["provider_tokens"] = 1
            elif name == "wrong checkpoint": changed[-2]["payload"]["checkpoints"][-1]["state"] = "unreached"
            elif name == "wrong stop": state["terminal_reason"] = "provider_token_limit"
            mutations[name] = changed
        for name, changed in mutations.items():
            with self.subTest(tamper=name): self.assertFalse(replay(changed))
        fault = next(e["payload"] for e in original if e["kind"] == "run_fault")
        stdout_ref = next(item for item in fault["objects"] if item["role"] == "provider_stdout")
        def tampered(reference):
            return codex_report(191500, fault["provider_usage"]["thread_id"]) if reference == stdout_ref else store.read_object(reference)
        self.assertFalse(replay(original, tampered))


if __name__ == "__main__":
    unittest.main()
