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
from open_cake_ir.lab.replay.refusals import ReplayRefusal

from open_cake_ir.evidence import EvidenceStore
from open_cake_ir.lab.faults import ReportedProviderUsage, RunProtocolFault, ProviderBoundaryDeclarationFault
from open_cake_ir.lab.provider_events import reported_codex_usage
from open_cake_ir.tasks.runtime import TaskLab
from tests.contracts._contexts import enter_class_context
from tests.contracts import test_lab as consumers
from tests.contracts._executor_fixture import SemanticExecutorFixture

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
    def test_cumulative_codex_counter_is_differenced_and_claude_is_per_invocation(self):
        from open_cake_ir.lab.provider_events import provider_token_delta, reported_provider_usage
        from open_cake_ir.lab.claude import CLAUDE_EVENT_CONTRACTS
        provider = {"event_contract": "tool_rich_candidate_v1"}
        totals = (266187, 535235, 872461, 1270198, 1737381)
        previous, deltas = 0, []
        for total in totals:
            deltas.append(provider_token_delta(total, provider=provider, previous_tokens=previous))
            previous += deltas[-1]
        self.assertEqual(deltas, [266187, 269048, 337226, 397737, 467183])
        self.assertEqual(sum(deltas), 1737381)
        self.assertEqual(provider_token_delta(535235, provider=provider, previous_tokens=535235), 0)
        for native, previous in ((12, 13), (True, 0), (12, -1), (12, True)):
            with self.subTest(native=native, previous=previous), self.assertRaises(ValueError):
                provider_token_delta(native, provider=provider, previous_tokens=previous)
        self.assertIsNone(reported_provider_usage(codex_report(12), provider=provider,
                                                  previous_tokens=13))
        for contract in CLAUDE_EVENT_CONTRACTS:
            self.assertEqual(provider_token_delta(12, provider={"event_contract": contract},
                                                   previous_tokens=100), 12)

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
        from open_cake_ir.lab.replay.provider import replay_fault_usage
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
        with self.assertRaisesRegex(ReplayRefusal, "run_fault"):
            replay_fault_usage(payload=payload, evidence=evidence,
                          provider={**provider, "model": "another-model"})
        with self.assertRaisesRegex(ReplayRefusal, "run_fault"):
            replay_fault_usage(payload=payload, evidence=evidence, provider=provider,
                          expected_thread_id="00000000-0000-0000-0000-000000000001")

    def test_claude_fault_quota_attribution_is_rederived_from_retained_stdout(self):
        from open_cake_ir.lab.replay.provider import replay_fault_usage
        from tests.contracts.test_claude_provider import ClaudeProviderContracts, CLAUDE_EVENT_CONTRACT
        fixture = ClaudeProviderContracts()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        events = fixture.events()
        events[-1].pop("structured_output")  # Failed Turn; the notice remains retained.
        events[1:1] = [{"type": "rate_limit_event", "uuid": "11111111-2222-3333-4444-555555555555",
            "session_id": THREAD, "rate_limit_info": {"status": "allowed_warning",
                "resetsAt": 1789455600, "rateLimitType": "seven_day", "utilization": 0.99,
                "isUsingOverage": False, "surpassedThreshold": 0.75}}]
        raw = fixture.raw(events)
        quota = {"status": "allowed_warning", "rateLimitType": "seven_day",
                 "resetsAt": 1789455600, "utilization": 0.99, "surpassedThreshold": 0.75}
        provider = {"event_contract": CLAUDE_EVENT_CONTRACT, "model": "exact-requested-model"}
        evidence = SimpleNamespace(read_object=lambda _: raw)
        base = {"stage": "provider", "provider_usage": {"status": "observed",
                "event_contract": CLAUDE_EVENT_CONTRACT, "thread_id": THREAD, "provider_tokens": 205},
                "objects": [{"role": "provider_stdout"}]}
        # An attribution equal to the one the retained stdout rederives replays, and
        # evidence sealed before the field exists replays unchanged without it.
        self.assertEqual(replay_fault_usage(payload={**base, "observed_quota": quota},
            evidence=evidence, provider=provider), 205)
        self.assertEqual(replay_fault_usage(payload=base, evidence=evidence, provider=provider), 205)
        # A declared attribution the retained stdout does not support is refused.
        for tampered in ({"utilization": 0.5}, {"status": "allowed"}, {"resetsAt": 1789455601},
                         {"rateLimitType": "five_hour"}, {"surpassedThreshold": 0.9},
                         {"utilization": 0.99, "fabricated": 1}):
            with self.subTest(tamper=tampered):
                with self.assertRaisesRegex(ReplayRefusal, "run_fault"):
                    replay_fault_usage(
                    payload={**base, "observed_quota": {**quota, **tampered}},
                    evidence=evidence, provider=provider)
        # An attribution with no notice behind it in the stream is fabricated.
        bare = fixture.raw(fixture.events())
        with self.assertRaisesRegex(ReplayRefusal, "run_fault"):
            replay_fault_usage(payload={**base, "observed_quota": quota},
            evidence=SimpleNamespace(read_object=lambda _: bare), provider=provider)
        # The observed wall: the last notice is the rejection that killed the process,
        # and an attribution naming the earlier warning instead is refused.
        events[2:2] = [{"type": "rate_limit_event", "uuid": "11111111-2222-3333-4444-555555555555",
            "session_id": THREAD, "rate_limit_info": {"status": "rejected",
                "resetsAt": 1789455600, "rateLimitType": "seven_day",
                "overageStatus": "rejected", "overageDisabledReason": "org_level_disabled",
                "isUsingOverage": False}}]
        died = fixture.raw(events)
        rejected = {"status": "rejected", "rateLimitType": "seven_day", "resetsAt": 1789455600}
        evidence_died = SimpleNamespace(read_object=lambda _: died)
        self.assertEqual(replay_fault_usage(payload={**base, "observed_quota": rejected},
            evidence=evidence_died, provider=provider), 205)
        with self.assertRaisesRegex(ReplayRefusal, "run_fault"):
            replay_fault_usage(payload={**base, "observed_quota": quota},
            evidence=evidence_died, provider=provider)

    def test_absent_notice_at_a_claude_fault_is_the_recorded_observation(self):
        """F-2026-09-16-001: gateway-transport deaths carry no rate-limit notice at all.

        The absence itself is what the fault seam records and what replay
        rederives; evidence sealed while the field stayed absent still replays
        unchanged, and the marker cannot stand in for a notice in either
        direction.
        """
        from open_cake_ir.lab.replay.provider import replay_fault_usage
        from tests.contracts.test_claude_provider import ClaudeProviderContracts, CLAUDE_EVENT_CONTRACT
        fixture = ClaudeProviderContracts()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        events = fixture.events()
        events[-1].pop("structured_output")  # Failed Turn; no notice is carried either.
        bare = fixture.raw(events)
        noticed = fixture.events()
        noticed[-1].pop("structured_output")
        noticed[1:1] = [{"type": "rate_limit_event", "uuid": "11111111-2222-3333-4444-555555555555",
            "session_id": THREAD, "rate_limit_info": {"status": "allowed_warning",
                "resetsAt": 1789455600, "rateLimitType": "seven_day", "utilization": 0.99,
                "isUsingOverage": False, "surpassedThreshold": 0.75}}]
        quota = {"status": "allowed_warning", "rateLimitType": "seven_day",
                 "resetsAt": 1789455600, "utilization": 0.99, "surpassedThreshold": 0.75}
        provider = {"event_contract": CLAUDE_EVENT_CONTRACT, "model": "exact-requested-model"}
        base = {"stage": "provider", "provider_usage": {"status": "observed",
                "event_contract": CLAUDE_EVENT_CONTRACT, "thread_id": THREAD, "provider_tokens": 205},
                "objects": [{"role": "provider_stdout"}]}
        bare_evidence = SimpleNamespace(read_object=lambda _: bare)
        # The recorded absence replays against the notice-free stream behind it,
        # and evidence sealed before the distinction exists replays unchanged.
        self.assertEqual(replay_fault_usage(
            payload={**base, "observed_quota": {"observed": "no_notice"}},
            evidence=bare_evidence, provider=provider), 205)
        self.assertEqual(replay_fault_usage(payload=base, evidence=bare_evidence, provider=provider), 205)
        # The marker cannot ride a stream that does carry a notice, and a notice
        # cannot masquerade as the recorded absence.
        evidence_noticed = SimpleNamespace(read_object=lambda _: fixture.raw(noticed))
        with self.assertRaisesRegex(ReplayRefusal, "run_fault"):
            replay_fault_usage(
            payload={**base, "observed_quota": {"observed": "no_notice"}},
            evidence=evidence_noticed, provider=provider)
        with self.assertRaisesRegex(ReplayRefusal, "run_fault"):
            replay_fault_usage(payload={**base, "observed_quota": quota},
            evidence=bare_evidence, provider=provider)

    def test_replayed_usage_refuses_boolean_and_float_token_witnesses(self):
        from open_cake_ir.lab.replay.provider import replay_fault_usage
        for native_tokens, claimed_tokens in ((0, False), (1, True), (191499, 191499.0)):
            raw = codex_report(native_tokens)
            payload = {"stage": "provider", "provider_usage": {"status": "observed",
                "event_contract": CONTRACT, "thread_id": THREAD, "provider_tokens": claimed_tokens},
                "objects": [{"role": "provider_stdout"}]}
            with self.subTest(native=native_tokens, claimed=claimed_tokens):
                with self.assertRaisesRegex(ReplayRefusal, "run_fault"):
                    replay_fault_usage(payload=payload,
                    evidence=SimpleNamespace(read_object=lambda _: raw),
                    provider={"event_contract": CONTRACT})

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
        # Calling the consumer setup directly does not invoke its per-test setUp.
        # Keep one explicit CPU Executor at preflight, execution and replay rather
        # than falling through to whichever released host/closure is current.
        cls.executor_fixture = enter_class_context(cls, SemanticExecutorFixture())
        class RuntimeFixture(consumers.EmpiricalSelectionContractTests):
            pass
        cls.fixture_type = RuntimeFixture
        RuntimeFixture.setUpClass()
        cls.addClassCleanup(RuntimeFixture.tearDownClass)
        cls.root, cls.parent = RuntimeFixture.root, RuntimeFixture.parent
        cls.lab = TaskLab(cls.root, clock=lambda: 0.0)
        cls.lock = cls.lab.preflight(cls.root / "contracts/studies/matched-search-infrastructure-template.json")
        if cls.lock.document["execution"]["executor_revision"] != dict(cls.executor_fixture.revision(cls.root).reference):
            raise AssertionError("failed-provider consumer must use its explicit CPU Executor")

    def campaign(self, *, fault_turn=2, tokens=191499, unknown=False, mismatch=False, stage="provider",
                 returned_identity_refusal=False, missing_stdout=False, quota=None, boundary=False):
        class Provider(consumers.FakeProvider):
            def turn(inner, request):
                if request.turn == fault_turn and stage == "provider" and not returned_identity_refusal:
                    thread = request.thread_id or THREAD
                    raw = b"truncated stdout" if unknown else codex_report(request.cumulative_provider_tokens + tokens, thread)
                    witness = reported_codex_usage(raw, event_contract=CONTRACT,
                                                   expected_thread_id=request.thread_id)
                    if mismatch:
                        witness = ReportedProviderUsage(CONTRACT, thread, request.cumulative_provider_tokens + tokens + 1)
                    if boundary:
                        # F-2026-09-16-002: the boundary fault's own type at the
                        # provider seam; the harness keeps a Codex usage stream so
                        # the synthetic campaign's accounting stays replayable.
                        raise ProviderBoundaryDeclarationFault(
                            "Claude terminal declared candidate_written without a witnessed write",
                            artifact_payloads={"provider_stdout": raw, "provider_stderr": b""},
                            reported_usage=witness,
                            observed_quota={"observed": "no_notice"})
                    raise RunProtocolFault("provider_fault", "synthetic provider format fault",
                        artifact_payloads={} if missing_stdout else {"provider_stdout": raw},
                        reported_usage=None if missing_stdout else witness,
                        observed_quota=quota)
                result = super().turn(request)
                events = [json.loads(line) for line in result.raw_events.splitlines()]
                rejected_return = returned_identity_refusal and request.turn == fault_turn
                events[-1]["usage"] = ({"input_tokens": request.cumulative_provider_tokens + tokens, "output_tokens": 0} if rejected_return
                                        else {"input_tokens": request.cumulative_provider_tokens + 94700, "output_tokens": 58})
                raw = b"\n".join(json.dumps(event).encode() for event in events)
                return replace(result, provider_tokens=tokens if rejected_return else 94758, raw_events=raw,
                               raw_events_sha256="f" * 64 if rejected_return else sha256(raw).hexdigest())

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
                self.lock.document["workload"]["canonical_sha256"], project_root=self.root))
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

    def test_observed_quota_attribution_is_retained_and_replays(self):
        quota = {"status": "allowed_warning", "rateLimitType": "seven_day",
                 "resetsAt": 1789455600, "utilization": 0.99, "surpassedThreshold": 0.75}
        campaign, store, events = self.campaign(quota=quota)
        fault = next(event["payload"] for event in events if event["kind"] == "run_fault")
        self.assertEqual(fault["observed_quota"], quota)
        self.assertEqual(fault["provider_usage"]["provider_tokens"], 191499)
        self.assertTrue(self.lab.audit(campaign).semantic_replay_passed)

    def test_settled_checkpoint_outlives_a_budget_boundary_declaration_fault(self):
        """F-2026-09-16-002: the boundary fault converts only what already settled.

        The fault observation stays in the ledger; the terminal goes to the
        settled checkpoint with the conversion named on the terminal event and
        the natural budget stop reason. This synthetic campaign is
        Codex-contract, so semantic replay must refuse the converted marker
        here: the marker belongs to the Claude contract whose adapter can raise
        the fault, which is exactly what the refusal asserts.
        """
        limit = self.lock.document["resolved_inputs"]["budget"]["limit"]
        campaign, store, events = self.campaign(boundary=True, tokens=limit - 94758)
        fault = next(event["payload"] for event in events if event["kind"] == "run_fault")
        self.assertEqual(fault["fault"], "provider_fault")
        self.assertEqual(fault["exception_type"], "ProviderBoundaryDeclarationFault")
        self.assertEqual(fault["terminal_provider_tokens"], limit)
        state = events[-2]["payload"]["ralph"]
        self.assertEqual(state["terminal_reason"], "provider_token_limit")
        terminal = events[-1]["payload"]
        self.assertEqual(terminal["protocol_adherence"], "adhered")
        self.assertEqual(terminal["boundary_diagnostic"],
                         {"turn": 2, "stage": "provider",
                          "diagnostic": "candidate_write_declared_unwitnessed"})
        self.assertEqual(terminal["endpoint_observation"], "qualified")
        self.assertFalse(self.lab.audit(campaign).semantic_replay_passed)

    def test_boundary_declaration_without_a_settled_checkpoint_stays_a_fault(self):
        """The declaration check still fails closed with nothing settled."""
        campaign, store, events = self.campaign(boundary=True, fault_turn=1, tokens=500)
        fault = next(event["payload"] for event in events if event["kind"] == "run_fault")
        self.assertEqual((fault["fault"], fault["exception_type"]),
                         ("provider_fault", "ProviderBoundaryDeclarationFault"))
        terminal = events[-1]["payload"]
        self.assertNotIn("boundary_diagnostic", terminal)
        self.assertEqual(terminal["protocol_adherence"], "provider_fault")
        self.assertEqual(terminal["endpoint_observation"], "missing")
        self.assertTrue(self.lab.audit(campaign).semantic_replay_passed)

    def test_returned_turn_archive_refusal_counts_usage_without_committing_completion(self):
        campaign, store, events = self.campaign(returned_identity_refusal=True)
        fault = next(event["payload"] for event in events if event["kind"] == "run_fault")
        self.assertEqual((fault["fault"], fault["exception_type"], fault["stage"]),
                         ("provider_fault", "ValueError", "provider"))
        # A type alone cannot be acted on; the harness's own account is retained.
        self.assertIsInstance(fault["exception_message"], str)
        self.assertTrue(fault["exception_message"])
        self.assertLessEqual(len(fault["exception_message"]), 2048)
        self.assertEqual(fault["terminal_provider_tokens"], 286257)
        self.assertEqual(fault["provider_usage"]["provider_tokens"], 191499)
        self.assertNotIn("provider_usage_witness_mismatch", fault)
        self.assertEqual([event["payload"]["turn"] for event in events if event["kind"] == "provider_turn_completed"], [1])
        self.assertFalse(any(event["payload"].get("turn") == 2 and event["kind"] != "run_fault" for event in events))
        self.assertTrue(self.lab.audit(campaign).semantic_replay_passed)

    def test_retained_fault_message_is_bounded_and_optional_for_older_evidence(self):
        from open_cake_ir.lab.replay import _fault_message_is_closed
        # Evidence sealed before the field replays unchanged.
        self.assertTrue(_fault_message_is_closed({"fault": "broker_fault"}))
        for message in (None, "evaluator result metadata differs", "x" * 2048):
            with self.subTest(message=type(message)):
                self.assertTrue(_fault_message_is_closed({"exception_message": message}))
        for message in ("", "x" * 2049, 7, ["text"]):
            with self.subTest(message=message if isinstance(message, int) else type(message)):
                self.assertFalse(_fault_message_is_closed({"exception_message": message}))

    def test_resumed_fault_without_stdout_does_not_reuse_the_prior_turn_raw_usage(self):
        campaign, store, events = self.campaign(missing_stdout=True)
        fault = next(event["payload"] for event in events if event["kind"] == "run_fault")
        self.assertEqual(fault["terminal_provider_tokens"], 94758)
        self.assertEqual(fault["provider_usage"], {"status": "unavailable", "provider_tokens": None})
        self.assertEqual(fault["terminal_provider_tokens_scope"], "known_subtotal")
        self.assertNotIn("objects", fault)
        self.assertEqual(events[-2]["payload"]["ralph"]["cumulative_provider_tokens"], 94758)
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
