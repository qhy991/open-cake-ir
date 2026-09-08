"""Public ownership and lazy boundary contracts for the Lab facade."""
from __future__ import annotations

import ast
import copy
import importlib
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from open_cake_ir import lab as public
from open_cake_ir.lab import contracts, core, replay
from open_cake_ir.lab._policies import _MATCHED_RALPH_EVIDENCE_POLICY_V1
from open_cake_ir.lab.checkpoints import project_checkpoints

ROOT = Path(__file__).resolve().parents[2]


class LabCoreArchitectureTests(unittest.TestCase):
    def make_lab(self):
        callbacks = {name: Mock(side_effect=AssertionError(name + " was called eagerly")) for name in (
            "workload_loader", "prepare_schedule", "validate_authoring", "manifest_parser", "clock"
        )}
        return public.Lab(ROOT, **callbacks), callbacks

    def test_public_records_have_one_definition_and_facade_keeps_six_fields(self):
        for name in (
            "StudyContract", "CampaignLock", "CampaignRef", "StudyReport", "AnalysisInclusion",
            "TurnRequest", "RunProvider", "RunEvaluator",
        ):
            with self.subTest(record=name):
                value = getattr(public, name)
                self.assertIs(value, getattr(contracts, name))
                self.assertEqual(value.__module__, "open_cake_ir.lab.contracts")
        instance, callbacks = self.make_lab()
        self.assertEqual(set(vars(instance)), {
            "_root", "_clock", "_load_workload", "_prepare_schedule", "_validate_authoring", "_parse_manifest"
        })
        for callback in callbacks.values():
            callback.assert_not_called()
        for old_helper in ("_empirical_filter", "_replay_launchable_candidate", "_resolve_compiler_reference"):
            self.assertFalse(hasattr(core, old_helper))
        environments = importlib.import_module("open_cake_ir.lab.environments")
        for moved in ("_EMPIRICAL_SELECTION", "_EmpiricalSelection", "_empirical_context"):
            self.assertFalse(hasattr(environments, moved))

    def test_invalid_evidence_path_precedes_dependency_and_writer_side_effects(self):
        instance, callbacks = self.make_lab()
        with patch("open_cake_ir.lab.execution.EvidenceStore.create") as create:
            with self.assertRaisesRegex(ValueError, "outside|checkout"):
                instance.execute(object(), ROOT / "forbidden-evidence", provider=object(),
                                 environments={}, evaluator=object())
        create.assert_not_called()
        for callback in callbacks.values():
            callback.assert_not_called()

    def test_zero_provider_fault_returns_before_task_and_empirical_resolution(self):
        instance, callbacks = self.make_lab()
        audit = SimpleNamespace(run_id="open_cake-1", protocol_adherence="harness_fault",
                                endpoint_observation="missing", endpoint=None)
        from tests.contracts._executor_fixture import compiler_reference
        lock = SimpleNamespace(run_order=(audit.run_id,), analysis_plan={}, document={
            "compiler_revision": compiler_reference(ROOT),
            "resolved_inputs": {
                "evidence_policy": copy.deepcopy(_MATCHED_RALPH_EVIDENCE_POLICY_V1),
                "budget": {"checkpoints": [80000]},
                "arm_environments": {"open_cake": {"candidate_selection": "must not resolve"}},
            },
            "execution": {"executor_revision": "must not load"},
        })
        from open_cake_ir.lab.ralph import RalphBudget, RalphController
        budget = {"limit": 80000, "maximum_turns": 2, "maximum_candidates_per_turn": 1,
                  "wall_time_seconds": 30, "active_authoring_time_seconds": 15,
                  "evaluation_limits": {"search": 2, "confirmatory": 2, "attribution": 2}, "checkpoints": [80000]}
        lock.document["resolved_inputs"]["budget"] = budget
        ralph = RalphController(RalphBudget.from_mapping(budget), searches_per_turn=1,
                                profile_each_search_survivor=False, clock=lambda: 0.0)
        events = [
            {"kind": "run_started", "payload": {"sequence": 1, "assigned_arm": "open_cake",
                "automatic_retries": 0, "replacement_run": False}},
            {"kind": "run_fault", "payload": {"fault": "harness_fault", "exception_type": "RuntimeError",
                "turn": 1, "stage": "provider", "terminal_provider_tokens": 0,
                "provider_usage": {"status": "unavailable", "provider_tokens": None},
                "terminal_provider_tokens_scope": "known_subtotal"}},
            {"kind": "checkpoints_projected", "payload": {
                "checkpoints": [asdict(item) for item in project_checkpoints(
                    turns=(), checkpoints=[80000], terminal_provider_tokens=0)],
                "ralph": dict(ralph.state_card(turn=1, cumulative_provider_tokens=0,
                                              feedback={"kind": "initial"}, terminal_reason="harness_fault")),
            }},
            {"kind": "run_terminal", "payload": {"protocol_adherence": "harness_fault",
                "endpoint_observation": "missing", "endpoint": None}},
        ]
        evidence = SimpleNamespace(replay_events=Mock(return_value=events))
        from open_cake_ir.lab.bindings import load_compiler_reference
        with patch("open_cake_ir.lab.bindings.load_compiler_reference", wraps=load_compiler_reference) as compiler_dependency, \
             patch.object(public.Lab, "task_package", side_effect=AssertionError("task package eager")) as package, \
             patch.object(replay.ExecutorRevision, "load_reference", side_effect=AssertionError("Executor eager")) as executor, \
             patch.object(replay, "_EmpiricalSelection", side_effect=AssertionError("empirical model eager")) as selection:
            self.assertTrue(instance._replay_matched_run(evidence, audit, lock))
        compiler_dependency.assert_called_once_with(ROOT, lock.document["compiler_revision"], "replay.compiler_revision")
        package.assert_not_called()
        executor.assert_not_called()
        selection.assert_not_called()
        for callback in callbacks.values():
            callback.assert_not_called()

    def test_reporting_receives_bound_callbacks_without_eager_audit_or_replay(self):
        instance, callbacks = self.make_lab()
        campaign = object()
        report = object()
        with patch("open_cake_ir.lab.reporting.audit_campaign", return_value=report) as aggregate:
            self.assertIs(instance.audit(campaign), report)
        self.assertEqual(aggregate.call_args.args, (campaign,))
        replay_run = aggregate.call_args.kwargs["replay_run"]
        self.assertIs(replay_run.__self__, instance)
        self.assertIs(replay_run.__func__, public.Lab._replay_matched_run)
        with patch("open_cake_ir.lab.reporting.threshold_view", return_value={}) as threshold, \
             patch.object(public.Lab, "audit", side_effect=AssertionError("audit eager")) as audit:
            self.assertEqual(instance.threshold_view(campaign, 1.0), {})
            self.assertIs(threshold.call_args.kwargs["audit_campaign"], audit)
        audit.assert_not_called()
        for callback in callbacks.values():
            callback.assert_not_called()

    def test_explicit_runtime_owner_imports_form_an_acyclic_graph(self):
        owners = {"core", "contracts", "_documents", "_policies", "bindings", "preflight", "execution",
                  "replay", "reporting", "selection", "archive", "environments",
                  "admission", "build", "candidate_filter", "evaluation_writer", "execution_admission",
                  "provider_documents", "provider_events", "provider_invocation", "providers",
                  "replay_artifacts", "replay_attempts", "replay_candidates", "replay_outcomes",
                  "replay_provider", "replay_selection", "run_completion", "runtime", "runtime_config",
                  "endpoints", "evaluation_lifecycle", "diagnoses", "reference_access"}
        edges = {name: set() for name in owners}
        def runtime_imports(node):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
                return
            if isinstance(node, ast.ImportFrom):
                yield node
            for child in ast.iter_child_nodes(node):
                yield from runtime_imports(child)
        for name in owners:
            path = ROOT / "src/open_cake_ir/lab" / (name + ".py")
            for imported in runtime_imports(ast.parse(path.read_text())):
                self.assertFalse(any(alias.name == "*" for alias in imported.names), name)
                if imported.level == 1:
                    targets = [imported.module.split(".")[0]] if imported.module else [alias.name for alias in imported.names]
                elif (imported.module or "").startswith("open_cake_ir.lab."):
                    targets = [imported.module.split(".")[2]]
                else:
                    targets = []
                edges[name].update(target for target in targets if target in owners)
        self.assertFalse(edges["selection"] & {"core", "contracts", "environments"})
        self.assertFalse(edges["replay"] & {"core", "execution", "preflight", "reporting"})
        self.assertFalse(edges["_documents"])
        for reader in ("replay", "replay_artifacts", "replay_attempts", "replay_candidates",
                       "replay_outcomes", "replay_provider", "replay_selection"):
            self.assertFalse(edges[reader] & {"execution", "evaluation_writer", "run_completion"})
        def visit(name, path):
            self.assertNotIn(name, path, " -> ".join((*path, name)))
            for dependency in edges[name]:
                visit(dependency, (*path, name))
        for name in owners:
            visit(name, ())


if __name__ == "__main__":
    unittest.main()
