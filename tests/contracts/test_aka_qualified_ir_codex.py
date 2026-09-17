from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.run_aka_qualified_ir_codex import (  # noqa: E402
    MODEL,
    PHASE0_CANARY_DERIVED_PARENT,
    PHASE0_CANARY_OPERATOR,
    PHASE0_CANARY_WORKLOAD,
    PHASE0_DISABLED_FEATURES,
    PHASE0_FAILURE_CATEGORIES,
    PHASE0_PERMISSION_PROFILE,
    PHASE0_RECEIPT_SCHEMA,
    Phase0Error,
    _phase0_assert_problem_copies,
    _phase0_assert_create_only,
    _phase0_codex_command,
    _phase0_failure_category,
    _phase0_model_receipt,
    _phase0_prompt,
    _phase0_result_invariants,
    _phase0_select_canary,
    _phase0_static_gate,
    parent_completion_schema,
    phase0_result_schema,
)


class AkaQualifiedIrCodexTests(unittest.TestCase):
    def test_parent_bridge_schema_is_strict_and_binds_source_coordinate(self) -> None:
        entry = {
            "parent_coordinate": "a" * 40 + ":path/to/data.jsonl:7",
            "record_field": "input",
        }
        schema = parent_completion_schema(entry)

        def assert_strict(node: object) -> None:
            if not isinstance(node, dict):
                return
            node_type = node.get("type")
            if node_type == "object":
                self.assertFalse(node.get("additionalProperties", True))
                properties = node.get("properties")
                self.assertIsInstance(properties, dict)
                self.assertEqual(set(node.get("required", [])), set(properties))
            if "const" in node:
                self.assertIn("type", node)
            for value in node.values():
                if isinstance(value, dict):
                    assert_strict(value)
                elif isinstance(value, list):
                    for item in value:
                        assert_strict(item)

        assert_strict(schema)
        original = schema["properties"]["original_parent"]["properties"]
        self.assertEqual(original["case_path"]["const"], entry["parent_coordinate"])
        self.assertEqual(original["record_field"]["const"], "input")

    @staticmethod
    def phase0_problem() -> dict[str, object]:
        return {
            "case_id": "case-a",
            "derived_parent_id": "parent-a",
            "artifact_refs": ["evidence/aka/bundles/a/sources/baseline/a.cu"],
            "provenance_refs": {"aka_git_commit": "a" * 40},
            "frozen_contract": {"workload": {"id": "n1"}},
            "assessment_scope": "fixed_instance",
        }

    @classmethod
    def phase0_result(cls) -> dict[str, object]:
        problem = cls.phase0_problem()
        return {
            "schema": "open-cake.aka-qualified-ir-result.v1",
            "case_id": problem["case_id"],
            "derived_parent_id": problem["derived_parent_id"],
            "artifact_refs": problem["artifact_refs"],
            "provenance_refs": problem["provenance_refs"],
            "frozen_contract": problem["frozen_contract"],
            "assessment_scope": problem["assessment_scope"],
            "owner": "schedule",
            "current_ir_expressibility": "expressible",
            "ir": {"description": "Complete fixed-instance Schedule.", "schedule_json": "{}"},
            "gap": None,
            "eligibility": {
                "gpu": {"eligible": False, "reason": "Static review only."},
                "optimization": {
                    "eligible": False,
                    "reason": "No optimization evidence.",
                },
                "training": {
                    "eligible": False,
                    "reason": "Portable parents are not training eligible.",
                },
            },
            "evidence_refs": ["problem.json"],
            "verification": None,
            "status": "reviewed",
            "reason": "Concrete current IR supplied for deterministic checking.",
        }

    def test_phase0_schema_is_strict_and_verifier_owns_conditionals(self) -> None:
        schema = phase0_result_schema(self.phase0_problem())

        def assert_typed_objects_are_strict(node: object) -> None:
            if not isinstance(node, dict):
                return
            node_type = node.get("type")
            is_object = node_type == "object" or (
                isinstance(node_type, list) and "object" in node_type
            )
            if is_object and "properties" in node and "required" in node:
                self.assertFalse(node.get("additionalProperties", True))
                properties = node.get("properties")
                self.assertIsInstance(properties, dict)
                self.assertEqual(set(node.get("required", [])), set(properties))
            for value in node.values():
                if isinstance(value, (dict, list)):
                    assert_typed_objects_are_strict(value)

        def assert_consts_are_scalar(node: object) -> None:
            if isinstance(node, dict):
                if "const" in node:
                    self.assertNotIsInstance(node["const"], (dict, list))
                for value in node.values():
                    assert_consts_are_scalar(value)
            elif isinstance(node, list):
                for value in node:
                    assert_consts_are_scalar(value)

        assert_typed_objects_are_strict(schema)
        assert_consts_are_scalar(schema)
        self.assertNotIn("uniqueItems", json.dumps(schema, sort_keys=True))
        validator = Draft202012Validator(schema)
        base = self.phase0_result()
        self.assertTrue(validator.is_valid(base))

        duplicate_evidence = deepcopy(base)
        duplicate_evidence["evidence_refs"] = ["problem.json", "problem.json"]
        self.assertTrue(validator.is_valid(duplicate_evidence))
        with self.assertRaisesRegex(Phase0Error, "references must be unique"):
            _phase0_result_invariants(duplicate_evidence)

        invalid = deepcopy(base)
        invalid["owner"] = "ir_gap"
        self.assertTrue(validator.is_valid(invalid))
        with self.assertRaises(Phase0Error):
            _phase0_result_invariants(invalid)

        invalid = deepcopy(base)
        invalid["current_ir_expressibility"] = "not_expressible"
        invalid["owner"] = "ir_gap"
        invalid["gap"] = {
            "earliest_gap": "operation body",
            "minimum_required_semantics": ["one missing effect"],
            "candidate_primitive": {
                "name": "candidate",
                "semantics": "minimum semantics only",
            },
            "counterexample": "The fixed case cannot be represented.",
        }
        self.assertTrue(validator.is_valid(invalid))
        with self.assertRaises(Phase0Error):
            _phase0_result_invariants(invalid)

        not_expressible = deepcopy(invalid)
        not_expressible["ir"] = None
        self.assertTrue(validator.is_valid(not_expressible))
        _phase0_result_invariants(not_expressible)
        duplicate_semantics = deepcopy(not_expressible)
        duplicate_semantics["gap"]["minimum_required_semantics"] = [  # type: ignore[index]
            "one missing effect",
            "one missing effect",
        ]
        self.assertTrue(validator.is_valid(duplicate_semantics))
        with self.assertRaisesRegex(Phase0Error, "semantics must be unique"):
            _phase0_result_invariants(duplicate_semantics)
        not_expressible["gap"]["candidate_primitive"] = None  # type: ignore[index]
        self.assertTrue(validator.is_valid(not_expressible))
        with self.assertRaises(Phase0Error):
            _phase0_result_invariants(not_expressible)

        composition = deepcopy(not_expressible)
        composition["owner"] = "program_composition"
        self.assertTrue(validator.is_valid(composition))
        _phase0_result_invariants(composition)

        insufficient = deepcopy(composition)
        insufficient["current_ir_expressibility"] = "insufficient_evidence"
        insufficient["owner"] = "insufficient_evidence"
        insufficient["status"] = "insufficient_evidence"
        self.assertTrue(validator.is_valid(insufficient))
        _phase0_result_invariants(insufficient)
        insufficient["status"] = "reviewed"
        self.assertTrue(validator.is_valid(insufficient))
        with self.assertRaises(Phase0Error):
            _phase0_result_invariants(insufficient)

        invalid = deepcopy(base)
        invalid["eligibility"]["training"]["eligible"] = True  # type: ignore[index]
        self.assertFalse(validator.is_valid(invalid))

        copied = self.phase0_result()
        _phase0_assert_problem_copies(self.phase0_problem(), copied)
        copied["artifact_refs"] = ["evidence/drifted.cu"]
        with self.assertRaisesRegex(Phase0Error, "artifact_refs differs"):
            _phase0_assert_problem_copies(self.phase0_problem(), copied)
        invalid = deepcopy(base)
        invalid["decorative_field"] = "not owned"
        self.assertFalse(validator.is_valid(invalid))

    def test_phase0_schema_is_stable_across_canonical_json_round_trip(self) -> None:
        problem = self.phase0_problem()
        frozen = problem["frozen_contract"]
        self.assertIsInstance(frozen, dict)
        frozen["producer_owned_extensions"] = {
            "semantics": {"zeta": 1, "alpha": 2},
            "contract": {"output": "y", "input": "x"},
        }
        round_tripped = json.loads(json.dumps(problem, sort_keys=True))
        self.assertEqual(
            phase0_result_schema(problem),
            phase0_result_schema(round_tripped),
        )

    def test_canary_selection_is_deterministic(self) -> None:
        target = {
            "case_id": "a-canary",
            "derived_parent_id": PHASE0_CANARY_DERIVED_PARENT,
            "taxonomy": {"operator": PHASE0_CANARY_OPERATOR},
            "qualification": {
                "stages": {
                    "correctness": {
                        "workloads": [
                            {"id": "n2048", "correct": True},
                            {"id": PHASE0_CANARY_WORKLOAD, "correct": True},
                        ]
                    }
                }
            },
        }
        later = deepcopy(target)
        later["case_id"] = "z-canary"
        later["derived_parent_id"] = "not-selected"
        unrelated = deepcopy(target)
        unrelated["case_id"] = "0-unrelated"
        unrelated["taxonomy"]["operator"] = "other"  # type: ignore[index]

        self.assertIs(_phase0_select_canary([later, unrelated, target]), target)
        self.assertIs(_phase0_select_canary([target, later, unrelated]), target)

    def test_item_command_fixes_treatment_and_covers_auth_deny(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            item_root = Path(temporary) / "item"
            open_cake_root = Path(temporary) / "open-cake"
            command = _phase0_codex_command(
                codex_bin=Path("/usr/bin/codex"),
                item_root=item_root,
            )

        joined = "\n".join(command)
        self.assertEqual(command[command.index("--model") + 1], MODEL)
        self.assertIn('model_reasoning_effort="max"', command)
        self.assertIn(
            f'default_permissions="{PHASE0_PERMISSION_PROFILE}"',
            command,
        )
        self.assertIn('"/home/qhy-sol/.codex"="deny"', joined)
        self.assertIn(
            f"permissions.{PHASE0_PERMISSION_PROFILE}.network.enabled=false",
            command,
        )
        self.assertIn("CUDA_VISIBLE_DEVICES=", command)
        self.assertIn("multi_agent", command)
        self.assertIn("multi_agent_v2", command)
        self.assertNotIn("code_mode_host", PHASE0_DISABLED_FEATURES)
        self.assertIn("shell_tool", PHASE0_DISABLED_FEATURES)
        self.assertIn("view_image", PHASE0_DISABLED_FEATURES)
        self.assertNotIn("tools.view_image=false", command)
        self.assertNotIn("--sandbox", command)
        self.assertNotIn("--permission-profile", command)
        self.assertNotIn("--add-dir", command)
        self.assertNotIn("danger-full-access", joined)
        self.assertNotIn("fallback", joined.lower())

    def test_tool_free_prompt_embeds_only_bounded_text_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            item_root = Path(temporary)
            documents = {
                "problem.json": '{"case_id":"case"}\n',
                "portable_record.json": '{"outcome":"qualified"}\n',
                "reference/compiler.json": '{"revision":"compiler"}\n',
                "reference/schedule.schema.json": '{"type":"object"}\n',
                "reference/AUTHORING_CONTRACT.md": "Authoring contract.\n",
                "evidence/source.cu": "__global__ void kernel() {}\n",
            }
            for relative, content in documents.items():
                target = item_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            prompt = _phase0_prompt(item_root)

        self.assertIn("do not call any nested tool", prompt)
        self.assertIn("for this revision it is sm_100a", prompt)
        self.assertIn("Omit program_map.traversal", prompt)
        self.assertIn("Do not declare an allocation for register scratch", prompt)
        self.assertIn("Access maps describe global-memory buffers only", prompt)
        self.assertIn(
            'BEGIN_UNTRUSTED_DOCUMENT "open-cake:compiler/revision.json"', prompt
        )
        self.assertIn(
            'BEGIN_UNTRUSTED_DOCUMENT "open-cake:compiler/targets/sm_100a.json"',
            prompt,
        )
        self.assertIn("BEGIN_UNTRUSTED_DOCUMENT \"problem.json\"", prompt)
        self.assertIn("BEGIN_UNTRUSTED_DOCUMENT \"evidence/source.cu\"", prompt)
        self.assertIn("__global__ void kernel()", prompt)
        self.assertLess(
            prompt.index('BEGIN_UNTRUSTED_DOCUMENT "reference/compiler.json"'),
            prompt.index('BEGIN_UNTRUSTED_DOCUMENT "problem.json"'),
        )

    def test_create_only_and_ambiguous_attempt_are_never_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_root = root / "new-run"
            _phase0_assert_create_only(run_root)
            run_root.mkdir()
            with self.assertRaisesRegex(Phase0Error, "create-only"):
                _phase0_assert_create_only(run_root)

            item_root = root / "item"
            model_root = item_root / "model"
            model_root.mkdir(parents=True)
            started = "2026-09-02T00:00:00+00:00"
            (model_root / "attempt.json").write_text(
                json.dumps(
                    {
                        "case_id": "item",
                        "started_at": started,
                        "command": "fixed command",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Phase0Error, "ambiguous rerun"):
                _phase0_model_receipt(item_root)

            receipt = {
                "schema": PHASE0_RECEIPT_SCHEMA,
                "case_id": "item",
                "model": MODEL,
                "reasoning_effort": "max",
                "authentication": "parent_cli_chatgpt",
                "permission_profile": PHASE0_PERMISSION_PROFILE,
                "started_at": started,
                "finished_at": "2026-09-02T00:01:00+00:00",
                "timeout_seconds": 60,
                "timed_out": False,
                "exit_code": 1,
                "failure_category": "rate_limit",
                "failure_detail": None,
                "events": "model/events.jsonl",
                "event_count": None,
                "stderr": "model/stderr.log",
                "final_response": None,
            }
            (model_root / "receipt.json").write_text(
                json.dumps(receipt), encoding="utf-8"
            )
            with self.assertRaises(Phase0Error) as raised:
                _phase0_model_receipt(item_root)
            self.assertEqual(raised.exception.category, "rate_limit")

    def test_failure_classes_remain_distinct(self) -> None:
        cases = (
            (
                {"timed_out": True, "exit_code": None, "stderr": "", "result_exists": False},
                "timeout",
            ),
            (
                {
                    "timed_out": False,
                    "exit_code": 1,
                    "stderr": "HTTP 429 rate limit",
                    "result_exists": False,
                },
                "rate_limit",
            ),
            (
                {
                    "timed_out": False,
                    "exit_code": 1,
                    "stderr": "subscription required",
                    "result_exists": False,
                },
                "subscription",
            ),
            (
                {
                    "timed_out": False,
                    "exit_code": 1,
                    "stderr": "not logged in",
                    "result_exists": False,
                },
                "account",
            ),
            (
                {
                    "timed_out": False,
                    "exit_code": 1,
                    "stderr": "DNS failure",
                    "result_exists": False,
                },
                "infra",
            ),
            (
                {
                    "timed_out": False,
                    "exit_code": 2,
                    "stderr": "error: unexpected argument '--permission-profile' found",
                    "result_exists": False,
                },
                "infra",
            ),
            (
                {
                    "timed_out": False,
                    "exit_code": 1,
                    "stderr": "unknown configuration field `tools.view_image`",
                    "result_exists": False,
                },
                "infra",
            ),
            (
                {
                    "timed_out": False,
                    "exit_code": 1,
                    "stderr": "invalid JSON response format",
                    "result_exists": False,
                },
                "schema",
            ),
            (
                {
                    "timed_out": False,
                    "exit_code": 1,
                    "stderr": "invalid_json_schema: uniqueItems is not permitted",
                    "result_exists": False,
                },
                "schema",
            ),
            (
                {
                    "timed_out": False,
                    "exit_code": 0,
                    "stderr": "",
                    "result_exists": False,
                },
                "task",
            ),
            (
                {
                    "timed_out": False,
                    "exit_code": 1,
                    "stderr": "provider rejected request",
                    "result_exists": False,
                },
                "provider",
            ),
        )
        for arguments, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(_phase0_failure_category(**arguments), expected)
        self.assertEqual(
            _phase0_failure_category(
                timed_out=False,
                exit_code=1,
                stderr="2026-09-02T14:23:43.443429Z patch failed",
                result_exists=False,
            ),
            "provider",
        )
        self.assertEqual(
            set(PHASE0_FAILURE_CATEGORIES),
            {
                "provider",
                "rate_limit",
                "account",
                "subscription",
                "infra",
                "timeout",
                "schema",
                "reviewer",
                "corpus",
                "task",
            },
        )

    def test_static_gate_only_parses_validates_and_lowers_when_eligible(self) -> None:
        class FakeCompiler:
            def __init__(self, assessment: SimpleNamespace, lowering=None) -> None:
                self.assessment = assessment
                self.lowering = lowering
                self.assess_calls = 0
                self.lower_calls = 0

            def assess(self, schedule):
                self.assess_calls += 1
                self.schedule = schedule
                return self.assessment

            def lower(self, assessment):
                self.lower_calls += 1
                return self.lowering

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            result = self.phase0_result()
            result["current_ir_expressibility"] = "not_expressible"
            result["owner"] = "program_composition"
            result["ir"] = None
            skipped = FakeCompiler(SimpleNamespace())
            verification = _phase0_static_gate(
                item_root=root,
                result=result,
                compiler_ref={},
                compiler=skipped,  # type: ignore[arg-type]
            )
            self.assertEqual(
                verification["parse"]["status"], "not_applicable"  # type: ignore[index]
            )
            self.assertEqual(skipped.assess_calls, 0)
            self.assertEqual(skipped.lower_calls, 0)

            invalid = self.phase0_result()
            invalid["ir"] = {"description": "bad", "schedule_json": "{"}
            with self.assertRaisesRegex(Phase0Error, "does not parse"):
                _phase0_static_gate(
                    item_root=root,
                    result=invalid,
                    compiler_ref={},
                    compiler=skipped,  # type: ignore[arg-type]
                )
            self.assertEqual(skipped.assess_calls, 0)

            rejected_root = root / "rejected"
            (rejected_root / "verifier").mkdir(parents=True)
            rejected = FakeCompiler(
                SimpleNamespace(
                    accepted=False,
                    lowering_eligible=False,
                    findings=[SimpleNamespace(code="schedule.invalid")],
                )
            )
            with mock.patch(
                "tools.run_aka_qualified_ir_codex._assessment",
                return_value={"accepted": False},
            ):
                with self.assertRaisesRegex(Phase0Error, "not accepted"):
                    _phase0_static_gate(
                        item_root=rejected_root,
                        result=self.phase0_result(),
                        compiler_ref={},
                        compiler=rejected,  # type: ignore[arg-type]
                    )
            self.assertEqual(rejected.lower_calls, 0)

            backend_root = root / "backend"
            (backend_root / "verifier").mkdir(parents=True)
            backend = FakeCompiler(
                SimpleNamespace(
                    accepted=True,
                    lowering_eligible=False,
                    findings=[],
                )
            )
            backend_result = self.phase0_result()
            backend_result["owner"] = "backend"
            with mock.patch(
                "tools.run_aka_qualified_ir_codex._assessment",
                return_value={"accepted": True, "lowering_eligible": False},
            ):
                verification = _phase0_static_gate(
                    item_root=backend_root,
                    result=backend_result,
                    compiler_ref={},
                    compiler=backend,  # type: ignore[arg-type]
                )
            self.assertEqual(verification["lower"]["status"], "blocked")  # type: ignore[index]
            self.assertEqual(backend.lower_calls, 0)

            lowering_root = root / "lowering"
            (lowering_root / "verifier").mkdir(parents=True)
            lowering = SimpleNamespace(
                compiler_revision_id="compiler-test",
                schedule_id="schedule-test",
                target="b200",
                route=SimpleNamespace(
                    backend=SimpleNamespace(value="cuda"),
                    entry_point="kernel",
                ),
                source="def kernel():\n    pass\n",
                toolchain_requirements={"source_language": "python"},
            )
            lowerable = FakeCompiler(
                SimpleNamespace(
                    accepted=True,
                    lowering_eligible=True,
                    findings=[],
                ),
                lowering,
            )
            with mock.patch(
                "tools.run_aka_qualified_ir_codex._assessment",
                return_value={"accepted": True, "lowering_eligible": True},
            ):
                verification = _phase0_static_gate(
                    item_root=lowering_root,
                    result=self.phase0_result(),
                    compiler_ref={},
                    compiler=lowerable,  # type: ignore[arg-type]
                )
            self.assertEqual(verification["lower"]["status"], "passed")  # type: ignore[index]
            self.assertEqual(lowerable.assess_calls, 1)
            self.assertEqual(lowerable.lower_calls, 1)
            self.assertTrue((lowering_root / "verifier/lowering.json").is_file())


if __name__ == "__main__":
    unittest.main()
