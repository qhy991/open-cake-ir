from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.review_aka_expressibility import (  # noqa: E402
    ReviewError,
    _git_closure_identity,
    _load_context,
    _validate_parent_completion,
    initialize,
    materialize,
    status,
    verify_review,
)


GAP = {
    "capability": "runtime scalar predicate",
    "irreducible_semantics": "Select one value from a runtime scalar comparison.",
    "hardware_commitment": "The predicate is evaluated per logical element.",
    "verification_rule": "The predicate and selected values must be type-compatible.",
    "analysis_requirement": "Track the predicate dependency and selected result dtype.",
    "lowering_requirement": "Emit one target-supported compare and select sequence.",
    "acceptance_case": "Admit one typed select and reject one mismatched result dtype.",
}


class AkaExpressibilityReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source_temporary = tempfile.TemporaryDirectory()
        self.work_temporary = tempfile.TemporaryDirectory()
        self.repository = Path(self.source_temporary.name) / "source"
        self.dataset = self.repository / "cuda_kernel_dataset_test"
        self.work_root = Path(self.work_temporary.name) / "review-work"
        subprocess.run(["git", "init", "-q", str(self.repository)], check=True)
        subprocess.run(
            ["git", "-C", str(self.repository), "config", "user.name", "Review Test"],
            check=True,
        )
        checker_commit = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        self.checker_identity = {
            "git_commit": checker_commit,
            "paths": [
                "tools/review_aka_expressibility.py",
                "tools/run_aka_expressibility_codex.py",
                "tools/audit_aka_corpus.py",
            ],
        }
        self.checker_patch = mock.patch(
            "tools.review_aka_expressibility._checker_identity",
            return_value=self.checker_identity,
        )
        self.checker_patch.start()
        subprocess.run(
            [
                "git",
                "-C",
                str(self.repository),
                "config",
                "user.email",
                "review@test.invalid",
            ],
            check=True,
        )

    def tearDown(self) -> None:
        self.checker_patch.stop()
        self.source_temporary.cleanup()
        self.work_temporary.cleanup()

    def write_rows(self, tasks: list[str]) -> str:
        counters: dict[str, int] = {}
        for task in tasks:
            index = counters.get(task, 0)
            counters[task] = index + 1
            path = self.dataset / f"categories/movement/copy/{task}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "instruction": f"Review bounded {task} case {index}.",
                "input": (
                    "__global__ void copy_%d(float *out, const float *in) "
                    "{ out[threadIdx.x] = in[threadIdx.x]; }" % index
                ),
                "reasoning": "Visible-source reasoning only.",
                "output": (
                    "__global__ void candidate_%d(float *out, const float *in) "
                    "{ out[threadIdx.x] = in[threadIdx.x]; }" % index
                    if task == "generation"
                    else "One bounded review output."
                ),
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        subprocess.run(["git", "-C", str(self.repository), "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", str(self.repository), "commit", "-q", "-m", "freeze"],
            check=True,
        )
        return subprocess.run(
            ["git", "-C", str(self.repository), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()

    def initialize(
        self,
        tasks: list[str] | None = None,
        *,
        record_format: str = "aka_v1_operator_sft",
        work_root: Path | None = None,
    ) -> Path:
        revision = self.write_rows(tasks or ["analysis"])
        target = work_root or self.work_root
        initialize(
            dataset_root=self.dataset,
            work_root=target,
            source_revision=revision,
            record_format=record_format,
        )
        return target

    def materialized_review(self, case_id: str = "case-000001") -> dict[str, object]:
        materialize(self.work_root, case_id)
        return json.loads(
            (
                self.work_root / "cases" / case_id / "review.template.json"
            ).read_text(encoding="utf-8")
        )

    def write_review(
        self, case_id: str, review: dict[str, object], *, work_root: Path | None = None
    ) -> None:
        root = work_root or self.work_root
        path = root / "cases" / case_id / "review.json"
        path.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")

    def qualified_parent_completion(
        self,
        case_id: str,
        *,
        work_root: Path | None = None,
        outcome: str = "qualified",
    ) -> Path:
        root = work_root or self.work_root
        case_input = json.loads(
            (root / "cases" / case_id / "input.json").read_text(encoding="utf-8")
        )
        completion_root = Path(self.work_temporary.name) / f"{root.name}-{case_id}-parent"
        for relative in (
            "baseline",
            "reference",
            "harness",
            "evidence/route",
            "evidence/mirror/stages/compile",
            "evidence/mirror/stages/correctness",
            "evidence/mirror/stages/sanitize",
        ):
            (completion_root / relative).mkdir(parents=True, exist_ok=True)
        artifacts = {
            "input.json": {"source": "visible record"},
            "task.json": {"task": "qualify one derived parent"},
            "evidence/route/route.json": {
                "locator": {"node_id": "fixed-node", "run_id": "run-1"}
            },
            "evidence/mirror/result.json": {
                "outcome": "completed",
                "validity": "valid",
            },
            "evidence/mirror/stages/compile/result.json": {
                "status": "passed",
                "validity": "valid",
            },
            "evidence/mirror/stages/correctness/result.json": {
                "status": "passed",
                "validity": "valid",
                "complete_output": True,
                "workloads": [{"correct": True}, {"correct": True}],
            },
            "evidence/mirror/stages/sanitize/result.json": {
                "status": "passed",
                "validity": "valid",
                "checks": ["memcheck", "racecheck"],
            },
        }
        for relative, value in artifacts.items():
            path = completion_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        completion = {
            "schema": "aka.kernel-parent-completion.v1",
            "case_id": f"{root.name}-{case_id}-completion",
            "original_parent": {
                "case_path": case_input["parent_coordinate"],
                "record_field": case_input["challenge_case"]["source_ref"][
                    "primary_field"
                ],
            },
            "recovery_mode": "contract_narrowed",
            "source": {
                "provenance": "visible_record",
                "repository": None,
                "revision": None,
                "path": None,
                "symbol": "copy_derived",
            },
            "taxonomy": {"category": "movement", "operator": "copy"},
            "derived_parent_id": "copy_contiguous_fp32_int32_v1",
            "semantics": {
                "inputs": ["one contiguous FP32 input"],
                "outputs": ["one contiguous FP32 output"],
                "computation": "Copy each input element to the matching output element.",
                "valid_domain": ["the element count is positive"],
                "material_unknowns": ["framework integration is not claimed"],
            },
            "contract": {
                "api": "Copy one contiguous FP32 array.",
                "dtypes": ["float32"],
                "index_types": ["int32"],
                "layouts": ["contiguous"],
                "optional_inputs": [],
                "invariants": ["complete outputs match the reference"],
                "exclusions": ["framework Tensor behavior is not claimed"],
                "launch_policy": "block=256 and grid=ceil(numel/256)",
            },
            "optimization_handoff": {
                "mechanism": "vectorized_copy",
                "hypothesis": "Aligned vector loads reduce instruction count.",
                "eligibility": ["aligned contiguous FP32 arrays"],
                "anti_conditions": ["unaligned tails"],
            },
            "artifacts": {
                "baseline": "baseline",
                "reference": "reference",
                "harness": "harness",
                "task": "task.json",
            },
            "qualification": (
                {
                    "locator": {"node_id": "fixed-node", "run_id": "run-1"},
                    "route": "evidence/route/route.json",
                    "node_result": "evidence/mirror/result.json",
                    "stages": {
                        "compile": "evidence/mirror/stages/compile/result.json",
                        "correctness": "evidence/mirror/stages/correctness/result.json",
                        "sanitize": "evidence/mirror/stages/sanitize/result.json",
                    },
                }
                if outcome == "qualified"
                else None
            ),
            "outcome": outcome,
            "missing_facts": [],
            "evidence": ["input.json", "evidence/route/route.json"],
            "training_route": (
                "augmentation_parent"
                if outcome == "qualified"
                else "augmentation_parent_pending_qualification"
            ),
            "training_eligibility": False,
            "next_action": "Start one fresh mechanism augmentation case.",
        }
        completion_path = completion_root / "parent_completion.json"
        completion_path.write_text(json.dumps(completion) + "\n", encoding="utf-8")
        validator = json.loads(
            (root / "manifest.json").read_text(encoding="utf-8")
        )["parent_completion_validator"]["path"]
        subprocess.run(
            [sys.executable, validator, str(completion_path), "--finalize"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        return completion_path

    def bind_completion(
        self,
        review: dict[str, object],
        case_id: str,
        *,
        work_root: Path | None = None,
        outcome: str = "qualified",
    ) -> None:
        review["parent_contract"] = {
            "status": "completion",
            "missing_facts": [],
            "completion_path": str(
                self.qualified_parent_completion(
                    case_id, work_root=work_root, outcome=outcome
                )
            ),
        }

    def copy_schedule(self, case_id: str, source: str, name: str = "schedule.json") -> str:
        destination = self.work_root / "cases" / case_id / name
        shutil.copyfile(ROOT / source, destination)
        return name

    @staticmethod
    def schedule_aspect(
        path: str, assessment_scope: str = "contract"
    ) -> dict[str, object]:
        return {
            "assessment_scope": assessment_scope,
            "owner_scope": "schedule",
            "owner_relation": None,
            "disposition": "schedule",
            "schedule": path,
            "missing_ir": None,
            "reason": "This case-local typed Schedule is the submitted candidate.",
        }

    def test_init_builds_one_reference_bundle_and_unknown_is_provisional(self) -> None:
        self.initialize()
        self.assertEqual(
            sorted(path.name for path in self.work_root.iterdir()),
            ["manifest.json", "reference"],
        )
        self.assertEqual(
            sorted(path.name for path in (self.work_root / "reference").iterdir()),
            [
                "AUTHORING_CONTRACT.md",
                "README.md",
                "TASK.md",
                "compiler.json",
                "review.schema.json",
                "schedule.schema.json",
            ],
        )
        manifest = json.loads(
            (self.work_root / "manifest.json").read_text(encoding="utf-8")
        )
        compiler_reference = json.loads(
            (self.work_root / "reference/compiler.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["checker"], self.checker_identity)
        self.assertEqual(compiler_reference["checker"], self.checker_identity)
        structured_schema = json.loads(
            (self.work_root / "reference/review.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertNotIn("oneOf", json.dumps(structured_schema, sort_keys=True))

        def assert_strict_output_schema(node: object) -> None:
            if not isinstance(node, dict):
                return
            if "const" in node or "enum" in node:
                self.assertIn("type", node)
            node_type = node.get("type")
            if node_type == "object" or (
                isinstance(node_type, list) and "object" in node_type
            ):
                self.assertFalse(node.get("additionalProperties", True))
                properties = node.get("properties")
                self.assertIsInstance(properties, dict)
                self.assertEqual(set(node.get("required", [])), set(properties))
            for value in node.values():
                if isinstance(value, dict):
                    assert_strict_output_schema(value)
                elif isinstance(value, list):
                    for item in value:
                        assert_strict_output_schema(item)

        assert_strict_output_schema(structured_schema)
        review = self.materialized_review()
        case_input = json.loads(
            (self.work_root / "cases/case-000001/input.json").read_text(
                encoding="utf-8"
            )
        )
        source_ref = case_input["challenge_case"]["source_ref"]
        self.assertEqual(
            case_input["parent_coordinate"],
            f"{source_ref['revision']}:{source_ref['path']}:{source_ref['line']}",
        )
        self.assertEqual(
            sorted(
                path.name
                for path in (self.work_root / "cases/case-000001").iterdir()
            ),
            ["input.json", "review.template.json"],
        )
        self.write_review("case-000001", review)

        result = verify_review(self.work_root, "case-000001")

        self.assertEqual(result["primary_class"], "unknown")
        self.assertEqual(
            result["complete_parent_expressibility"]["assessment_scope"],
            "contract",
        )
        self.assertEqual(
            result["delta_expressibility"]["assessment_scope"], "contract"
        )
        self.assertEqual(result["delta_expressibility"]["classification"], "not_applicable")
        self.assertEqual(result["semantic_binding"], "reviewer_claimed")
        self.assertEqual(result["compiler_maturity"], "draft")
        self.assertEqual(result["gpu_test"], "not_run")
        self.assertEqual(result["review_state"], "checked")

        (self.work_root / "reference/README.md").write_text(
            "changed\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ReviewError, "reference bundle changed"):
            status(self.work_root)

    def test_next_is_create_only_and_incomplete_case_is_not_materialized(self) -> None:
        self.initialize(["analysis", "analysis"])
        first = materialize(self.work_root)
        incomplete = self.work_root / "cases/case-000002"
        incomplete.mkdir()
        (incomplete / "input.json").write_text("{}\n", encoding="utf-8")

        aggregate = status(self.work_root)

        self.assertEqual(first["case_id"], "case-000001")
        self.assertEqual(aggregate["materialized"], 1)
        self.assertEqual(aggregate["incomplete"], 1)
        with self.assertRaisesRegex(ReviewError, "incomplete"):
            materialize(self.work_root)

    def test_contract_missing_and_schedule_gap_candidate_are_distinct(self) -> None:
        self.initialize(["analysis", "analysis"])
        missing = self.materialized_review("case-000001")
        missing["parent_contract"] = {
            "status": "missing",
            "missing_facts": ["The launch geometry and dtype specialization are absent."],
            "completion_path": None,
        }
        self.write_review("case-000001", missing)

        gap = self.materialized_review("case-000002")
        gap["complete_parent"] = {
            "assessment_scope": "contract",
            "owner_scope": "schedule",
            "owner_relation": None,
            "disposition": "schedule_gap",
            "schedule": None,
            "missing_ir": GAP,
            "reason": "A runtime predicate is an irreducible missing capability.",
        }
        self.write_review("case-000002", gap)
        with self.assertRaisesRegex(ReviewError, "runnable canonical"):
            verify_review(self.work_root, "case-000002")
        self.bind_completion(gap, "case-000002")
        self.write_review("case-000002", gap)

        missing_result = verify_review(self.work_root, "case-000001")
        gap_result = verify_review(self.work_root, "case-000002")

        self.assertEqual(missing_result["primary_class"], "contract_missing")
        self.assertEqual(gap_result["primary_class"], "schedule_gap_candidate")
        self.assertEqual(
            gap_result["complete_parent_expressibility"]["compiler_check"],
            "not_run",
        )

    def test_parent_completion_is_fixed_validator_only_not_custody(self) -> None:
        self.initialize()
        review = self.materialized_review()
        self.bind_completion(review, "case-000001")
        self.write_review("case-000001", review)

        result = verify_review(self.work_root, "case-000001")

        self.assertEqual(
            result["parent_contract"]["status"],
            "qualified_by_parent_validator",
        )
        self.assertEqual(
            result["parent_contract"]["independent_node_custody"], "not_assessed"
        )

        review["parent_contract"]["completion_path"] = "parent_completion.json"
        self.write_review("case-000001", review)
        with self.assertRaisesRegex(ReviewError, "absolute"):
            verify_review(self.work_root, "case-000001")

    def test_runnable_unqualified_parent_allows_only_provisional_ir_review(self) -> None:
        self.initialize()
        review = self.materialized_review()
        self.bind_completion(
            review, "case-000001", outcome="runnable_unqualified"
        )
        review["complete_parent"] = {
            "assessment_scope": "contract",
            "owner_scope": "schedule",
            "owner_relation": None,
            "disposition": "schedule_gap",
            "schedule": None,
            "missing_ir": GAP,
            "reason": "The runnable parent requires one bounded runtime scalar.",
        }
        self.write_review("case-000001", review)

        result = verify_review(self.work_root, "case-000001")

        self.assertEqual(
            result["parent_contract"]["status"],
            "runnable_by_parent_validator",
        )
        self.assertEqual(result["primary_class"], "schedule_gap_candidate")
        self.assertEqual(result["semantic_binding"], "reviewer_claimed")
        self.assertEqual(result["gpu_test"], "not_run")

    def test_parent_validator_timeout_fails_closed(self) -> None:
        self.initialize()
        self.materialized_review()
        completion = self.qualified_parent_completion("case-000001")
        context = _load_context(self.work_root)
        case_input = json.loads(
            (self.work_root / "cases/case-000001/input.json").read_text(
                encoding="utf-8"
            )
        )
        with mock.patch(
            "tools.review_aka_expressibility.subprocess.run",
            side_effect=subprocess.TimeoutExpired("parent-validator", 30),
        ):
            with self.assertRaisesRegex(ReviewError, "exceeded 30s"):
                _validate_parent_completion(context, str(completion), case_input)

    def test_case_local_lowerable_schedule_is_only_a_candidate(self) -> None:
        self.initialize()
        review = self.materialized_review()
        self.bind_completion(review, "case-000001")
        schedule = self.copy_schedule(
            "case-000001", "corpus/schedules/flash-kmeans-assignment-full.json"
        )
        review["complete_parent"] = self.schedule_aspect(schedule)
        self.write_review("case-000001", review)

        result = verify_review(self.work_root, "case-000001")

        self.assertEqual(result["primary_class"], "schedule_candidate_lowerable")
        checked = result["complete_parent_expressibility"]
        self.assertEqual(checked["compiler_check"], "lowerable")
        self.assertTrue(checked["compiler_evidence"]["assessment"]["accepted"])
        self.assertIsNotNone(checked["compiler_evidence"]["lowering"])
        self.assertNotIn("no_gap", json.dumps(result))
        self.assertNotIn("described", json.dumps(result))

    def test_fixed_instance_lowering_does_not_claim_complete_parent(self) -> None:
        self.initialize()
        review = self.materialized_review()
        self.bind_completion(review, "case-000001")
        schedule = self.copy_schedule(
            "case-000001", "corpus/schedules/flash-kmeans-assignment-full.json"
        )
        review["complete_parent"] = self.schedule_aspect(
            schedule, "fixed_instance"
        )
        self.write_review("case-000001", review)

        result = verify_review(self.work_root, "case-000001")

        self.assertEqual(result["primary_class"], "fixed_instance_candidate_lowerable")
        checked = result["complete_parent_expressibility"]
        self.assertEqual(checked["assessment_scope"], "fixed_instance")
        self.assertEqual(checked["compiler_check"], "lowerable")
        self.assertNotEqual(checked["classification"], "schedule_candidate_lowerable")

        review["complete_parent"] = {
            "assessment_scope": "fixed_instance",
            "owner_scope": "schedule",
            "owner_relation": None,
            "disposition": "schedule_gap",
            "schedule": None,
            "missing_ir": GAP,
            "reason": "This fixed instance needs one closed missing capability.",
        }
        self.write_review("case-000001", review)
        gap = verify_review(self.work_root, "case-000001")
        self.assertEqual(gap["primary_class"], "fixed_instance_gap_candidate")
        self.assertEqual(
            gap["complete_parent_expressibility"]["compiler_check"], "not_run"
        )

    def test_fixed_instance_unknown_and_redirect_stay_in_the_fixed_lane(self) -> None:
        self.initialize(["optimization_positive"])
        review = self.materialized_review()
        review["complete_parent"]["assessment_scope"] = "fixed_instance"
        self.write_review("case-000001", review)
        with self.assertRaisesRegex(ReviewError, "fixed-instance claim requires"):
            verify_review(self.work_root, "case-000001")

        self.bind_completion(review, "case-000001")
        self.write_review("case-000001", review)
        unknown = verify_review(self.work_root, "case-000001")
        self.assertEqual(unknown["primary_class"], "fixed_instance_unknown")

        review["complete_parent"] = {
            "assessment_scope": "fixed_instance",
            "owner_scope": "program",
            "owner_relation": "Two launches form this one fixed endpoint.",
            "disposition": "owner_redirect",
            "schedule": None,
            "missing_ir": None,
            "reason": "Even this fixed instance is larger than one Schedule.",
        }
        review["delta"] = {
            "assessment_scope": "fixed_instance",
            "owner_scope": "portfolio",
            "owner_relation": "One fixed binding still selects a sealed specialist.",
            "disposition": "owner_redirect",
            "schedule": None,
            "missing_ir": None,
            "reason": "The fixed delta is a specialist-selection decision.",
        }
        self.write_review("case-000001", review)
        redirected = verify_review(self.work_root, "case-000001")
        self.assertEqual(
            redirected["primary_class"],
            "fixed_instance_program_redirect_candidate",
        )
        self.assertEqual(
            redirected["delta_expressibility"]["classification"],
            "fixed_instance_portfolio_redirect_candidate",
        )

        review["delta"] = {
            **review["delta"],
            "assessment_scope": "fixed_instance",
            "owner_scope": "unknown",
            "owner_relation": None,
            "disposition": "not_applicable",
            "reason": "This lane is not applicable.",
        }
        self.write_review("case-000001", review)
        with self.assertRaisesRegex(ReviewError, "cannot be not_applicable"):
            verify_review(self.work_root, "case-000001")

    def test_case_local_backend_blocked_schedule_is_only_a_candidate(self) -> None:
        self.initialize()
        review = self.materialized_review()
        self.bind_completion(review, "case-000001")
        schedule = self.copy_schedule(
            "case-000001",
            "corpus/schedules/flash-kmeans-assignment-full-shape-drift.json",
        )
        review["complete_parent"] = self.schedule_aspect(schedule)
        self.write_review("case-000001", review)

        result = verify_review(self.work_root, "case-000001")

        self.assertEqual(
            result["primary_class"], "schedule_candidate_backend_blocked"
        )
        checked = result["complete_parent_expressibility"]
        self.assertEqual(checked["compiler_check"], "backend_blocked")
        self.assertIsNone(checked["compiler_evidence"]["lowering"])
        self.assertNotIn('"backend_gap"', json.dumps(result))

    def test_compiler_root_schedule_and_invalid_local_schedule_are_rejected(self) -> None:
        self.initialize()
        review = self.materialized_review()
        self.bind_completion(review, "case-000001")
        review["complete_parent"] = self.schedule_aspect(
            str(ROOT / "corpus/schedules/flash-kmeans-assignment-full.json")
        )
        self.write_review("case-000001", review)
        with self.assertRaisesRegex(ReviewError, "case-relative"):
            verify_review(self.work_root, "case-000001")

        schedule = self.copy_schedule(
            "case-000001",
            "corpus/schedules/flash-kmeans-b32-smoke-dtype-drift-v2.json",
        )
        review["complete_parent"] = self.schedule_aspect(schedule)
        self.write_review("case-000001", review)
        with self.assertRaisesRegex(ReviewError, "invalid authoring"):
            verify_review(self.work_root, "case-000001")

    def test_program_and_portfolio_redirects_are_provisional(self) -> None:
        self.initialize(["optimization_positive"])
        review = self.materialized_review()
        self.bind_completion(review, "case-000001")
        review["complete_parent"] = {
            "assessment_scope": "contract",
            "owner_scope": "program",
            "owner_relation": "Two launches and their dependency edge form the endpoint.",
            "disposition": "owner_redirect",
            "schedule": None,
            "missing_ir": None,
            "reason": "The complete endpoint is larger than one Schedule.",
        }
        review["delta"] = {
            "assessment_scope": "contract",
            "owner_scope": "portfolio",
            "owner_relation": "A runtime shape guard selects one specialist.",
            "disposition": "owner_redirect",
            "schedule": None,
            "missing_ir": None,
            "reason": "The delta changes dispatch rather than kernel semantics.",
        }
        self.write_review("case-000001", review)

        result = verify_review(self.work_root, "case-000001")

        self.assertEqual(result["primary_class"], "program_redirect_candidate")
        self.assertEqual(
            result["delta_expressibility"]["classification"],
            "portfolio_redirect_candidate",
        )
        self.assertEqual(
            result["complete_parent_expressibility"]["compiler_check"], "not_run"
        )
        self.assertEqual(result["delta_expressibility"]["compiler_check"], "not_run")

    def test_delta_role_is_derived_for_v1_and_v2(self) -> None:
        self.initialize(["analysis", "optimization_neutral", "optimization_positive"])
        analysis = self.materialized_review("case-000001")
        optimization = self.materialized_review("case-000002")
        self.assertEqual(analysis["delta"]["disposition"], "not_applicable")
        self.assertEqual(optimization["delta"]["disposition"], "unknown")
        optimization["delta"]["disposition"] = "not_applicable"
        self.write_review("case-000002", optimization)
        with self.assertRaisesRegex(ReviewError, "not_applicable conflicts"):
            verify_review(self.work_root, "case-000002")

        v2_root = Path(self.work_temporary.name) / "review-work-v2"
        initialize(
            dataset_root=self.dataset,
            work_root=v2_root,
            source_revision=json.loads(
                (self.work_root / "manifest.json").read_text(encoding="utf-8")
            )["source"]["revision"],
            record_format="aka_v2_review_projection",
        )
        materialize(v2_root, "case-000002")
        v2_neutral = json.loads(
            (v2_root / "cases/case-000002/review.template.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            v2_neutral["complete_parent"]["disposition"], "not_applicable"
        )
        self.assertEqual(v2_neutral["delta"]["disposition"], "not_applicable")
        self.write_review("case-000002", v2_neutral, work_root=v2_root)
        self.assertEqual(
            verify_review(v2_root, "case-000002")["delta_expressibility"][
                "classification"
            ],
            "not_applicable",
        )

        materialize(v2_root, "case-000003")
        v2_positive = json.loads(
            (v2_root / "cases/case-000003/review.template.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(v2_positive["complete_parent"]["disposition"], "unknown")
        self.assertEqual(v2_positive["delta"]["disposition"], "unknown")

    def test_source_ref_compiler_identity_and_validator_drift_fail_closed(self) -> None:
        self.initialize()
        review = self.materialized_review()
        review["source_ref"]["line"] = 2
        self.write_review("case-000001", review)
        with self.assertRaisesRegex(ReviewError, "source_ref differs"):
            verify_review(self.work_root, "case-000001")

        manifest_path = self.work_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["checker"]["git_commit"] = "0" * 40
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ReviewError, "checker Git/raw-byte identity drifted"):
            status(self.work_root)

        manifest["checker"] = self.checker_identity
        manifest["compiler"]["git_commit"] = "0" * 40
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ReviewError, "identity drifted"):
            status(self.work_root)

        manifest["compiler"]["git_commit"] = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        manifest["parent_completion_validator"]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ReviewError, "validator identity drifted"):
            status(self.work_root)

    def test_raw_git_closure_defeats_assume_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            subprocess.run(["git", "init", "-q", str(repository)], check=True)
            subprocess.run(
                ["git", "-C", str(repository), "config", "user.name", "Closure Test"],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "config",
                    "user.email",
                    "closure@test.invalid",
                ],
                check=True,
            )
            (repository / "checker.py").write_text("checker = 1\n", encoding="utf-8")
            (repository / "compiler.py").write_text("compiler = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repository), "add", "-A"], check=True)
            subprocess.run(
                ["git", "-C", str(repository), "commit", "-q", "-m", "freeze"],
                check=True,
            )

            identity = _git_closure_identity(
                repository, ("checker.py", "compiler.py"), "test closure"
            )
            self.assertEqual(identity["paths"], ["checker.py", "compiler.py"])
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "update-index",
                    "--assume-unchanged",
                    "checker.py",
                ],
                check=True,
            )
            (repository / "checker.py").write_text("checker = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ReviewError, "raw worktree bytes differ"):
                _git_closure_identity(
                    repository, ("checker.py", "compiler.py"), "test closure"
                )

    def test_cases_symlink_is_rejected(self) -> None:
        self.initialize()
        outside = Path(self.work_temporary.name) / "outside-cases"
        outside.mkdir()
        (self.work_root / "cases").symlink_to(outside, target_is_directory=True)

        with self.assertRaisesRegex(ReviewError, "non-symlink directory"):
            materialize(self.work_root)

    def test_checked_cache_is_create_only_and_status_rederives_it(self) -> None:
        self.initialize()
        review = self.materialized_review()
        self.write_review("case-000001", review)
        verify_review(self.work_root, "case-000001", finalize=True)

        aggregate = status(self.work_root)

        self.assertEqual(aggregate["checked"], 1)
        self.assertEqual(aggregate["primary_class_counts"], {"unknown": 1})
        with self.assertRaisesRegex(ReviewError, "create-only"):
            verify_review(self.work_root, "case-000001", finalize=True)

        checked_path = self.work_root / "cases/case-000001/checked.json"
        cached = json.loads(checked_path.read_text(encoding="utf-8"))
        cached["primary_class"] = "schedule_candidate_lowerable"
        checked_path.write_text(json.dumps(cached) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ReviewError, "differs from rederived"):
            status(self.work_root)


if __name__ == "__main__":
    unittest.main()
