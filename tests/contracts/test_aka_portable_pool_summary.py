from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.summarize_aka_portable_parent_pool import markdown, summarize  # noqa: E402


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class AkaPortablePoolSummaryTests(unittest.TestCase):
    def test_terminal_pool_projects_claim_boundaries_and_gap_signals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(
                root / "pool.final.json",
                {
                    "schema": "open-cake.aka-portable-parent-pool-final.v1",
                    "source_revision": "a" * 40,
                    "selected": 2,
                    "max_workers": 2,
                    "counts": {"completed": 2},
                    "finished_at": "2026-08-29T00:00:00+00:00",
                },
            )
            write(
                root / "pool.json",
                {
                    "model": "gpt-5.6-sol",
                    "reasoning_effort": "max",
                    "codex": {"version": "codex-cli 0.test"},
                    "implementation": {"git_commit": "b" * 40},
                },
            )
            entries = []
            for index in (1, 2):
                case_id = f"case-{index:06d}"
                entries.append(
                    {
                        "queue_case_id": case_id,
                        "portable_case_id": f"portable-{index}",
                        "derived_parent_id": f"derived-{index}",
                        "source_identity": ["categories/copy/analysis.jsonl", index, "input"],
                    }
                )
                case_root = root / "cases" / case_id
                write(
                    case_root / "worker.finished.json",
                    {"status": "completed", "exit_code": 0},
                )
                write(
                    case_root / f"reviews/runs/{case_id}/receipt.json",
                    {
                        "exit_code": 0,
                        "timed_out": False,
                        "model": "gpt-5.6-sol",
                        "reasoning_effort": "max",
                    },
                )
                write(
                    case_root / f"parents/cases/{case_id}/PARENT_DONE.json",
                    {
                        "outcome": "runnable_unqualified",
                        "counts_as_runnable_bundle": True,
                        "counts_as_executable_parent": False,
                    },
                )
                gap = index == 1
                write(
                    case_root / f"reviews/cases/{case_id}/checked.json",
                    {
                        "schema": "open-cake.aka-expressibility-checked.v4",
                        "primary_class": (
                            "schedule_gap_candidate"
                            if gap
                            else "fixed_instance_candidate_lowerable"
                        ),
                        "semantic_binding": "reviewer_claimed",
                        "gpu_test": "not_run",
                        "review_state": "checked",
                        "parent_contract": {
                            "status": "runnable_by_parent_validator"
                        },
                        "complete_parent_expressibility": {
                            "assessment_scope": (
                                "contract"
                                if gap
                                else "fixed_instance"
                            ),
                            "classification": (
                                "schedule_gap_candidate"
                                if gap
                                else "fixed_instance_candidate_lowerable"
                            ),
                            "owner_scope": "schedule",
                            "owner_relation": None,
                            "compiler_check": "not_run" if gap else "lowerable",
                            "reason": (
                                "Runtime scalar extent and masked indexed store are missing."
                                if gap
                                else "One fixed Schedule lowers."
                            ),
                            "missing_ir": (
                                {
                                    "capability": "Runtime scalar indexed store",
                                    "irreducible_semantics": "One runtime masked scatter.",
                                }
                                if gap
                                else None
                            ),
                        },
                        "delta_expressibility": {
                            "classification": "not_applicable"
                        },
                    },
                )
            write(root / "pool.plan.json", {"selected": 2, "entries": entries})

            result = summarize(root, expected_count=2)

            self.assertEqual(result["records"], 2)
            self.assertEqual(
                result["primary_class_counts"],
                {
                    "fixed_instance_candidate_lowerable": 1,
                    "schedule_gap_candidate": 1,
                },
            )
            self.assertEqual(
                result["assessment_scope_counts"],
                {
                    "contract": 1,
                    "fixed_instance": 1,
                },
            )
            self.assertEqual(
                result["retrieval_signal_counts"]["runtime_parameterization"],
                1,
            )
            self.assertEqual(
                result["retrieval_signal_counts"][
                    "indexed_addressing_or_scatter"
                ],
                1,
            )
            self.assertEqual(
                result["claim_boundary"]["semantic_binding"], "reviewer_claimed"
            )
            rendered = markdown(result)
            self.assertIn("## Retrieval-only signals", rendered)
            self.assertIn("fixed_instance_candidate_lowerable", rendered)
            self.assertNotIn("## Overlapping gap signals", rendered)

            legacy_path = (
                root
                / "cases/case-000002/reviews/cases/case-000002/checked.json"
            )
            legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
            legacy["schema"] = "open-cake.aka-expressibility-checked.v2"
            legacy["primary_class"] = "schedule_candidate_lowerable"
            legacy["complete_parent_expressibility"][
                "classification"
            ] = "schedule_candidate_lowerable"
            legacy["complete_parent_expressibility"].pop("assessment_scope")
            write(legacy_path, legacy)

            legacy_result = summarize(root, expected_count=2)
            self.assertEqual(
                legacy_result["assessment_scope_counts"]["legacy_unspecified"], 1
            )


if __name__ == "__main__":
    unittest.main()
