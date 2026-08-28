from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler  # noqa: E402
from open_cake_ir.lab import (  # noqa: E402
    qsa_compiler_feedback,
    qsa_evaluation_feedback,
    qsa_next_turn_request,
)


def _completed() -> dict[str, object]:
    return {
        "schema": "kernelinfra.run-result.v1",
        "task_id": "open-cake-qsa-prefill-t32768-seed-v1",
        "outcome": "completed",
        "validity": "valid",
        "terminal_reason": None,
        "stages": [
            {"id": "compile", "status": "passed", "summary": "compiled"},
            {"id": "correctness", "status": "passed", "summary": "correct"},
            {"id": "benchmark", "status": "passed", "summary": "measured"},
        ],
        "workloads": [
            {
                "id": "qsa-prefill-t32768",
                "correct": True,
                "candidate_ms": 7.0,
                "baseline_ms": 0.7,
                "speedup": 0.1,
                "stable": True,
                "candidate_samples_ms": [7.0] * 125,
                "baseline_samples_ms": [0.7] * 125,
            }
        ],
        "metrics": {
            "correctness": {
                "passed": True,
                "match_fraction": 1.0,
                "max_abs_diff": 0.01,
            },
            "profile": {
                "kind": "ncu_kernel_attribution",
                "kernel_name": "score_topk",
                "occupancy": {"resident_ctas_per_sm": 1},
            },
        },
    }


class QsaFeedbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def test_compiler_feedback_preserves_local_finding_and_route(self) -> None:
        path = ROOT / "corpus/schedules/top-k-b8-smoke.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        next(buffer for buffer in document["buffers"] if buffer["name"] == "scores")[
            "shape"
        ] = [8, 192]
        next(buffer for buffer in document["buffers"] if buffer["name"] == "score_row")[
            "shape"
        ] = [192]
        feedback = qsa_compiler_feedback(self.compiler.assess(document))

        self.assertTrue(feedback["actionable"])
        self.assertEqual(feedback["routed_to"], "candidate")
        finding = next(
            item
            for item in feedback["findings"]
            if item["code"] == "TOP_K_SOURCE_UNLOWERABLE"
        )
        self.assertEqual(finding["path"], "operations[1].reads")
        self.assertTrue(finding["blocks_lowering"])

    def test_completed_feedback_is_bounded_and_keeps_actionable_metrics(self) -> None:
        feedback = qsa_evaluation_feedback(_completed(), arm="open_cake")

        self.assertEqual(feedback["kind"], "evaluation")
        self.assertTrue(feedback["actionable"])
        self.assertEqual(feedback["timing"]["candidate_ms"], 7.0)
        self.assertEqual(feedback["profile"]["kernel_name"], "score_topk")
        self.assertNotIn("candidate_samples_ms", feedback["timing"])

    def test_infrastructure_fault_never_tells_the_agent_to_edit(self) -> None:
        result = _completed()
        result.update(
            outcome="infra_error",
            validity="unknown",
            terminal_reason="SSH observer unavailable",
        )
        result["stages"][-1] = {
            "id": "benchmark",
            "status": "failed",
            "summary": "collector timed out",
        }
        feedback = qsa_evaluation_feedback(result, arm="direct_cuda")

        self.assertEqual(feedback["kind"], "infrastructure_fault")
        self.assertFalse(feedback["actionable"])
        self.assertIn("retain the candidate unchanged", feedback["instruction"])
        self.assertNotIn("profile", feedback)

    def test_compile_rejection_uses_the_stage_owned_route(self) -> None:
        result = _completed()
        result.update(outcome="rejected", validity="invalid")
        result["stages"] = [
            {"id": "compile", "status": "failed", "summary": "ptxas rejected"}
        ]
        result["workloads"] = []
        result["metrics"] = {
            "compile": {
                "kind": "compiler",
                "stage": "compile",
                "routed_to": "verifier",
                "diagnostic": "unsupported instruction",
            }
        }
        feedback = qsa_evaluation_feedback(result, arm="open_cake")

        self.assertEqual(feedback["routed_to"], "verifier")
        self.assertEqual(feedback["compiler"]["diagnostic"], "unsupported instruction")

    def test_terminal_result_becomes_same_thread_next_turn_feedback(self) -> None:
        request = qsa_next_turn_request(
            run_id="open_cake-1",
            arm="open_cake",
            turn=2,
            cumulative_provider_tokens=1200,
            thread_id="019d1111-2222-7333-8444-555555555555",
            maximum_candidates_per_turn=3,
            result=_completed(),
        )

        self.assertEqual(request.turn, 2)
        self.assertEqual(request.thread_id, "019d1111-2222-7333-8444-555555555555")
        self.assertEqual(request.feedback["kind"], "evaluation")


if __name__ == "__main__":
    unittest.main()
