from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import evaluate_qsa_candidate as evaluator  # noqa: E402
from open_cake_ir.compiler import Compiler, EmpiricalCostModel  # noqa: E402
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
        cls.produced = cls._produce_static_profiles()

    @classmethod
    def _produce_static_profiles(cls) -> dict[str, object]:
        # This fresh Program binds current sources. Historical Program files remain
        # untouched; corpus admission and the external build are software fixtures,
        # never evidence of released-Executor or GPU acceptance.
        program = json.loads((ROOT / evaluator._PROGRAM_PATH).read_text())
        program["program_id"] = "qsa-feedback-software-projection-fixture"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for relative in (
                "compiler/targets/sm_100a.json",
                evaluator._WORKLOAD_PATH,
                *(node["schedule"] for node in program["nodes"]),
            ):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, destination)
            for node in program["nodes"]:
                assessment = cls.compiler.assess_file(root / node["schedule"])
                node["schedule_sha256"] = assessment.schedule_sha256
                node["lowering_source_sha256"] = cls.compiler.lower(assessment).source_sha256
            program_path = root / evaluator._PROGRAM_PATH
            program_path.parent.mkdir(parents=True, exist_ok=True)
            program_path.write_text(json.dumps(program))
            candidate = {"nodes": [
                {"id": node["id"], "schedule": node["schedule"]}
                for node in program["nodes"]
            ]}

            def build(request):
                return SimpleNamespace(artifact_payloads={
                    "lowered_source": request.source,
                    "ptx": b"software fixture, not compiled PTX",
                    "cubin": b"software fixture, not a launchable kernel",
                    "launch_manifest": json.dumps({
                        "kernel_name": request.entry_point,
                        "grid": [1, 1, 1], "block": [1, 1, 1],
                        "dynamic_shared_memory_bytes": 0,
                    }).encode(),
                })

            with (
                patch.object(evaluator.Compiler, "load", return_value=cls.compiler),
                patch.object(cls.compiler, "check_corpus", return_value=SimpleNamespace(passed=True)),
                patch.object(evaluator.TritonToolchainBuilder, "build", side_effect=build),
            ):
                metrics = evaluator._compile_open_cake(root, root, candidate, root / "built")
            retained = json.loads((root / "built/static-profile.json").read_text())
            if retained != {"schema_version": 1, **metrics}:
                raise AssertionError("producer receipt differs from returned compile metrics")
            return dict(metrics)

    def _produced_result(self) -> dict[str, object]:
        result = _completed()
        result["metrics"]["compile"] = json.loads(json.dumps(self.produced))
        return result

    def _cli(self, *arguments: str) -> dict[str, object]:
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "tools/project_qsa_feedback.py"), *arguments],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

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

    def test_static_profile_preserves_numeric_abstention_explanations(self) -> None:
        result = _completed()
        result["metrics"]["compile"] = {
            "kind": "open_cake_program_static_profile",
            "nodes": {
                "score_topk": {
                    "work": {"flops": 100, "compulsory_read_bytes": 20},
                    "residency": {
                        "ctas_per_sm_upper_bound": 8,
                        "binding_resource": "threads",
                        "logical_register_pressure_per_thread": 72,
                        "bounds": [],
                    },
                    "lowering": {
                        "generated_source_bytes": 7310,
                        "top_k": [{"k": 512, "merge_width": 1024}],
                        "explicit_barrier_count": 0,
                        "tile_loop_count": 1,
                        "runtime_indexed_buffers": [],
                    },
                    "ncu_metrics": [
                        {
                            "metric": "launch__registers_per_thread",
                            "estimate_kind": "unknown",
                            "value": None,
                            "unit": "register/thread",
                            "coverage": "compiled backend allocation",
                            "reasons": ["logical register pressure proxy=72"],
                            "missing": ["compiled-kernel register allocation"],
                        },
                        {
                            "metric": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                            "estimate_kind": "unknown",
                            "value": None,
                            "unit": "%",
                            "coverage": "uncalibrated",
                            "reasons": [],
                            "missing": ["B200 calibration"],
                        },
                    ],
                }
            },
        }
        feedback = qsa_evaluation_feedback(result, arm="open_cake")
        node = feedback["compiler"]["nodes"]["score_topk"]

        self.assertEqual(node["residency"]["logical_register_pressure_per_thread"], 72)
        self.assertEqual(node["ncu_estimates"], [])
        self.assertEqual(
            node["ncu_abstentions"],
            result["metrics"]["compile"]["nodes"]["score_topk"]["ncu_metrics"],
        )

    def test_real_producer_keeps_local_findings_and_profile_domains(self) -> None:
        feedback = qsa_evaluation_feedback(self._produced_result(), arm="open_cake")
        for node_id, profile in self.produced["nodes"].items():
            with self.subTest(node=node_id):
                node = feedback["compiler"]["nodes"][node_id]
                self.assertEqual(node["findings"], profile["findings"])
                self.assertEqual(node["residency"], profile["residency"])
                self.assertEqual(node["work"], profile["work"])
                self.assertEqual(node["abstentions"], profile["abstentions"])
                self.assertEqual(node["ncu_abstentions"], [
                    metric for metric in profile["ncu_metrics"]
                    if metric["estimate_kind"] == "unknown"
                ])
                self.assertEqual(node["ncu_estimates"], [
                    metric for metric in profile["ncu_metrics"]
                    if metric["estimate_kind"] != "unknown"
                ])
                self.assertNotIn("empirical_cost", node)
        topk = feedback["compiler"]["nodes"]["score_topk"]
        assessment = self.compiler.assess_file(ROOT / "corpus/schedules/qsa-score-topk-t32768.json")
        self.assertEqual(topk["findings"], qsa_compiler_feedback(assessment)["findings"])
        finding = next(item for item in topk["findings"] if item["code"] == "RESIDENCY_BOUND")
        self.assertEqual(finding["path"], "roles")
        self.assertFalse(finding["blocks_acceptance"])
        self.assertFalse(finding["blocks_lowering"])
        register = next(item for item in topk["ncu_abstentions"] if item["metric"] == "launch__registers_per_thread")
        self.assertIsNone(register["value"])
        self.assertIn("compiled-kernel register allocation", register["missing"])
        self.assertTrue(any("tl.dot" in note for note in topk["abstentions"]))
        self.assertTrue(any("tl.topk" in note for note in topk["abstentions"]))
        self.assertTrue(topk["residency"]["bounds"])

    def test_public_evaluation_and_turn_cli_preserve_bounded_producer_feedback(self) -> None:
        result = self._produced_result()
        topk = result["metrics"]["compile"]["nodes"]["score_topk"]
        topk["raw_report"] = "EXCLUDED_RAW_REPORT"
        topk["lowering"]["source"] = "EXCLUDED_LOWERED_SOURCE"
        result["artifacts"] = {"ptx": "EXCLUDED_ASSEMBLY"}
        expected = qsa_evaluation_feedback(result, arm="open_cake")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(result))
            evaluation = self._cli("evaluation", "--arm", "open_cake", str(path))
            turn = self._cli(
                "turn", "--arm", "open_cake", "--run-id", "open_cake-1",
                "--turn", "2", "--cumulative-provider-tokens", "1200",
                "--thread-id", "same-thread", "--maximum-candidates-per-turn", "3", str(path),
            )
        self.assertEqual(evaluation, expected)
        self.assertEqual(turn, {
            "run_id": "open_cake-1", "arm": "open_cake", "turn": 2,
            "cumulative_provider_tokens": 1200, "thread_id": "same-thread",
            "maximum_candidates_per_turn": 3, "feedback": expected,
        })
        self.assertNotIn("EXCLUDED_", json.dumps(turn))
        self.assertNotIn("candidate_samples_ms", turn["feedback"]["timing"])

    def test_public_compiler_cli_retains_local_findings_and_unknown_metric_details(self) -> None:
        path = ROOT / "corpus/schedules/qsa-score-topk-t32768.json"
        assessment = self.compiler.assess_file(path)
        feedback = self._cli("compiler", "--revision", "compiler/revision.json", str(path))
        self.assertEqual(feedback["findings"], qsa_compiler_feedback(assessment)["findings"])
        profile = self.compiler.profile(assessment).as_dict()
        self.assertEqual(feedback["static_profile"], profile)

    def test_empirical_next_turn_keeps_decisions_and_leaves_supplier_payloads_in_source(self) -> None:
        schedule = json.loads((ROOT / "corpus/schedules/qsa-score-topk-t32768.json").read_text())
        assessment = self.compiler.assess(schedule)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        model_path = Path(directory.name) / "model.json"
        result_path = Path(directory.name) / "result.json"
        for covered in (True, False):
            with self.subTest(covered=covered):
                document = {
                    "schema_version": 2, "model_id": "synthetic-feedback-transport-only",
                    "compiler_revision_id": assessment.compiler_revision_id if covered else "other-revision",
                    "compiler_revision_sha256": assessment.compiler_revision_sha256,
                    "target": "sm_100a",
                    "context": {
                        "timer": "synthetic; no measurement", "cache_protocol": "synthetic",
                        "runtime": {
                            "compiler_version": "synthetic",
                            "supplier_detail": "EXCLUDED_RUNTIME_PAYLOAD" * 100,
                        },
                        "input_scope": "software fixture only",
                    },
                    "reported_evidence": {
                        "kind": "synthetic; no calibration qualification",
                        "raw_samples_us": [10.0] * 10000,
                        "raw_report": "EXCLUDED_EMPIRICAL_REPORT",
                    },
                    "curves": [{
                        "template": schedule,
                        "varying_dimensions": [{"buffer": "index_q", "dimension": 0}],
                        "extent_multiple": 32768,
                        "points": [{"extent": 32768, "kernel_us": 10}, {"extent": 65536, "kernel_us": 20}],
                        "relative_error_envelope": 0.1,
                    }],
                }
                source_text = json.dumps(document)
                model_path.write_text(source_text)
                model = EmpiricalCostModel.load(model_path)
                profile = self.compiler.profile(assessment, cost_model=model).as_dict()
                result = self._produced_result()
                result["metrics"]["compile"]["nodes"]["score_topk"] = profile
                request = qsa_next_turn_request(
                    run_id="open_cake-1", arm="open_cake", turn=2,
                    cumulative_provider_tokens=1200, thread_id="same-thread",
                    maximum_candidates_per_turn=3, result=result,
                )
                cost = request.feedback["compiler"]["nodes"]["score_topk"]["empirical_cost"]
                expected = {
                    key: value for key, value in profile["empirical_cost"].items()
                    if key not in {"context", "reported_evidence"}
                }
                self.assertEqual(cost, expected)
                self.assertEqual(cost["covered"], covered)
                if covered:
                    self.assertEqual(cost["predicted_kernel_us"], 10)
                    self.assertEqual(cost["empirical_range_us"], [9, 11])
                else:
                    self.assertIsNone(cost["predicted_kernel_us"])
                    self.assertIsNone(cost["empirical_range_us"])
                    self.assertIn("Revision", cost["reason"])
                compiler_feedback = qsa_compiler_feedback(assessment, static_profile=profile)
                self.assertEqual(compiler_feedback["static_profile"]["empirical_cost"], expected)
                for metrics in (
                    result["metrics"]["compile"],
                    # Retained Compiler feedback may contain the full original profile.
                    {**compiler_feedback, "static_profile": profile},
                ):
                    result["metrics"]["compile"] = metrics
                    result_path.write_text(json.dumps(result))
                    turn = self._cli(
                        "turn", "--arm", "open_cake", "--run-id", "open_cake-1",
                        "--turn", "2", "--cumulative-provider-tokens", "1200",
                        "--thread-id", "same-thread", str(result_path),
                    )
                    self.assertEqual(turn["thread_id"], "same-thread")
                    projected = turn["feedback"]["compiler"]
                    node = (projected["nodes"]["score_topk"]
                            if "nodes" in projected else projected["static_profile"])
                    self.assertEqual(node["empirical_cost"], expected)
                    self.assertNotIn("raw_samples_us", json.dumps(turn))
                    self.assertNotIn("EXCLUDED_", json.dumps(turn))
                self.assertEqual(model_path.read_text(), source_text)
                self.assertEqual(profile["empirical_cost"]["context"], document["context"])
                self.assertEqual(profile["empirical_cost"]["reported_evidence"], document["reported_evidence"])

    def test_direct_cuda_keeps_its_own_diagnostics_without_cake_profile(self) -> None:
        result = _completed()
        result["metrics"]["compile"] = {
            "kind": "direct_cuda_toolchain", "static_profile": None,
            "reason": "direct CUDA has no Open Cake Schedule authority",
        }
        feedback = qsa_evaluation_feedback(result, arm="direct_cuda")
        self.assertEqual(feedback["compiler"], result["metrics"]["compile"])
        self.assertEqual(feedback["profile"], result["metrics"]["profile"])
        self.assertNotIn("nodes", feedback["compiler"])

    def test_malformed_profiles_still_fail_at_the_projection_boundary(self) -> None:
        for metrics in ({}, ["metric-name"], [{"metric": 7}]):
            with self.subTest(metrics=metrics):
                result = self._produced_result()
                result["metrics"]["compile"]["nodes"]["score_topk"]["ncu_metrics"] = metrics
                with self.assertRaisesRegex(ValueError, "QSA static profile"):
                    qsa_evaluation_feedback(result, arm="open_cake")

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
