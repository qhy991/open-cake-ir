from __future__ import annotations

import importlib.util

import json
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from open_cake_ir.tasks.workloads import load_workload

from open_cake_ir.evaluation import BrokerAttempt, EvaluationProtocol, EvaluationReceipt, LaunchableCandidate, LaunchObservation, NCU_ATTRIBUTION_METRICS, PairedTimingProtocol, WorkloadContract, build_ncu_attribution_profile, derive_paired_timing, evaluate_with_admission_recovery
from open_cake_ir.tasks.flash_kmeans.cuda_manifest import CudaLaunchManifest, parse_cuda_launch_manifest
from open_cake_ir.tasks.flash_kmeans.portfolio import ExactShapeDispatcher, PortfolioArtifact, PortfolioCaseObservation, evaluate_portfolio_observations, replay_portfolio_receipt
from open_cake_ir.tasks.flash_kmeans.workload import assignment_raw_sha256, classify_flash_kmeans_output, flash_kmeans_metrics, flash_kmeans_oracle, generate_flash_kmeans_case, tensor_raw_sha256
from open_cake_ir.tasks.flash_kmeans.correctness import audit_flash_kmeans_assignment
from open_cake_ir.tasks.flash_kmeans.evaluation import evaluate_flash_kmeans
from open_cake_ir.tasks.tinygemm.evaluation import evaluate_tinygemm, tinygemm_metrics, tinygemm_oracle
from open_cake_ir.tasks.flash_kmeans.legacy import replay_legacy_r45_result


def _profile_fixture(candidate_sha256: str, case_id: str, kernel_name: str) -> bytes:
    values = (95, 2, 2, 16, 8, 61.0, 24.0, 41.0, 37.0, 18.0, 3.0)
    lines = ['"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"']
    for index, (metric, value) in enumerate(zip(NCU_ATTRIBUTION_METRICS, values)):
        unit = "%" if "pct" in metric else "count"
        lines.append(f'"{index}","{kernel_name}","{metric}","{unit}","{value}"')
    return build_ncu_attribution_profile(
        candidate_sha256=candidate_sha256,
        case_id=case_id,
        kernel_name=kernel_name,
        ncu_version="2026.1.1.0",
        ncu_executable_sha256="e" * 64,
        stdout=("\n".join(lines) + "\n").encode(),
        stderr=b"==PROF== fixture\n",
    )


class EvaluationContractTests(unittest.TestCase):
    def test_evaluation_receipt_cannot_contradict_raw_correctness_or_timing(self) -> None:
        launch = b'{"candidate_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}'
        with self.assertRaisesRegex(ValueError, "correctness disposition"):
            self._evaluation_receipt_with_raw(
                launch,
                correctness={"passed": False},
                correctness_passed=True,
                samples=[1.0],
                pooled=1.0,
            )
        with self.assertRaisesRegex(ValueError, "timing summary"):
            self._evaluation_receipt_with_raw(
                launch,
                correctness={"passed": True},
                correctness_passed=True,
                samples=[999.0],
                pooled=1.0,
            )

    @staticmethod
    def _evaluation_receipt_with_raw(
        launch,
        *,
        correctness,
        correctness_passed,
        samples,
        pooled,
    ):
        from open_cake_ir.evaluation import EvaluationReceipt

        return EvaluationReceipt(
            candidate_sha256="a" * 64,
            workload_sha256="b" * 64,
            evaluation_protocol_sha256="c" * 64,
            purpose="search",
            case_id="headline_b32",
            correctness_passed=correctness_passed,
            correctness=correctness,
            kernel_calls=1,
            fallback_calls=0,
            launch_receipt_sha256=sha256(launch).hexdigest(),
            timing={"measurement_quality_passed": True, "pooled_median_ms": pooled},
            artifact_payloads={
                "correctness_output": json.dumps(correctness).encode(),
                "launch_receipt": launch,
                "timing_samples": json.dumps({"cohorts_ms": [samples]}).encode(),
            },
        )

    def test_actual_r45_raw_result_binds_its_own_cubins_and_launch_specs(self) -> None:
        workload = load_workload(
            ROOT / "contracts/workloads/flash-kmeans-assign-v2.json"
        )
        raw = json.loads((ROOT / "tests/fixtures/r45-portfolio-result.json").read_text())

        artifact, receipt = replay_legacy_r45_result(
            raw,
            workload,
            expected_raw_fixture_sha256="30cfb93234adb72cff8f790811ca03dc6575b3f301f46abda8fc19f42d0f08b8",
            expected_result_sha256="2eab2ad2c9e386195901cef861f26f2d0264aa3544df6d191039026b41294e32",
            legacy_contract_sha256="ed03090039bd5187cd8bc2026586ff014e303ce9ba7d4e15645a176286511aa8",
            broker_job_id="gpuq-abbdd239d57f",
        )

        by_case = {entry.case_id: entry for entry in artifact.entries}
        self.assertEqual(
            by_case["b32_smoke"].candidate.artifact_roles["cubin"],
            "67c9c82a61533c34b568748836bd15ab8217fde100a451f477de644c8fcda652",
        )
        self.assertTrue(receipt.correctness_supported)
        self.assertTrue(receipt.stable_kernel_performance_supported)
        self.assertFalse(receipt.stable_dispatcher_performance_supported)
        self.assertEqual(
            receipt.pooled_medians_ms["headline_b32"],
            {"kernel": 1.156062, "dispatcher": 1.283412},
        )

    def test_legacy_r45_fixture_and_self_hash_preimage_are_checked_independently(self) -> None:
        workload = load_workload(
            ROOT / "contracts/workloads/flash-kmeans-assign-v2.json"
        )
        raw = json.loads((ROOT / "tests/fixtures/r45-portfolio-result.json").read_text())
        result_sha256 = "2eab2ad2c9e386195901cef861f26f2d0264aa3544df6d191039026b41294e32"
        common = {
            "legacy_contract_sha256": (
                "ed03090039bd5187cd8bc2026586ff014e303ce9ba7d4e15645a176286511aa8"
            ),
            "broker_job_id": "gpuq-abbdd239d57f",
        }

        with self.assertRaisesRegex(ValueError, "raw fixture"):
            replay_legacy_r45_result(
                raw,
                workload,
                expected_raw_fixture_sha256="0" * 64,
                expected_result_sha256=result_sha256,
                **common,
            )

        changed_self_hash = json.loads(json.dumps(raw))
        changed_self_hash["result_sha256"] = "0" * 64
        changed_self_hash_sha256 = sha256(
            json.dumps(
                changed_self_hash,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "embedded result"):
            replay_legacy_r45_result(
                changed_self_hash,
                workload,
                expected_raw_fixture_sha256=changed_self_hash_sha256,
                expected_result_sha256=result_sha256,
                **common,
            )

        changed_preimage = json.loads(json.dumps(raw))
        changed_preimage["status"] = "passed"
        changed_preimage_sha256 = sha256(
            json.dumps(
                changed_preimage,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "canonical result preimage"):
            replay_legacy_r45_result(
                changed_preimage,
                workload,
                expected_raw_fixture_sha256=changed_preimage_sha256,
                expected_result_sha256=result_sha256,
                **common,
            )

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires optional Torch for Flash-KMeans CPU tensors/oracle")
    def test_two_launchable_candidates_cross_one_common_evaluation_interface(self) -> None:
        workload = load_workload(
            ROOT / "contracts/workloads/flash-kmeans-assign.json"
        )
        protocol = EvaluationProtocol(
            protocol_id="duplicate-tie-confirmatory-v1",
            purpose="confirmatory",
            workload_sha256=workload.canonical_sha256,
            case_id="duplicate_tie",
            timing="none",
        )

        class OracleLauncher:
            def __init__(self, workload_contract: WorkloadContract) -> None:
                self.workload = workload_contract

            def launch(self, candidate, tokens, centroids, centroid_sq):
                output = flash_kmeans_oracle(
                    self.workload,
                    tokens,
                    centroids,
                    case_id="duplicate_tie",
                )
                receipt = sha256(
                    (candidate.candidate_sha256 + candidate.entry_point).encode()
                ).hexdigest()
                return LaunchObservation(output, 1, 0, receipt)

        candidates = (
            LaunchableCandidate(
                candidate_sha256="1" * 64,
                target="sm_100a",
                entry_point="open_cake_entry",
                artifact_roles={"cubin": "2" * 64},
                launch_spec_sha256="3" * 64,
            ),
            LaunchableCandidate(
                candidate_sha256="4" * 64,
                target="sm_100a",
                entry_point="direct_cuda_entry",
                artifact_roles={"cubin": "5" * 64},
                launch_spec_sha256="6" * 64,
            ),
        )

        receipts = [
            evaluate_flash_kmeans(
                candidate,
                workload,
                protocol,
                OracleLauncher(workload),
                device="cpu",
            )
            for candidate in candidates
        ]

        self.assertEqual([receipt.correctness_passed for receipt in receipts], [True, True])
        self.assertEqual(receipts[0].correctness, receipts[1].correctness)
        self.assertNotEqual(receipts[0].candidate_sha256, receipts[1].candidate_sha256)
        self.assertFalse(hasattr(receipts[0], "arm"))

    def test_compiler_expanded_source_is_not_the_lowering_input_artifact(self) -> None:
        with self.assertRaisesRegex(ValueError, "distinct artifacts"):
            LaunchableCandidate(
                candidate_sha256="1" * 64,
                target="sm_100a",
                entry_point="kernel",
                artifact_roles={
                    "lowered_source": "2" * 64,
                    "compiler_expanded_source": "2" * 64,
                    "cubin": "3" * 64,
                },
                launch_spec_sha256="4" * 64,
            )

    def test_manifest_driven_dispatch_rejects_unsupported_shape_before_launch(self) -> None:
        workload = load_workload(
            ROOT / "contracts/workloads/flash-kmeans-assign-v2.json"
        )
        candidate = LaunchableCandidate(
            candidate_sha256="1" * 64,
            target="sm_100a",
            entry_point="kernel",
            artifact_roles={"cubin": "2" * 64},
            launch_spec_sha256="3" * 64,
        )
        artifact = PortfolioArtifact.build(
            workload,
            "4" * 64,
            {"b32_smoke": candidate},
        )
        launches: list[str] = []

        class Launcher:
            def launch(self, selected, arguments):
                launches.append(selected.candidate_sha256)

        class Tensor:
            def __init__(self, shape, dtype, pointer):
                self.shape = shape
                self.dtype = dtype
                self.device = "cuda:0"
                self.is_cuda = True
                self._pointer = pointer

            def is_contiguous(self):
                return True

            def data_ptr(self):
                return self._pointer

        dispatcher = ExactShapeDispatcher(artifact, Launcher())
        supported = (
            Tensor((32, 512, 128), "torch.bfloat16", 1),
            Tensor((32, 1024, 128), "torch.bfloat16", 2),
            Tensor((32, 1024), "torch.float32", 3),
            Tensor((32, 512), "torch.int32", 4),
        )
        dispatcher.dispatch(supported)
        unsupported = (
            Tensor((2, 512, 128), "torch.bfloat16", 5),
            Tensor((2, 1024, 128), "torch.bfloat16", 6),
            Tensor((2, 1024), "torch.float32", 7),
            Tensor((2, 512), "torch.int32", 8),
        )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            dispatcher.dispatch(unsupported)

        self.assertEqual(len(launches), 1)
        self.assertEqual(dispatcher.kernel_calls, 1)
        self.assertEqual(dispatcher.rejections, 1)
        self.assertEqual(dispatcher.fallback_calls, 0)

    def test_portfolio_correctness_survives_unstable_dispatcher_measurement(self) -> None:
        workload = load_workload(
            ROOT / "contracts/workloads/flash-kmeans-assign-v2.json"
        )
        candidates = {
            case_id: LaunchableCandidate(
                candidate_sha256=str(index) * 64,
                target="sm_100a",
                entry_point=f"kernel_{case_id}",
                artifact_roles={"cubin": str(index + 3) * 64},
                launch_spec_sha256=str(index + 6) * 64,
            )
            for index, case_id in enumerate(
                ("headline_b32", "b32_smoke", "public_b1"), start=1
            )
        }
        artifact = PortfolioArtifact.build(workload, "a" * 64, candidates)
        stable = tuple(tuple([1.0] * 25) for _ in range(5))
        unstable_three = tuple(
            tuple([1.0, 2.0] * 12 + [1.0]) if index < 3 else tuple([1.0] * 25)
            for index in range(5)
        )
        unstable_two = tuple(
            tuple([1.0, 2.0] * 12 + [1.0]) if index < 2 else tuple([1.0] * 25)
            for index in range(5)
        )
        observations = {
            "headline_b32": PortfolioCaseObservation(
                True,
                True,
                True,
                stable,
                stable,
                {
                    "direct_preflight": {"passed": True},
                    "dispatcher_preflight": {"passed": True},
                    "postflight": {"passed": True},
                },
                candidates["headline_b32"].canonical_sha256,
                candidates["headline_b32"].artifact_roles["cubin"],
                candidates["headline_b32"].launch_spec_sha256,
                {
                    "candidate_record_sha256": candidates["headline_b32"].canonical_sha256,
                    "cubin_sha256": candidates["headline_b32"].artifact_roles["cubin"],
                    "launch_spec_sha256": candidates["headline_b32"].launch_spec_sha256,
                    "module_loaded": True,
                    "gpu_uuid": "GPU-fixture",
                    "broker_job_id": "gpuq-000000000001",
                },
            ),
            "b32_smoke": PortfolioCaseObservation(
                True,
                True,
                True,
                stable,
                unstable_three,
                {
                    "direct_preflight": {"passed": True},
                    "dispatcher_preflight": {"passed": True},
                    "postflight": {"passed": True},
                },
                candidates["b32_smoke"].canonical_sha256,
                candidates["b32_smoke"].artifact_roles["cubin"],
                candidates["b32_smoke"].launch_spec_sha256,
                {
                    "candidate_record_sha256": candidates["b32_smoke"].canonical_sha256,
                    "cubin_sha256": candidates["b32_smoke"].artifact_roles["cubin"],
                    "launch_spec_sha256": candidates["b32_smoke"].launch_spec_sha256,
                    "module_loaded": True,
                    "gpu_uuid": "GPU-fixture",
                    "broker_job_id": "gpuq-000000000001",
                },
            ),
            "public_b1": PortfolioCaseObservation(
                True,
                True,
                True,
                stable,
                unstable_two,
                {
                    "direct_preflight": {"passed": True},
                    "dispatcher_preflight": {"passed": True},
                    "postflight": {"passed": True},
                },
                candidates["public_b1"].canonical_sha256,
                candidates["public_b1"].artifact_roles["cubin"],
                candidates["public_b1"].launch_spec_sha256,
                {
                    "candidate_record_sha256": candidates["public_b1"].canonical_sha256,
                    "cubin_sha256": candidates["public_b1"].artifact_roles["cubin"],
                    "launch_spec_sha256": candidates["public_b1"].launch_spec_sha256,
                    "module_loaded": True,
                    "gpu_uuid": "GPU-fixture",
                    "broker_job_id": "gpuq-000000000001",
                },
            ),
        }
        route_counts = {
            "automatic_retries": 0,
            "compiler_invocations": 3,
            "module_loads": 3,
            "module_unloads": 3,
            "direct_preflight_calls": 3,
            "dispatcher_preflight_calls": 3,
            "cupti_candidate_calls": 540,
            "host_dispatch_calls": 450,
            "l2_flush_calls": 450,
            "dispatcher_postflight_calls": 3,
            "unsupported_probes": 1,
            "candidate_kernel_calls": 999,
            "dispatcher_kernel_calls": 456,
            "fallback_calls": 0,
            "selections": {case_id: 152 for case_id in observations},
        }

        receipt = evaluate_portfolio_observations(
            artifact,
            observations,
            evaluation_protocol_sha256="f" * 64,
            route_counts=route_counts,
            unsupported_kernel_call_delta=0,
        )

        self.assertTrue(receipt.correctness_supported)
        self.assertTrue(receipt.stable_kernel_performance_supported)
        self.assertFalse(receipt.stable_dispatcher_performance_supported)
        self.assertEqual(
            receipt.dispatcher_measurement_quality_by_case,
            {"headline_b32": "stable", "b32_smoke": "unstable", "public_b1": "unstable"},
        )
        replayed = replay_portfolio_receipt(
            receipt.document,
            artifact,
            expected_protocol_sha256="f" * 64,
        )
        self.assertEqual(replayed.canonical_sha256, receipt.canonical_sha256)
        tampered = json.loads(json.dumps(receipt.document))
        tampered["dispatcher_measurement_quality_by_case"]["b32_smoke"] = "stable"
        with self.assertRaisesRegex(ValueError, "projections"):
            replay_portfolio_receipt(
                tampered,
                artifact,
                expected_protocol_sha256="f" * 64,
            )

    def test_only_exact_zero_work_admission_failure_can_resubmit_once(self) -> None:
        candidate = LaunchableCandidate(
            candidate_sha256="1" * 64,
            target="sm_100a",
            entry_point="kernel",
            artifact_roles={"cubin": "2" * 64},
            launch_spec_sha256="3" * 64,
        )
        receipt = type("ReceiptFixture", (), {})
        valid_receipt = self._evaluation_receipt(candidate.candidate_sha256)
        calls: list[int] = []

        def submit(number):
            calls.append(number)
            return BrokerAttempt(
                job_id=f"gpuq-{number:012x}",
                mode="exclusive",
                candidate_sha256=candidate.candidate_sha256,
                manifest_sha256="4" * 64,
                policy_sha256="5" * 64,
                evaluator_arguments_sha256="6" * 64,
                admitted=number == 2,
                error="gpu_admission_differs" if number == 1 else None,
                compiler_invocations=0,
                module_loads=0,
                preflight_calls=0,
                kernel_calls=0 if number == 1 else 1,
                timing_samples=0 if number == 1 else 125,
                fallback_calls=0,
                receipt=None if number == 1 else valid_receipt,
            )

        logical = evaluate_with_admission_recovery(candidate, submit)

        self.assertEqual(calls, [1, 2])
        self.assertEqual(len(logical.attempts), 2)
        self.assertIs(logical.final_receipt, valid_receipt)
        self.assertIsNotNone(receipt)

        with self.assertRaisesRegex(ValueError, "contradicts admission"):
            BrokerAttempt(
                job_id="gpuq-000000000003",
                mode="exclusive",
                candidate_sha256=candidate.candidate_sha256,
                manifest_sha256="4" * 64,
                policy_sha256="5" * 64,
                evaluator_arguments_sha256="6" * 64,
                admitted=False,
                error="gpu_admission_differs",
                compiler_invocations=0,
                module_loads=0,
                preflight_calls=0,
                kernel_calls=0,
                timing_samples=0,
                fallback_calls=0,
                receipt=valid_receipt,
            )

    @staticmethod
    def _evaluation_receipt(candidate_sha256):
        from open_cake_ir.evaluation import EvaluationReceipt

        return EvaluationReceipt(
            candidate_sha256=candidate_sha256,
            workload_sha256="7" * 64,
            evaluation_protocol_sha256="8" * 64,
            purpose="confirmatory",
            case_id="headline_b32",
            correctness_passed=True,
            correctness={},
            kernel_calls=1,
            fallback_calls=0,
            launch_receipt_sha256="9" * 64,
            timing={"measurement_quality_passed": True, "pooled_median_ms": 1.0},
        )

    def test_workload_semantics_and_oracle_are_closed_authority(self) -> None:
        original = json.loads(
            (ROOT / "contracts/workloads/flash-kmeans-assign.json").read_text()
        )
        mutations = (
            lambda document: document.update(
                {"semantics": {"definition": "do_the_opposite"}}
            ),
            lambda document: document.update({"oracle": {"kind": "trust_candidate"}}),
            lambda document: document.update({"validation": {}}),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as directory:
                document = json.loads(json.dumps(original))
                mutate(document)
                path = Path(directory) / "workload.json"
                path.write_text(json.dumps(document))
                with self.assertRaisesRegex(ValueError, "semantics|oracle|validation"):
                    load_workload(path)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires optional Torch for Flash-KMeans CPU tensors/oracle")
    def test_flash_kmeans_workload_generates_and_oracles_the_declared_tie_case(self) -> None:
        workload = load_workload(ROOT / "contracts/workloads/flash-kmeans-assign.json")

        tokens, centroids = generate_flash_kmeans_case(workload, "duplicate_tie", device="cpu")
        assignments = flash_kmeans_oracle(
            workload, tokens, centroids, case_id="duplicate_tie"
        )
        metrics = flash_kmeans_metrics(
            workload,
            tokens,
            centroids,
            assignments,
            assignments,
            case_id="duplicate_tie",
        )
        token_digest = tensor_raw_sha256(tokens)
        centroid_digest = tensor_raw_sha256(centroids)
        assignment_digest = assignment_raw_sha256(assignments)

        self.assertEqual(assignments.tolist(), [[0, 1, 2, 0]])
        self.assertEqual(str(assignments.dtype), "torch.int32")
        self.assertTrue(metrics["exact_match"])
        self.assertTrue(metrics["tie_aware_distance_match"])
        self.assertEqual(metrics["mismatch_count"], 0)
        self.assertEqual(token_digest[1], 1024)
        self.assertEqual(centroid_digest[1], 1024)
        self.assertNotEqual(token_digest[0], centroid_digest[0])
        self.assertEqual(assignment_digest[1], 16)
        self.assertEqual(
            assignment_digest[0],
            sha256(assignments.numpy().tobytes(order="C")).hexdigest(),
        )

        invalid = assignments.clone()
        invalid[0, 0] = 999
        passed, rejection = classify_flash_kmeans_output(
            workload,
            tokens,
            centroids,
            invalid,
            assignments,
            case_id="duplicate_tie",
        )
        self.assertFalse(passed)
        self.assertEqual(
            rejection,
            {
                "schema_version": 1,
                "failure_code": "candidate_output_contract_violation",
            },
        )

    def test_direct_cuda_candidate_exposes_one_closed_host_launch_manifest(self) -> None:
        source = (ROOT / "contracts/scaffolds/direct-cuda-headline-v1.cu").read_bytes()

        manifest = parse_cuda_launch_manifest(source)

        self.assertIsInstance(manifest, CudaLaunchManifest)
        self.assertEqual(manifest.target, "sm_100a")
        self.assertEqual(manifest.kernel_name, "cake_flash_kmeans_assign")
        self.assertEqual(manifest.grid, (512, 32, 1))
        self.assertEqual(manifest.block, (64, 1, 1))
        self.assertEqual(manifest.dynamic_shared_memory_bytes, 0)

    def test_r37_both_arms_cross_the_same_correctness_interface(self) -> None:
        workload = load_workload(ROOT / "contracts/workloads/flash-kmeans-assign.json")
        observed = {}
        for arm, filename in (
            ("cake_ir", "r37-cake-correctness-result.json"),
            ("cuda_ptx", "r37-cuda-correctness-result.json"),
        ):
            raw = json.loads((ROOT / "tests/fixtures" / filename).read_text())
            observed[arm] = audit_flash_kmeans_assignment(raw, workload, case_id="headline_b32")

        for observation in observed.values():
            self.assertTrue(observation.correctness_passed)
            self.assertFalse(observation.exact_match)
            self.assertEqual(observation.mismatch_count, 2)
            self.assertEqual(observation.near_tie_mismatch_count, 2)
            self.assertTrue(observation.tie_aware_distance_match)
            self.assertEqual(observation.max_chosen_distance_excess, 0.0)
            self.assertEqual(observation.output_sha256, "ee335f602c3e67ed34519bfa59ab130a42eef8676fcca5d18987916297d5c75f")
            self.assertEqual(observation.kernel_calls, 1)
            self.assertEqual(observation.fallback_calls, 0)
        self.assertEqual(observed["cake_ir"], observed["cuda_ptx"])

    def test_workload_contracts_own_semantics_without_study_policy(self) -> None:
        flash = load_workload(ROOT / "contracts/workloads/flash-kmeans-assign.json")
        tiny = load_workload(ROOT / "contracts/workloads/tinygemm2-stage4-v2.json")

        self.assertEqual(flash.case_ids, ("tail_nk", "batched_tail", "b32_smoke", "duplicate_tie", "public_b1", "headline_b32"))
        self.assertEqual(flash.case("headline_b32")["shape"], {"B": 32, "N": 65536, "K": 1024, "D": 128})
        self.assertEqual(tiny.case_ids, ("stage4_n8_m1024_k1024",))
        materialization = tiny.document["provenance"][1]
        materialization_path = ROOT / materialization["path"]
        self.assertEqual(
            sha256(materialization_path.read_bytes()).hexdigest(),
            materialization["raw_sha256"],
        )
        attempt = json.loads(materialization_path.read_text())
        parent = attempt["retained_parent_output"]
        parent_source = ROOT / parent["source_path"]
        parent_bytes = parent_source.read_bytes()
        self.assertEqual(sha256(parent_bytes).hexdigest(), parent["source_raw_sha256"])
        self.assertTrue(parent_bytes.endswith(b"\n"))
        self.assertEqual(
            sha256(parent_bytes[:-1]).hexdigest(),
            parent["legacy_source_raw_sha256"],
        )
        for contract in (flash, tiny):
            self.assertNotIn("provider", contract.document)
            self.assertNotIn("budget", contract.document)
            self.assertNotIn("estimand", contract.document)

    def test_tinygemm_v2_fails_closed_before_non_cuda_launch(self) -> None:
        workload = load_workload(
            ROOT / "contracts/workloads/tinygemm2-stage4-v2.json"
        )
        protocol = EvaluationProtocol(
            protocol_id="tinygemm-confirmatory-v1",
            purpose="confirmatory",
            workload_sha256=workload.canonical_sha256,
            case_id="stage4_n8_m1024_k1024",
            timing="none",
        )
        candidate = LaunchableCandidate(
            candidate_sha256="a" * 64,
            target="sm_100a",
            entry_point="tinygemm",
            artifact_roles={"cubin": "b" * 64},
            launch_spec_sha256="c" * 64,
        )

        class MustNotLaunch:
            def launch(self, selected, input_tensor, weight, bias):
                raise AssertionError("candidate launched before materialization admission")

        with self.assertRaisesRegex(ValueError, "exact materialization requires CUDA"):
            evaluate_tinygemm(
                candidate,
                workload,
                protocol,
                MustNotLaunch(),
                device="cpu",
            )

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires optional Torch for CPU tensor/oracle checks")
    def test_tinygemm_separates_parent_exactness_from_oracle_tolerance(self) -> None:
        import torch

        document = json.loads(
            (ROOT / "contracts/workloads/tinygemm2-stage4-v2.json").read_text()
        )
        parent_output = torch.zeros((8, 1024), dtype=torch.bfloat16)
        oracle = parent_output.clone()
        oracle[0, 0] = 0.0078125
        parent_raw = parent_output.view(torch.uint8).numpy().tobytes(order="C")
        document["cases"][0]["materialized"]["parent_output"] = {
            "sha256": sha256(parent_raw).hexdigest(),
            "size_bytes": len(parent_raw),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workload.json"
            path.write_text(json.dumps(document))
            workload = load_workload(path)

        metrics = tinygemm_metrics(
            workload,
            parent_output,
            oracle,
            case_id="stage4_n8_m1024_k1024",
        )
        self.assertTrue(metrics["bitwise_parent_equal"])
        self.assertTrue(metrics["tolerance_equal"])
        self.assertGreater(metrics["maximum_absolute_error"], 0.0)

        old_self_consistent = tinygemm_metrics(
            workload,
            oracle,
            oracle,
            case_id="stage4_n8_m1024_k1024",
        )
        self.assertFalse(old_self_consistent["bitwise_parent_equal"])
        self.assertTrue(old_self_consistent["tolerance_equal"])

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires optional Torch for CPU tensor/oracle checks")
    def test_tinygemm_oracle_replays_cpu_fp32_linear_bytes(self) -> None:
        import torch

        document = json.loads(
            (ROOT / "contracts/workloads/tinygemm2-stage4-v2.json").read_text()
        )
        generator = torch.Generator().manual_seed(17)
        input_tensor = torch.randn((8, 1024), generator=generator).to(torch.bfloat16)
        weight = torch.randn((1024, 1024), generator=generator).to(torch.bfloat16)
        bias = torch.randn((1024,), generator=generator).to(torch.bfloat16)
        expected = torch.nn.functional.linear(
            input_tensor.cpu().float(),
            weight.cpu().float(),
            bias.cpu().float(),
        ).to(torch.bfloat16)
        expected_raw = expected.view(torch.uint8).numpy().tobytes(order="C")
        document["cases"][0]["materialized"]["fp32_linear_bf16_oracle"] = {
            "sha256": sha256(expected_raw).hexdigest(),
            "size_bytes": len(expected_raw),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workload.json"
            path.write_text(json.dumps(document))
            workload = load_workload(path)

        observed = tinygemm_oracle(
            workload,
            input_tensor,
            weight,
            bias,
            case_id="stage4_n8_m1024_k1024",
        )

        self.assertEqual(observed.device.type, "cpu")
        self.assertTrue(torch.equal(observed, expected))

    def test_current_tinygemm_observation_replays_the_frozen_plan(self) -> None:
        result = json.loads(
            (
                ROOT
                / "inventory/V25_TINYGEMM2_V2_B200_CORRECTNESS_RESULT_20260825.json"
            ).read_text()
        )
        plan_path = ROOT / result["plan"]["path"]
        observation_path = ROOT / result["raw_observation"]["path"]
        plan = json.loads(plan_path.read_text())
        observation = json.loads(observation_path.read_text())
        self.assertEqual(observation["plan"], result["plan"])
        self.assertEqual(plan["route"]["backend"], "checked_cuda_asset")
        self.assertFalse(plan["route"]["generated"])
        self.assertEqual(plan["execution"]["automatic_retries"], 0)
        self.assertEqual(plan["execution"]["timing"], "none")
        self.assertEqual(len(observation["authority_byte_checks"]), 17)
        self.assertTrue(all(observation["authority_byte_checks"].values()))
        self.assertEqual(
            set(observation["materialized"]), set(plan["expected"]["materialized"])
        )
        close = observation["launch"]["close_receipt"]
        self.assertEqual(close["kernel_calls"], 1)
        self.assertEqual(close["fallback_calls"], 0)
        self.assertTrue(close["module_unloaded_after_synchronize"])
        self.assertEqual(observation["correctness"], {
            "bitwise_parent_equal": True,
            "maximum_absolute_error": 0.000244140625,
            "tolerance_equal": True,
        })
        self.assertTrue(observation["passed"])
        self.assertFalse(observation["performance_measured"])
        self.assertFalse(observation["scientific_claim_authorized"])
        summary = result["result"]
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(
            summary["authority_byte_check_count"],
            len(observation["authority_byte_checks"]),
        )
        self.assertTrue(summary["all_authority_byte_checks_passed"])
        self.assertTrue(summary["all_materialized_receipts_passed"])
        self.assertEqual(summary["kernel_calls"], close["kernel_calls"])
        self.assertEqual(summary["fallback_calls"], close["fallback_calls"])
        self.assertEqual(
            summary["module_unloaded_after_synchronize"],
            close["module_unloaded_after_synchronize"],
        )
        self.assertEqual(summary["bitwise_parent_equal"], True)
        self.assertEqual(
            summary["maximum_absolute_error"],
            observation["correctness"]["maximum_absolute_error"],
        )
        self.assertEqual(summary["tolerance_equal"], True)
        self.assertEqual(result["disposition"], {
            "passed": observation["passed"],
            "checked_asset_partition": "passed",
            "historical_r31_relabelled": False,
            "performance_claim_authorized": observation[
                "scientific_claim_authorized"
            ],
        })

    def test_r39_raw_samples_rederive_the_fixed_pair_observation(self) -> None:
        raw = json.loads((ROOT / "tests/fixtures/r39-paired-timing-result.json").read_text())
        protocol = PairedTimingProtocol(
            arms=("cake", "direct_cuda"),
            pair_order=(
                ("cake", "direct_cuda"),
                ("direct_cuda", "cake"),
                ("cake", "direct_cuda"),
                ("direct_cuda", "cake"),
                ("cake", "direct_cuda"),
            ),
            samples_per_cohort=25,
            route_calls_per_cohort=36,
            maximum_cv=0.05,
            materiality_ratio=1.02,
            required_pair_wins=4,
        )

        observation = derive_paired_timing(raw["observations"]["measurements"], protocol)

        self.assertTrue(observation.measurement_quality_passed)
        self.assertEqual(observation.pair_wins, {"cake": 5, "direct_cuda": 0})
        self.assertEqual(observation.pooled_sample_counts, {"cake": 125, "direct_cuda": 125})
        self.assertEqual(
            observation.pooled_medians_ms,
            {"cake": 1.154594, "direct_cuda": 11.192384},
        )
        self.assertEqual(observation.speedup, 9.693783269270412)
        self.assertEqual(observation.classification, "first_arm_faster")


if __name__ == "__main__":
    unittest.main()


class AttributionAssayTest(unittest.TestCase):
    """Profiler evidence is a separate assay that structurally cannot report a latency.

    Nsight Compute serialises kernels and replays them, inflating every span it observes.
    A harness that profiled the timed run and then reported its latency would be reporting
    the profiler's overhead as the candidate's cost. Rather than documenting that, the
    attribution purpose refuses to carry timing at all, so the mistake is unrepresentable.
    """

    def _receipt(self, **changes):
        fields = {
            "candidate_sha256": "a" * 64,
            "workload_sha256": "b" * 64,
            "evaluation_protocol_sha256": "c" * 64,
            "purpose": "attribution",
            "case_id": "headline_b32",
            "correctness_passed": True,
            "correctness": {"tie_aware_distance_match": True},
            "kernel_calls": 1,
            "fallback_calls": 0,
            "launch_receipt_sha256": "d" * 64,
            "timing": None,
        }
        fields.update(changes)
        return EvaluationReceipt(**fields)

    def test_an_attribution_receipt_may_not_carry_timing(self) -> None:
        self._receipt()  # timing None is the only admissible shape
        with self.assertRaisesRegex(ValueError, "identity or route differs"):
            self._receipt(timing={"median_ms": 1.0})

    def test_an_attribution_protocol_may_not_declare_a_timing_method(self) -> None:
        EvaluationProtocol(
            protocol_id="attribution-v1",
            purpose="attribution",
            workload_sha256="b" * 64,
            case_id="headline_b32",
            timing="none",
        )
        with self.assertRaisesRegex(ValueError, "EvaluationProtocol differs"):
            EvaluationProtocol(
                protocol_id="attribution-v1",
                purpose="attribution",
                workload_sha256="b" * 64,
                case_id="headline_b32",
                timing="paired_cupti",
            )

    def test_the_artifact_roles_differ_from_a_timed_assay(self) -> None:
        # A timed assay owes timing samples; an attribution assay owes a profile. Asking
        # for both would mean one of them was produced by a run that could not produce it.
        launch = json.dumps({"purpose": "attribution"}, sort_keys=True).encode()
        correctness = json.dumps(
            {"passed": True, "metrics": {"tie_aware_distance_match": True}},
            sort_keys=True,
        ).encode()
        with self.assertRaisesRegex(ValueError, "artifact custody differs"):
            self._receipt(
                launch_receipt_sha256=sha256(launch).hexdigest(),
                artifact_payloads={
                    "correctness_output": correctness,
                    "launch_receipt": launch,
                    "timing_samples": b"[]",
                },
            )

    def test_profile_summary_is_recomputed_from_retained_ncu_csv(self) -> None:
        launch = json.dumps(
            {"candidate_sha256": "a" * 64, "purpose": "attribution"},
            sort_keys=True,
        ).encode()
        correctness = json.dumps(
            {"passed": True, "metrics": {"tie_aware_distance_match": True}},
            sort_keys=True,
        ).encode()
        profile = _profile_fixture("a" * 64, "headline_b32", "kernel")
        receipt = self._receipt(
            launch_receipt_sha256=sha256(launch).hexdigest(),
            artifact_payloads={
                "correctness_output": correctness,
                "launch_receipt": launch,
                "profile": profile,
            },
        )
        self.assertEqual(
            receipt.attribution_feedback["occupancy"]["binding_resources"],
            ["registers", "shared_memory"],
        )
        self.assertEqual(
            receipt.attribution_feedback["signals"]["long_scoreboard_stall_pct"],
            18.0,
        )

        tampered = json.loads(profile)
        tampered["summary"]["occupancy"]["resident_ctas_per_sm"] = 3.0
        with self.assertRaisesRegex(ValueError, "projection differs"):
            self._receipt(
                launch_receipt_sha256=sha256(launch).hexdigest(),
                artifact_payloads={
                    "correctness_output": correctness,
                    "launch_receipt": launch,
                    "profile": json.dumps(
                        tampered, sort_keys=True, separators=(",", ":")
                    ).encode(),
                },
            )

    def test_profiled_launch_must_itself_pass_correctness(self) -> None:
        launch = b'{"candidate_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}'
        correctness = json.dumps(
            {"passed": False, "metrics": {"tie_aware_distance_match": False}},
            sort_keys=True,
        ).encode()
        with self.assertRaisesRegex(ValueError, "identity or route differs"):
            self._receipt(
                correctness_passed=False,
                correctness={"tie_aware_distance_match": False},
                launch_receipt_sha256=sha256(launch).hexdigest(),
                artifact_payloads={
                    "correctness_output": correctness,
                    "launch_receipt": launch,
                    "profile": _profile_fixture("a" * 64, "headline_b32", "kernel"),
                },
            )
