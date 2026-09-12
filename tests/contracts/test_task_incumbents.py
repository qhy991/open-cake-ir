"""Task incumbents advance only through audited material confirmations."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping
from unittest.mock import patch

from open_cake_ir.evaluation import LaunchableCandidate
from open_cake_ir.evaluation.paired import candidate_identity
from open_cake_ir.evidence import EvidenceStore
from open_cake_ir.evidence import custody
from open_cake_ir.lab.incumbents import (
    TaskIncumbentKey,
    TaskIncumbentRegistry,
    _bound_audit_report,
    promote_task_incumbent,
)
from open_cake_ir.lab.execution import _baseline_comparison_feedback
from open_cake_ir.serialization import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[2]


class TaskIncumbentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.custody = self.base / "custody"
        environment = patch.dict(
            "os.environ", {custody.ENVIRONMENT: str(self.custody)}
        )
        environment.start()
        self.addCleanup(environment.stop)
        self.registry = self.base / "incumbents"
        self.protocol = {
            "case_id": "primary",
            "search_evaluation": "correctness_then_paired_metal",
            "confirmatory_evaluation": "fresh_fixed_candidate_correctness_then_paired_metal",
            "validation_case_ids": ["primary"],
            "paired_timing": {
                "kind": "fixed_baseline_paired_metal_v2",
                "arms": ["candidate", "baseline"],
                "pair_order": [["candidate", "baseline"]],
                "samples_per_cohort": 1,
                "route_calls_per_cohort": 2,
                "maximum_cv": 0.05,
                "materiality_ratio": 1.05,
                "required_pair_wins": 1,
                "dispatches_per_sample": 1,
                "maximum_relative_iqr": 0.1,
            },
        }

    @staticmethod
    def candidate(character: str) -> tuple[LaunchableCandidate, dict[str, bytes]]:
        manifest = canonical_json_bytes(
            {"target": "apple_gpu_family9", "kernel_name": "fixture_kernel"}
        )
        payloads = {
            "launch_manifest": manifest,
            "metal_binary_archive": f"archive-{character}".encode(),
            "metal_build_report": canonical_json_bytes({"fixture": character}),
        }
        candidate = LaunchableCandidate(
            candidate_sha256=character * 64,
            target="apple_gpu_family9",
            entry_point="fixture_kernel",
            artifact_roles={
                role: sha256(payload).hexdigest()
                for role, payload in payloads.items()
            },
            launch_spec_sha256=sha256(manifest).hexdigest(),
            artifact_payloads=payloads,
        )
        return candidate, payloads

    def campaign(
        self,
        name: str,
        character: str,
        baseline: Mapping[str, object],
        *,
        classification: str = "first_arm_faster",
        speedup: float = 1.2,
    ):
        candidate, payloads = self.candidate(character)
        evidence = EvidenceStore.create(self.base / f"evidence-{name}")
        lock_document = {
            "workload": {
                "workload_id": "fixture-r2-c8",
                "canonical_sha256": "1" * 64,
            },
            "execution": {
                "target": "apple_gpu_family9",
                "fixed_baseline": {"candidate": dict(baseline)},
            },
            "evaluation_protocol": self.protocol,
            "resolved_inputs": {
                "arm_environments": {
                    "open_cake": {"lowering_route": {"backend": "metal"}}
                }
            },
        }
        authority = {"fixture": name}
        lock_sha = sha256(canonical_json_bytes(authority)).hexdigest()
        lock = SimpleNamespace(
            claim_scope="artifact_optimization_only",
            canonical_sha256=lock_sha,
            document=lock_document,
        )
        run = evidence.start_run(
            "open_cake-1", authority_sha256=lock_sha, authority=authority
        )
        artifact_refs = []
        for role, payload in payloads.items():
            media_type = (
                "application/json" if role.endswith("report") or role == "launch_manifest"
                else "application/octet-stream"
            )
            artifact_refs.append(evidence.put(payload, media_type=media_type).reference(role))
        run.append(
            "launchable_candidate_sealed",
            {
                "turn": 1,
                "candidate_sha256": candidate.candidate_sha256,
                "candidate_record_sha256": candidate.canonical_sha256,
                "objects": artifact_refs,
            },
        )
        timing = {
            "kind": "fixed_baseline_paired_metal_v2",
            "classification": classification,
            "measurement_quality_passed": True,
            "speedup": speedup,
            "pooled_median_ms": 1.0,
            "pooled_medians_ms": {"candidate": 1.0, "baseline": speedup},
            "pair_wins": {"candidate": 1, "baseline": 0},
            "tied_pairs": 0,
            "pooled_sample_counts": {"candidate": 1, "baseline": 1},
        }
        receipt_payload = canonical_json_bytes(
            {
                "candidate_sha256": candidate.candidate_sha256,
                "purpose": "confirmatory",
                "correctness_passed": True,
                "kernel_calls": 1,
                "fallback_calls": 0,
                "timing": timing,
            }
        )
        receipt = evidence.put(receipt_payload, media_type="application/json")
        run.append(
            "candidate_evaluated",
            {
                "turn": 1,
                "purpose": "confirmatory",
                "candidate_sha256": candidate.candidate_sha256,
                "objects": [receipt.reference("evaluation_receipt")],
            },
        )
        run.seal(
            protocol_adherence="adhered",
            endpoint_observation="qualified",
            endpoint={"candidate_sha256": candidate.candidate_sha256},
        )
        lock_path = self.base / f"lock-{name}.json"
        lock_path.write_text("{}")
        report = SimpleNamespace(
            campaign_complete=True,
            archive_integrity_passed=True,
            filesystem_custody_verified=True,
            semantic_replay_passed=True,
            descriptive={
                "semantic_replay_by_run": {"open_cake-1": True},
                "promoted_artifacts": {
                    "open_cake-1": {"candidate_sha256": candidate.candidate_sha256,
                        "turn": 1, "confirmed_latency_ms": 1.0,
                        "evaluation_receipt_sha256": receipt.sha256}
                }
            },
        )
        lab = SimpleNamespace(
            reference_campaign=lambda *_: object(), audit=lambda _: report
        )
        return lock, lock_path, evidence.root, candidate

    def promote(self, fixture):
        lock, lock_path, evidence_root, _ = fixture
        source = EvidenceStore.open(evidence_root)
        confirmation = next(event for event in source.replay_events("open_cake-1")
                            if event["kind"] == "candidate_evaluated")
        receipt_sha256 = next(item["sha256"] for item in confirmation["payload"]["objects"]
                              if item["role"] == "evaluation_receipt")
        with patch(
            "open_cake_ir.lab.incumbents.CampaignLock.load", return_value=lock
        ):
            return promote_task_incumbent(
                project_root=ROOT,
                registry_root=self.registry,
                campaign_lock_path=lock_path,
                evidence_root=evidence_root,
                lab=SimpleNamespace(
                    reference_campaign=lambda *_: object(),
                    audit=lambda _: SimpleNamespace(
                        campaign_complete=True,
                        archive_integrity_passed=True,
                        filesystem_custody_verified=True,
                        semantic_replay_passed=True,
                        descriptive={
                            "semantic_replay_by_run": {"open_cake-1": True},
                            "promoted_artifacts": {
                                "open_cake-1": {
                                    "candidate_sha256": fixture[3].candidate_sha256,
                                    "turn": 1,
                                    "confirmed_latency_ms": 1.0,
                                    "evaluation_receipt_sha256": receipt_sha256,
                                }
                            }
                        },
                    ),
                ),
            )

    def test_material_winner_advances_chain_and_materializes_exact_bundle(self):
        baseline, _ = self.candidate("0")
        first_fixture = self.campaign("first", "a", candidate_identity(baseline))
        first = self.promote(first_fixture)
        self.assertEqual(first["generation"], 0)
        key = TaskIncumbentKey.from_campaign(first_fixture[0])
        registry = TaskIncumbentRegistry.open(self.registry)
        materialized = registry.materialize(key, self.base / "materialized")
        self.assertIsNotNone(materialized)
        assert materialized is not None
        bundle_path, record = materialized
        bundle = json.loads(bundle_path.read_text())
        self.assertEqual(bundle["candidate"], record["candidate"])
        self.assertEqual(
            set(bundle["artifact_paths"]),
            set(record["candidate"]["artifact_roles"]),
        )

        second_fixture = self.campaign("second", "b", record["candidate"])
        second = self.promote(second_fixture)
        self.assertEqual(second["generation"], 1)
        self.assertEqual(second["predecessor"]["run_id"], first["run_id"])
        self.assertEqual(
            TaskIncumbentRegistry.open(self.registry).current(key)["candidate"][
                "candidate_sha256"
            ],
            "b" * 64,
        )

    def test_wrong_baseline_close_null_and_copied_registry_are_refused(self):
        baseline, _ = self.candidate("0")
        first_fixture = self.campaign("first", "a", candidate_identity(baseline))
        self.promote(first_fixture)
        wrong = self.campaign("wrong", "b", candidate_identity(baseline))
        with self.assertRaisesRegex(ValueError, "not the current"):
            self.promote(wrong)
        close = self.campaign(
            "close", "c", candidate_identity(baseline),
            classification="close_null", speedup=1.01,
        )
        with self.assertRaisesRegex(ValueError, "material confirmed win"):
            self.promote(close)

        copied = self.base / "copied-incumbents"
        shutil.copytree(self.registry, copied)
        key = TaskIncumbentKey.from_campaign(first_fixture[0])
        with self.assertRaisesRegex(ValueError, "custody-audited"):
            TaskIncumbentRegistry.open(copied).current(key)

    def test_empty_cells_fall_back_independently_by_backend_and_exact_target(self):
        EvidenceStore.create(self.registry)
        metal = TaskIncumbentKey.from_values(
            workload_id="fixture",
            workload_sha256="1" * 64,
            case_id="primary",
            target="apple_gpu_family9",
            backend="metal",
            evaluation_protocol=self.protocol,
        )
        m1 = TaskIncumbentKey.from_values(
            workload_id="fixture",
            workload_sha256="1" * 64,
            case_id="primary",
            target="apple_gpu_family7",
            backend="metal",
            evaluation_protocol=self.protocol,
        )
        triton = TaskIncumbentKey.from_values(
            workload_id="fixture",
            workload_sha256="1" * 64,
            case_id="primary",
            target="sm_100a",
            backend="triton",
            evaluation_protocol=self.protocol,
        )
        self.assertEqual(len({metal.canonical_sha256, m1.canonical_sha256,
                              triton.canonical_sha256}), 3)
        registry = TaskIncumbentRegistry.open(self.registry)
        for key in (metal, m1, triton):
            self.assertIsNone(registry.current(key))
            self.assertIsNone(registry.materialize(key, self.base / key.target))

    def test_next_turn_feedback_names_black_box_incumbent_and_measured_gap(self):
        lock = SimpleNamespace(document={"execution": {"fixed_baseline": {
            "selection": {"source": "task_incumbent"}}}})
        feedback = _baseline_comparison_feedback(lock, {
            "pooled_medians_ms": {"candidate": 0.8, "baseline": 1.0},
            "speedup": 1.25,
            "measurement_quality_passed": True,
        })
        self.assertEqual(feedback, {
            "source": "task_incumbent",
            "baseline_latency_ms": 1.0,
            "candidate_speedup": 1.25,
            "measurement_quality_passed": True,
        })

    def test_historical_promotion_audits_through_the_campaign_bound_source(self):
        lock_path = self.base / "historical-lock.json"
        lock_path.write_text("{}")
        evidence = self.base / "historical-evidence"
        evidence.mkdir()
        lock = SimpleNamespace(
            study_id="historical-study",
            claim_scope="artifact_optimization_only",
            document={"execution": {"executor_revision": {
                "executor_id": "historical-executor",
                "path": "runtime/executors/historical.json",
                "canonical_sha256": "1" * 64,
            }}},
        )
        executor = SimpleNamespace(document={"host_environment": {"python": {
            "invocation_path": "/historical/python"}}})
        report = {"study_id": lock.study_id, "claim_scope": lock.claim_scope,
                  "descriptive": {}}
        completed = SimpleNamespace(returncode=0, stdout=json.dumps(report), stderr="")
        with patch("open_cake_ir.lab.incumbents.ExecutorRevision.load_reference",
                   return_value=executor) as load, \
             patch("open_cake_ir.lab.incumbents.subprocess.run",
                   return_value=completed) as run:
            self.assertEqual(
                _bound_audit_report(ROOT, lock, lock_path, evidence), report
            )
        load.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command[:2], ["/historical/python", "-I"])
        self.assertEqual(
            command[2], str(ROOT / "src/open_cake_ir/evaluation/source_bootstrap.py")
        )
        self.assertEqual(command[-4:], ["--lock", str(lock_path),
                                        "--evidence-root", str(evidence)])


if __name__ == "__main__":
    unittest.main()
