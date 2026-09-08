"""Synthetic CPU protocol fixtures only: no GPU, real broker or measured timings.

Compiler inputs are copied unchanged from the admitted release; the Executor is an
explicit CPU semantic fixture and its dummy host is never admitted. No release cycle
or approval is manufactured. Target compilation, broker observation and evaluator
process execution are replaced; collector/fitter and model/ordering owners execute.
"""
from __future__ import annotations

import copy
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from open_cake_ir.tasks.workloads import load_workload
sys.path.insert(0, str(ROOT / "tools"))
import open_cake_ir.tasks.flash_kmeans.calibrate as instrument
import open_cake_ir.tasks.evaluate as common
from open_cake_ir.compiler import Compiler, EmpiricalCostModel
from open_cake_ir.compiler.toolchain import TritonCompilation
from open_cake_ir.evaluation import WorkloadContract
from open_cake_ir.tasks.flash_kmeans import environment as environments
from open_cake_ir.lab.executor import ExecutorRevision
from open_cake_ir.lab.selection import _empirical_context
from tests.contracts._executor_fixture import SemanticExecutorFixture, compiler_reference
from open_cake_ir.tasks.flash_kmeans.seed import ExactShape, KernelSeed


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class FlashCalibrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="SYNTHETIC-flash-contract-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name).resolve()
        cls.project = cls.root / "synthetic-source"
        cls.project.mkdir()
        # Preserve the actual frozen Compiler bytes and identity. This fixture
        # exercises lowering/model semantics, not the independent release workflow.
        manifest = json.loads((ROOT / "compiler/revision.lock.json").read_bytes())
        paths = {record["path"] for record in manifest["sources"]}
        paths.update(reference["path"] for reference in manifest["target_definitions"].values())
        paths.update(manifest[key]["path"] for key in ("corpus_manifest", "corpus_gate", "release_approval"))
        paths.update({"compiler/revision.lock.json", "compiler/revision.json", "compiler/source_set.json",
                      "contracts/workloads/flash-kmeans-assign-v2.json", "contracts/kernel-seeds/r42-cake-r1-turn1-v3.json"})
        # The child-process supervision probes need the actual Python runtime code.
        shutil.copytree(ROOT / "src", cls.project / "src", ignore=shutil.ignore_patterns("__pycache__"))
        for relative in paths:
            destination = cls.project / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copyfile(ROOT / relative, destination)
        cls.compiler_ref = compiler_reference(cls.project)
        cls.compiler = Compiler.load(cls.project, cls.project / cls.compiler_ref["path"])
        executor = SemanticExecutorFixture().revision(cls.project)
        # Declared collector/worker capabilities in a CPU-only dependency double;
        # these are not source-digest records or a released Executor descriptor.
        cls.executor = replace(executor, document={**executor.document, "sources": [
            {"path": "src/open_cake_ir/tasks/flash_kmeans/calibrate.py"},
            {"path": "src/open_cake_ir/tasks/evaluate.py"},
        ]})
        cls.workload = load_workload(cls.project / "contracts/workloads/flash-kmeans-assign-v2.json")
        cls.seed = KernelSeed.load(cls.project, cls.project / "contracts/kernel-seeds/r42-cake-r1-turn1-v3.json")

    def setUp(self):
        self.local = tempfile.TemporaryDirectory(prefix="run-", dir=self.root)
        self.addCleanup(self.local.cleanup)
        self.directory = Path(self.local.name)
        self.addCleanup(patch.stopall)
        patch.object(instrument, "ROOT", self.project).start()
        def admit_cpu_executor(project_root, reference, context):
            self.assertEqual(Path(project_root).resolve(), self.project)
            self.assertEqual(reference, dict(self.executor.reference), context)
            return self.executor
        patch.object(ExecutorRevision, "load_reference", side_effect=admit_cpu_executor).start()
        self.calls = []

    def plan(self):
        pool = [{"id": name, "schedule": f"schedules/{name}.json"} for name in ("n128-k64-w4", "n256-k64-w4", "n256-k128-w8", "n128-k128-w8")]
        return {"schema_version": 1, "plan_id": "SYNTHETIC-not-a-GPU-plan", "state": "frozen", "model_id": "SYNTHETIC-no-performance-evidence", "compiler_revision": self.compiler_ref, "executor_revision": dict(self.executor.reference), "workload": {"path": "contracts/workloads/flash-kmeans-assign-v2.json", "workload_id": self.workload.workload_id, "canonical_sha256": self.workload.canonical_sha256}, "case_id": "b32_smoke", "target": "sm_100a", "pool": pool, "baseline": pool[2]["id"], "observations": instrument.observation_order(pool, pool[2]["id"]), "acceptance": {"maximum_cohort_cv": .05, "maximum_within_observation_median_ratio": 1.05, "maximum_baseline_drift_ratio": 1.05, "maximum_mape": .10, "maximum_relative_error": .20, "top_k": 2, "maximum_top_k_regret_ratio": 1.05}}

    def _synthetic_compilation(self, source, requirements):
        payloads = {role: f"SYNTHETIC NON-EXECUTABLE {role}".encode() for role in ("source", "ttir", "ttgir", "llir", "ptx", "cubin")}
        payloads["source"] = source + b"\n# SYNTHETIC compiler expansion, not executable evidence\n"
        payloads["cubin"] = b"\x7fELF SYNTHETIC NON-EXECUTABLE " + repr(requirements).encode()
        return TritonCompilation(source, "sm_100a", requirements["kernel_entry_point"], payloads, requirements["compile_options"]["num_warps"] * 32, 0, "SYNTHETIC")

    def _fake_evaluator(self, command, **kwargs):
        self.calls.append(command)
        self.assertEqual(command[:2], [self.executor.document["host_environment"]["python"]["invocation_path"], str(self.project / "src/open_cake_ir/tasks/evaluate.py")])
        self.assertEqual(command[2], "--request")
        self.assertEqual(command[4], "--output")
        self.assertEqual(kwargs["cwd"], self.project)
        self.assertEqual(kwargs["env"]["GPUQ_JOB_ID"], "gpuq-000000000001")
        self.assertNotIn("OPENAI_API_KEY", kwargs["env"])
        self.assertNotIn("start_new_session", kwargs)
        directory = Path(command[3]).parent
        request = json.loads(Path(command[3]).read_text())
        # Exercise the real worker's independent request admission before the
        # CPU-only evaluator supplies any synthetic measurements.
        with patch.object(common, "ROOT", self.project):
            authority = common._load_authority(Path(command[3]))
        self.assertEqual(authority.request["compiler_revision"], self.active_plan["compiler_revision"])
        self.assertEqual(authority.workload.canonical_sha256, self.workload.canonical_sha256)
        _candidate, manifest = instrument._candidate(directory, self.active_plan, self.project)
        self.assertEqual(manifest.hidden_null_pointer_parameters, 2)
        spec = next(r for r in self.active_plan["observations"] if r["id"] == directory.name)
        index = [p["id"] for p in self.active_plan["pool"]].index(spec["candidate_id"])
        median = self.sample(spec, index)
        values = [[median] * 25 for _ in range(5)]
        metrics = {"SYNTHETIC_CPU_FIXTURE_NOT_ORACLE_EVIDENCE": True}
        result = common._base_result("gpuq-000000000001")
        result.update(admitted=True)
        result["counters"].update(module_loads=1, preflight_calls=1, kernel_calls=181, timing_samples=125)
        result["receipt"] = {"correctness_passed": True, "correctness": metrics, "kernel_calls": 1, "fallback_calls": 0, "timing": {"measurement_quality_passed": True, "pooled_median_ms": median, "cohort_count": 5, "samples_per_cohort": 25}, "artifacts": {"correctness_output": "correctness-output.json", "launch_receipt": "launch-receipt.json", "timing_samples": "timing-samples.json"}}
        write(directory / "correctness-output.json", {"metrics": metrics, "output_sha256": "0" * 64, "output_size_bytes": 65536})
        write(directory / "launch-receipt.json", {"job_id": result["job_id"], "gpu_uuid": "GPU-SYNTHETIC-NO-DEVICE", "candidate_sha256": request["candidate_sha256"], "correctness_launches": 1, "fallback_calls": 0})
        write(directory / "timing-samples.json", {"cohorts_ms": values})
        if self.reject == directory.name:
            result["receipt"]["correctness_passed"] = False
        write(Path(command[5]), result)
        kwargs["stdout"].write(b"SYNTHETIC evaluator stdout")
        kwargs["stderr"].write(b"SYNTHETIC evaluator stderr")
        return subprocess.CompletedProcess(command, 0)

    def fixture(self, *, sample=None, reject=None):
        self.active_plan = plan = self.plan()
        self.sample = sample or (lambda spec, index: (.010 + index * .003) * (1.01 if spec["split"] == "calibration" else 1))
        self.reject = reject
        run = self.directory / "run"
        candidate = run / "candidate"
        write(candidate / "plan.json", plan)
        for spec, choices in zip(plan["pool"], ((128, 64, 4), (256, 64, 4), (256, 128, 8), (128, 128, 8))):
            bn, bk, warps = choices
            seed = replace(self.seed, block_n=bn, block_k=bk, num_warps=warps, num_stages=3)
            schedule = seed.schedule_for(plan["case_id"], ExactShape.from_mapping(self.workload.case(plan["case_id"])["shape"]))
            schedule["schedule_id"] = spec["id"]
            # The historical seed's base metadata predates Workload v2. Bind only
            # this new derived Schedule; never edit or relabel the frozen seed.
            schedule["metadata"]["workload_contract_sha256"] = self.workload.canonical_sha256
            write(candidate / spec["schedule"], schedule)
        judge = {"identity": f"{self.executor.executor_id}@{self.executor.canonical_sha256}", "cwd": str(self.project), "command": [self.executor.document["host_environment"]["python"]["invocation_path"], str(self.project / "src/open_cake_ir/tasks/flash_kmeans/calibrate.py"), "collect"]}
        task = {"schema": "kernelinfra.task.v1", "task_id": "SYNTHETIC-no-GPU-task", "workloads": [self.workload.workload_id], "comparison": {"primary_workloads": [self.workload.workload_id], "relative_noise_floor": .05}, "stages": [{"id": "compile", "kind": "compile", "execution": "local", "judge": judge}, {"id": "collection", "kind": "judge", "resources": {"mode": "exclusive", "gpu_count": 1, "run_timeout_s": 3600}, "judge": judge}]}
        write(run / "task.json", task)
        principal = {"pid": 200, "parent_pid": 100, "uid": 321, "gid": 654, "broker_peer": [100, 321, 654], "broker_socket": "/SYNTHETIC/no-broker.sock", "job_id": None, "visible_device": "7", "run_id": "SYNTHETIC-NOT-A-GPU-RUN"}
        node_ids = {"run_id": principal["run_id"], "task_id": task["task_id"], "task_sha256": "a" * 64, "candidate_sha256": "b" * 64}
        write(run / "request.json", {"schema": "kernelinfra.request.v1", **node_ids})
        write(run / "state.json", {"schema": "kernelinfra.state.v1", **node_ids, "state": "running", "stage_id": "collection", "stage_kind": "judge", "stage_index": 1, "broker_job_id": "gpuq-000000000001", "gpu_ids": [7], "run_dir": str(run), "terminal_at": None})
        stage_summaries = []
        for spec in task["stages"]:
            stage = run / "stages" / spec["id"]
            stage.mkdir(parents=True)
            env = {"KERNELINFRA_RUN_DIR": str(run), "KERNELINFRA_CANDIDATE_DIR": str(candidate), "KERNELINFRA_TASK": str(run / "task.json"), "KERNELINFRA_STAGE_DIR": str(stage), "KERNELINFRA_STAGE_ID": spec["id"], "KERNELINFRA_STAGE_KIND": spec["kind"], "KERNELINFRA_RESULT": str(stage / "result.json"), "KERNELINFRA_RUN_ID": principal["run_id"], "OPENAI_API_KEY": "SYNTHETIC-SECRET-MUST-NOT-REACH-CHILD"}
            if spec["kind"] == "judge":
                env.update(CUDA_VISIBLE_DEVICES=principal["visible_device"])
            with patch.dict(os.environ, env, clear=True), patch.object(ExecutorRevision, "admit_host", return_value=object()), patch.object(environments, "compile_triton", side_effect=self._synthetic_compilation), patch.object(instrument, "_broker_principal", return_value=principal), patch.object(instrument.subprocess, "run", side_effect=self._fake_evaluator):
                code = instrument.collect()
            result = instrument._read(stage / "result.json")
            write(stage / "receipt.json", {"schema": "kernelinfra.stage-receipt.v1", "run_id": principal["run_id"], "stage_id": spec["id"], "stage_kind": spec["kind"], "execution": spec.get("execution", "broker"), "judge_identity": judge["identity"], "exit_code": code, "judge_result_valid": True, "error": None, "broker_job_id": principal["job_id"] if spec["kind"] == "judge" else None, "gpu_ids": [7] if spec["kind"] == "judge" else []})
            stage_summaries.append({"id": spec["id"], "kind": spec["kind"], "status": result["status"], "validity": result["validity"]})
            if code:
                break
        completed = len(stage_summaries) == 2 and all(r["status"] == "passed" for r in stage_summaries)
        write(run / "result.json", {"schema": "kernelinfra.run-result.v1", **node_ids, "outcome": "completed" if completed else "infra_error", "validity": "valid" if completed else "unknown", "frontier_eligible": False, "stages": stage_summaries})
        return run

    def fit(self, run, name="fit-output"):
        output = self.directory / name
        code = instrument.fit(run, output)
        return code, output, instrument._read(output / "audit.json")

    def test_complete_cpu_orchestration_and_fitter_use_exact_common_owners(self):
        run = self.fixture()
        self.assertEqual(len(self.calls), 17, instrument._read(run / "stages/collection/result.json"))
        code, output, audit = self.fit(run)
        self.assertEqual(code, 0, audit)
        document = instrument._read(output / "model.json")
        self.assertEqual(len(document["curves"]), 4)
        self.assertTrue(all(len(c["points"]) == 1 and c["points"][0]["extent"] == 512 for c in document["curves"]))
        self.assertEqual(audit["validation"]["audit"]["selected_ids"], [p["id"] for p in self.active_plan["pool"][:2]])
        self.assertEqual(audit["validation"]["audit"]["metrics"]["top_k_regret_ratio"], 1)
        self.assertEqual(audit["validation"]["audit"]["metrics"]["descriptive_range_coverage_fraction"], 1)
        self.assertEqual(document["context"], _empirical_context(self.executor, workload_sha256=self.workload.canonical_sha256, case_id="b32_smoke"))
        model = EmpiricalCostModel(document)
        schedule = copy.deepcopy(document["curves"][0]["template"])
        for buffer in schedule["buffers"]:
            if buffer["name"] in {"tokens", "assignments"}:
                buffer["shape"][1] += 128
        prediction = model.estimate(schedule, compiler_revision_id=self.compiler_ref["revision_id"], compiler_revision_sha256=self.compiler_ref["canonical_sha256"], target="sm_100a")
        self.assertFalse(prediction["covered"])
        judge = instrument._read(run / "stages/collection/result.json")
        self.assertFalse(any("candidate_ms" in row or "baseline_ms" in row for row in judge["workloads"]))

    def test_common_worker_refuses_missing_or_foreign_compiler_reference(self):
        run = self.fixture()
        directory = run / "stages/collection/aa-01"
        original = instrument._read(directory / "request.json")
        self.assertEqual(original["compiler_revision"], self.active_plan["compiler_revision"])
        for kind in ("missing", "foreign"):
            request = copy.deepcopy(original)
            if kind == "missing":
                del request["compiler_revision"]
            else:
                request["compiler_revision"]["revision_id"] = "foreign-compiler-fixture"
            path = directory / (kind + "-request.json")
            write(path, request)
            with self.subTest(kind=kind), patch.object(common, "ROOT", self.project), \
                 patch.object(common, "load_workload") as workload:
                with self.assertRaisesRegex(ValueError, "compiler_revision|Compiler Revision"):
                    common._load_authority(path)
                workload.assert_not_called()

    def test_oracle_failure_stops_collection_and_retains_failed_observation(self):
        run = self.fixture(reject="aa-02")
        self.assertEqual(len(self.calls), 2)
        result = instrument._read(run / "stages/collection/result.json")
        self.assertEqual((result["status"], result["validity"]), ("failed", "invalid"))
        self.assertTrue((run / "stages/collection/aa-02/stdout.log").is_file())
        code, output, audit = self.fit(run)
        self.assertEqual(code, 1)
        self.assertFalse((output / "model.json").exists())

    def test_historical_seed_metadata_requires_new_derived_workload_binding(self):
        run = self.fixture()
        plan = self.active_plan
        first = run / "candidate" / plan["pool"][0]["schedule"]
        schedule = instrument._read(first)
        original = self.seed.schedule_for(plan["case_id"], ExactShape.from_mapping(self.workload.case(plan["case_id"])["shape"]))
        schedule["metadata"]["workload_contract_sha256"] = original["metadata"]["workload_contract_sha256"]
        self.assertNotEqual(schedule["metadata"]["workload_contract_sha256"], self.workload.canonical_sha256)
        write(first, schedule)
        stage = self.directory / "refused-prospective-build"
        stage.mkdir()
        compiler = Compiler.load(self.project, self.project / self.compiler_ref["path"])
        with patch.dict(os.environ, {}, clear=True), patch.object(ExecutorRevision, "admit_host", return_value=object()), patch.object(environments, "compile_triton", side_effect=self._synthetic_compilation), self.assertRaisesRegex(ValueError, "Schedule Workload binding differs"):
            instrument._compile(run / "candidate", stage, plan, compiler, self.executor, self.workload)
        self.assertFalse((stage / "build.json").exists())

    def test_aa_drift_stops_before_pool_measurement(self):
        run = self.fixture(sample=lambda spec, index: .02 if spec["id"] == "aa-04" else .01)
        self.assertEqual(len(self.calls), 4)
        self.assertIn("A/A", instrument._read(run / "stages/collection/result.json")["summary"])
        self.assertFalse((run / "stages/collection/fit-n128-k64-w4").exists())

    def test_final_anchor_drift_refuses_model(self):
        run = self.fixture(sample=lambda spec, index: .02 if spec["split"] == "anchor" else .01)
        self.assertEqual(len(self.calls), 17)
        self.assertIn("anchor", instrument._read(run / "stages/collection/result.json")["summary"])
        code, output, _ = self.fit(run)
        self.assertEqual(code, 1)
        self.assertFalse((output / "model.json").exists())

    def test_audit_changes_neither_fitted_point_nor_calibration_envelope(self):
        run = self.fixture(sample=lambda spec, index: (.01 + index * .003) * (1.01 if spec["split"] == "calibration" else 1.08 if spec["split"] == "audit" else 1))
        code, output, audit = self.fit(run)
        self.assertEqual(code, 0, audit)
        model = instrument._read(output / "model.json")
        self.assertAlmostEqual(model["curves"][0]["points"][0]["kernel_us"], 10)
        self.assertAlmostEqual(model["curves"][0]["relative_error_envelope"], .01)
        self.assertEqual(audit["validation"]["audit"]["metrics"]["descriptive_range_coverage_fraction"], 0)

    def test_audit_point_error_refuses_model_without_adjusting_calibration(self):
        run = self.fixture(sample=lambda spec, index: (.01 + index * .003) * (1.5 if spec["split"] == "audit" else 1))
        code, output, audit = self.fit(run)
        self.assertEqual(code, 1, audit)
        self.assertTrue(audit["validation"]["calibration"]["passed"])
        self.assertFalse(audit["validation"]["audit"]["passed"])
        self.assertFalse((output / "model.json").exists())

    def test_top_two_regret_uses_existing_stable_point_order(self):
        def samples(spec, index):
            return [.010, .0101, .0102, .0103][index] if spec["split"] != "audit" else [.0105, .0106, .0095, .0103][index]
        run = self.fixture(sample=samples)
        code, output, audit = self.fit(run)
        self.assertEqual(code, 1, audit)
        metrics = audit["validation"]["audit"]["metrics"]
        self.assertLess(metrics["maximum_relative_error"], .20)
        self.assertGreater(metrics["top_k_regret_ratio"], 1.05)
        self.assertFalse((output / "model.json").exists())

    def test_unknown_or_mismatched_node_receipt_cannot_emit_model(self):
        run = self.fixture()
        path = run / "stages/collection/receipt.json"
        original = instrument._read(path)
        for index, mutation in enumerate(({"judge_result_valid": False}, {"gpu_ids": [8]}, {"exit_code": 1}, {"execution": "local"}, {"broker_job_id": "gpuq-OTHER"}, {"judge_identity": "different-Executor"}, {"error": "lost observation"})):
            write(path, {**original, **mutation})
            with self.subTest(mutation=mutation):
                code, output, audit = self.fit(run, f"refused-{index}")
                self.assertEqual(code, 1, audit)
                self.assertFalse((output / "model.json").exists())

    def test_missing_corrupt_or_rebound_launch_artifacts_refuse_model(self):
        run = self.fixture()
        directory = run / "stages/collection/fit-n128-k64-w4"
        cubin = directory / "candidate-cubin"
        original = cubin.read_bytes()
        cubin.unlink()
        self.assertEqual(self.fit(run, "missing")[0], 1)
        cubin.write_bytes(original + b"CORRUPT")
        self.assertEqual(self.fit(run, "corrupt")[0], 1)
        cubin.write_bytes(original)
        request = instrument._read(directory / "request.json")
        request["case_id"] = "headline_b32"
        write(directory / "request.json", request)
        self.assertEqual(self.fit(run, "rebound")[0], 1)

    def test_raw_counts_types_quality_and_gpu_changes_are_rejected(self):
        run = self.fixture()
        directory = run / "stages/collection/fit-n128-k64-w4"
        path = directory / "timing-samples.json"
        original = instrument._read(path)
        variants = [[], [[.01] * 24] * 5, [[True] * 25] * 5, [[float("nan")] * 25] * 5, [[-.01] * 25] * 5, [[.01] * 24 + [.02]] * 5, [[.01 * (1 + i * .1)] * 25 for i in range(5)]]
        for i, cohorts in enumerate(variants):
            write(path, {"cohorts_ms": cohorts})
            with self.subTest(variant=i):
                code, output, audit = self.fit(run, f"bad-raw-{i}")
                self.assertEqual(code, 1, audit)
                self.assertFalse((output / "model.json").exists())
        write(path, original)
        launch = instrument._read(directory / "launch-receipt.json")
        write(directory / "launch-receipt.json", {**launch, "gpu_uuid": "GPU-SYNTHETIC-OTHER"})
        self.assertEqual(self.fit(run, "other-gpu")[0], 1)

    def test_plan_split_context_and_create_only_output_boundaries(self):
        run = self.fixture()
        code, output, audit = self.fit(run)
        self.assertEqual(code, 0, audit)
        before = (output / "model.json").read_bytes()
        with self.assertRaises(FileExistsError):
            instrument.fit(run, output)
        self.assertEqual((output / "model.json").read_bytes(), before)
        path = run / "candidate/plan.json"
        plan = instrument._read(path)
        plan["observations"][4]["split"] = "audit"
        write(path, plan)
        self.assertEqual(self.fit(run, "split-refusal")[0], 1)
        write(path, self.active_plan)
        build_path = run / "stages/compile/build.json"
        build = instrument._read(build_path)
        build["context"]["timer"] = "different timer"
        write(build_path, build)
        self.assertEqual(self.fit(run, "context-refusal")[0], 1)

    def test_broker_identity_checks_observations_without_fixed_service_uid(self):
        record = {"broker_peer": [100, 321, 654], "parent_pid": 100, "uid": 321, "gid": 654, "job_id": "gpuq-000000000002", "run_id": "SYNTHETIC-run", "visible_device": "SYNTHETIC-one-GPU"}
        instrument._principal(record, record["job_id"], record["run_id"])
        for mutation in ({"parent_pid": 99}, {"uid": 999}, {"gid": 987}, {"visible_device": "0,1"}, {"job_id": None}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                instrument._principal({**record, **mutation}, record["job_id"], record["run_id"])

    def test_mandatory_streams_cannot_be_removed_from_the_supplied_inventory(self):
        run = self.fixture()
        stage = run / "stages/collection"
        inventory = instrument._read(stage / "result.json")
        for index, name in enumerate(("stdout.log", "stderr.log")):
            path = stage / "aa-01" / name
            original = path.read_bytes()
            path.unlink()
            changed = copy.deepcopy(inventory)
            changed["artifacts"].pop(f"aa-01/{name}")
            write(stage / "result.json", changed)
            code, output, audit = self.fit(run, f"missing-stream-{index}")
            self.assertEqual(code, 1, audit)
            self.assertFalse((output / "model.json").exists())
            path.mkdir()
            self.assertEqual(self.fit(run, f"nonfile-stream-{index}")[0], 1)
            path.rmdir()
            path.write_bytes(original)
        write(stage / "result.json", inventory)

    def test_evaluator_schema_requires_integer_and_redundant_commit_field_is_refused(self):
        run = self.fixture()
        path = run / "stages/collection/aa-01/result.json"
        record = instrument._read(path)
        for index, version in enumerate((True, 1.0)):
            write(path, {**record, "schema_version": version})
            code, output, audit = self.fit(run, f"schema-{index}")
            self.assertEqual(code, 1, audit)
            self.assertFalse((output / "model.json").exists())
        write(path, record)
        for relative in ("candidate/plan.json", "stages/compile/plan.json", "stages/collection/plan.json", "stages/compile/build.json"):
            path = run / relative
            document = instrument._read(path)
            self.assertNotIn("source_commit", document)
            write(path, {**document, "source_commit": "f" * 40})
        self.assertEqual(self.fit(run, "removed-unbound-label")[0], 1)

    def test_node_admission_publishes_real_job_to_child_without_exported_environment(self):
        run = self.fixture()
        self.assertEqual(len(self.calls), 17)
        record = instrument._read(run / "stages/collection/execution-context.json")
        self.assertIsNone(record["exported_job_id"])
        self.assertEqual(record["job_id"], "gpuq-000000000001")
        self.assertEqual(record["node_admission"]["broker_job_id"], record["job_id"])
        self.assertEqual(self.fit(run)[0], 0)
        task = instrument._read(run / "task.json")
        state = instrument._read(run / "state.json")
        queued = {**state, "state": "queued", "gpu_ids": []}
        states = iter((queued, state))
        original_read = instrument._read
        def read(path):
            return copy.deepcopy(next(states)) if Path(path) == run / "state.json" else original_read(path)
        with patch.object(instrument, "_read", side_effect=read), patch.object(instrument.time, "sleep") as wait:
            observed = instrument._node_admission(run, task, {**record, "job_id": None}, time.monotonic() + 1)
        wait.assert_called_once_with(.05)
        self.assertEqual(observed["job_id"], state["broker_job_id"])

    def test_node_admission_refuses_unrelated_malformed_terminal_or_unpublished_state(self):
        run = self.fixture()
        task = instrument._read(run / "task.json")
        principal = instrument._read(run / "stages/collection/execution-context.json")
        principal["job_id"] = None
        path = run / "state.json"
        original = instrument._read(path)
        for mutation in ({"run_id": "other-run"}, {"task_id": "other-task"}, {"candidate_sha256": "c" * 64}, {"stage_id": "compile"}, {"stage_kind": "benchmark"}, {"stage_index": True}, {"state": "completed"}, {"terminal_at": "finished"}, {"broker_job_id": "gpuq-not-a-real-assignment"}, {"gpu_ids": [8]}, {"gpu_ids": [7, 8]}, {"run_dir": "/other-run"}):
            write(path, {**original, **mutation})
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                instrument._node_admission(run, task, copy.deepcopy(principal), time.monotonic() + 1)
        write(path, {**original, "state": "queued", "gpu_ids": []})
        with self.assertRaises(TimeoutError):
            instrument._node_admission(run, task, copy.deepcopy(principal), time.monotonic() - 1)
        write(path, original)
        with self.assertRaisesRegex(ValueError, "exported broker job"):
            instrument._node_admission(run, task, {**principal, "job_id": "gpuq-000000000002"}, time.monotonic() + 1)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            instrument._node_admission(run, task, copy.deepcopy(principal), time.monotonic() + 1)

    def test_retained_node_admission_must_match_final_node_receipts(self):
        run = self.fixture()
        path = run / "stages/collection/execution-context.json"
        original = instrument._read(path)
        for index, mutation in enumerate(({"broker_job_id": "gpuq-000000000002"}, {"candidate_sha256": "c" * 64}, {"task_sha256": "d" * 64}, {"gpu_ids": [8]}, {"stage_id": "other"})):
            changed = copy.deepcopy(original)
            changed["node_admission"].update(mutation)
            write(path, changed)
            code, output, audit = self.fit(run, f"node-refusal-{index}")
            self.assertEqual(code, 1, audit)
            self.assertFalse((output / "model.json").exists())

    @unittest.skipUnless(os.name == "posix", "broker process groups require POSIX")
    def test_owned_cpu_evaluator_inherits_group_and_is_reaped_on_timeout_or_cancellation(self):
        run = self.fixture()
        cpu_root = self.directory / "owned-CPU-process-fixture"
        evaluator = cpu_root / "src/open_cake_ir/tasks/evaluate.py"
        evaluator.parent.mkdir(parents=True)
        evaluator.write_text("""import json,os,time
from pathlib import Path
Path(os.environ['FLASH_CALIBRATION_CPU_PROBE']).write_text(json.dumps({'pid':os.getpid(),'pgid':os.getpgrp()}))
print('owned CPU child; no CUDA or evaluation',flush=True)
time.sleep(30)
""")
        controller_code = """import json,signal,sys,types,subprocess
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import open_cake_ir.tasks.flash_kmeans.calibrate as i
source,run,cpu,stage,outcome=map(Path,sys.argv[2:7])
plan=i._read(run/'candidate/plan.json')
name=plan['baseline']
candidate,_=i._candidate(run/'stages/compile'/name,plan,source)
plan['observations']=plan['observations'][:1]
i.ROOT=cpu
i._broker_principal=lambda:{'CPU_TEST_ONLY':True,'job_id':'gpuq-000000000002'}
i._node_admission=lambda run,task,observed,deadline:observed
executor=types.SimpleNamespace(document={'host_environment':{'python':{'invocation_path':sys.executable}}})
task={'stages':[{}, {'resources':{'run_timeout_s':int(sys.argv[7])}}]}
def stop(signum,frame):
    raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,stop)
try:
    i._measure(run,stage,plan,executor,{name:(candidate,{})},task)
except subprocess.TimeoutExpired:
    outcome.write_text('timeout-reaped')
except SystemExit:
    outcome.write_text('group-cancelled')
    raise
"""
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                stage = self.directory / f"owned-stage-{cancel}"
                stage.mkdir()
                probe = self.directory / f"owned-child-{cancel}.json"
                outcome = self.directory / f"owned-outcome-{cancel}.txt"
                command = [sys.executable, "-c", controller_code, str(ROOT / "tools"), str(self.project), str(run), str(cpu_root), str(stage), str(outcome), "30" if cancel else "4"]
                environment = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1", "FLASH_CALIBRATION_CPU_PROBE": str(probe)}
                process = subprocess.Popen(command, env=environment, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    deadline = time.monotonic() + 5
                    while not probe.exists() and process.poll() is None and time.monotonic() < deadline:
                        time.sleep(.02)
                    self.assertTrue(probe.exists(), "owned CPU child did not start")
                    child = instrument._read(probe)
                    self.assertEqual(child["pgid"], process.pid)
                    if cancel:
                        self.assertEqual(os.getpgid(child["pid"]), process.pid)
                        os.killpg(process.pid, signal.SIGTERM)  # Only this test's group.
                    stdout, stderr = process.communicate(timeout=6)
                    self.assertEqual(outcome.read_text(), "group-cancelled" if cancel else "timeout-reaped", (stdout, stderr))
                    with self.assertRaises(ProcessLookupError):
                        os.kill(child["pid"], 0)
                    self.assertIn(b"owned CPU child", (stage / "aa-01/stdout.log").read_bytes())
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.communicate()


if __name__ == "__main__":
    unittest.main()
