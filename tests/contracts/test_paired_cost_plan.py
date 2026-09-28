"""Synthetic frozen-plan admission; no compilation, GPU or performance result."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import calibrate_paired_cost as instrument

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.toolchain import TritonCompilation
from open_cake_ir.evaluation.paired import candidate_identity
from open_cake_ir.lab.bindings import load_baseline_bundle
from open_cake_ir.lab.executor import ExecutorRevision
from open_cake_ir.serialization import canonical_json_bytes
from open_cake_ir.tasks.workloads import load_workload
from open_cake_ir.tasks import evaluate as worker


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class PairedCostPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        cls.executor = ExecutorRevision.for_target(ROOT, "sm_103a")
        cls.workload = load_workload(ROOT / "contracts/workloads/gemm-bias-bf16-fp32-v2.json")
        cls.base = json.loads((ROOT / "corpus/schedules/gemm-bias-b1-smoke-b300.json").read_text())

    def fixture(self, directory):
        root = Path(directory).resolve()
        snapshot = root / "candidate"
        baseline_root = snapshot / "baseline"
        identity = {"kind": "bubblewrap_triton_kernel_v1",
                    "python": self.executor.document["host_environment"]["python"]["invocation_path"],
                    "triton_version": self.executor.document["host_environment"]["packages"]["triton"],
                    "fixture": "synthetic; no executable compiler"}
        baseline_schedule = copy.deepcopy(self.base)
        baseline_schedule["schedule_id"] = "synthetic-baseline-b300"
        baseline_schedule["metadata"]["workload_contract_sha256"] = self.workload.canonical_sha256
        baseline_root.mkdir(parents=True, exist_ok=True)
        (baseline_root / "schedule.json").write_bytes(canonical_json_bytes(baseline_schedule))
        candidates = []
        for cap in (64, 96, 128):
            name = f"tile-{cap}"
            schedule = copy.deepcopy(self.base)
            schedule["schedule_id"] = name
            schedule["residency"]["registers_per_thread"] = cap
            schedule["metadata"]["workload_contract_sha256"] = self.workload.canonical_sha256
            relative = f"schedules/{name}.json"
            write(snapshot / relative, schedule)
            candidates.append({"id": name, "schedule": relative})
        toolchain_config = {"python": self.executor.document["host_environment"]["python"]["invocation_path"],
                            "bubblewrap": "/SYNTHETIC/no-bubblewrap",
                            "runtime_roots": ["/SYNTHETIC/no-runtime"],
                            "build_environment": {},
                            "triton_version": self.executor.document["host_environment"]["packages"]["triton"],
                            "timeout_seconds": 600}
        write(snapshot / "toolchain.json", toolchain_config)
        plan = {"schema_version": 2, "state": "frozen",
                "plan_id": "synthetic-paired-plan", "model_id": "synthetic-paired-model",
                "compiler_revision": {"path": "compiler/revision.json",
                                      "revision_id": self.compiler._revision.revision_id},
                "executor_revision": dict(self.executor.reference),
                "study_path": "contracts/studies/matched-search-triton-b300-gemm-optimization-template.json",
                "workload": {"path": "contracts/workloads/gemm-bias-bf16-fp32-v2.json",
                             "workload_id": self.workload.workload_id,
                             "canonical_sha256": self.workload.canonical_sha256},
                "case_id": "primary", "target": "sm_103a",
                "baseline_schedule_path": "baseline/schedule.json", "candidates": candidates,
                "observations": instrument.observation_order([row["id"] for row in candidates]),
                "varying_dimensions": [{"buffer": name, "dimension": 0} for name in ("a", "c")],
                "toolchain_identity": identity,
                "toolchain_config_path": "toolchain.json",
                "acceptance": {"maximum_baseline_drift_ratio": 1.05,
                               "maximum_mape": .1, "maximum_relative_error": .2,
                               "maximum_top2_regret_ratio": 1.05,
                               "envelope_allowance": .05}}
        write(snapshot / "plan.json", plan)
        return snapshot, plan

    def task_fixture(self, run, plan):
        python = self.executor.document["host_environment"]["python"]["invocation_path"]
        tool = str(ROOT / "tools/calibrate_paired_cost.py")
        stages = []
        for name, kind, execution, action in (
            ("compile", "compile", "local", "collect-compile"),
            ("collection", "judge", "broker", "collect-device"),
        ):
            stage = {"id": name, "kind": kind, "execution": execution,
                     "judge": {"identity": self.executor.executor_id, "cwd": str(ROOT),
                               "command": [python, tool, action]}}
            if name == "collection":
                stage["resources"] = {"mode": "exclusive", "gpu_count": 1,
                                       "run_timeout_s": 3600}
            stages.append(stage)
        task = {"schema": "kernelinfra.task.v1", "task_id": plan["plan_id"],
                "workloads": [self.workload.workload_id], "stages": stages}
        write(run / "task.json", task)
        return task

    class FakeIsolatedCompiler:
        def __init__(self, identity):
            self.identity = identity
            self.checked = False
            self.calls = 0

        def check_executor(self, executor, *, author_workspace):
            self.checked = executor.document["target"] == "sm_103a" and Path(author_workspace).is_dir()

        def compile(self, source, requirements):
            self.calls += 1
            artifacts = {role: f"SYNTHETIC NONEXECUTABLE {role}".encode()
                         for role in ("ttir", "ttgir", "llir", "ptx", "cubin")}
            artifacts["cubin"] = b"\x7fELF SYNTHETIC NONEXECUTABLE " + sha256(source).digest()
            artifacts["source"] = source + b"\n# synthetic compiler expansion\n"
            return TritonCompilation(
                source, "sm_103a", requirements["kernel_entry_point"], artifacts,
                requirements["compile_options"]["num_warps"] * 32, 0,
                self.identity["triton_version"], "cubin",
            )

    def test_frozen_b300_study_and_complete_schedule_pool_are_admitted(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, _ = self.fixture(directory)
            checked = instrument.check_plan(snapshot)
            self.assertEqual(checked["target"], "sm_103a")
            self.assertEqual(checked["candidate_count"], 3)
            self.assertEqual(checked["observation_count"], 9)
            self.assertEqual(checked["baseline_binding"], "compile-stage")
            self.assertEqual(checked["paired_kind"], instrument.PAIRED_KIND)

    def test_changed_order_and_schedule_abi_are_refused_before_device(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, original = self.fixture(directory)
            changed = copy.deepcopy(original)
            changed["observations"] = changed["observations"][::-1]
            write(snapshot / "plan.json", changed)
            with self.assertRaisesRegex(ValueError, "observation order"):
                instrument.check_plan(snapshot)
            write(snapshot / "plan.json", original)
            source = snapshot / original["candidates"][0]["schedule"]
            schedule = json.loads(source.read_text())
            next(buffer for buffer in schedule["buffers"] if buffer["name"] == "a")["shape"][0] = 256
            write(source, schedule)
            with self.assertRaises(ValueError):
                instrument.check_plan(snapshot)

    def test_corpus_placeholder_workload_identity_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, plan = self.fixture(directory)
            source = snapshot / plan["candidates"][0]["schedule"]
            schedule = json.loads(source.read_text())
            schedule["metadata"]["workload_contract_sha256"] = "0" * 64
            write(source, schedule)
            with self.assertRaisesRegex(ValueError, "Workload binding"):
                instrument.check_plan(snapshot)

    def test_calibration_snapshot_cannot_live_in_another_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, _ = self.fixture(directory)
            (Path(directory).resolve() / ".git").mkdir()
            with self.assertRaisesRegex(ValueError, "outside every checkout"):
                instrument.check_plan(snapshot)

    def test_baseline_schedule_is_admitted_before_compile(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, plan = self.fixture(directory)
            source = snapshot / plan["baseline_schedule_path"]
            schedule = json.loads(source.read_text())
            schedule["metadata"]["workload_contract_sha256"] = "0" * 64
            write(source, schedule)
            with self.assertRaisesRegex(ValueError, "Workload binding"):
                instrument.check_plan(snapshot)

    def test_compile_stage_owns_one_baseline_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, plan = self.fixture(directory)
            stage = Path(directory).resolve() / "compile"
            stage.mkdir()
            with patch.object(ExecutorRevision, "admit_host", return_value=object()), \
                    patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "", "GPUQ_JOB_ID": ""}):
                index = instrument.prepare_compile(
                    snapshot, stage, self.FakeIsolatedCompiler(plan["toolchain_identity"]))
            baseline = load_baseline_bundle(ROOT, stage / "baseline/candidate.json")
            self.assertEqual(index["baseline"], candidate_identity(baseline))
            self.assertEqual(json.loads(index["context"]["input_scope"])[
                "baseline_candidate_record_sha256"], baseline.canonical_sha256)

    def test_cpu_compile_seals_all_three_common_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, plan = self.fixture(directory)
            stage = Path(directory).resolve() / "compile"
            stage.mkdir()
            isolated = self.FakeIsolatedCompiler(plan["toolchain_identity"])
            with patch.object(ExecutorRevision, "admit_host", return_value=object()), \
                    patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "", "GPUQ_JOB_ID": ""}):
                index = instrument.prepare_compile(snapshot, stage, isolated)
            self.assertTrue(isolated.checked)
            self.assertEqual(isolated.calls, 4)
            self.assertEqual(len(index["candidates"]), 3)
            self.assertEqual(json.loads((stage / "compile-index.json").read_text()), index)
            self.assertTrue((stage / "baseline/candidate.json").is_file())
            for spec in plan["candidates"]:
                candidate = load_baseline_bundle(ROOT, stage / spec["id"] / "candidate.json")
                self.assertIn("cubin", candidate.artifact_payloads)
                self.assertIn("launch_manifest", candidate.artifact_payloads)
                self.assertEqual(candidate.target, "sm_103a")

    def test_cpu_compile_refuses_unpinned_toolchain_and_visible_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, plan = self.fixture(directory)
            stage = Path(directory).resolve() / "compile"
            stage.mkdir()
            isolated = self.FakeIsolatedCompiler({**plan["toolchain_identity"], "triton_version": "other"})
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "", "GPUQ_JOB_ID": ""}):
                with self.assertRaisesRegex(ValueError, "isolated toolchain differs"):
                    instrument.prepare_compile(snapshot, stage, isolated)
            self.assertEqual(list(stage.iterdir()), [])
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0", "GPUQ_JOB_ID": ""}):
                with self.assertRaisesRegex(ValueError, "must not inherit a GPU"):
                    instrument.prepare_compile(snapshot, stage, self.FakeIsolatedCompiler(plan["toolchain_identity"]))
            self.assertEqual(list(stage.iterdir()), [])

    def test_gpu_infra_local_entry_retains_compiled_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, plan = self.fixture(directory)
            stage = Path(directory).resolve() / "compile"
            stage.mkdir()
            (stage / "stdout.log").write_text("synthetic GPU Infra stage stream")
            isolated = self.FakeIsolatedCompiler(plan["toolchain_identity"])
            environment = {"KERNELINFRA_CANDIDATE_DIR": str(snapshot),
                           "KERNELINFRA_STAGE_DIR": str(stage),
                           "KERNELINFRA_RESULT": str(stage / "result.json"),
                           "KERNELINFRA_STAGE_KIND": "compile",
                           "KERNELINFRA_STAGE_ID": "compile",
                           "CUDA_VISIBLE_DEVICES": "", "GPUQ_JOB_ID": ""}
            with patch.dict(os.environ, environment), \
                    patch.object(ExecutorRevision, "admit_host", return_value=object()), \
                    patch.object(instrument, "IsolatedTritonCompiler", return_value=isolated):
                self.assertEqual(instrument.collect_compile(), 0)
            result = json.loads((stage / "result.json").read_text())
            self.assertEqual((result["status"], result["validity"]), ("passed", "valid"))
            self.assertEqual(result["metrics"]["candidate_count"], 3)
            self.assertIn("baseline/candidate.json", result["artifacts"])
            self.assertIn("tile-64/candidate.json", result["artifacts"])
            self.assertEqual(isolated.calls, 4)

    def test_device_stage_requires_own_task_and_node_assignment(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, plan = self.fixture(directory)
            run = snapshot.parent
            task = self.task_fixture(run, plan)
            checked = instrument._task(run, plan, self.executor, self.workload)
            self.assertEqual(checked, task)
            identity = {"run_id": "SYNTHETIC-NOT-A-GPU-RUN",
                        "task_id": plan["plan_id"],
                        "task_sha256": "a" * 64, "candidate_sha256": "b" * 64}
            write(run / "request.json", {"schema": "kernelinfra.request.v1", **identity})
            state = {"schema": "kernelinfra.state.v1", **identity,
                     "stage_id": "collection", "stage_kind": "judge", "stage_index": 1,
                     "state": "running", "broker_job_id": "gpuq-123456789abc",
                     "gpu_ids": [7], "run_dir": str(run), "terminal_at": None}
            write(run / "state.json", state)
            environment = {"KERNELINFRA_RUN_ID": identity["run_id"],
                           "CUDA_VISIBLE_DEVICES": "7"}
            with patch.dict(os.environ, environment, clear=True):
                assignment = instrument._node_assignment(run, task)
            self.assertEqual(assignment["broker_job_id"], state["broker_job_id"])
            self.assertEqual(assignment["physical_gpu"], 7)
            state["candidate_sha256"] = "c" * 64
            write(run / "state.json", state)
            with patch.dict(os.environ, environment, clear=True), \
                    self.assertRaisesRegex(ValueError, "another run or stage"):
                instrument._node_assignment(run, task)

    def test_device_replays_cpu_compiled_candidate_seals(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, plan = self.fixture(directory)
            run = snapshot.parent
            stage = run / "stages/compile"
            stage.mkdir(parents=True)
            isolated = self.FakeIsolatedCompiler(plan["toolchain_identity"])
            with patch.object(ExecutorRevision, "admit_host", return_value=object()), \
                    patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "", "GPUQ_JOB_ID": ""}):
                instrument.prepare_compile(snapshot, stage, isolated)
            write(stage / "result.json", {"schema": "kernelinfra.stage-result.v1",
                                          "status": "passed", "validity": "valid"})
            write(stage / "receipt.json", {"execution": "local", "exit_code": 0,
                                           "judge_result_valid": True})
            checked = instrument.check_plan(snapshot)
            compiled, baseline, context = instrument._compiled_candidates(
                run, plan, checked, self.compiler, self.workload)
            self.assertEqual(set(compiled), {row["id"] for row in plan["candidates"]})
            self.assertEqual(json.loads(context["input_scope"])[
                "baseline_candidate_record_sha256"], baseline.canonical_sha256)
            baseline_record_path = stage / "baseline/candidate.json"
            baseline_record = json.loads(baseline_record_path.read_text())
            original_record = copy.deepcopy(baseline_record)
            baseline_record["candidate"]["artifact_roles"]["cubin"] = "0" * 64
            write(baseline_record_path, baseline_record)
            with self.assertRaises(ValueError):
                instrument._compiled_candidates(run, plan, checked,
                                                self.compiler, self.workload)
            write(baseline_record_path, original_record)
            index = json.loads((stage / "compile-index.json").read_text())
            index["candidates"][0]["candidate_id"] = "another"
            write(stage / "compile-index.json", index)
            with self.assertRaisesRegex(ValueError, "compiled Schedule differs"):
                instrument._compiled_candidates(run, plan, checked,
                                                self.compiler, self.workload)

    def test_paired_request_uses_common_worker_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, plan = self.fixture(directory)
            run = snapshot.parent
            stage = run / "stages/compile"
            stage.mkdir(parents=True)
            isolated = self.FakeIsolatedCompiler(plan["toolchain_identity"])
            with patch.object(ExecutorRevision, "admit_host", return_value=object()), \
                    patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "", "GPUQ_JOB_ID": ""}):
                instrument.prepare_compile(snapshot, stage, isolated)
            candidate = load_baseline_bundle(ROOT, stage / "tile-64/candidate.json")
            baseline = load_baseline_bundle(ROOT, stage / "baseline/candidate.json")
            study = json.loads((ROOT / plan["study_path"]).read_text())
            observation = run / "stages/collection/fit-tile-64"
            observation.parent.mkdir()
            request = instrument._seal_observation(
                observation, candidate, baseline, plan, study["evaluation_protocol"])
            admitted = worker._load_authority(observation / "request.json")
            self.assertEqual(admitted.candidate.canonical_sha256, candidate.canonical_sha256)
            self.assertEqual(admitted.baseline.canonical_sha256, baseline.canonical_sha256)
            self.assertEqual(admitted.request["evaluation_protocol"], study["evaluation_protocol"])
            self.assertEqual(request["purpose"], "search")

    def test_broker_controller_observes_frozen_nine_without_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, plan = self.fixture(directory)
            run = snapshot.parent
            task = self.task_fixture(run, plan)
            compile_stage = run / "stages/compile"
            compile_stage.mkdir(parents=True)
            isolated = self.FakeIsolatedCompiler(plan["toolchain_identity"])
            with patch.object(ExecutorRevision, "admit_host", return_value=object()), \
                    patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "", "GPUQ_JOB_ID": ""}):
                instrument.prepare_compile(snapshot, compile_stage, isolated)
            write(compile_stage / "result.json", {"schema": "kernelinfra.stage-result.v1",
                                                   "status": "passed", "validity": "valid"})
            write(compile_stage / "receipt.json", {"execution": "local", "exit_code": 0,
                                                    "judge_result_valid": True})
            stage = run / "stages/collection"
            stage.mkdir()
            identity = {"run_id": "SYNTHETIC-NOT-A-GPU-RUN", "task_id": task["task_id"],
                        "task_sha256": "a" * 64, "candidate_sha256": "b" * 64}
            write(run / "request.json", {"schema": "kernelinfra.request.v1", **identity})
            write(run / "state.json", {"schema": "kernelinfra.state.v1", **identity,
                                       "stage_id": "collection", "stage_kind": "judge", "stage_index": 1,
                                       "state": "running", "broker_job_id": "gpuq-123456789abc",
                                       "gpu_ids": [7], "run_dir": str(run), "terminal_at": None})
            seen = []
            real_run = subprocess.run

            def fake_evaluator(command, **kwargs):
                if len(command) < 2 or command[1] != str(ROOT / "src/open_cake_ir/tasks/evaluate.py"):
                    return real_run(command, **kwargs)
                self.assertNotIn("start_new_session", kwargs)
                self.assertEqual(kwargs["env"]["GPUQ_JOB_ID"], "gpuq-123456789abc")
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "7")
                seen.append(Path(command[3]).parent.name)
                Path(command[5]).write_text('{"synthetic":true}')
                return subprocess.CompletedProcess(command, 0)

            def fake_observation(directory, spec, plan, policy, candidate, baseline, job_id, schedule):
                self.assertEqual(directory.name, spec["id"])
                return {"candidate_id": spec["candidate_id"], "split": spec["split"],
                        "schedule": schedule,
                        "observation": {"baseline_us": 100, "gpu_uuid": "GPU-SYNTHETIC",
                                        "candidate_us": 10}}

            environment = {"KERNELINFRA_RUN_DIR": str(run),
                           "KERNELINFRA_CANDIDATE_DIR": str(snapshot),
                           "KERNELINFRA_STAGE_DIR": str(stage),
                           "KERNELINFRA_RESULT": str(stage / "result.json"),
                           "KERNELINFRA_TASK": str(run / "task.json"),
                           "KERNELINFRA_RUN_ID": identity["run_id"],
                           "KERNELINFRA_STAGE_KIND": "judge",
                           "KERNELINFRA_STAGE_ID": "collection",
                           "CUDA_VISIBLE_DEVICES": "7",
                           "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
            with patch.dict(os.environ, environment, clear=True), \
                    patch.object(ExecutorRevision, "admit_host", return_value=object()), \
                    patch.object(instrument, "_broker_parent", return_value=(100, 321, 654)), \
                    patch.object(instrument.subprocess, "run", side_effect=fake_evaluator), \
                    patch.object(instrument, "_observed_evaluation", side_effect=fake_observation):
                self.assertEqual(instrument.collect_device(), 0)
            self.assertEqual(seen, [row["id"] for row in plan["observations"]])
            result = json.loads((stage / "result.json").read_text())
            self.assertEqual((result["status"], result["validity"]), ("passed", "valid"))
            self.assertEqual(result["metrics"]["observation_count"], 9)

    def test_terminal_fit_checks_receipts_before_publishing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, plan = self.fixture(directory)
            run = snapshot.parent
            task = self.task_fixture(run, plan)
            compile_stage = run / "stages/compile"
            compile_stage.mkdir(parents=True)
            isolated = self.FakeIsolatedCompiler(plan["toolchain_identity"])
            with patch.object(ExecutorRevision, "admit_host", return_value=object()), \
                    patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "", "GPUQ_JOB_ID": ""}):
                instrument.prepare_compile(snapshot, compile_stage, isolated)
            run_id = "SYNTHETIC-NOT-A-GPU-RUN"
            node = {"run_id": run_id, "task_id": task["task_id"],
                    "task_sha256": "a" * 64, "candidate_sha256": "b" * 64}
            write(run / "request.json", {"schema": "kernelinfra.request.v1", **node})
            write(run / "state.json", {"schema": "kernelinfra.state.v1", **node,
                                       "state": "completed", "terminal_at": "SYNTHETIC",
                                       "run_dir": str(run)})
            checked = instrument.check_plan(snapshot)
            baseline = load_baseline_bundle(ROOT, compile_stage / "baseline/candidate.json")
            candidates = {spec["id"]: load_baseline_bundle(ROOT, compile_stage / spec["id"] / "candidate.json")
                          for spec in plan["candidates"]}
            policy = json.loads((ROOT / plan["study_path"]).read_text())["evaluation_protocol"]
            assignment = {"run_id": run_id, "broker_job_id": "gpuq-123456789abc",
                          "physical_gpu": 7,
                          "node_state": {"schema": "kernelinfra.state.v1", **node,
                                         "stage_id": "collection", "stage_kind": "judge",
                                         "stage_index": 1}}
            stage = run / "stages/collection"
            stage.mkdir()
            write(stage / "execution-context.json", {**assignment,
                                                      "broker_peer": [100, 321, 654], "uid": 321})
            rows = []
            for spec in plan["observations"]:
                candidate = candidates[spec["candidate_id"]]
                observation_dir = stage / spec["id"]
                instrument._seal_observation(observation_dir, candidate, baseline, plan, policy)
                (observation_dir / "result.json").write_text('{"synthetic":true}')
                candidate_index = [item["id"] for item in plan["candidates"]].index(spec["candidate_id"])
                factor = {"fit": 1, "calibration": 1.02, "audit": 1.03}[spec["split"]]
                schedule_file = next(item["schedule"] for item in plan["candidates"]
                                     if item["id"] == spec["candidate_id"])
                row = {"candidate_id": spec["candidate_id"], "split": spec["split"],
                       "schedule": json.loads((snapshot / schedule_file).read_text()),
                       "observation": {"candidate_record_sha256": candidate.canonical_sha256,
                                       "baseline_record_sha256": baseline.canonical_sha256,
                                       "workload_sha256": self.workload.canonical_sha256,
                                       "case_id": "primary",
                                       "evaluation_protocol_sha256": sha256(canonical_json_bytes(policy)).hexdigest(),
                                       "job_id": assignment["broker_job_id"],
                                       "gpu_uuid": "GPU-SYNTHETIC",
                                       "candidate_us": (10 + candidate_index * 10) * factor,
                                       "baseline_us": 100, "sample_count": 250}}
                write(observation_dir / "observation.json", row)
                rows.append(row)
            compile_index = json.loads((compile_stage / "compile-index.json").read_text())
            write(stage / "observations.json", {"rows": rows, "context": compile_index["context"],
                                                "assignment": assignment})
            for name, kind, execution in (("compile", "compile", "local"),
                                          ("collection", "judge", "broker")):
                directory = run / "stages" / name
                receipt = {"schema": "kernelinfra.stage-receipt.v1", "run_id": run_id,
                           "stage_id": name, "stage_kind": kind, "execution": execution,
                           "judge_identity": self.executor.executor_id,
                           "exit_code": 0, "judge_result_valid": True, "error": None,
                           "gpu_ids": [7] if name == "collection" else [],
                           "broker_job_id": assignment["broker_job_id"] if name == "collection" else None}
                write(directory / "receipt.json", receipt)
                artifacts = {file.relative_to(directory).as_posix(): file.relative_to(directory).as_posix()
                             for file in directory.rglob("*") if file.is_file()
                             and file not in {directory / "result.json", directory / "receipt.json"}}
                write(directory / "result.json", {"schema": "kernelinfra.stage-result.v1",
                                                  "status": "passed", "validity": "valid",
                                                  "artifacts": artifacts})
            write(run / "result.json", {"schema": "kernelinfra.run-result.v1", **node,
                                        "outcome": "completed", "validity": "valid",
                                        "frontier_eligible": False,
                                        "stages": [{"id": name, "kind": kind, "status": "passed", "validity": "valid"}
                                                   for name, kind in (("compile", "compile"),
                                                                      ("collection", "judge"))]})
            by_directory = {spec["id"]: row for spec, row in zip(plan["observations"], rows, strict=True)}
            with patch.object(instrument, "_observed_evaluation",
                              side_effect=lambda directory, *args: by_directory[directory.name]):
                self.assertEqual(instrument.fit_run(run, run / "fit-output"), 0)
            model = json.loads((run / "fit-output/model.json").read_text())
            self.assertEqual(len(model["curves"]), 3)
            self.assertEqual(model["reported_evidence"]["run_id"], run_id)
            receipt_path = stage / "receipt.json"
            tampered = json.loads(receipt_path.read_text())
            tampered["broker_job_id"] = "gpuq-aaaaaaaaaaaa"
            write(receipt_path, tampered)
            with patch.object(instrument, "_observed_evaluation",
                              side_effect=lambda directory, *args: by_directory[directory.name]):
                self.assertEqual(instrument.fit_run(run, run / "tampered-output"), 1)
            failure = json.loads((run / "tampered-output/audit.json").read_text())
            self.assertFalse(failure["passed"])
            self.assertIn("broker assignment", failure["reason"])


if __name__ == "__main__":
    unittest.main()
