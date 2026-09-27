"""CPU-only synthetic artifact relations; none of these fixtures is a GPU result."""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("empirical_fitter", ROOT / "tools/calibrate_empirical_cost.py")
instrument = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(instrument)
from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.performance.compiled_resources import CompiledResources


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))


class FitterBindingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def fixture(self, root, target="sm_100a"):
        run = root / "run"
        base = json.loads((ROOT / "corpus/schedules/fma-b8-smoke.json").read_text())
        base["target"] = target
        cases, curves, rows = [], [], []
        duration_by_id = {}
        collector = Path(instrument.__file__).read_bytes()
        assessment = self.compiler.assess(base)
        revision = assessment.compiler_revision_id
        for variant, cap in enumerate((64, 96, 128)):
            curve_id = f"synthetic-r{cap}"
            curves.append({"id": curve_id, "extent_multiple": 8, "varying_dimensions": [{"buffer": name, "dimension": 0} for name in ("a", "b", "c", "y")]})
            for extent, split in ((8, "fit"), (32, "fit"), (16, "calibration"), (24, "audit")):
                document = copy.deepcopy(base)
                document["residency"]["registers_per_thread"] = cap
                document["schedule_id"] = f"{curve_id}-x{extent}"
                for buffer in document["buffers"]:
                    if buffer["space"] == "global":buffer["shape"][0] = extent
                case = {"id": document["schedule_id"], "curve_id": curve_id, "extent": extent, "split": split, "schedule": f"schedules/{document['schedule_id']}.json", "family": "fma", "workload_id": f"synthetic-x{extent}"}
                cases.append(case)
                write(run / "candidate" / case["schedule"], document)
                assessment = self.compiler.assess(document)
                lowering = self.compiler.lower(assessment)
                # This synthetic ELF-marked blob tests identity checks, not execution.
                cubin = b"\x7fELF-SYNTHETIC-NOT-EXECUTABLE-" + case["id"].encode()
                resource = CompiledResources(lowering.source_sha256, sha256(cubin).hexdigest(), target, lowering.toolchain_requirements["kernel_entry_point"], 128, 16, 0, 0, 0, 0, "synthetic", "synthetic")
                duration = 10 + variant + extent / 16
                duration_by_id[case["id"]] = duration
                row = {**case, "grid": list(lowering.toolchain_requirements["grid"]), "profile": {"compiled_resources": resource.as_dict()}, "correct": True, "inputs_unchanged": True, "quality_passed": True, "samples_us": []}
                rows.append(row)
                path = run / "stages/compile" / f"{len(rows)-1:04d}"
                write(path / "schedule.json", json.loads(assessment.schedule_bytes))
                (path / "lowered.py").write_bytes(lowering.source.encode())
                (path / "kernel.cubin").write_bytes(cubin)
                write(path / "launch.json", {"entry_point": resource.entry_point,
                                              "threads_per_cta": resource.threads_per_cta,
                                              "dynamic_shared_bytes": resource.dynamic_shared_bytes,
                                              "grid": row["grid"]})
        plan = {"state": "frozen", "collector_sha256": sha256(collector).hexdigest(), "compiler_revision_id": revision, "target": target, "device_name": self.compiler._revision.targets[target].device_names[0], "multiprocessor_count": 1, "broker_uid": 1000, "cuobjdump": "/usr/bin/true", "container_image_id": "sha256:" + "0" * 64, "expected_runtime": {"compiler_version": "synthetic"}, "model_id": "synthetic-boundary-test", "input_scope": "SYNTHETIC CPU CONTRACT TEST ONLY; no measured performance", "cases": cases, "curves": curves, "sampling": {"warmup": 1, "rounds": 2, "repetitions": 2, "l2_flush_bytes": 268435456}, "acceptance": {"maximum_cohort_cv": .05, "maximum_repeat_median_ratio": 1.05}, "model_acceptance": {"maximum_mape": .1, "maximum_relative_error": .2, "maximum_top2_regret_ratio": 1.05, "maximum_candidates_per_turn": 3, "envelope_allowance": .05}}
        write(run / "candidate/plan.json", plan)
        compile_stage = run / "stages/compile"
        (compile_stage / "collector.py").write_bytes(collector)
        write(compile_stage / "plan.json", plan)
        write(compile_stage / "observations.json", {"schema_version": 1, "rows": [
            {key: value for key, value in row.items() if key in {*case, "grid", "profile"}}
            for row, case in zip(rows, cases, strict=True)
        ], "compiler_version": "synthetic", "inspector_version": "synthetic"})
        write(compile_stage / "receipt.json", {"execution": "local", "exit_code": 0,
                                                "judge_result_valid": True})
        stages = [{"id": "compile", "status": "passed", "validity": "valid"}]
        for phase in ("correctness", "collection"):
            path = run / "stages" / phase
            path.mkdir(parents=True, exist_ok=True)
            (path / "collector.py").write_bytes(collector)
            write(path / "plan.json", plan)
            write(path / "observations.json", {"schema_version": 1, "runtime": {"compiler_version": "synthetic"}, "device_checks_passed": True, "rows": rows})
            write(path / "receipt.json", {"execution": "broker", "exit_code": 0, "judge_result_valid": True, "broker_job_id": "synthetic-not-an-actual-job", "gpu_ids": [0]})
            write(path / "execution-context.json", {"broker_job_id": "synthetic-not-an-actual-job",
                                                         "physical_gpu": 0, "run_id": "SYNTHETIC-NOT-A-GPU-RUN",
                                                         "uid": 1000})
            stages.append({"id": phase, "status": "passed", "validity": "valid"})
        write(run / "result.json", {"outcome": "completed", "validity": "valid", "run_id": "SYNTHETIC-NOT-A-GPU-RUN", "stages": stages})
        for repetition in range(2):
            order = []
            for round_index in range(2):
                indices = [(round_index + repetition * 7 + offset) % len(rows) for offset in range(len(rows))]
                order.extend(indices[::-1] if repetition else indices)
            events = []
            for position, index in enumerate(order):
                row = rows[index]
                common = {"device": 0, "context": 1, "stream": 7}
                events.extend([
                    {"cat": "kernel", "name": "FillFunctor<unsigned char>", "ts": position * 100, "dur": 1, "args": common},
                    {"cat": "kernel", "name": row["profile"]["compiled_resources"]["entry_point"], "ts": position * 100 + 2, "dur": duration_by_id[row["id"]], "args": {**common, "grid": row["grid"], "block": [128, 1, 1], "correlation": position + 1}},
                    {"cat": "cuda_driver", "name": "cuLaunchKernel", "args": {"correlation": position + 1}},
                ])
            path = run / "stages/collection"
            write(path / f"cupti-trace-{repetition}.json", {"traceEvents": events})
            write(path / f"launch-order-{repetition}.json", [rows[index]["id"] for index in order])
        return run

    def test_positive_complete_fitter_contract(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            run = self.fixture(root)
            self.assertEqual(instrument._fit(run, root / "output"), 0)
            model = json.loads((root / "output/model.json").read_text())
            self.assertEqual(len(model["curves"]), 3)
            self.assertEqual(model["reported_evidence"]["validation"]["max_top2_regret_ratio"], 1)

    def test_b300_fitter_binds_its_own_target(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            run = self.fixture(root, target="sm_103a")
            self.assertEqual(instrument._fit(run, root / "output"), 0)
            model = json.loads((root / "output/model.json").read_text())
            self.assertEqual(model["target"], "sm_103a")
            target, binary_version = instrument._target_contract(self.compiler, json.loads((run / "candidate/plan.json").read_text()))
            self.assertEqual(target.compute_capability, (10, 3))
            self.assertEqual(binary_version, 103)

    def test_b300_plan_is_checked_before_device_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.fixture(root, target="sm_103a")
            plan, _ = instrument._check_plan(run / "candidate")
            self.assertEqual(plan["target"], "sm_103a")
            self.assertEqual(len(plan["cases"]), 12)
            for mutation, message in ((lambda p: p.update(device_name="NVIDIA B200"), "device name"),
                                      (lambda p: p["cases"][0].update(family="gemm_bias"), "ABI"),
                                      (lambda p: p["cases"][0].update(extent=7), "extent alignment")):
                changed = json.loads((run / "candidate/plan.json").read_text())
                mutation(changed)
                write(run / "candidate/plan.json", changed)
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    instrument._check_plan(run / "candidate")
                write(run / "candidate/plan.json", plan)
            schedule = run / "candidate" / plan["cases"][1]["schedule"]
            drifted = json.loads(schedule.read_text())
            drifted["residency"]["registers_per_thread"] += 1
            write(schedule, drifted)
            with self.assertRaisesRegex(ValueError, "curve template drifts"):
                instrument._check_plan(run / "candidate")

    def test_local_compile_stage_prepares_without_a_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.fixture(root, target="sm_103a")
            stage = run / "stages/compile"
            shutil.rmtree(stage)
            stage.mkdir()
            fake_torch = types.SimpleNamespace(
                set_num_threads=lambda count: self.assertEqual(count, 1),
                save=lambda value, path: Path(path).write_text(json.dumps(value)),
            )

            def compile_source(source, requirements):
                binary = b"\x7fELF-SYNTHETIC-NONEXECUTABLE-" + sha256(source).digest()
                return types.SimpleNamespace(
                    source=source, target="sm_103a",
                    entry_point=requirements["kernel_entry_point"],
                    artifacts={"cubin": binary, "ptx": b"SYNTHETIC PTX"},
                    threads_per_cta=requirements["compile_options"]["num_warps"] * 32,
                    dynamic_shared_bytes=0, compiler_version="synthetic",
                )

            def inspect(compilation, _cuobjdump):
                return CompiledResources(
                    sha256(compilation.source).hexdigest(),
                    sha256(compilation.artifacts["cubin"]).hexdigest(),
                    "sm_103a", compilation.entry_point,
                    compilation.threads_per_cta, 16, 0, 0, 0, 0,
                    "synthetic", "synthetic",
                )

            environment = {"KERNELINFRA_RUN_DIR": str(run),
                           "KERNELINFRA_STAGE_DIR": str(stage),
                           "KERNELINFRA_CANDIDATE_DIR": str(run / "candidate"),
                           "KERNELINFRA_RESULT": str(stage / "result.json"),
                           "KERNELINFRA_STAGE_KIND": "compile",
                           "KERNELINFRA_STAGE_ID": "compile",
                           "CUDA_VISIBLE_DEVICES": ""}
            with patch.dict(os.environ, environment), patch.dict(sys.modules, {"torch": fake_torch}), \
                    patch.object(instrument, "compile_triton", side_effect=compile_source), \
                    patch.object(instrument, "inspect_triton_resources", side_effect=inspect), \
                    patch.object(instrument, "_cpu_case", return_value=([1, 2, 3], 4, 0.0)):
                instrument._prepare()
            result = json.loads((stage / "result.json").read_text())
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["metrics"]["case_count"], 12)
            self.assertEqual(len(json.loads((stage / "observations.json").read_text())["rows"]), 12)

    def test_three_member_subsets_expose_regret_hidden_by_full_pool(self):
        audit = [
            {"id": "a", "workload_id": "w", "predicted_us": 1, "observed_us": 1},
            {"id": "b", "workload_id": "w", "predicted_us": 2, "observed_us": 20},
            {"id": "c", "workload_id": "w", "predicted_us": 3, "observed_us": 100},
            {"id": "d", "workload_id": "w", "predicted_us": 4, "observed_us": 2},
        ]
        regrets = instrument._candidate_set_regrets(audit, 3)
        self.assertEqual(len(regrets), 4)
        self.assertEqual(max(row["top2_regret_ratio"] for row in regrets), 10)
        self.assertIn(["b", "c", "d"], [row["candidate_set"] for row in regrets])

    def test_prediction_ties_follow_provider_order_at_cut(self):
        audit = [
            {"id": "a", "workload_id": "w", "predicted_us": 1, "observed_us": 100},
            {"id": "b", "workload_id": "w", "predicted_us": 1, "observed_us": 10},
            {"id": "c", "workload_id": "w", "predicted_us": 1, "observed_us": 1},
        ]
        regrets = instrument._candidate_set_regrets(audit, 3)
        self.assertEqual(regrets[0]["provider_order"], ["a", "b", "c"])
        self.assertEqual(regrets[0]["top2_regret_ratio"], 10)

    def test_broker_receipt_must_match_device_stage_assignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.fixture(root)
            context = run / "stages/collection/execution-context.json"
            value = json.loads(context.read_text())
            value["physical_gpu"] = 7
            write(context, value)
            with self.assertRaisesRegex(ValueError, "broker assignment differs"):
                instrument._fit(run, root / "output")

    def test_container_stage_maps_only_the_broker_assigned_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.fixture(root, target="sm_103a")
            stage = run / "stages/collection"
            write(run / "state.json", {"run_id": "synthetic-run", "stage_id": "collection",
                                        "state": "running", "broker_job_id": "gpuq-synthetic",
                                        "gpu_ids": [3]})
            environment = {"KERNELINFRA_RUN_DIR": str(run),
                           "KERNELINFRA_STAGE_DIR": str(stage),
                           "KERNELINFRA_CANDIDATE_DIR": str(run / "candidate"),
                           "KERNELINFRA_RESULT": str(stage / "result.json"),
                           "KERNELINFRA_TASK": str(run / "task.json"),
                           "KERNELINFRA_STAGE_KIND": "profile",
                           "KERNELINFRA_STAGE_ID": "collection",
                           "KERNELINFRA_RUN_ID": "synthetic-run",
                           "CUDA_VISIBLE_DEVICES": "3"}
            commands = []
            def fake_run(command, **_kwargs):
                commands.append(command)
                return types.SimpleNamespace(returncode=0)
            with patch.dict(os.environ, environment, clear=True), \
                    patch.object(instrument, "_broker_parent", return_value=(123, 1000, 1000)), \
                    patch.object(instrument.subprocess, "run", side_effect=fake_run):
                instrument._collect_container()
            self.assertEqual(len(commands), 1)
            self.assertIn("device=3", commands[0])
            self.assertIn("CUDA_VISIBLE_DEVICES=0", commands[0])
            self.assertIn("TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor", commands[0])
            self.assertTrue(any(value.startswith("USER=") for value in commands[0]))
            self.assertNotIn("device=all", commands[0])
            context = json.loads((stage / "broker-container.json").read_text())
            self.assertEqual(context["physical_gpu"], 3)
            self.assertEqual(context["broker_job_id"], "gpuq-synthetic")

    def test_daemon_state_binds_broker_job_to_the_running_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            state = {"run_id": "synthetic-run", "stage_id": "collection",
                     "state": "running", "broker_job_id": "gpuq-synthetic",
                     "gpu_ids": [3]}
            write(run / "state.json", state)
            with patch.dict(os.environ, {"KERNELINFRA_RUN_ID": "synthetic-run"}):
                self.assertEqual(instrument._node_broker_assignment(run, "collection", 3), "gpuq-synthetic")
                with self.assertRaisesRegex(ValueError, "daemon broker assignment"):
                    instrument._node_broker_assignment(run, "collection", 2, timeout_s=0)

    def test_local_container_stage_exposes_no_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.fixture(root, target="sm_103a")
            stage = run / "stages/compile"
            environment = {"KERNELINFRA_RUN_DIR": str(run),
                           "KERNELINFRA_STAGE_DIR": str(stage),
                           "KERNELINFRA_CANDIDATE_DIR": str(run / "candidate"),
                           "KERNELINFRA_RESULT": str(stage / "result.json"),
                           "KERNELINFRA_TASK": str(run / "task.json"),
                           "KERNELINFRA_STAGE_KIND": "compile",
                           "KERNELINFRA_STAGE_ID": "compile",
                           "KERNELINFRA_RUN_ID": "synthetic-run",
                           "CUDA_VISIBLE_DEVICES": ""}
            commands = []
            with patch.dict(os.environ, environment), \
                    patch.object(instrument, "_run_named_container", side_effect=lambda command, _name: commands.append(command)):
                instrument._compile_container()
            self.assertEqual(len(commands), 1)
            self.assertNotIn("--gpus", commands[0])
            self.assertIn("NVIDIA_VISIBLE_DEVICES=void", commands[0])
            self.assertIn("CUDA_VISIBLE_DEVICES=", commands[0])
            self.assertIn("TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor", commands[0])

    def test_changed_stage_templates_cannot_relabel_original_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.fixture(root)
            for path in (run / "stages/compile").glob("*/schedule.json"):
                value = json.loads(path.read_text())
                value["residency"]["registers_per_thread"] = 128
                write(path, value)
            with self.assertRaisesRegex(ValueError, "Schedule differs"):
                instrument._fit(run, root / "output")
            self.assertFalse((root / "output").exists())

    def test_missing_or_corrupt_artifacts_refuse_model_generation(self):
        for filename, action in [("kernel.cubin", "delete"), ("kernel.cubin", "corrupt"), ("lowered.py", "corrupt")]:
            with self.subTest(filename=filename, action=action), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run = self.fixture(root)
                path = run / "stages/compile/0000" / filename
                if action == "delete":path.unlink()
                else:path.write_bytes(path.read_bytes() + b"changed")
                with self.assertRaises((ValueError, OSError)):instrument._fit(run, root / "output")
                self.assertFalse((root / "output").exists())

    def test_changed_compiled_binary_with_updated_identity_still_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self.fixture(root)
            path = run / "stages/compile/0000/kernel.cubin"
            changed = path.read_bytes() + b"different-compiled-binary"
            path.write_bytes(changed)
            observation = run / "stages/compile/observations.json"
            value = json.loads(observation.read_text())
            value["rows"][0]["profile"]["compiled_resources"]["cubin_sha256"] = sha256(changed).hexdigest()
            write(observation, value)
            with self.assertRaisesRegex(ValueError, "compiled observation differs"):
                instrument._fit(run, root / "output")
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":unittest.main()
