"""CPU protocol tests; no GPU or provider qualification is inferred."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.evaluation.gpuq import observe_allocation, validate_allocation
from open_cake_ir.evaluation.paired import admit_device_identity, PAIRED_HIP_KIND
from open_cake_ir.lab import gpu_infra as infra
from tools import kernel_experiment, launch_task, launch_task_matrix

JOB = "gpuq-123456789abc"
COMMIT = "a" * 40


def environment(backend="amd", ids="0"):
    value = {"GPUQ_JOB_ID": JOB, "GPUQ_MODE": "exclusive", "GPUQ_BACKEND": backend,
             "GPUQ_DEVICE_IDS": ids, "GPUQ_OCCUPANCY_SCOPE": "cooperative" if backend == "metal" else "system"}
    if backend in {"amd", "hygon"}:
        value["HIP_VISIBLE_DEVICES"] = ids
    elif backend == "nvidia":
        value["CUDA_VISIBLE_DEVICES"] = ids
    return value


class AllocationTests(unittest.TestCase):
    def test_each_vendor_requires_its_own_broker_and_scope(self):
        for target, backend in (("sm_103a", "nvidia"), ("apple_gpu_family7", "metal"),
                                ("gfx938", "hygon"), ("gfx1151", "amd")):
            with self.subTest(target=target), patch.dict(os.environ, environment(backend), clear=True):
                value = observe_allocation(target)
                self.assertEqual(value["job_id"], JOB)
                for key, wrong in (("backend", "unknown"), ("mode", "shared"),
                                   ("device_ids", [True]), ("device_ids", [0, 1])):
                    with self.assertRaises(ValueError):
                        validate_allocation({**value, key: wrong}, target=target, job_id=JOB)
        with patch.dict(os.environ, environment("hygon"), clear=True), self.assertRaises(ValueError):
            observe_allocation("gfx1151")

    def test_conflicting_visibility_and_placeholder_job_refuse(self):
        for changed in ({"ROCR_VISIBLE_DEVICES": "1"}, {"HIP_VISIBLE_DEVICES": "1"},
                        {"GPUQ_JOB_ID": "gpuq-000000000000"}, {"GPUQ_MODE": "shared"}):
            with patch.dict(os.environ, {**environment(), **changed}, clear=True), self.assertRaises(ValueError):
                observe_allocation("gfx1151")

    def test_hip_paired_evidence_requires_matching_allocation_on_both_records(self):
        with patch.dict(os.environ, environment(), clear=True):
            allocation = observe_allocation("gfx1151")
        raw = {"kind": PAIRED_HIP_KIND, "job_id": JOB, "gpu_uuid": "fixture",
               "broker_allocation": allocation}
        launch = {"job_id": JOB, "gpu_uuid": "fixture", "broker_allocation": allocation}
        participants = {"candidate": {"target": "gfx1151"}}
        admit_device_identity(raw, launch, participants)
        for broken in ({**launch, "broker_allocation": None}, {**launch, "job_id": "gpuq-abcdef123456"}):
            with self.assertRaises(ValueError):
                admit_device_identity(raw, broken, participants)


class InfraTransportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.input = self.root / "original"
        self.input.mkdir()
        (self.input / "artifact").write_bytes(b"sealed candidate")
        (self.input / "workload.json").write_text("{}")
        self.request = {"target": "gfx1151", "case_id": "primary", "purpose": "search",
            "artifact_paths": {"hsaco": "artifact"}, "workload_path": str(self.input / "workload.json")}
        (self.input / "request.json").write_text(json.dumps(self.request))

    def node(self):
        return {"broker": {"backend": "amd", "instance_id": "fixture",
            "probe_error": None, "gpus": [{"id": 0, "state": "idle"}],
            "occupancy_scope": "system", "allocation_environment": "gpuq_v1"}}

    def test_preflight_rejects_old_or_wrong_backend_before_submit(self):
        for change in ({"allocation_environment": None}, {"backend": "hygon"}, {"probe_error": "offline"}):
            node = self.node()
            node["broker"].update(change)
            with patch.object(infra, "invoke", return_value=SimpleNamespace(returncode=0, stdout=json.dumps(node), stderr="")) as command:
                with self.assertRaises(ValueError):
                    infra.preflight("kernelctl", Path("/socket"), "gfx1151")
                self.assertEqual(command.call_count, 1)

    def test_submit_snapshots_then_observes_one_run_and_returns_raw_worker_artifacts(self):
        calls = []
        run_id = "cake-gfx1151-abcdef123456"
        run_dir = self.root / "runs" / run_id
        worker = run_dir / "stages/evaluation/worker"
        worker.mkdir(parents=True)
        (worker / "correctness.json").write_text('{"passed":false}')
        result = {"job_id": JOB, "mode": "exclusive", "receipt": {
            "correctness_passed": False, "artifacts": {"correctness_output": "correctness.json"}}}
        (worker / "result.json").write_text(json.dumps(result))
        def invoke(client, argv, **kwargs):
            calls.append(argv)
            if argv[0] == "node-status":
                payload = json.dumps(self.node())
            elif argv[0] == "task-check":
                task = json.loads(Path(argv[1]).read_bytes())
                self.assertEqual(task["stages"][0]["execution"], "broker")
                self.assertEqual(task["stages"][0]["resources"]["mode"], "exclusive")
                payload = ""
            elif argv[0] == "submit":
                snap = Path(argv[-1])
                (self.input / "artifact").write_bytes(b"next candidate")
                self.assertEqual((snap / "artifact").read_bytes(), b"sealed candidate")
                payload = run_id
            else:
                self.assertEqual(argv[0], "wait")
                self.assertIn(run_id, argv)
                payload = json.dumps({"run_id": run_id, "run_dir": str(run_dir),
                                      "state": "rejected", "broker_job_id": JOB})
            return SimpleNamespace(returncode=0, stdout=payload, stderr="")
        args = SimpleNamespace(kernelctl="kernelctl", socket=Path("/socket"),
            evidence_root=self.root / "evidence", request=self.input / "request.json",
            output=self.input / "result.json", worker_module="open_cake_ir.tasks.evaluate", timeout=30)
        with patch.object(infra, "invoke", side_effect=invoke), patch.object(infra, "checkout_commit", return_value=COMMIT):
            self.assertEqual(infra.submit(args), 0)
        self.assertEqual(json.loads(args.output.read_bytes()), result)
        self.assertEqual((self.input / "correctness.json").read_bytes(), (worker / "correctness.json").read_bytes())
        self.assertEqual(sum(c[0] == "submit" for c in calls), 1)

    def test_snapshot_refuses_escape_and_reserved_input_names(self):
        for name in ("../workload.json", "workload.json"):
            document = {**self.request, "artifact_paths": {"hsaco": name}}
            (self.input / "request.json").write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                infra.snapshot_request(self.input / "request.json", self.root / str(len(name)))

    def test_stage_wrong_target_never_invokes_worker_and_reports_unknown(self):
        stage_dir = self.root / "stage"
        stage_dir.mkdir()
        env = {**environment(), "KERNELINFRA_RESULT": str(stage_dir / "result.json"),
            "KERNELINFRA_STAGE_DIR": str(stage_dir), "KERNELINFRA_CANDIDATE_DIR": str(self.input)}
        with patch.dict(os.environ, env, clear=True), patch.object(infra, "checkout_commit", return_value=COMMIT), \
             patch.object(infra.subprocess, "run") as worker:
            result = infra.stage(SimpleNamespace(commit=COMMIT, target="gfx938", worker_module="open_cake_ir.tasks.evaluate"))
        self.assertEqual(result, 1)
        worker.assert_not_called()
        self.assertEqual(json.loads((stage_dir / "result.json").read_bytes())["validity"], "unknown")

    def test_stage_keeps_worker_inputs_in_stage_and_does_not_promote_timing(self):
        stage_dir = self.root / "stage"
        stage_dir.mkdir()
        env = {**environment(), "KERNELINFRA_RESULT": str(stage_dir / "result.json"),
            "KERNELINFRA_STAGE_DIR": str(stage_dir), "KERNELINFRA_CANDIDATE_DIR": str(self.input)}
        def evaluate(argv, **kwargs):
            request_path = Path(argv[argv.index("--request") + 1])
            request = json.loads(request_path.read_bytes())
            self.assertEqual(request_path.parent, stage_dir / "worker")
            self.assertEqual(Path(request["workload_path"]).read_bytes(), b"{}")
            self.assertEqual((request_path.parent / "artifact").read_bytes(), b"sealed candidate")
            Path(argv[argv.index("--output") + 1]).write_text(json.dumps({
                "job_id": JOB, "mode": "exclusive", "admitted": True, "error": None,
                "receipt": {"correctness_passed": True, "timing": {"pooled_median_ms": 1.0}}}))
            return SimpleNamespace(returncode=0)
        with patch.dict(os.environ, env, clear=True), patch.object(infra, "checkout_commit", return_value=COMMIT), \
             patch.object(infra.subprocess, "run", side_effect=evaluate):
            self.assertEqual(infra.stage(SimpleNamespace(commit=COMMIT, target="gfx1151", worker_module="open_cake_ir.tasks.evaluate")), 0)
        result = json.loads((stage_dir / "result.json").read_bytes())
        self.assertEqual(result["validity"], "valid")
        self.assertNotIn("workloads", result)
        self.assertNotIn("timing", result)
        self.assertEqual(json.loads((self.input / "request.json").read_bytes()), self.request)

    def test_unknown_observation_never_resubmits_or_returns_success(self):
        actions = []
        def invoke(client, argv, **kwargs):
            actions.append(argv[0])
            payload = json.dumps(self.node()) if argv[0] == "node-status" else (
                "cake-gfx1151-abcdef123456" if argv[0] == "submit" else "")
            return SimpleNamespace(returncode=1 if argv[0] == "wait" else 0, stdout=payload, stderr="offline")
        args = SimpleNamespace(kernelctl="kernelctl", socket=Path("/socket"),
            evidence_root=self.root / "evidence", request=self.input / "request.json",
            output=self.input / "result.json", worker_module="open_cake_ir.tasks.evaluate", timeout=30)
        with patch.object(infra, "invoke", side_effect=invoke), patch.object(infra, "checkout_commit", return_value=COMMIT):
            with self.assertRaisesRegex(ValueError, "observation unknown"):
                infra.submit(args)
        self.assertEqual(actions.count("submit"), 1)
        self.assertNotIn("cancel", actions)
        self.assertFalse(args.output.exists())


class ExperimentInputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        ref = self.root / "kernel.py"
        ref.write_text("# reviewed reference\n")
        node = {"transport": "local", "project_root": str(kernel_experiment.ROOT),
            "python": "/fixture/python", "kernelctl": "/fixture/kernelctl", "socket": "/fixture/socket",
            "workspace": str(self.root / "node-run")}
        self.config = {"schema_version": 1, "objective": "Reproduce the supplied structure",
            "provider": {"harness": "codex", "model": "fixture", "effort": "high"},
            "budget": {"turns": 2, "token_budget": 10000, "wall_seconds": 120},
            "references": [{"path": str(ref), "source": "reference repository at fixture commit"}],
            "cells": [{"id": "metal", "task": "rmsnorm", "backend": "metal-m1-pro",
                       "rows": 2, "columns": 8, "node": node}]}

    def test_prepared_multitarget_inputs_include_reference_and_rules(self):
        second = deepcopy(self.config["cells"][0])
        second.update(id="b300", backend="triton-b300")
        second["node"].update(transport="ssh", host="B300-M2", workspace="/tmp/cake-b300-fixture")
        second["node"]["provider_executable"] = "/opt/codex/bin/codex"
        self.config["cells"].append(second)
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(self.config))
        output = self.root / "experiment"
        with patch.object(kernel_experiment, "checkout_commit", return_value=COMMIT):
            kernel_experiment.prepare(config_path, output)
        self.assertEqual((output / "AGENTS.md").read_bytes(), kernel_experiment.POLICY.read_bytes())
        self.assertIn("reviewed reference", (output / "scaffold.md").read_text())
        self.assertIn("sm_103a", (output / "TASK.md").read_text())
        with patch.object(kernel_experiment, "checkout_commit", return_value=COMMIT), \
             patch.object(kernel_experiment.subprocess, "run", return_value=SimpleNamespace(returncode=255)) as launch:
            self.assertEqual(kernel_experiment.run_cell(output, "b300"), 255)
            self.assertEqual(launch.call_count, 1)
            argv = launch.call_args.args[0]
            self.assertEqual(argv[0], "ssh")
            self.assertIn("B300-M2", argv)
            payload = json.loads(launch.call_args.kwargs["input"])
            self.assertEqual(payload["cell"]["node"]["provider_executable"], "/opt/codex/bin/codex")
            with self.assertRaises(FileExistsError):
                kernel_experiment.run_cell(output, "b300")
        receipt = json.loads((output / "launches/b300/transport.json").read_bytes())
        self.assertEqual(receipt["observation"], "failed_or_unknown_no_retry")

    def test_node_bootstrap_passes_explicit_native_provider(self):
        node = deepcopy(self.config["cells"][0]["node"])
        node["provider_executable"] = "/opt/codex/bin/codex"
        node["http_proxy"] = "http://127.0.0.1:17990"
        node["qualification"] = "/external/receipt.json"
        node["qualification_anchor"] = "/external/anchor.json"
        payload = {"cell": {**self.config["cells"][0], "node": node},
            "source_commit": COMMIT, "scaffold": "rules", "provider": self.config["provider"],
            "budget": self.config["budget"]}
        import io
        with patch("sys.stdin", io.StringIO(json.dumps(payload))), \
             patch("subprocess.run", return_value=SimpleNamespace(returncode=0)) as execute:
            with self.assertRaises(SystemExit) as result:
                exec(kernel_experiment._NODE, {})
        self.assertEqual(result.exception.code, 0)
        command = execute.call_args.args[0]
        self.assertEqual(command[command.index("--provider-executable") + 1], "/opt/codex/bin/codex")
        self.assertEqual(command[command.index("--qualification") + 1], "/external/receipt.json")
        self.assertEqual(command[command.index("--qualification-anchor") + 1], "/external/anchor.json")
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            self.assertEqual(execute.call_args.kwargs["env"][key], node["http_proxy"])

    def test_proxy_credentials_and_non_http_routes_are_not_task_metadata(self):
        for proxy in ("http://user:secret@host:7890", "file:///tmp/proxy", "http://host", "http://host:7890/path"):
            config = deepcopy(self.config)
            config["cells"][0]["node"]["http_proxy"] = proxy
            with self.assertRaises(ValueError):
                kernel_experiment.validate(config)

    def test_qualification_requires_both_external_locators(self):
        config = deepcopy(self.config)
        config['cells'][0]['node']['qualification'] = '/external/receipt.json'
        with self.assertRaisesRegex(ValueError, 'supplied together'):
            kernel_experiment.validate(config)

    def test_duplicate_workspace_and_undeclared_target_refuse(self):
        for mutation in ("workspace", "backend"):
            config = deepcopy(self.config)
            cell = deepcopy(config["cells"][0])
            cell["id"] = "second"
            if mutation == "backend":
                cell["backend"] = "triton-unknown"
            config["cells"].append(cell)
            with self.assertRaises(ValueError):
                kernel_experiment.validate(config)

    def test_matrix_forwards_rules_and_single_allocator(self):
        args = SimpleNamespace(backend="triton-b300", harness="codex", model="fixture", effort="high",
            turns=2, token_budget=10000, max_candidates=1, searches_per_turn=1, wall_seconds=120,
            maximum_cv=None, required_pair_wins=None, dispatches_per_sample=None, gpu_run=None,
            broker_socket=None, rows=2, columns=8, provider_executable=None, provider_revision=None,
            incumbent_registry=None, agents_md=Path("/rules/AGENTS.md"), kernelctl=Path("/bin/kernelctl"),
            infra_socket=Path("/tmp/kernel.sock"))
        command = launch_task_matrix._command(args, "rmsnorm", self.root / "run", None)
        for flag in ("--agents-md", "--kernelctl", "--infra-socket"):
            self.assertIn(flag, command)
        self.assertNotIn("--gpu-run", command)

    def test_runtime_infra_path_never_wraps_local_broker(self):
        executor = SimpleNamespace(document={"host_environment": {"python": {"invocation_path": "/fixture/python"}}})
        with patch.object(launch_task.shutil, "which", return_value=__file__), \
             patch.object(launch_task, "_launch_toolchain", return_value=SimpleNamespace(runtime_section=lambda *a: {})):
            config = launch_task._runtime_config(self.root, executor, "/provider", "metal",
                allocation="local_broker", kernelctl="kernelctl", infra_socket=Path("/socket"))
        command = config["broker"]["command"]
        self.assertIn("open_cake_ir.lab.gpu_infra", command)
        self.assertNotIn("open_cake_ir.evaluation.local_broker", command)


if __name__ == "__main__":
    unittest.main()
