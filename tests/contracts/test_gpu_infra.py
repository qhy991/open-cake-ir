"""CPU protocol tests; no GPU or provider qualification is inferred."""
from copy import deepcopy
import io
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
        self.assertFalse((output / "cells").exists())
        with patch.object(kernel_experiment, "checkout_commit", return_value=COMMIT), \
             patch.object(kernel_experiment.subprocess, "run", return_value=SimpleNamespace(returncode=255)) as launch:
            self.assertEqual(kernel_experiment.run_cell(output, "b300"), 255)
            self.assertEqual(launch.call_count, 1)
            argv = launch.call_args.args[0]
            self.assertEqual(argv[0], "ssh")
            self.assertIn("B300-M2", argv)
            payload = json.loads(launch.call_args.kwargs["input"])
            self.assertEqual(payload["cell"]["node"]["provider_executable"], "/opt/codex/bin/codex")
            self.assertEqual(payload["scaffold"], (output / "scaffold.md").read_text())
            with self.assertRaises(FileExistsError):
                kernel_experiment.run_cell(output, "b300")
        receipt = json.loads((output / "launches/b300/transport.json").read_bytes())
        self.assertEqual(receipt["observation"], "failed_or_unknown_no_retry")

    def _per_cell_config(self):
        config = deepcopy(self.config)
        config["schema_version"] = 2
        references = config.pop("references")
        first = config["cells"][0]
        first.update(id="rmsnorm-r1", references=references)
        second = deepcopy(first)
        second.update(id="silu-r1", task="silu")
        second["node"]["workspace"] = str(self.root / "silu-run")
        reference = self.root / "silu-reference.py"
        reference.write_text("# silu-only mechanism\n")
        second["references"] = [{"path": str(reference), "source": "silu reference at fixture commit"}]
        config["cells"].append(second)
        return config

    def _prepare_experiment(self, config, name="experiment"):
        config_path = self.root / f"{name}.json"
        config_path.write_text(json.dumps(config))
        output = self.root / name
        with patch.object(kernel_experiment, "checkout_commit", return_value=COMMIT):
            kernel_experiment.prepare(config_path, output)
        return output

    def _launch_payload(self, output, cell_id):
        with patch.object(kernel_experiment, "checkout_commit", return_value=COMMIT), \
             patch.object(kernel_experiment.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as launch:
            self.assertEqual(kernel_experiment.run_cell(output, cell_id), 0)
        launch.assert_called_once()
        return json.loads(launch.call_args.kwargs["input"])

    def _budget_command(self, budget, *, version=2, name="budget"):
        config = deepcopy(self.config) if version == 1 else self._per_cell_config()
        config["cells"] = config["cells"][:1]
        config["cells"][0]["node"]["workspace"] = str(self.root / f"{name}-run")
        config["budget"] = budget
        output = self._prepare_experiment(config, name)
        self.assertEqual(json.loads((output / "experiment.json").read_bytes())["budget"], budget)
        payload = self._launch_payload(output, config["cells"][0]["id"])
        self.assertEqual(payload["budget"], budget)
        self.assertEqual(json.loads((output / "launches" / config["cells"][0]["id"]
                                    / "request.json").read_bytes())["budget"], budget)
        with patch("sys.stdin", io.StringIO(json.dumps(payload))), \
             patch("subprocess.run", return_value=SimpleNamespace(returncode=0)) as execute:
            with self.assertRaises(SystemExit) as result:
                exec(kernel_experiment._NODE, {})
        self.assertEqual(result.exception.code, 0)
        self.assertEqual(execute.call_count, 2)  # source checkout, then the task launcher
        command = execute.call_args.args[0]
        for key, value in budget.items():
            flag = "--" + key.replace("_", "-")
            if value is None:
                self.assertNotIn(flag, command)
            else:
                self.assertEqual(command[command.index(flag) + 1], str(value))
        return command

    def _resolved_budget(self, command, *, refusal=None):
        # Exercise the real CLI parser and budget owner; stop before stack admission.
        captured = []
        original = launch_task.task_run_inputs
        def inputs(*args, **kwargs):
            value = original(*args, **kwargs)
            captured.append(value)
            return value
        with patch.object(launch_task, "_provider_executable", return_value=Path("/fixture/provider")), \
             patch.object(infra, "preflight", return_value={}), \
             patch.object(launch_task, "task_run_inputs", side_effect=inputs), \
             patch.object(launch_task, "_admit_stack", side_effect=RuntimeError("stop before execution")) as admit, \
             patch.object(launch_task, "_prepare_baseline") as baseline, \
             patch.object(launch_task, "_qualify") as qualify, \
             patch.object(launch_task, "execute_run_from_config") as run:
            with self.assertRaisesRegex(ValueError if refusal else RuntimeError,
                                        refusal or "stop before execution"):
                launch_task.main(command[2:])
            if refusal:
                admit.assert_not_called()
            else:
                admit.assert_called_once()
            baseline.assert_not_called()
            qualify.assert_not_called()
            run.assert_not_called()
        return captured[0]["budget"] if captured else None

    def test_v2_explicit_budget_survives_prepare_transport_and_real_task_inputs(self):
        budget = {"turns": 8, "token_budget": 150000, "wall_seconds": 3600,
                  "max_candidates": 3, "searches_per_turn": 3,
                  "max_compilations": 24, "confirmation_seconds": 600}
        observed = self._resolved_budget(self._budget_command(budget))
        self.assertEqual(observed, {"unit": "provider_tokens", "limit": 150000,
            "checkpoints": [150000], "maximum_turns": 8, "maximum_candidates_per_turn": 3,
            "maximum_compilations": 24, "confirmation_wall_time_seconds": 600.0,
            "wall_time_seconds": 3600, "active_authoring_time_seconds": 1800.0,
            "evaluation_limits": {"search": 24, "attribution": 24, "confirmatory": 8}})

    def test_omitted_v2_controls_preserve_v1_cli_defaults_and_null_tokens(self):
        budget = {**self.config["budget"], "token_budget": None}
        legacy = self._budget_command(budget, version=1, name="legacy")
        current = self._budget_command(budget, name="current")
        for command in (legacy, current):
            for flag in ("--max-candidates", "--searches-per-turn", "--max-compilations", "--confirmation-seconds"):
                self.assertNotIn(flag, command)
        observed = self._resolved_budget(current)
        self.assertEqual(observed, self._resolved_budget(legacy))
        self.assertIsNone(observed["limit"])
        self.assertEqual(observed["checkpoints"], [])

    def test_v2_budget_controls_can_each_override_one_cli_default(self):
        base = self.config["budget"]
        defaults = self._resolved_budget(self._budget_command(base, name="defaults"))
        for field, value, projected in (
            ("max_candidates", 4, "maximum_candidates_per_turn"),
            ("searches_per_turn", 1, "evaluation_limits"),
            ("max_compilations", 7, "maximum_compilations"),
            ("confirmation_seconds", 12.5, "confirmation_wall_time_seconds"),
        ):
            with self.subTest(field=field):
                command = self._budget_command({**base, field: value}, name=field)
                observed = self._resolved_budget(command)
                expected = deepcopy(defaults)
                expected[projected] = ({"search": base["turns"], "attribution": base["turns"],
                                        "confirmatory": base["turns"]}
                                       if field == "searches_per_turn" else value)
                self.assertEqual(observed, expected)

    def test_partial_override_conflicts_are_refused_by_real_budget_owner_before_execution(self):
        for field, value in (("max_candidates", 1), ("searches_per_turn", 4)):
            with self.subTest(field=field):
                command = self._budget_command({**self.config["budget"], field: value}, name=field)
                self.assertIsNone(self._resolved_budget(command,
                    refusal="searches per Turn must fit the candidate budget"))

    def test_v1_rejects_v2_budget_controls(self):
        for field in ("max_candidates", "searches_per_turn", "max_compilations", "confirmation_seconds"):
            config = deepcopy(self.config)
            config["budget"][field] = 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                kernel_experiment.validate(config)

    def test_v2_budget_rejects_invalid_explicit_values_and_pairs(self):
        config = self._per_cell_config()
        for field in ("turns", "wall_seconds", "token_budget", "max_candidates", "searches_per_turn", "max_compilations"):
            for value in (True, 0, -1, "3", 1.5, float("nan"), float("inf"), None):
                if field == "token_budget" and value is None:
                    continue
                invalid = deepcopy(config)
                invalid["budget"][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    kernel_experiment.validate(invalid)
        for value in (True, None, "10", 0, -1, float("nan"), float("inf"), -float("inf"), 120, 121, 10 ** 400):
            invalid = deepcopy(config)
            invalid["budget"]["confirmation_seconds"] = value
            with self.subTest(confirmation_seconds=value), self.assertRaises(ValueError):
                kernel_experiment.validate(invalid)
        config["budget"].update(max_candidates=2, searches_per_turn=3)
        with self.assertRaisesRegex(ValueError, "searches_per_turn must fit max_candidates"):
            kernel_experiment.validate(config)

    def test_v2_delivers_selected_task_snapshots_and_custom_policy_only(self):
        config = self._per_cell_config()
        custom = self.root / "rmsnorm-AGENTS.md"
        custom.write_text("# Task-specific instructions\nUse the RMSNorm-only tactic.\n")
        first, second = config["cells"]
        first["agents_md"] = str(custom)
        original_custom = custom.read_text()
        original_policy = kernel_experiment.POLICY.read_text()
        output = self._prepare_experiment(config)
        self.assertEqual((output / "AGENTS.md").read_text(), original_policy)
        self.assertFalse((output / "scaffold.md").exists())
        self.assertFalse((output / "references").exists())
        for cell in config["cells"]:
            self.assertIn(cell["id"], (output / "TASK.md").read_text())
            directory = output / "cells" / cell["id"]
            self.assertTrue((directory / "TASK.md").is_file())
            self.assertTrue((directory / "references").is_dir())
            self.assertTrue((directory / "scaffold.md").is_file())
        first_inputs = output / "cells" / first["id"]
        second_inputs = output / "cells" / second["id"]
        self.assertEqual((first_inputs / "AGENTS.md").read_text(), original_custom)
        self.assertEqual((second_inputs / "AGENTS.md").read_text(), original_policy)

        # The external originals are not runtime authorities after preparation.
        for cell in config["cells"]:
            Path(cell["references"][0]["path"]).write_text("changed after preparation\n")
        custom.write_text("changed after preparation\n")
        first_payload = self._launch_payload(output, first["id"])
        second_payload = self._launch_payload(output, second["id"])
        self.assertEqual(first_payload["cell"]["task"], "rmsnorm")
        self.assertEqual(second_payload["cell"]["task"], "silu")
        self.assertEqual(first_payload["scaffold"], (first_inputs / "scaffold.md").read_text())
        self.assertEqual(second_payload["scaffold"], (second_inputs / "scaffold.md").read_text())
        self.assertIn("reviewed reference", first_payload["scaffold"])
        self.assertNotIn("silu-only mechanism", first_payload["scaffold"])
        self.assertIn(original_custom, first_payload["scaffold"])
        self.assertNotIn(original_policy, first_payload["scaffold"])
        self.assertIn("silu-only mechanism", second_payload["scaffold"])
        self.assertNotIn("reviewed reference", second_payload["scaffold"])
        self.assertNotIn("RMSNorm-only tactic", second_payload["scaffold"])
        self.assertIn(original_policy, second_payload["scaffold"])
        for payload in (first_payload, second_payload):
            self.assertNotIn("changed after preparation", payload["scaffold"])

    def test_v2_shares_material_only_when_each_cell_lists_it(self):
        config = self._per_cell_config()
        common = self.root / "common.md"
        common.write_text("Shared FP32 numerical guidance.\n")
        declared = {"path": str(common), "source": "reviewed common guidance"}
        for cell in config["cells"]:
            cell["references"].append(deepcopy(declared))
        unlisted = deepcopy(config["cells"][1])
        unlisted["id"] = "silu-r2-without-common"
        unlisted["node"]["workspace"] = str(self.root / "silu-run-two")
        unlisted["references"] = unlisted["references"][:1]
        config["cells"].append(unlisted)
        output = self._prepare_experiment(config)
        for cell in config["cells"]:
            payload = self._launch_payload(output, cell["id"])
            if cell["id"] == unlisted["id"]:
                self.assertNotIn("Shared FP32 numerical guidance", payload["scaffold"])
            else:
                self.assertIn("Shared FP32 numerical guidance", payload["scaffold"])

    def test_v2_missing_cell_scaffold_never_falls_back_or_launches(self):
        config = self._per_cell_config()
        output = self._prepare_experiment(config)
        (output / "cells" / config["cells"][0]["id"] / "scaffold.md").unlink()
        (output / "scaffold.md").write_text("Root fallback must not be used.\n")
        with patch.object(kernel_experiment, "checkout_commit", return_value=COMMIT), \
             patch.object(kernel_experiment.subprocess, "run") as launch:
            with self.assertRaises((OSError, ValueError)):
                kernel_experiment.run_cell(output, config["cells"][0]["id"])
        launch.assert_not_called()
        self.assertFalse((output / "launches" / config["cells"][0]["id"]).exists())

    def test_reference_schema_versions_refuse_mixed_or_missing_authority(self):
        cases = []
        for field in ("references", "agents_md"):
            config = deepcopy(self.config)
            config["cells"][0][field] = (deepcopy(config["references"]) if field == "references"
                                        else str(self.root / "custom-AGENTS.md"))
            cases.append((f"v1-cell-{field}", config))
        config = self._per_cell_config()
        config["references"] = deepcopy(self.config["references"])
        cases.append(("v2-root-references", config))
        config = self._per_cell_config()
        del config["cells"][0]["references"]
        cases.append(("v2-missing-cell-references", config))
        for references in ([], None, "reference.py", [{}]):
            config = self._per_cell_config()
            config["cells"][0]["references"] = references
            cases.append((f"v2-invalid-references-{references!r}", config))
        for name, config in cases:
            with self.subTest(case=name), self.assertRaises(ValueError):
                kernel_experiment.validate(config)

    def test_v2_custom_policy_requires_an_absolute_regular_file(self):
        for path in ("", "relative.md", "/tmp/../AGENTS.md", None):
            config = self._per_cell_config()
            config["cells"][0]["agents_md"] = path
            with self.subTest(path=path), self.assertRaises(ValueError):
                kernel_experiment.validate(config)
        custom = self.root / "custom-AGENTS.md"
        custom.write_text("Reviewed task instructions.\n")
        link = self.root / "linked-AGENTS.md"
        link.symlink_to(custom)
        directory = self.root / "policy-directory"
        directory.mkdir()
        for index, path in enumerate((self.root / "missing-AGENTS.md", link, directory)):
            config = self._per_cell_config()
            config["cells"][0]["agents_md"] = str(path)
            with self.subTest(path=path), self.assertRaises((OSError, ValueError)):
                self._prepare_experiment(config, name=f"invalid-policy-{index}")

    def test_codex_home_binding_is_absolute_and_provider_specific(self):
        for home in ("relative", ""):
            config = deepcopy(self.config)
            config["cells"][0]["node"]["codex_home"] = home
            with self.assertRaises(ValueError):
                kernel_experiment.validate(config)
        config = deepcopy(self.config)
        config["cells"][0]["node"]["codex_home"] = "/tmp/node-codex"
        kernel_experiment.validate(config)
        config["provider"]["harness"] = "claude-code"
        with self.assertRaisesRegex(ValueError, "only valid for the Codex"):
            kernel_experiment.validate(config)

    def test_node_bootstrap_passes_explicit_native_provider(self):
        node = deepcopy(self.config["cells"][0]["node"])
        node["provider_executable"] = "/opt/codex/bin/codex"
        node["http_proxy"] = "http://127.0.0.1:17990"
        codex_home = self.root / "node-codex-home"
        codex_home.mkdir()
        node["codex_home"] = str(codex_home)
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
        self.assertEqual(execute.call_args.kwargs["env"]["CODEX_HOME"], str(codex_home))
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            self.assertEqual(execute.call_args.kwargs["env"][key], node["http_proxy"])

    def test_claude_response_aliases_survive_the_remote_bootstrap_as_separate_flags(self):
        config = deepcopy(self.config)
        config['provider'] = {'harness': 'claude-code', 'model': 'kimi-k3', 'effort': 'high',
                              'response_model_aliases': ['moonshotai/kimi-k3']}
        kernel_experiment.validate(config)
        payload = {'cell': config['cells'][0], 'source_commit': COMMIT, 'scaffold': 'rules',
                   'provider': config['provider'], 'budget': config['budget']}
        import io
        with patch('sys.stdin', io.StringIO(json.dumps(payload))), \
             patch('subprocess.run', return_value=SimpleNamespace(returncode=0)) as execute:
            with self.assertRaises(SystemExit) as result:
                exec(kernel_experiment._NODE, {})
        self.assertEqual(result.exception.code, 0)
        command = execute.call_args.args[0]
        self.assertEqual(command[command.index('--model')+1], 'kimi-k3')
        self.assertEqual(command[command.index('--response-model-alias')+1], 'moonshotai/kimi-k3')
        self.assertNotIn('--response-model-aliases', command)
        for aliases in ('moonshotai/kimi-k3', ['kimi-k3'], ['alias', 'alias']):
            config['provider']['response_model_aliases'] = aliases
            with self.assertRaises(ValueError): kernel_experiment.validate(config)
        config['provider']['response_model_aliases'] = ['moonshotai/kimi-k3']
        config['provider']['harness'] = 'codex'
        with self.assertRaisesRegex(ValueError, 'Claude provider'): kernel_experiment.validate(config)

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
            broker_socket=None, rows=2, columns=8, depth=None, provider_executable=None, provider_revision=None,
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
