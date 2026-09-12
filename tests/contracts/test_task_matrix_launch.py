"""The matrix driver reuses authority and delegates every task to launch_task."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools import launch_task_matrix as matrix


class TaskMatrixLaunchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "matrix"

    def args(self, *tasks):
        result = ["--backend", "metal-m4", "--model", "exact-model",
                  "--harness", "claude-code", "--effort", "high",
                  "--workspace-root", str(self.root)]
        for task in tasks:
            result.extend(("--task", task))
        return result

    def test_registered_order_is_unique_and_depth_is_owned_by_depth_tasks(self):
        self.assertEqual(len(matrix.ALL_TASKS), len(set(matrix.ALL_TASKS)))
        args = type("Args", (), dict(backend="metal-m4", harness="claude-code",
            model="m", effort="high", turns=1, token_budget=10, max_candidates=1,
            searches_per_turn=1, dispatches_per_sample=64, wall_seconds=10,
            rows=2, columns=8, depth=17, provider_executable=None,
            provider_revision=None, incumbent_registry=None, gpu_run=None, broker_socket=None,
            maximum_cv=None, required_pair_wins=None))()
        ordinary = matrix._command(args, "rmsnorm", self.root / "a", None)
        contraction = matrix._command(args, "gemm", self.root / "b", None)
        legacy = matrix._command(args, "gemm_bias", self.root / "c", None)
        self.assertNotIn("--depth", ordinary)
        for command in (contraction, legacy):
            self.assertEqual(command[command.index("--depth") + 1], "17")

    def test_registry_is_one_matrix_input_but_each_task_resolves_its_own_cell(self):
        registry = self.root.parent / "incumbents"
        args = type("Args", (), dict(backend="metal-m4", harness="claude-code",
            model="m", effort="high", turns=1, token_budget=10, max_candidates=1,
            searches_per_turn=1, dispatches_per_sample=64, wall_seconds=10,
            rows=2, columns=8, depth=17, provider_executable=None,
            provider_revision=None, incumbent_registry=registry, gpu_run=None, broker_socket=None, maximum_cv=None, required_pair_wins=None))()
        command = matrix._command(args, "rmsnorm", self.root / "a", None)
        self.assertEqual(
            command[command.index("--incumbent-registry") + 1], str(registry)
        )
        prepared = self.root / "prepared-baseline.json"
        execution = matrix._command(args, "rmsnorm", self.root / "a", None, prepared)
        self.assertNotIn("--incumbent-registry", execution)
        self.assertNotIn("--fixed-baseline-bundle", execution)
        self.assertEqual(execution[execution.index("--prepared-baseline") + 1], str(prepared))

    def test_first_qualification_is_reused_and_task_faults_do_not_stop_the_matrix(self):
        calls = []
        preparations = []
        def run(command, **kwargs):
            workspace = Path(command[command.index("--workspace") + 1])
            workspace.mkdir(parents=True)
            if '--baseline-only' in command:
                preparations.append(command)
                bundle = workspace / 'baseline.json'
                bundle.write_text('{}')
                (workspace / 'prepared-baseline.json').write_text('{}')
                return subprocess.CompletedProcess(command, 0, (str(bundle)+'\n').encode(), b'')
            self.assertEqual(len(preparations), 2)
            calls.append(command)
            self.assertIn('--prepared-baseline', command)
            task = command[command.index('--task') + 1]
            self.assertEqual(command[command.index('--prepared-baseline') + 1],
                             str(self.root / 'baseline-preflight' / task / 'prepared-baseline.json'))
            if len(calls) == 1:
                (workspace / "provider-qualification.json").write_text("{}")
                (workspace / "provider-anchor.json").write_text("{}")
            return subprocess.CompletedProcess(command, 0 if len(calls) == 1 else 1,
                                               f"task-{len(calls)}".encode(), b"fault")
        with patch.object(matrix.subprocess, "run", side_effect=run):
            self.assertEqual(matrix.main(self.args("rmsnorm", "layernorm")), 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][calls[0].index('--token-budget')+1], '3000000')
        self.assertEqual(calls[0][calls[0].index('--turns')+1], '32')
        self.assertEqual(calls[0][calls[0].index('--wall-seconds')+1], '28800')
        self.assertNotIn("--qualification", calls[0])
        self.assertIn("--qualification", calls[1])
        rows = [json.loads(line) for line in (self.root / "task-results.jsonl").read_text().splitlines()]
        self.assertEqual([row["task"] for row in rows], ["rmsnorm", "layernorm"])
        self.assertEqual([row["qualification_reused"] for row in rows], [False, True])
        terminal = json.loads((self.root / "terminal.json").read_text())
        self.assertEqual(terminal["status"], "completed_with_task_faults")
        self.assertEqual(terminal["task_count_attempted"], 2)

    def test_baseline_failure_stops_before_any_provider_or_campaign(self):
        completed = subprocess.CompletedProcess([], 1, b"", b"common setup fault")
        with patch.object(matrix.subprocess, "run", return_value=completed) as run:
            self.assertEqual(matrix.main(self.args("rmsnorm", "layernorm")), 1)
        run.assert_called_once()
        self.assertFalse((self.root / "layernorm").exists())
        terminal = json.loads((self.root / "terminal.json").read_text())
        self.assertEqual(terminal["status"], "stopped_before_baseline_preflight")
        self.assertEqual(terminal["task_count_attempted"], 0)
        self.assertEqual(terminal["provider_calls"], 0)

    def test_qualification_failure_after_all_baselines_stops_the_matrix(self):
        def run(command, **kwargs):
            if '--baseline-only' in command:
                workspace = Path(command[command.index('--workspace')+1])
                workspace.mkdir(parents=True)
                bundle = workspace/'baseline.json'; bundle.write_text('{}')
                (workspace / 'prepared-baseline.json').write_text('{}')
                return subprocess.CompletedProcess(command, 0, (str(bundle)+'\n').encode(), b'')
            return subprocess.CompletedProcess(command, 1, b'', b'qualification fault')
        with patch.object(matrix.subprocess, 'run', side_effect=run) as mocked:
            self.assertEqual(matrix.main(self.args('rmsnorm','layernorm')), 1)
        self.assertEqual(mocked.call_count, 3)
        terminal=json.loads((self.root/'terminal.json').read_text())
        self.assertEqual(terminal['status'], 'stopped_before_provider_qualification')
        self.assertEqual(terminal['task_count_attempted'], 1)

    def test_duplicate_task_selection_refuses_before_creating_the_root(self):
        with self.assertRaises(SystemExit), patch.object(matrix.subprocess, "run") as run:
            matrix.main(self.args("rmsnorm", "rmsnorm"))
        run.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_explicit_cuda_broker_options_are_forwarded_without_metal_defaults(self):
        args = self.args('silu')
        args[args.index('--backend')+1] = 'triton-b300'
        args += ['--gpu-run','/unit-test/gpu-run','--broker-socket','/unit-test/gpu.sock']
        with patch.object(matrix.subprocess,'run',return_value=subprocess.CompletedProcess([],1,b'',b'')) as run:
            matrix.main(args)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index('--gpu-run')+1], '/unit-test/gpu-run')
        self.assertEqual(command[command.index('--broker-socket')+1], '/unit-test/gpu.sock')
        self.assertNotIn('--dispatches-per-sample', command)

    def test_unsupported_requested_factory_fails_before_workspace_or_provider(self):
        args = self.args('silu','rmsnorm')
        args[args.index('--backend')+1] = 'triton-b300'
        with self.assertRaises(SystemExit),patch.object(matrix.subprocess,'run') as run:
            matrix.main(args)
        run.assert_not_called()
        self.assertFalse(self.root.exists())


if __name__ == "__main__":
    unittest.main()
