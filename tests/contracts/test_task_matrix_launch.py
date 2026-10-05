"""The matrix driver reuses authority and delegates every task to launch_task."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
from threading import Barrier, Event, Lock
import unittest

from open_cake_ir.tasks import devices
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
        args.response_model_alias = ["vendor/m"]
        declared = matrix._command(args, "rmsnorm", self.root / "alias", None)
        self.assertEqual(declared[declared.index("--response-model-alias") + 1], "vendor/m")
        self.assertNotIn("--response-model-alias", ordinary)
        self.assertNotIn("--depth", ordinary)
        for command in (contraction, legacy):
            self.assertEqual(command[command.index("--depth") + 1], "17")

    def test_tinygemm_depth_matches_preflight_command_and_record(self):
        task = matrix.launch_task.TINYGEMM_TASK
        for requested, expected in ((3072, 3072), (None, 720)):
            with self.subTest(requested=requested), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory).resolve()
                args = self.args(task)
                args[args.index('--backend') + 1] = 'triton-b300'
                args[args.index('--workspace-root') + 1] = str(Path(directory)/'matrix')
                if requested is not None: args += ['--depth', str(requested)]
                with patch.object(matrix.launch_task, 'create_task', wraps=matrix.launch_task.create_task) as factory, \
                     patch.object(matrix.subprocess, 'run', return_value=subprocess.CompletedProcess([],1,b'',b'bounded CPU stop')) as run:
                    self.assertEqual(matrix.main(args), 1)
                self.assertEqual(factory.call_args.kwargs['depth'], requested)
                command = run.call_args.args[0]
                if requested is None: self.assertNotIn('--depth', command)
                else: self.assertEqual(command[command.index('--depth')+1], str(requested))
                record = json.loads((Path(directory)/'matrix/matrix.json').read_text())
                self.assertEqual(record['task_shapes'][task]['K'], expected)
        self.assertEqual(matrix._depth('gemm', None), 256)

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
        self.assertNotIn('--token-budget', calls[0])
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

    def test_parallel_tasks_overlap_after_one_preflight_and_preserve_every_result(self):
        tasks = ('rmsnorm', 'layernorm', 'softmax')
        barrier, lock = Barrier(3), Lock()
        preparations, calls = [], []
        def run(command, **kwargs):
            workspace = Path(command[command.index('--workspace') + 1])
            workspace.mkdir(parents=True)
            task = command[command.index('--task') + 1]
            if '--baseline-only' in command:
                (workspace / 'prepared-baseline.json').write_text('{}')
                with lock: preparations.append(task)
                return subprocess.CompletedProcess(command, 0, b'prepared', b'')
            self.assertEqual(set(preparations), set(tasks))
            if '--preflight-only' in command:
                (workspace / 'provider-qualification.json').write_text('{}')
                (workspace / 'provider-anchor.json').write_text('{}')
                return subprocess.CompletedProcess(command, 0, b'qualified', b'')
            self.assertIn('--qualification', command)
            self.assertEqual(command[command.index('--qualification') + 1],
                             str(self.root / 'provider-preflight/provider-qualification.json'))
            self.assertEqual(command[command.index('--turns') + 1], '20')
            self.assertEqual(command[command.index('--wall-seconds') + 1], '28800')
            self.assertNotIn('--token-budget', command)
            with lock: calls.append(task)
            barrier.wait(timeout=5)  # All three task processes must start before any returns.
            return subprocess.CompletedProcess(command, int(task == 'layernorm'),
                                               task.encode(), b'fault' if task == 'layernorm' else b'')
        with patch.object(matrix.subprocess, 'run', side_effect=run):
            self.assertEqual(matrix.main(self.args(*tasks) + [
                '--parallel-tasks', '3', '--local-queue-seconds', '1200', '--turns', '20']), 1)
        self.assertEqual(set(calls), set(tasks))
        rows = [json.loads(line) for line in (self.root / 'task-results.jsonl').read_text().splitlines()]
        self.assertEqual({r['task']: r['position'] for r in rows},
                         dict(zip(tasks, (1, 2, 3))))
        self.assertTrue(all(r['qualification_reused'] for r in rows))
        self.assertEqual((self.root / 'layernorm.stdout').read_bytes(), b'layernorm')
        self.assertEqual(json.loads((self.root / 'terminal.json').read_text())['task_fault_count'], 1)

    def test_parallel_host_slots_are_bounded_and_refilled_after_completion(self):
        entered, release = Barrier(3), Event()
        lock = Lock()
        active = maximum = 0
        def job(value):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            if value < 3: entered.wait(timeout=5)
            with lock: active -= 1
            return value
        results = list(matrix._bounded_map(range(9), job, 3, release))
        self.assertEqual(set(results), set(range(9)))
        self.assertEqual(maximum, 3)

    def test_parallel_preflight_failure_starts_no_optimization_run(self):
        optimized = []
        def run(command, **kwargs):
            workspace = Path(command[command.index('--workspace') + 1])
            workspace.mkdir(parents=True)
            if '--baseline-only' in command:
                (workspace / 'prepared-baseline.json').write_text('{}')
                return subprocess.CompletedProcess(command, 0, b'', b'')
            if '--preflight-only' in command:
                return subprocess.CompletedProcess(command, 1, b'', b'qualification refused')
            optimized.append(command)
            raise AssertionError('optimization before qualification')
        with patch.object(matrix.subprocess, 'run', side_effect=run):
            self.assertEqual(matrix.main(self.args('rmsnorm', 'layernorm') + [
                '--parallel-tasks', '2', '--local-queue-seconds', '1200']), 1)
        self.assertEqual(optimized, [])
        terminal = json.loads((self.root / 'terminal.json').read_text())
        self.assertEqual(terminal['task_count_attempted'], 0)
        self.assertEqual(terminal['status'], 'stopped_before_provider_qualification')

    def test_parallel_local_tasks_refuse_zero_queue_before_creating_inputs(self):
        with self.assertRaises(SystemExit), patch.object(matrix.subprocess, 'run') as run:
            matrix.main(self.args('rmsnorm') + ['--parallel-tasks', '2'])
        run.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_supplied_qualification_is_reused_without_another_provider_preflight(self):
        receipt, anchor = self.root.parent / 'receipt.json', self.root.parent / 'anchor.json'
        receipt.write_text('{}'); anchor.write_text('{}')
        calls = []
        def run(command, **kwargs):
            workspace = Path(command[command.index('--workspace') + 1])
            workspace.mkdir(parents=True)
            if '--baseline-only' in command:
                (workspace / 'prepared-baseline.json').write_text('{}')
            else:
                self.assertNotIn('--preflight-only', command)
                self.assertEqual(command[command.index('--qualification') + 1], str(receipt))
                calls.append(command)
            return subprocess.CompletedProcess(command, 0, b'', b'')
        with patch.object(matrix.subprocess, 'run', side_effect=run):
            self.assertEqual(matrix.main(self.args('rmsnorm', 'layernorm') + [
                '--parallel-tasks', '2', '--local-queue-seconds', '1200',
                '--qualification', str(receipt), '--qualification-anchor', str(anchor)]), 0)
        self.assertEqual(len(calls), 2)

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

    def test_gfx1151_gelu_pair_now_reaches_baseline_preparation(self):
        # The last unsupported pair in the default matrix is now admitted. Exercise
        # its real preparation path; do not invent another refused pair for this test.
        self.assertEqual(devices.tanh_contract('triton-gfx1151'), 'ocml.tanh.f32')
        args = self.args('gelu_tanh', 'gelu_tanh_backward')
        args[args.index('--backend')+1] = 'triton-gfx1151'
        with patch.object(matrix.subprocess, 'run', side_effect=RuntimeError('baseline boundary')), \
                self.assertRaisesRegex(RuntimeError, 'baseline boundary'):
            matrix.main(args)

    def test_the_pair_this_case_used_to_rest_on_is_supported_now(self):
        """gelu_tanh on triton-dcu is admitted, so it cannot carry the refusal case.

        Stated as its own assertion so the change that admitted it is visible here,
        rather than the previous case silently being re-aimed at a different target.
        """
        self.assertEqual(
            devices.tanh_contract('triton-dcu'), 'ocml.tanh.f32')


if __name__ == "__main__":
    unittest.main()
