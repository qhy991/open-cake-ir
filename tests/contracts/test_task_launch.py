"""Thin entry/composition wiring with CPU doubles; no live provider or Metal calls."""
from __future__ import annotations

import contextlib
import io
import json
import shutil
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from open_cake_ir.compiler.corpus import CorpusGateReport
from open_cake_ir.compiler.target import Target
from open_cake_ir.evaluation.paired import PAIRED_KIND, PAIRED_METAL_BATCHED_KIND, paired_protocol
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.lab.contracts import StudyContract, StudyReport
from open_cake_ir.evidence import RunAudit
from open_cake_ir.tasks.normalization.study import study_template
from open_cake_ir.tasks.workloads import create_task
from tools import launch_task

ROOT = Path(__file__).resolve().parents[2]


class TaskLaunchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.workspace = self.directory / "new-task"

    def args(self):
        return ["--task", "rmsnorm", "--backend", "metal-m1-pro", "--model", "exact-unit-test-model",
                "--harness", "claude-code", "--effort", "high", "--workspace", str(self.workspace),
                "--rows", "2", "--columns", "7", "--turns", "2", "--token-budget", "12000"]

    def test_all_user_treatment_fields_are_required_before_any_preparation(self):
        for flag in ("--task", "--backend", "--model", "--harness", "--effort", "--workspace"):
            args = self.args()
            index = args.index(flag)
            del args[index:index+2]
            with self.subTest(flag=flag), patch.object(launch_task, "_admit_stack") as admit:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    launch_task.main(args)
                admit.assert_not_called()
                self.assertFalse(self.workspace.exists())

    def test_codex_npm_wrapper_resolves_only_its_own_native_dependency(self):
        package = self.directory/'codex-package'
        (package/'bin').mkdir(parents=True)
        shim = package/'bin/codex.js'
        shim.write_text('// Node entrypoint fixture, never executed')
        (package/'package.json').write_text(json.dumps({'name':'@openai/codex','bin':{'codex':'bin/codex.js'}}))
        dependency = package/'node_modules/@openai/codex-darwin-arm64'
        native = dependency/'vendor/aarch64-apple-darwin/bin/codex'
        native.parent.mkdir(parents=True)
        native.write_bytes(b'CPU native fixture, never executed')
        native.chmod(0o700)
        (dependency/'package.json').write_text('{}')
        with patch.object(launch_task.shutil,'which',side_effect=lambda name: '/unit-test/node' if name=='node' else str(shim)), \
             patch.object(launch_task.platform,'system',return_value='Darwin'), \
             patch.object(launch_task.platform,'machine',return_value='arm64'), \
             patch.object(launch_task.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=str(dependency/'package.json'))) as node:
            self.assertEqual(launch_task._provider_executable('codex',None),native)
        args = node.call_args.args[0]
        self.assertEqual(args[:2],['/unit-test/node','-e'])
        self.assertEqual(args[-2:],[str(shim),'@openai/codex-darwin-arm64'])
        with patch.object(launch_task.shutil,'which',return_value=str(native)):
            self.assertEqual(launch_task._provider_executable('codex',native),native)

    def test_codex_missing_dependency_uses_own_vendor_but_other_resolution_errors_refuse(self):
        package = self.directory/'codex-package'
        shim = package/'bin/codex.js'
        shim.parent.mkdir(parents=True)
        shim.write_text('// CPU shim fixture')
        (package/'package.json').write_text(json.dumps({'name':'@openai/codex','bin':{'codex':'bin/codex.js'}}))
        native = package/'vendor/aarch64-apple-darwin/bin/codex'
        native.parent.mkdir(parents=True)
        native.write_bytes(b'CPU binary fixture')
        native.chmod(0o700)
        for status in (44,1):
            with patch.object(launch_task.shutil,'which',side_effect=lambda name: '/unit-test/node' if name=='node' else str(shim)), \
                 patch.object(launch_task.platform,'system',return_value='Darwin'), \
                 patch.object(launch_task.platform,'machine',return_value='arm64'), \
                 patch.object(launch_task.subprocess,'run',return_value=SimpleNamespace(returncode=status,stdout='')):
                if status == 44:
                    self.assertEqual(launch_task._provider_executable('codex',None),native)
                else:
                    with self.assertRaisesRegex(ValueError,'no alternative installation'):
                        launch_task._provider_executable('codex',None)

    def test_contraction_tasks_default_to_their_own_admitting_shape(self):
        """F-2026-09-10-014: the elementwise tile this launcher uses by default is refused
        by the Metal verifier for every contraction task (the starter materializes the
        full second operand, over the lane-owned storage bound). Absent flags resolve to
        the contraction contract's own extents; explicit flags still win, and every other
        family keeps the elementwise tile.
        """
        self.assertEqual(launch_task._default_shape("gemm", None, None), (1024, 64))
        self.assertEqual(launch_task._default_shape("attention_decode", None, None), (1024, 64))
        self.assertEqual(launch_task._default_shape("gemm_silu", 128, 32), (128, 32))
        self.assertEqual(launch_task._default_shape("pairwise_sqdist", 128, None), (128, 64))
        self.assertEqual(launch_task._default_shape("layernorm", None, None), (128, 1024))
        self.assertEqual(launch_task._default_shape("adadelta", None, 512), (128, 512))

    def test_stable_study_has_real_task_inputs_and_declared_metal_assay(self):
        document, source = create_task("layernorm", rows=2, columns=7)
        workload_path, starter = self.directory/"workload.json", self.directory/"starter.py"
        workload_path.write_text(json.dumps(document))
        starter.write_text(source)
        workload = WorkloadContract(document)
        for harness in ("codex", "claude-code"):
            with self.subTest(harness=harness):
                study = study_template(ROOT, workload, workload_path, starter, harness=harness,
                                       model="exact-model", effort="high", turns=2, token_budget=12000)
                path = self.directory/f"{harness}-study.json"
                path.write_text(json.dumps(study))
                loaded = StudyContract.load(path)
                self.assertEqual(loaded.state, "template")
                self.assertEqual(set(study["arms"]), {"open_cake"})
                self.assertEqual(study["allocation"]["order"], ["open_cake-1"])
                self.assertEqual(study["workload"]["path"], str(workload_path))
                self.assertEqual(study["arms"]["open_cake"]["schedule_skeleton"]["path"], str(starter))
                self.assertEqual(study["execution"]["executor_revision"], {"binding":"current_release"})
                self.assertEqual(study["execution"]["fixed_baseline"], {"binding":"campaign_lock"})
                self.assertEqual(study["arms"]["open_cake"]["provider"]["qualification"], {"binding":"campaign_lock"})
                policy = study["evaluation_protocol"]
                assay = paired_protocol(policy)
                self.assertEqual(policy["paired_timing"]["kind"], PAIRED_METAL_BATCHED_KIND)
                # Amortization is declared, so a sample is one dispatch of a batched buffer.
                self.assertEqual(policy["paired_timing"]["dispatches_per_sample"], 64)
                self.assertEqual(assay.dispatches_per_sample, 64)
                self.assertEqual(policy["validation_case_ids"], list(workload.case_ids))
                self.assertEqual((len(assay.pair_order), assay.samples_per_cohort, assay.route_calls_per_cohort), (10,25,28))
                self.assertEqual((assay.maximum_cv, assay.materiality_ratio, assay.required_pair_wins), (0.05,1.05,6))
                self.assertEqual(study["execution"]["gpu"]["mode"], "local_serialized")
                self.assertIsNone(study["analysis_plan"]["estimand"])

    def test_each_supported_hardware_uses_its_own_existing_assay_and_mode(self):
        for backend, device in launch_task.DEVICE_BACKENDS.items():
            with self.subTest(backend=backend):
                document, source = create_task("silu", backend=backend, rows=2, columns=8)
                workload = WorkloadContract(document)
                path, starter = self.directory/f"{backend}.json", self.directory/f"{backend}.py"
                path.write_text(json.dumps(document)); starter.write_text(source)
                study = study_template(ROOT, workload, path, starter, harness="claude-code",
                                       model="exact-model", effort="high", turns=2, token_budget=12000)
                policy = study["evaluation_protocol"]
                assay = paired_protocol(policy)
                metal = device["route"] == "metal"
                self.assertEqual(study["execution"]["target"], device["target"])
                self.assertEqual(policy["validation_case_ids"], list(workload.case_ids))
                if device["timing_source"] is None:
                    # A backend with no named timing source states the coverage
                    # limitation instead of inheriting another target's timer. The
                    # template used to read `metal ? metal : cupti`, so the DCU study
                    # said `paired_cupti` on a machine with no CUPTI installed.
                    self.assertIsNone(assay)
                    self.assertNotIn("paired_timing", policy)
                    self.assertEqual(policy["measurement_coverage"]["timed_assay"],
                                     "unavailable")
                    self.assertIn(device["target"],
                                  policy["measurement_coverage"]["reason"])
                    for stage in ("search_evaluation", "confirmatory_evaluation"):
                        self.assertNotIn("paired", policy[stage])
                    continue
                # Two axes, read separately: the route picks the paired assay and its
                # cohort shape, the allocation picks how the device is reached.
                local = device["allocation"] == "local_broker"
                self.assertEqual(study["execution"]["gpu"]["mode"],
                                 "local_serialized" if local else "exclusive")
                self.assertEqual(policy["paired_timing"]["kind"], PAIRED_METAL_BATCHED_KIND if metal else PAIRED_KIND)
                self.assertEqual(assay.route_calls_per_cohort, 28 if metal else 42)
                self.assertEqual((len(assay.pair_order), assay.samples_per_cohort,
                                  assay.maximum_cv, assay.materiality_ratio, assay.required_pair_wins),
                                 (10, 25, 0.05, 1.05, 6))
                if not metal:
                    frozen = json.loads((ROOT/'contracts/studies/matched-search-triton-b300-optimization-template.json').read_text())
                    self.assertEqual(policy['paired_timing'], frozen['evaluation_protocol']['paired_timing'])
                    self.assertNotIn('dispatches_per_sample', policy['paired_timing'])
                    self.assertNotIn('maximum_relative_iqr', policy['paired_timing'])
                    with self.assertRaisesRegex(ValueError, 'Metal command-buffer'):
                        study_template(ROOT, workload, path, starter, harness="claude-code",
                                       model="exact-model", effort="high", dispatches_per_sample=64)

    def test_a_triton_route_can_be_reached_by_a_local_broker(self):
        """The DCU case: Triton's toolchain, an Apple-shaped allocator, no gpu-run.

        Reading the allocator off the route refused this combination outright, with
        "Triton execution requires the existing gpu-run allocator" -- a CUDA cluster
        allocator a Hygon DCU has no use for.
        """
        executor = SimpleNamespace(document={"host_environment": {
            "python": {"invocation_path": "/unit-test/python"}, "packages": {"triton": "3.6.0"},
            "tools": {"build_tools": [{"kind": "bwrap", "path": "/usr/bin/true"}]}}})
        with patch.object(launch_task, '_triton_runtime_roots', return_value=['/unit-test', '/usr']), \
             patch.object(launch_task.shutil, 'which', return_value=None):
            runtime = launch_task._runtime_config(
                self.workspace, executor, Path('/unit-test/provider'), 'triton',
                allocation='local_broker')
        command = runtime['broker']['command']
        self.assertIn('open_cake_ir.evaluation.local_broker', command)
        self.assertNotIn('gpu-run', ' '.join(command))
        # Its own lock and its own job prefix; a DCU run is not recorded as a Metal one.
        self.assertEqual(command[command.index('--kind') + 1], 'hip')
        # The toolchain is still Triton's, because the route did not change.
        self.assertEqual(runtime['toolchain']['triton_version'], '3.6.0')
        self.assertNotIn('output_root', runtime['toolchain'])
        with self.assertRaisesRegex(ValueError, 'gpu_run allocation'):
            launch_task._runtime_config(
                self.workspace, executor, Path('/unit-test/provider'), 'triton',
                allocation='local_broker', gpu_run=Path('/unit-test/gpu-run'))

    def test_the_jail_is_found_on_this_host_and_refused_by_name_when_absent(self):
        """`/usr/bin/bwrap` was a constant; it is a fact about a host, not about bwrap.

        The DCU container installs its own userspace, and a path written against another
        distribution surfaces as a FileNotFoundError from inside the isolated compiler,
        naming a path no one on this host chose.
        """
        jail = self.directory / 'bwrap'
        jail.write_bytes(b'#!/bin/sh\n')
        declared = {"tools": {"build_tools": [
            {"kind": "hipcc", "path": "/opt/dtk/bin/hipcc"}, {"kind": "bwrap", "path": str(jail)}]}}
        with patch.object(launch_task.shutil, 'which', return_value=None):
            # A host that pins it is believed over the conventional location.
            self.assertEqual(launch_task._bubblewrap(declared), str(jail))
            # Released HIP and CUDA descriptors pin no jail: HIP_BUILD_TOOLS is closed and
            # has no bwrap, so every captured host is silent here and PATH decides.
            with self.assertRaisesRegex(ValueError, 'bubblewrap'):
                launch_task._bubblewrap({"tools": {"build_tools": []}})
        with patch.object(launch_task.shutil, 'which', return_value=str(jail)):
            self.assertEqual(launch_task._bubblewrap({}), str(jail))

    def test_a_hip_host_declares_the_environment_its_jail_would_otherwise_lose(self):
        """The jail runs --clearenv, and two DTK facts live only in /opt/dtk/env.sh.

        Measured on the DCU, one after the other: without LD_LIBRARY_PATH the jailed build
        failed with "libgalaxyhip.so.5: cannot open shared object file" while /opt was
        mounted and the file was sitting in it, because ldconfig does not know /opt/dtk;
        with it, the build reached the AMDGCN compile and clang-18 reported "cannot find
        ROCm device library", because ROCM_PATH was gone too. Same kind of fact twice, so
        one declaration rather than a field each. A CUDA host declares none -- torch finds
        its libraries through RPATH.
        """
        environment = {"LD_LIBRARY_PATH": "/opt/dtk/lib:/opt/hyhal/lib",
                       "ROCM_PATH": "/opt/dtk"}
        hip = SimpleNamespace(document={"host_environment": {
            "python": {"invocation_path": "/unit-test/python"}, "packages": {"triton": "3.6.0"},
            "tools": {"build_tools": [{"kind": "bwrap", "path": "/usr/bin/true"}]},
            "runtime": {"backend": "hip", "build_environment": environment}}})
        cuda = SimpleNamespace(document={"host_environment": {
            "python": {"invocation_path": "/unit-test/python"}, "packages": {"triton": "3.6.0"},
            "tools": {"build_tools": [{"kind": "bwrap", "path": "/usr/bin/true"}]}}})
        with patch.object(launch_task, "_triton_runtime_roots", return_value=["/opt", "/usr"]):
            self.assertEqual(launch_task._triton_toolchain_config(hip)["build_environment"],
                             environment)
            self.assertEqual(launch_task._triton_toolchain_config(cuda)["build_environment"], {})
        # Every absolute path the declaration names is mounted, so the jail is never
        # pointed at something it cannot see.
        roots = [Path(root) for root in
                 launch_task._triton_runtime_roots(Path("/usr/local/bin/python"),
                                                   ("/opt/dtk/lib", "/opt/hyhal/lib", "/opt/dtk"))]
        for declared in (Path("/opt/dtk/lib"), Path("/opt/hyhal/lib"), Path("/opt/dtk")):
            self.assertTrue(any(declared == root or declared.is_relative_to(root)
                                for root in roots), (declared, roots))

    def test_cuda_runtime_uses_existing_allocator_and_same_isolated_toolchain(self):
        executor = SimpleNamespace(document={"host_environment": {
            "python": {"invocation_path": "/unit-test/python"}, "packages": {"triton": "3.6.0"}}})
        with patch.object(launch_task, '_triton_runtime_roots', return_value=['/unit-test', '/usr']), \
             patch.object(launch_task.shutil, 'which', return_value='/usr/bin/true'):
            runtime = launch_task._runtime_config(self.workspace, executor, Path('/unit-test/provider'), 'triton',
                allocation='gpu_run',
                gpu_run=Path('/unit-test/gpu-run'), broker_socket=Path('/unit-test/broker.sock'))
            with patch.object(launch_task, 'IsolatedTritonCompiler') as compiler, \
                 patch.object(launch_task, 'TritonToolchainBuilder'):
                launch_task._triton_builder(executor, Mock())
            self.assertEqual(compiler.call_args.kwargs, runtime['toolchain'])
        path = self.directory/'runtime.json'; path.write_text(json.dumps(runtime))
        from open_cake_ir.lab.runtime_config import load_runtime_config
        self.assertEqual(load_runtime_config(path, toolchain_kind='triton')['toolchain'], runtime['toolchain'])
        command = runtime['broker']['command']
        for flag, value in (('--mode','exclusive'),('--gpu-count','1'),('--estimate','unknown'),
                            ('--socket','/unit-test/broker.sock'),('--queue-timeout','1800s'),('--run-timeout','1800s')):
            self.assertEqual(command[command.index(flag)+1], value)
        self.assertEqual(command[command.index('--')+1:], ['/unit-test/python', '-I',
            str(ROOT/'src/open_cake_ir/evaluation/source_bootstrap.py'), 'open_cake_ir.tasks.evaluate'])
        self.assertNotIn('open_cake_ir.evaluation.local_broker', command)
        self.assertNotIn('--env', command)
        self.assertGreater(runtime['broker']['timeout_seconds'], 3600)
        with patch.object(launch_task.shutil, 'which', return_value=None), \
                self.assertRaisesRegex(ValueError, 'gpu-run'):
            launch_task._runtime_config(self.workspace, executor, Path('/unit-test/provider'), 'triton',
                                        allocation='gpu_run')
        with self.assertRaisesRegex(ValueError, 'gpu_run allocation'):
            launch_task._runtime_config(self.workspace, executor, Path('/unit-test/provider'), 'metal',
                                        allocation='local_broker',
                                        broker_socket=Path('/unit-test/broker.sock'))
        with self.assertRaisesRegex(ValueError, 'allocation'):
            launch_task._runtime_config(self.workspace, executor, Path('/unit-test/provider'), 'triton',
                                        allocation='slurm')

    def test_a_broken_allocator_is_found_before_a_token_is_spent(self):
        """Measured at the cost of a full authoring turn, so it is checked by using it.

        The launcher's gpu-run default socket was not the socket the B300 host's broker
        listens on. The campaign authored a candidate, sealed it, reached the allocator
        and faulted with "cannot reach broker" -- 77103 provider tokens for a run that
        could never be evaluated (F-2026-09-16-003). The allocator is the one participant
        a launch cannot check by reading a file.
        """
        false = shutil.which("false") or "/usr/bin/false"
        runtime = {"broker": {"cwd": str(self.directory),
                              "command": [false, "--label", "x", "--", "evaluator"]}}
        with self.assertRaisesRegex(ValueError, "refused a trivial lease"):
            launch_task._admit_allocator(runtime)
        # The probe replaces the evaluator with /bin/true and keeps everything the
        # campaign will actually pass the allocator.
        true = shutil.which("true") or "/usr/bin/true"
        recorded = {}

        def record(probe, **kwargs):
            recorded["probe"] = probe
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch.object(launch_task.subprocess, "run", record):
            launch_task._admit_allocator(
                {"broker": {"cwd": str(self.directory),
                            "command": [true, "--mode", "exclusive", "--", "evaluator",
                                        "--request", "r"]}})
        self.assertEqual(recorded["probe"],
                         [true, "--mode", "exclusive", "--", "/bin/true"])
        # A local broker takes no lease, so there is nothing to probe and nothing is run.
        with patch.object(launch_task.subprocess, "run") as never:
            launch_task._admit_allocator(
                {"broker": {"cwd": str(self.directory), "command": ["python", "-m", "b"]}})
        never.assert_not_called()

    def test_failed_full_gate_prevents_executor_and_provider_work(self):
        self.workspace.mkdir()
        gate = CorpusGateReport("unit-fixture", "fixture", "not-live", False, ())
        compiler = SimpleNamespace(commit="0" * 40, check_corpus=lambda: gate)
        with patch.object(launch_task.Compiler, "load", return_value=compiler), \
             patch.object(launch_task, "resolve_executor") as executor:
            with self.assertRaisesRegex(ValueError, "full Corpus Gate"):
                launch_task._admit_stack(ROOT, self.workspace, "apple_gpu_family7")
            executor.assert_not_called()
        self.assertFalse(json.loads((self.workspace/"compiler-gate.json").read_text())["passed"])

    def test_nonmetal_executor_is_refused_before_archive_helper_admission(self):
        self.workspace.mkdir()
        gate = CorpusGateReport("unit-fixture", "fixture", "not-live", True, ())
        compiler = SimpleNamespace(commit="0" * 40, check_corpus=lambda: gate)
        executor = SimpleNamespace(document={"host_environment":{"kind":"cuda"}})
        with patch.object(launch_task.Compiler,"load",return_value=compiler), \
             patch.object(launch_task,"resolve_executor",return_value=executor), \
             patch.object(launch_task.MetalArchiveHost,"from_executor") as host:
            with self.assertRaisesRegex(ValueError,"Metal host capture"):
                launch_task._admit_stack(ROOT,self.workspace,"apple_gpu_family7")
            host.assert_not_called()

    def test_a_gpu_route_refuses_the_metal_host_and_never_builds_a_metal_archive(self):
        """The mirror of the check above, now that a route selects the stack.

        A CUDA host carries no `kind` at all, so the two routes are separated by whether
        the host names itself Metal rather than by a marker each one owns. Both refusals
        land before any host helper is touched.
        """
        gate = CorpusGateReport("unit-fixture", "fixture", "not-live", True, ())
        compiler = SimpleNamespace(commit="0" * 40, check_corpus=lambda: gate)
        metal = SimpleNamespace(document={"host_environment": {"kind": "metal"}})
        # Each admission writes its own gate report, so give each one a fresh workspace.
        first, second, third = (self.workspace.parent / name for name in ("a", "b", "c"))
        for directory in (first, second, third):
            directory.mkdir(parents=True)
        with patch.object(launch_task.Compiler, "load", return_value=compiler), \
             patch.object(launch_task, "resolve_executor", return_value=metal), \
             patch.object(launch_task.MetalArchiveHost, "from_executor") as host:
            with self.assertRaisesRegex(ValueError, "requires a GPU host capture"):
                launch_task._admit_stack(ROOT, first, "sm_103a", "triton")
            host.assert_not_called()
        # And a host with no `kind` is what the CUDA capture actually writes, so the GPU
        # route must accept it while the Metal route must not.
        cuda = SimpleNamespace(document={"host_environment": {"packages": {}}})
        with patch.object(launch_task.Compiler, "load", return_value=compiler), \
             patch.object(launch_task, "resolve_executor", return_value=cuda), \
             patch.object(launch_task.MetalArchiveHost, "from_executor") as host:
            with self.assertRaisesRegex(ValueError, "Metal host capture"):
                launch_task._admit_stack(ROOT, second, "sm_103a", "metal")
            host.assert_not_called()
            compiler_, executor_, resolved_host, _ = launch_task._admit_stack(
                ROOT, third, "sm_103a", "triton")
            self.assertIsNone(resolved_host)
            host.assert_not_called()

    def test_stale_executor_refusal_names_the_required_host_capture(self):
        self.workspace.mkdir()
        gate = CorpusGateReport("unit-fixture", "fixture", "not-live", True, ())
        compiler = SimpleNamespace(commit="0" * 40, check_corpus=lambda: gate)
        with patch.object(launch_task.Compiler, "load", return_value=compiler), \
             patch.object(launch_task, "resolve_executor", side_effect=ValueError("source differs")), \
             patch.object(launch_task.MetalArchiveHost, "from_executor") as host:
            # The refusal names the target whose capture is stale, not one vendor: the
            # same boundary refuses a CUDA or HIP target and used to report it as a
            # missing Metal Executor.
            with self.assertRaisesRegex(
                    ValueError,
                    "committed host capture for 'apple_gpu_family7'; source differs"):
                launch_task._admit_stack(ROOT, self.workspace, "apple_gpu_family7")
            host.assert_not_called()
        second = self.directory / "second-task"
        second.mkdir()
        with patch.object(launch_task.Compiler, "load", return_value=compiler), \
             patch.object(launch_task, "resolve_executor", side_effect=ValueError("source differs")):
            with self.assertRaisesRegex(ValueError, "committed host capture for 'gfx938'"):
                launch_task._admit_stack(ROOT, second, "gfx938", "triton")

    def test_executor_for_another_apple_gpu_is_refused_before_archive_helper_admission(self):
        self.workspace.mkdir()
        gate = CorpusGateReport("unit-fixture", "fixture", "not-live", True, ())
        compiler = SimpleNamespace(commit="0" * 40, check_corpus=lambda: gate)
        executor = SimpleNamespace(document={"host_environment": {"kind": "metal",
                                                                 "host": {"target": "apple_gpu_family7"}}})
        with patch.object(launch_task.Compiler, "load", return_value=compiler), \
             patch.object(launch_task, "resolve_executor", return_value=executor), \
             patch.object(launch_task.MetalArchiveHost, "from_executor") as host:
            with self.assertRaisesRegex(ValueError, "apple_gpu_family7.*not the requested 'apple_gpu_family8'"):
                launch_task._admit_stack(ROOT, self.workspace, "apple_gpu_family8")
            host.assert_not_called()

    def test_each_backend_binds_its_own_exact_target_and_admitted_device(self):
        expected = {"metal-m1-pro": ("apple_gpu_family7", "Apple M1 Pro"),
                    "metal-m2": ("apple_gpu_family8", "Apple M2"),
                    "metal-m4": ("apple_gpu_family9", "Apple M4"),
                    "triton-b200": ("sm_100a", "NVIDIA B200"),
                    "triton-b300": ("sm_103a", "NVIDIA B300"),
                    "triton-dcu": ("gfx938", "BW1101")}
        # This family read an Apple-only registry until the task families were given one
        # device registry, so a normalization Workload could be frozen for Metal alone.
        self.assertEqual(set(launch_task.BACKENDS), set(expected))
        for backend, (target, device) in expected.items():
            # Metal stripes any width; Triton needs a power-of-two span.
            columns = 8 if launch_task.DEVICE_BACKENDS[backend]["power_of_two_width"] else 7
            with self.subTest(backend=backend):
                document, source = create_task("rmsnorm", backend=backend, rows=2, columns=columns)
                workload_path = self.directory / f"{backend}-workload.json"
                starter = self.directory / f"{backend}-starter.py"
                workload_path.write_text(json.dumps(document))
                starter.write_text(source)
                workload = WorkloadContract(document)
                self.assertEqual(workload.target, target)
                self.assertIn(backend, workload.workload_id)
                self.assertIn(f'target="{target}"', source)
                study = study_template(ROOT, workload, workload_path, starter, harness="claude-code",
                                       model="exact-model", effort="high", turns=2, token_budget=12000)
                self.assertEqual(study["execution"]["target"], target)
                self.assertEqual(study["analysis_plan"]["performance_reporting"], "task_efficiency_v1")
                # Metal runs serialized on one local device; the CUDA and AMD routes
                # take the GPU exclusively. The mode follows the route, as the study
                # template has always had it, so the expectation follows it too.
                # The allocation decides this, not the route: a DCU lowers through
                # Triton and serializes one local device.
                mode = ("local_serialized"
                        if launch_task.DEVICE_BACKENDS[backend]["allocation"] == "local_broker"
                        else "exclusive")
                self.assertEqual(study["execution"]["gpu"],
                                 {"name": device, "count": 1, "mode": mode})
                # The Compiler target is the independent authority preflight checks this
                # against. A Target may admit more than one marketing name for one device
                # -- sm_103a declares both NVIDIA B300 spellings -- so the registry's name
                # has to be one of them rather than the only one.
                self.assertIn(
                    device,
                    Target.load(ROOT / "compiler/targets" / f"{target}.json").device_names)

    def test_qualification_uses_shared_entry_with_exact_model_effort_and_python_source(self):
        self.workspace.mkdir()
        args = SimpleNamespace(qualification=None,harness="claude-code",model="exact-test-model",effort="high",
                               provider_revision=None,max_candidates=3,wall_seconds=900)
        source = self.workspace/'starter.py'
        source.write_text('# high-level CPU fixture source\n')
        version = SimpleNamespace(stdout='Claude fixture version',returncode=0)
        completed = SimpleNamespace(stdout='CPU process double',stderr='',returncode=0)
        with patch.object(launch_task.subprocess,'run',side_effect=(version,completed)) as process:
            receipt,anchor = launch_task._qualify(ROOT,self.workspace,args,Path('/unit-test/claude'),source)
        command = process.call_args_list[1].args[0]
        self.assertEqual(command[1],str(ROOT/'tools/qualify_codex_provider.py'))
        for flag,expected in (('--harness','claude-code'),('--model','exact-test-model'),('--reasoning-effort','high'),
                              ('--python-source',str(source)),('--feature-policy','provider_defaults_optimization')):
            self.assertEqual(command[command.index(flag)+1],expected)
        self.assertNotIn('--fixture-only',command)
        self.assertEqual(receipt,self.workspace/'provider-qualification.json')
        self.assertEqual(anchor,self.workspace/'provider-anchor.json')
        self.assertFalse(receipt.exists())  # The process was mocked, so no capability was manufactured.

    def _wiring(self, preflight_error=None, *, fixture_receipt=False, preflight_only=False, report=None, expected_exit=0,
                backend="metal-m1-pro", baseline_only=False, baseline_error=None, extra_args=(), incumbent=False,
                prepared_selection=None, selection_error=None):
        # Authority doubles are never persisted as qualification receipts or Evidence.
        executor = SimpleNamespace(document={"host_environment":{"python":{"invocation_path":"/unit-test/python"},
                                                                 "packages":{"triton":"3.6.0"}}})
        receipt = SimpleNamespace(qualified=True, scope="zero_gpu_contract_fixture_only" if fixture_receipt else "live_two_turn_tool_rich_provider")
        lock = SimpleNamespace(document={"unit_test_lock":True})
        lab = Mock()
        lab.preflight.side_effect = preflight_error
        lab.preflight.return_value = lock
        values = dict(study_id="fixture", claim_scope="artifact_optimization_only", system_qualification_passed=None,
            estimand=None, campaign_complete=True, archive_integrity_passed=True,
            filesystem_custody_verified=True, semantic_replay_passed=True, estimand_available=False,
            missing_run_count=0, estimate=None, uncertainty=None, run_inclusion=(),
            descriptive={"performance": {"policy": "task_efficiency_v1", "rows": [],
                "missing": ["no qualified candidate"], "ranking_scope": "same task and target",
                "threshold_status": "not_defined"}},
            run_audits=(SimpleNamespace(protocol_adherence="adhered", endpoint_observation="no_qualified_candidate"),))
        if report is not None:
            values.update(vars(report))
        values["run_audits"] = tuple(RunAudit(run_id="fixture", authority_sha256=None, archive_integrity=True,
            filesystem_custody_verified=True, event_count=0, protocol_adherence=audit.protocol_adherence,
            endpoint_observation=audit.endpoint_observation, endpoint=None, terminal_seal_sha256=None, findings=())
            for audit in values["run_audits"])
        lab.audit.return_value = StudyReport(**values)
        args = self.args() + (["--preflight-only"] if preflight_only else [])
        if incumbent:
            args += ["--incumbent-registry", str(self.directory / "incumbents")]
        if prepared_selection is not None:
            args += ["--prepared-baseline", str(self.directory / "prepared-baseline.json")]
        registry = Mock()
        key = SimpleNamespace(as_dict=lambda: {"fixture": "exact-key"})
        registry.key_for_launch.return_value = key
        registry.materialize.return_value = (
            self.directory / "incumbent.json",
            {"run_id": "incumbent-fixture"},
        )
        args.extend(extra_args)
        if baseline_only:
            args.append('--baseline-only')
        if backend != "metal-m1-pro":
            args[args.index('--backend')+1] = backend
            args[args.index('--task')+1] = 'silu'
            args[args.index('--columns')+1] = '8'
            args += ['--gpu-run', '/usr/bin/true', '--broker-socket', '/unit-test/broker.sock']
        with patch.object(launch_task.shutil, "which", return_value="/usr/bin/true"), \
             patch.object(launch_task, "_admit_stack", return_value=(Mock(),executor,Mock(),{"fixture":"compiler"})) as admit, \
             patch.object(launch_task, "_prepare_baseline", return_value=self.directory/"baseline.json") as baseline, \
             patch.object(launch_task, "load_baseline_bundle", return_value=Mock()), \
             patch.object(launch_task, "load_prepared_baseline", return_value=(
                 self.directory / "incumbent.json", Mock(), prepared_selection)), \
             patch.object(launch_task, "candidate_identity", return_value={"unit_test_candidate": True}), \
             patch.object(launch_task, "admit_baseline_selection", side_effect=selection_error), \
             patch.object(launch_task, "validate_pair_candidates", side_effect=baseline_error) as validate_baseline, \
             patch.object(launch_task, "_qualify", return_value=(self.directory/"receipt.json",self.directory/"anchor.json")) as qualify, \
             patch.object(launch_task.ProviderQualificationReceipt, "load", return_value=receipt), \
             patch.object(launch_task.TaskIncumbentRegistry, "open_if_exists",
                          return_value=registry if incumbent == "present" else None), \
             patch.object(launch_task.TaskIncumbentRegistry, "key_for_launch", return_value=key), \
             patch.object(launch_task, "TaskLab", return_value=lab), \
             patch.object(launch_task, 'admit_cohort_payload') as payload, \
             patch.object(launch_task, "execute_matched_from_config", return_value=SimpleNamespace(evidence_root="unit-test-campaign")) as execute, \
             contextlib.redirect_stdout(io.StringIO()) as stdout:
            if preflight_error or fixture_receipt or baseline_error or selection_error:
                with self.assertRaises(ValueError): launch_task.main(args)
                execute.assert_not_called()
            else:
                self.assertEqual(launch_task.main(args), expected_exit)
                if baseline_only:
                    qualify.assert_not_called()
                    lab.preflight.assert_not_called()
                    execute.assert_not_called()
                    selected = 'incumbent.json' if incumbent == 'present' or prepared_selection is not None else 'baseline.json'
                    self.assertIn(str(self.directory / selected), stdout.getvalue())
                elif preflight_only:
                    execute.assert_not_called()
                    lab.audit.assert_not_called()
                else:
                    execute.assert_called_once_with(ROOT, lock, self.workspace/"runtime.json", self.workspace/"campaign-evidence")
                    lab.audit.assert_called_once_with(execute.return_value)
                    self.assertIn("unit-test-campaign", stdout.getvalue())
                    self.assertIn("Task performance:", stdout.getvalue())
                    saved = json.loads((self.workspace / "report.json").read_text())
                    self.assertEqual(saved["descriptive"], lab.audit.return_value.descriptive)
                    self.assertEqual(saved["run_audits"][0]["endpoint_observation"],
                                     lab.audit.return_value.run_audits[0].endpoint_observation)
            admit.assert_called_once()
            if incumbent == "present" or prepared_selection is not None:
                baseline.assert_not_called()
            else:
                baseline.assert_called_once()
            validate_baseline.assert_called_once()
            if baseline_error or selection_error:
                qualify.assert_not_called()
                lab.preflight.assert_not_called()
            elif not baseline_only:
                qualify.assert_called_once()
            if backend == 'metal-m1-pro':
                payload.assert_called_once()
            else:
                payload.assert_not_called()
        return lab, lock

    def test_public_cuda_launch_wires_cupti_exclusive_broker_and_triton_config(self):
        self._wiring(preflight_only=True, backend='triton-b300')
        study = json.loads((self.workspace/'study.json').read_text())
        runtime = json.loads((self.workspace/'runtime.json').read_text())
        self.assertEqual(study['evaluation_protocol']['paired_timing']['kind'], PAIRED_KIND)
        self.assertEqual(study['evaluation_protocol']['paired_timing']['maximum_cv'], 0.15)
        self.assertEqual(study['evaluation_protocol']['paired_timing']['required_pair_wins'], 9)
        self.assertEqual(study['execution']['gpu']['mode'], 'exclusive')
        self.assertEqual(runtime['toolchain']['triton_version'], '3.6.0')
        self.assertNotIn('output_root', runtime['toolchain'])
        self.assertNotIn('open_cake_ir.evaluation.local_broker', runtime['broker']['command'])

    def test_baseline_only_builds_without_provider_qualification_or_campaign(self):
        self._wiring(baseline_only=True, backend='triton-b300')

    def test_baseline_only_needs_no_provider_and_writes_no_runtime_binding(self):
        """The flag's own help says it stops before provider qualification.

        It resolved the provider executable anyway, three statements before the workspace
        was created, and refused a DCU baseline build for not having `claude` installed in
        a compile container -- a refusal about the stage the flag exists to skip. The
        runtime config is the same: it binds the provider and the allocator, and nothing
        on this path reads it.
        """
        executor = SimpleNamespace(document={"host_environment": {
            "python": {"invocation_path": "/unit-test/python"}, "packages": {"triton": "3.6.0"},
            "tools": {"build_tools": [{"kind": "bwrap", "path": "/usr/bin/true"}]}}})
        args = self.args() + ["--baseline-only"]
        args[args.index("--backend") + 1] = "triton-dcu"
        args[args.index("--columns") + 1] = "8"
        with patch.object(launch_task.shutil, "which", return_value=None) as which, \
             patch.object(launch_task, "_admit_stack",
                          return_value=(Mock(), executor, Mock(), {"fixture": "compiler"})), \
             patch.object(launch_task, "_prepare_baseline",
                          return_value=self.directory / "baseline.json"), \
             patch.object(launch_task, "load_baseline_bundle", return_value=Mock()), \
             patch.object(launch_task, "candidate_identity", return_value={"unit_test": True}), \
             patch.object(launch_task, "admit_baseline_selection"), \
             patch.object(launch_task, "validate_pair_candidates"), \
             patch.object(launch_task, "_qualify") as qualify, \
             patch.object(launch_task, "TaskLab") as lab, \
             contextlib.redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(launch_task.main(args), 0)
        qualify.assert_not_called()
        lab.assert_not_called()
        self.assertIn(str(self.directory / "baseline.json"), stdout.getvalue())
        self.assertFalse((self.workspace / "runtime.json").exists())
        # Nothing looked for a harness on PATH; the only `which` this path could make is
        # the jail's, and the Executor host declared that one.
        self.assertNotIn("claude", [call.args[0] for call in which.call_args_list])
        self.assertTrue((self.workspace / "prepared-baseline.json").exists())

    def test_baseline_abi_failure_precedes_provider_qualification(self):
        self._wiring(baseline_error=ValueError('baseline ABI differs'), backend='triton-b300')

    def test_explicit_timing_values_are_frozen_in_the_study(self):
        self._wiring(preflight_only=True, backend='triton-b300',
            extra_args=('--maximum-cv','0.12','--required-pair-wins','8'))
        document=json.loads((self.workspace/'study.json').read_text())
        assay=paired_protocol(document['evaluation_protocol'])
        self.assertEqual((assay.maximum_cv,assay.required_pair_wins,assay.materiality_ratio), (0.12,8,1.05))
        StudyContract.load(self.workspace/'study.json')

    def test_launcher_wires_existing_preflight_and_composer_with_persistent_actor_root(self):
        lab, _ = self._wiring()
        lab.preflight.assert_called_once_with(self.workspace/"study.json", execution_bindings_path=self.workspace/"execution-bindings.json")
        runtime = json.loads((self.workspace/"runtime.json").read_text())
        self.assertEqual(runtime["provider"]["workspace_root"], str(self.workspace/"actors"))
        self.assertEqual(runtime["toolchain"], {"output_root":str(self.workspace/"builds")})
        self.assertEqual(runtime["broker"]["command"], ["/unit-test/python", "-I", str(ROOT / "src/open_cake_ir/evaluation/source_bootstrap.py"),
                         "open_cake_ir.evaluation.local_broker", "--kind", "metal",
                         "--worker-module", "open_cake_ir.tasks.evaluate"])
        self.assertEqual(set(json.loads((self.workspace/"execution-bindings.json").read_text())),
                         {"schema_version","qualification_path","qualification_anchor_path","runtime_config_path","fixed_baseline_bundle_path","fixed_baseline_selection"})
        self.assertFalse((self.workspace/"actors").exists())  # The existing composer creates it once.
        with self.assertRaises(FileExistsError): launch_task._new_workspace(self.workspace)

    def test_launcher_freezes_the_exact_task_incumbent_as_the_next_baseline(self):
        self._wiring(incumbent="present")
        selection = json.loads((self.workspace / "baseline-selection.json").read_text())
        self.assertEqual(selection, {
            "schema_version": 1,
            "policy": "exact_incumbent_or_reference",
            "source": "task_incumbent",
            "incumbent_key": {"fixture": "exact-key"},
            "promotion_run_id": "incumbent-fixture",
            "registry_root": str(self.directory / "incumbents"),
        })
        bindings = json.loads((self.workspace / "execution-bindings.json").read_text())
        self.assertEqual(
            bindings["fixed_baseline_bundle_path"],
            str(self.directory / "incumbent.json"),
        )

    def test_missing_incumbent_cell_explicitly_uses_the_starter_reference(self):
        self._wiring(incumbent="missing")
        selection = json.loads((self.workspace / "baseline-selection.json").read_text())
        self.assertEqual(selection["policy"], "exact_incumbent_or_reference")
        self.assertEqual(selection["source"], "starter_reference")
        self.assertIsNone(selection["promotion_run_id"])
        self.assertEqual(
            selection["registry_root"], str(self.directory / "incumbents")
        )

    def test_prepared_incumbent_preserves_selection_and_skips_rebuilding(self):
        selection = {
            "schema_version": 1, "policy": "exact_incumbent_or_reference",
            "source": "task_incumbent", "incumbent_key": {"fixture": "prepared-key"},
            "promotion_run_id": "prepared-promotion", "registry_root": str(self.directory / "incumbents"),
        }
        self._wiring(preflight_only=True, prepared_selection=selection)
        bindings = json.loads((self.workspace / "execution-bindings.json").read_text())
        self.assertEqual(bindings["fixed_baseline_selection"], selection)
        self.assertEqual(bindings["fixed_baseline_bundle_path"], str(self.directory / "incumbent.json"))

    def test_prepared_selection_refusal_precedes_provider_qualification(self):
        self._wiring(prepared_selection={"fixture": "stale-selection"},
                     selection_error=ValueError("fixed baseline is not the selected current incumbent"))

    def test_launcher_returns_nonzero_for_a_recorded_provider_fault(self):
        self._wiring(report=SimpleNamespace(campaign_complete=True, archive_integrity_passed=True,
            filesystem_custody_verified=True, semantic_replay_passed=True,
            run_audits=(SimpleNamespace(protocol_adherence="provider_fault", endpoint_observation="missing"),)),
            expected_exit=1)

    def test_launcher_rejects_missing_or_unverified_outcomes_but_accepts_adhered_rejections(self):
        good = dict(campaign_complete=True, archive_integrity_passed=True,
            filesystem_custody_verified=True, semantic_replay_passed=True,
            run_audits=(SimpleNamespace(protocol_adherence="adhered", endpoint_observation="no_qualified_candidate"),))
        self.assertEqual(launch_task._campaign_exit_code(SimpleNamespace(**good)), 0)
        for field in ("campaign_complete", "archive_integrity_passed", "filesystem_custody_verified", "semantic_replay_passed", "run_audits"):
            with self.subTest(field=field):
                self.assertEqual(launch_task._campaign_exit_code(SimpleNamespace(**{**good, field: () if field == "run_audits" else False})), 1)
        for adherence in ("harness_fault", "custody_violation", "contamination", "broker_fault"):
            with self.subTest(adherence=adherence):
                self.assertEqual(launch_task._campaign_exit_code(SimpleNamespace(**{**good,
                    "run_audits": (SimpleNamespace(protocol_adherence=adherence),)})), 1)

    def test_preflight_refusal_never_reaches_execution(self):
        self._wiring(ValueError("unit-test preflight refusal"))
        self.assertFalse((self.workspace/"campaign-lock.json").exists())

    def test_fixture_qualification_never_reaches_preflight(self):
        lab, _ = self._wiring(fixture_receipt=True)
        lab.preflight.assert_not_called()

    def test_preflight_only_stops_at_reviewable_lock(self):
        self._wiring(preflight_only=True)
        self.assertTrue((self.workspace/"campaign-lock.json").exists())


if __name__ == "__main__":
    unittest.main()
