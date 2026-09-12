"""Thin entry/composition wiring with CPU doubles; no live provider or Metal calls."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from open_cake_ir.compiler.corpus import CorpusGateReport
from open_cake_ir.compiler.target import Target
from open_cake_ir.evaluation.paired import PAIRED_METAL_BATCHED_KIND, paired_protocol
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.lab.contracts import StudyContract
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

    def test_failed_full_gate_prevents_executor_and_provider_work(self):
        self.workspace.mkdir()
        gate = CorpusGateReport("unit-fixture", "fixture", "not-live", False, ())
        compiler = SimpleNamespace(state="released", check_corpus=lambda: gate)
        with patch.object(launch_task.Compiler, "load", return_value=compiler), \
             patch.object(launch_task, "resolve_executor") as executor:
            with self.assertRaisesRegex(ValueError, "full Corpus Gate"):
                launch_task._admit_stack(ROOT, self.workspace, "apple_gpu_family7")
            executor.assert_not_called()
        self.assertFalse(json.loads((self.workspace/"compiler-gate.json").read_text())["passed"])

    def test_nonmetal_executor_is_refused_before_archive_helper_admission(self):
        self.workspace.mkdir()
        gate = CorpusGateReport("unit-fixture", "fixture", "not-live", True, ())
        compiler = SimpleNamespace(state="released", check_corpus=lambda: gate)
        executor = SimpleNamespace(document={"host_environment":{"kind":"cuda"}})
        with patch.object(launch_task.Compiler,"load",return_value=compiler), \
             patch.object(launch_task,"resolve_executor",return_value=executor), \
             patch.object(launch_task.MetalArchiveHost,"from_executor") as host:
            with self.assertRaisesRegex(ValueError,"released Metal Executor"):
                launch_task._admit_stack(ROOT,self.workspace,"apple_gpu_family7")
            host.assert_not_called()

    def test_a_gpu_route_refuses_the_metal_host_and_never_builds_a_metal_archive(self):
        """The mirror of the check above, now that a route selects the stack.

        A CUDA host carries no `kind` at all, so the two routes are separated by whether
        the host names itself Metal rather than by a marker each one owns. Both refusals
        land before any host helper is touched.
        """
        gate = CorpusGateReport("unit-fixture", "fixture", "not-live", True, ())
        compiler = SimpleNamespace(state="released", check_corpus=lambda: gate)
        metal = SimpleNamespace(document={"host_environment": {"kind": "metal"}})
        # Each admission writes its own gate report, so give each one a fresh workspace.
        first, second, third = (self.workspace.parent / name for name in ("a", "b", "c"))
        for directory in (first, second, third):
            directory.mkdir(parents=True)
        with patch.object(launch_task.Compiler, "load", return_value=compiler), \
             patch.object(launch_task, "resolve_executor", return_value=metal), \
             patch.object(launch_task.MetalArchiveHost, "from_executor") as host:
            with self.assertRaisesRegex(ValueError, "bound to a GPU host"):
                launch_task._admit_stack(ROOT, first, "sm_103a", "triton")
            host.assert_not_called()
        # And a host with no `kind` is what the CUDA capture actually writes, so the GPU
        # route must accept it while the Metal route must not.
        cuda = SimpleNamespace(document={"host_environment": {"packages": {}}})
        with patch.object(launch_task.Compiler, "load", return_value=compiler), \
             patch.object(launch_task, "resolve_executor", return_value=cuda), \
             patch.object(launch_task.MetalArchiveHost, "from_executor") as host:
            with self.assertRaisesRegex(ValueError, "released Metal Executor"):
                launch_task._admit_stack(ROOT, second, "sm_103a", "metal")
            host.assert_not_called()
            compiler_, executor_, resolved_host, _ = launch_task._admit_stack(
                ROOT, third, "sm_103a", "triton")
            self.assertIsNone(resolved_host)
            host.assert_not_called()

    def test_stale_executor_refusal_names_the_required_release_boundary(self):
        self.workspace.mkdir()
        gate = CorpusGateReport("unit-fixture", "fixture", "not-live", True, ())
        compiler = SimpleNamespace(state="released", check_corpus=lambda: gate)
        with patch.object(launch_task.Compiler, "load", return_value=compiler), \
             patch.object(launch_task, "resolve_executor", side_effect=ValueError("source differs")), \
             patch.object(launch_task.MetalArchiveHost, "from_executor") as host:
            with self.assertRaisesRegex(ValueError, "released Metal Executor matching this source; source differs"):
                launch_task._admit_stack(ROOT, self.workspace, "apple_gpu_family7")
            host.assert_not_called()

    def test_executor_for_another_apple_gpu_is_refused_before_archive_helper_admission(self):
        self.workspace.mkdir()
        gate = CorpusGateReport("unit-fixture", "fixture", "not-live", True, ())
        compiler = SimpleNamespace(state="released", check_corpus=lambda: gate)
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
                    "metal-m4": ("apple_gpu_family9", "Apple M4")}
        self.assertEqual(set(launch_task.BACKENDS), set(expected))
        for backend, (target, device) in expected.items():
            with self.subTest(backend=backend):
                document, source = create_task("rmsnorm", backend=backend, rows=2, columns=7)
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
                self.assertEqual(study["execution"]["gpu"], {"name": device, "count": 1, "mode": "local_serialized"})
                # The Compiler target is the independent authority preflight checks this against.
                self.assertEqual(Target.load(ROOT / "compiler/targets" / f"{target}.json").device_names, (device,))

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

    def _wiring(self, preflight_error=None, *, fixture_receipt=False,
                preflight_only=False, report=None, expected_exit=0,
                incumbent=False):
        # Authority doubles are never persisted as qualification receipts or Evidence.
        executor = SimpleNamespace(document={"host_environment":{"python":{"invocation_path":"/unit-test/python"}}})
        receipt = SimpleNamespace(qualified=True, scope="zero_gpu_contract_fixture_only" if fixture_receipt else "live_two_turn_tool_rich_provider")
        lock = SimpleNamespace(document={"unit_test_lock":True})
        lab = Mock()
        lab.preflight.side_effect = preflight_error
        lab.preflight.return_value = lock
        lab.audit.return_value = report or SimpleNamespace(campaign_complete=True, archive_integrity_passed=True,
            filesystem_custody_verified=True, semantic_replay_passed=True,
            run_audits=(SimpleNamespace(protocol_adherence="adhered", endpoint_observation="no_qualified_candidate"),))
        args = self.args() + (["--preflight-only"] if preflight_only else [])
        if incumbent:
            args += ["--incumbent-registry", str(self.directory / "incumbents")]
        registry = Mock()
        key = SimpleNamespace(as_dict=lambda: {"fixture": "exact-key"})
        registry.key_for_launch.return_value = key
        registry.materialize.return_value = (
            self.directory / "incumbent.json",
            {"run_id": "incumbent-fixture"},
        )
        with patch.object(launch_task.shutil, "which", return_value="/usr/bin/true"), \
             patch.object(launch_task, "_admit_stack", return_value=(Mock(),executor,Mock(),{"fixture":"compiler"})) as admit, \
             patch.object(launch_task, "_prepare_baseline", return_value=self.directory/"baseline.json") as baseline, \
             patch.object(launch_task, "_qualify", return_value=(self.directory/"receipt.json",self.directory/"anchor.json")) as qualify, \
             patch.object(launch_task.ProviderQualificationReceipt, "load", return_value=receipt), \
             patch.object(launch_task.TaskIncumbentRegistry, "open_if_exists",
                          return_value=registry if incumbent == "present" else None), \
             patch.object(launch_task.TaskIncumbentRegistry, "key_for_launch", return_value=key), \
             patch.object(launch_task, "TaskLab", return_value=lab), \
             patch.object(launch_task, "execute_matched_from_config", return_value=SimpleNamespace(evidence_root="unit-test-campaign")) as execute, \
             contextlib.redirect_stdout(io.StringIO()) as stdout:
            if preflight_error or fixture_receipt:
                with self.assertRaises(ValueError): launch_task.main(args)
                execute.assert_not_called()
            else:
                self.assertEqual(launch_task.main(args), expected_exit)
                if preflight_only:
                    execute.assert_not_called()
                    lab.audit.assert_not_called()
                else:
                    execute.assert_called_once_with(ROOT, lock, self.workspace/"runtime.json", self.workspace/"campaign-evidence")
                    lab.audit.assert_called_once_with(execute.return_value)
                    self.assertIn("unit-test-campaign", stdout.getvalue())
            admit.assert_called_once()
            if incumbent == "present":
                baseline.assert_not_called()
            else:
                baseline.assert_called_once()
            qualify.assert_called_once()
        return lab, lock

    def test_launcher_wires_existing_preflight_and_composer_with_persistent_actor_root(self):
        lab, _ = self._wiring()
        lab.preflight.assert_called_once_with(self.workspace/"study.json", execution_bindings_path=self.workspace/"execution-bindings.json")
        runtime = json.loads((self.workspace/"runtime.json").read_text())
        self.assertEqual(runtime["provider"]["workspace_root"], str(self.workspace/"actors"))
        self.assertEqual(runtime["toolchain"], {"output_root":str(self.workspace/"builds")})
        self.assertEqual(runtime["broker"]["command"], ["/unit-test/python", "-I", str(ROOT / "src/open_cake_ir/evaluation/source_bootstrap.py"),
                         "open_cake_ir.evaluation.local_broker", "--worker-module", "open_cake_ir.tasks.evaluate"])
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
