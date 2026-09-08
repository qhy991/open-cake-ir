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
from open_cake_ir.evaluation.paired import PAIRED_METAL_KIND, paired_protocol
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.lab.core import StudyContract
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
                self.assertEqual(policy["paired_timing"]["kind"], PAIRED_METAL_KIND)
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
                launch_task._admit_stack(ROOT, self.workspace)
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
                launch_task._admit_stack(ROOT,self.workspace)
            host.assert_not_called()

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

    def _wiring(self, preflight_error=None, *, fixture_receipt=False, preflight_only=False):
        # Authority doubles are never persisted as qualification receipts or Evidence.
        executor = SimpleNamespace(document={"host_environment":{"python":{"invocation_path":"/unit-test/python"}}})
        receipt = SimpleNamespace(qualified=True, scope="zero_gpu_contract_fixture_only" if fixture_receipt else "live_two_turn_tool_rich_provider")
        lock = SimpleNamespace(document={"unit_test_lock":True})
        lab = Mock()
        lab.preflight.side_effect = preflight_error
        lab.preflight.return_value = lock
        args = self.args() + (["--preflight-only"] if preflight_only else [])
        with patch.object(launch_task.shutil, "which", return_value="/usr/bin/true"), \
             patch.object(launch_task, "_admit_stack", return_value=(Mock(),executor,Mock())) as admit, \
             patch.object(launch_task, "_prepare_baseline", return_value=self.directory/"baseline.json") as baseline, \
             patch.object(launch_task, "_qualify", return_value=(self.directory/"receipt.json",self.directory/"anchor.json")) as qualify, \
             patch.object(launch_task.ProviderQualificationReceipt, "load", return_value=receipt), \
             patch.object(launch_task, "TaskLab", return_value=lab), \
             patch.object(launch_task, "execute_matched_from_config", return_value=SimpleNamespace(evidence_root="unit-test-campaign")) as execute, \
             contextlib.redirect_stdout(io.StringIO()):
            if preflight_error or fixture_receipt:
                with self.assertRaises(ValueError): launch_task.main(args)
                execute.assert_not_called()
            else:
                self.assertEqual(launch_task.main(args), 0)
                if preflight_only: execute.assert_not_called()
                else: execute.assert_called_once_with(ROOT, lock, self.workspace/"runtime.json", self.workspace/"campaign-evidence")
            admit.assert_called_once()
            baseline.assert_called_once()
            qualify.assert_called_once()
        return lab, lock

    def test_launcher_wires_existing_preflight_and_composer_with_persistent_actor_root(self):
        lab, _ = self._wiring()
        lab.preflight.assert_called_once_with(self.workspace/"study.json", execution_bindings_path=self.workspace/"execution-bindings.json")
        runtime = json.loads((self.workspace/"runtime.json").read_text())
        self.assertEqual(runtime["provider"]["workspace_root"], str(self.workspace/"actors"))
        self.assertEqual(runtime["toolchain"], {"output_root":str(self.workspace/"builds")})
        self.assertEqual(runtime["broker"]["command"], ["/unit-test/python","-m","open_cake_ir.evaluation.local_broker","--worker-module","open_cake_ir.tasks.evaluate"])
        self.assertEqual(set(json.loads((self.workspace/"execution-bindings.json").read_text())),
                         {"schema_version","qualification_path","qualification_anchor_path","runtime_config_path","fixed_baseline_bundle_path"})
        self.assertFalse((self.workspace/"actors").exists())  # The existing composer creates it once.
        with self.assertRaises(FileExistsError): launch_task._new_workspace(self.workspace)

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
