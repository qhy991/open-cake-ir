"""Verify common composer wiring with CPU authority doubles, never live qualification."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from open_cake_ir.lab.task_package import TaskPackage
from open_cake_ir.tasks import compose
from open_cake_ir.tasks.normalization.study import canonical, evaluation_policy, SCAFFOLD
from open_cake_ir.tasks.workloads import create_task
from open_cake_ir.evaluation.workload import WorkloadContract

ROOT = Path(__file__).resolve().parents[2]


class MetalTaskCompositionTests(unittest.TestCase):
    def test_metal_claude_composes_existing_broker_and_persistent_actor_workspace(self):
        self.check_composition('metal-m1-pro')

    def test_cuda_claude_composes_one_triton_environment_with_the_existing_builder(self):
        self.check_composition('triton-b300')

    def check_composition(self, backend):
        metal = backend.startswith('metal-')
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            workload_document, _ = create_task("rmsnorm" if metal else "silu", backend=backend,
                                                rows=2, columns=7 if metal else 8)
            workload = WorkloadContract(workload_document)
            workload_path = directory / "workload.json"
            workload_path.write_bytes(canonical(workload_document))
            qualification_path = directory / "qualification.json"
            qualification_path.write_text('{}')  # Loader below is an explicit authority double.
            anchor = {"qualification_receipt_sha256":"qualification-fixture"}
            anchor_path = directory / "anchor.json"
            anchor_path.write_bytes(canonical(anchor))
            executable = Path('/usr/bin/true').resolve()
            provider = {"harness":"claude-code", "model":"exact-test-model", "reasoning_effort":"high",
                "executable_sha256":sha256(executable.read_bytes()).hexdigest(),
                "qualification":{"path":str(qualification_path),"canonical_sha256":"qualification-fixture"},
                "qualification_anchor":{"path":str(anchor_path),"canonical_sha256":sha256(canonical(anchor)).hexdigest()},
                "removed_environment":["OPENAI_API_KEY","ANTHROPIC_API_KEY"]}
            open_arm = {"provider":provider,"lowering_route":{"backend":"metal" if metal else "triton",
                                                                       "entry_point":"cake_task"},
                "scaffold":{"path":SCAFFOLD,"sha256":sha256((ROOT/SCAFFOLD).read_bytes()).hexdigest()},
                "toolchain_sha256":"metal-toolchain-fixture"}
            runtime = {"schema_version":1,
                "provider":{"executable":str(executable),"workspace_root":str(directory/'actors')},
                "toolchain":{"output_root":str(directory/'builds')},
                "broker":{"command":[str(executable)],"cwd":str(ROOT),"timeout_seconds":30,
                          "service_user":"test-user","service_group":"test-group"}}
            if not metal:
                runtime['toolchain'] = {'python':'/unit-test/python','bubblewrap':'/unit-test/bwrap',
                    'runtime_roots':['/unit-test'],'triton_version':'3.6.0','timeout_seconds':600}
            runtime_path = directory/'runtime.json'
            runtime_path.write_bytes(canonical(runtime))
            execution = {"broker_execution_sha256":"broker-fixture",
                "fixed_baseline":{"bundle_path":str(directory/'baseline.json'),"candidate":{}},
                "runtime_config":{"path":str(runtime_path),"sha256":sha256(runtime_path.read_bytes()).hexdigest()}}
            lock = SimpleNamespace(study_kind='matched_search',claim_scope='artifact_optimization_only',run_order=('open_cake-1',),
                document={"resolved_inputs":{"arm_environments":{"open_cake":open_arm},"budget":{}},
                    "workload":{"path":str(workload_path),"canonical_sha256":workload.canonical_sha256},
                    "compiler_revision":{"path":"compiler/revision.lock.json","canonical_sha256":"compiler-fixture"},
                    "evaluation_protocol":evaluation_policy(workload),"execution":execution})
            compiler = SimpleNamespace(state='released',check_corpus=lambda:SimpleNamespace(passed=True,compiler_revision_sha256='compiler-fixture'))
            receipt = SimpleNamespace(scope='live_two_turn_tool_rich_provider',canonical_sha256='qualification-fixture',provider_revision='unit-fixture')
            toolchain = Mock(canonical_sha256='metal-toolchain-fixture')
            package = TaskPackage('open_cake-1','open_cake','# CPU wiring fixture\n','# CPU wiring fixture\n')
            lab = Mock()
            lab.task_package.return_value = package
            lab.execute.return_value = 'CPU wiring result'
            baseline = object()
            with patch.object(compose,'_admit_executor',return_value=(Mock(),{})), \
                 patch.object(compose.ProviderQualificationReceipt,'load',return_value=receipt), \
                 patch.object(compose.MetalArchiveHost,'from_executor',return_value=Mock()) as archive, \
                 patch.object(compose,'MetalToolchainBuilder',return_value=toolchain) as build, \
                 patch('open_cake_ir.lab.triton_build.IsolatedTritonCompiler',return_value=toolchain) as triton, \
                 patch.object(compose,'NvccToolchainBuilder') as nvcc, \
                 patch.object(compose,'broker_execution_sha256',return_value='broker-fixture'), \
                 patch.object(compose.Compiler,'load',return_value=compiler), \
                 patch.object(compose,'OpenCakeEnvironment') as environment, \
                 patch.object(compose,'load_baseline_bundle',return_value=baseline), \
                 patch.object(compose,'candidate_identity',return_value={}), \
                 patch.object(compose,'CommandBrokerSubmitter') as submitter, \
                 patch.object(compose,'BoundedBrokerEvaluator') as evaluator, \
                 patch.object(compose,'TaskLab',return_value=lab), \
                 patch.object(compose,'ClaudeRunProvider') as provider_type, \
                 patch.object(compose,'CodexInvocationBuilder') as codex:
                self.assertEqual(compose.execute_matched_from_config(ROOT,lock,runtime_path,directory/'evidence'),'CPU wiring result')
            codex.assert_not_called()
            if metal:
                archive.assert_called_once()
                build.assert_called_once()
                triton.assert_not_called()
                self.assertEqual(environment.call_args.args[1],toolchain)
            else:
                archive.assert_not_called()
                build.assert_not_called()
                triton.assert_called_once_with(**runtime['toolchain'])
                self.assertIs(environment.call_args.args[1]._isolated,toolchain)
                toolchain.check_executor.assert_called_once()
            nvcc.assert_not_called()
            self.assertEqual(submitter.call_args.kwargs['workload_path'],workload_path)
            self.assertIs(submitter.call_args.kwargs['baseline'],baseline)
            self.assertEqual(set(lab.execute.call_args.kwargs['environments']),{'open_cake'})
            self.assertIs(lab.execute.call_args.kwargs['evaluator'],evaluator.return_value)
            builders = provider_type.call_args.kwargs['builders']
            self.assertEqual(set(builders),{'open_cake-1'})
            builder = builders['open_cake-1']
            self.assertEqual(builder.configuration['model'],'exact-test-model')
            self.assertEqual(builder.configuration['reasoning_effort'],'high')
            self.assertEqual(builder.workspace,directory/'actors/open_cake-1')
            self.assertEqual({p.name for p in builder.workspace.iterdir()},{'TASK.md','AGENTS.md'})


if __name__ == '__main__':
    unittest.main()
