"""Full preflight policy with explicit CPU authority doubles, never qualification.

Compiler assessment/lowering and task input bindings are real. Only unreleased
Executor/provider resolution and the nonexecuted binary baseline are test doubles.
No live receipt, Executor descriptor or Campaign Lock is published by these tests.
"""
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.evaluation import LaunchableCandidate, WorkloadContract
from open_cake_ir.evaluation.metal_manifest import MetalTensorLaunchManifest
from open_cake_ir.evaluation.paired import candidate_identity
from open_cake_ir.lab import preflight, admission
from open_cake_ir.lab.incumbents import TaskIncumbentKey
from open_cake_ir.lab.provider_policy import provider_configuration
from open_cake_ir.tasks.normalization.study import canonical, study_template, SCAFFOLD
from open_cake_ir.tasks.runtime import TaskLab
from open_cake_ir.tasks.workloads import create_task

ROOT = Path(__file__).resolve().parents[2]


class MetalPreflightTests(unittest.TestCase):
    def fixture(self, directory, harness):
        document, source = create_task('rmsnorm', rows=2, columns=7)
        workload = WorkloadContract(document)
        workload_path, starter = directory/'workload.json', directory/'starter.py'
        workload_path.write_bytes(canonical(document)); starter.write_text(source)
        study = study_template(ROOT, workload, workload_path, starter, harness=harness,
                               model='exact-test-model', effort='high', turns=2)
        study_path = directory/'study.json'; study_path.write_bytes(canonical(study))
        provider = study['arms']['open_cake']['provider']
        receipt_path = directory/'receipt-double.json'; receipt_path.write_text('{}')
        receipt_identity = 'a'*64
        anchor = {'schema_version':1, 'kind': 'provider_qualification_evidence_anchor' if harness=='claude-code'
                  else 'codex_provider_qualification_evidence_anchor', 'run_id':'cpu-double',
                  'evidence_root':str(directory/'not-created'), 'authority_sha256':'b'*64,
                  'qualification_receipt_sha256':receipt_identity, 'immediate_audit_integrity':True,
                  'terminal_seal_sha256':'c'*64}
        anchor_path = directory/'anchor-double.json'; anchor_path.write_bytes(canonical(anchor))
        provider.update(revision='cpu-provider-double', executable_sha256='d'*64,
            qualification={'path':str(receipt_path),'canonical_sha256':receipt_identity},
            qualification_anchor={'path':str(anchor_path),'canonical_sha256':sha256(canonical(anchor)).hexdigest()})
        if harness=='codex':
            provider['code_mode_host']={'path':'/cpu-test-only/no-host', 'sha256':'e'*64}
        configuration = provider_configuration(provider, 'artifact_optimization_only', arms=study['arms'])
        receipt = SimpleNamespace(provider_revision=provider['revision'], executable_sha256='d'*64,
            configuration_sha256=sha256(canonical(configuration)).hexdigest(), initial_and_resume_equivalent=True,
            file_lifecycle_observed=True, usage_observed=True, qualified=True,
            scope='live_two_turn_tool_rich_provider', canonical_sha256=receipt_identity)
        compiler = Compiler.load(ROOT, ROOT/'compiler/revision.lock.json')
        lowering = compiler.lower(compiler.assess(frontend.parse(source).document))
        requirements = lowering.toolchain_requirements
        manifest = MetalTensorLaunchManifest.for_workload(workload, 'primary', target=workload.target,
            kernel_name=lowering.route.entry_point, grid=requirements['threadgroups_per_grid'],
            block=requirements['threads_per_threadgroup'])
        payloads = {'lowered_source':lowering.source.encode(), 'metal_binary_archive':b'CPU test only, nonexecutable',
                    'launch_manifest':canonical(manifest.as_dict())}
        candidate = LaunchableCandidate(lowering.schedule_sha256, workload.target, lowering.route.entry_point,
            {key:sha256(value).hexdigest() for key,value in payloads.items()}, manifest.canonical_sha256, payloads)
        executor = SimpleNamespace(reference={'executor_id':'cpu-test-double','path':'runtime/not-released.json',
                                              'canonical_sha256':'f'*64})
        study['arms']['open_cake']['toolchain_sha256']='1'*64
        study['execution'].update(executor_revision=executor.reference, broker_execution_sha256='2'*64,
            fixed_baseline={'bundle_path':str(directory/'baseline-double.json'),'candidate':candidate_identity(candidate)},
            runtime_config={'path':str(directory/'runtime-double.json'),'sha256':'3'*64})
        return study_path, study, executor, receipt, candidate

    def test_full_preflight_reaches_lock_and_python_package_for_both_harnesses(self):
        for harness in ('codex','claude-code'):
            with self.subTest(harness=harness), tempfile.TemporaryDirectory() as temporary:
                directory=Path(temporary).resolve()
                path,study,executor,receipt,candidate=self.fixture(directory,harness)
                with patch.object(preflight,'resolve_execution_bindings',return_value=(study,executor)), \
                     patch.object(admission.ProviderQualificationReceipt,'load',return_value=receipt), \
                     patch.object(admission,'load_baseline_bundle',return_value=candidate):
                    lock=TaskLab(ROOT).preflight(path)
                self.assertEqual(lock.run_order,('open_cake-1',))
                self.assertIsNone(lock.estimand)
                package=TaskLab(ROOT).task_package(lock,'open_cake-1')
                self.assertIn('schedule-starter.py',package.task_markdown)
                self.assertIn('```python',package.task_markdown)
                self.assertEqual(study['arms']['open_cake']['scaffold']['path'],
                                 'contracts/scaffolds/python-artifact-optimization-v2.md')
                self.assertIn((ROOT/SCAFFOLD).read_text().strip(), package.task_markdown)
                self.assertIn('warps=[0]', package.task_markdown)
                self.assertIn('tile=1', package.task_markdown)
                self.assertIn('coalesced=False', package.task_markdown)
                self.assertFalse((directory/'campaign-lock.json').exists())

    def test_full_preflight_refuses_a_baseline_from_different_lowered_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            path,study,executor,receipt,candidate=self.fixture(Path(temporary).resolve(),'claude-code')
            payloads=dict(candidate.artifact_payloads)
            payloads['lowered_source'] += b'\n// different source\n'
            candidate=LaunchableCandidate(candidate.candidate_sha256,candidate.target,candidate.entry_point,
                {key:sha256(value).hexdigest() for key,value in payloads.items()},candidate.launch_spec_sha256,payloads)
            study['execution']['fixed_baseline']['candidate']=candidate_identity(candidate)
            with patch.object(preflight,'resolve_execution_bindings',return_value=(study,executor)), \
                 patch.object(admission.ProviderQualificationReceipt,'load',return_value=receipt), \
                 patch.object(admission,'load_baseline_bundle',return_value=candidate):
                with self.assertRaisesRegex(ValueError,'fixed baseline differs'):
                    TaskLab(ROOT).preflight(path)

    def test_incumbent_baseline_may_differ_from_starter_but_must_be_registry_current(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary).resolve()
            path,study,executor,receipt,candidate=self.fixture(directory,'claude-code')
            payloads=dict(candidate.artifact_payloads)
            payloads['lowered_source'] += b'\n// independently promoted implementation\n'
            incumbent=LaunchableCandidate('9'*64,candidate.target,candidate.entry_point,
                {key:sha256(value).hexdigest() for key,value in payloads.items()},
                candidate.launch_spec_sha256,payloads)
            workload=WorkloadContract(json.loads(Path(study['workload']['path']).read_text()))
            key=TaskIncumbentKey.from_values(workload_id=workload.workload_id,
                workload_sha256=workload.canonical_sha256,case_id='primary',target=workload.target,
                backend='metal',evaluation_protocol=study['evaluation_protocol'])
            run_id=f'incumbent-{key.canonical_sha256}-000000000000'
            selection={'schema_version':1,'policy':'exact_incumbent_or_reference',
                'source':'task_incumbent','incumbent_key':key.as_dict(),
                'promotion_run_id':run_id,'registry_root':str(directory/'incumbents')}
            study['execution']['fixed_baseline']={
                'bundle_path':str(directory/'baseline-double.json'),
                'candidate':candidate_identity(incumbent),'selection':selection}
            current={'run_id':run_id,'candidate':candidate_identity(incumbent)}
            registry=SimpleNamespace(current=lambda observed: current)
            with patch.object(preflight,'resolve_execution_bindings',return_value=(study,executor)), \
                 patch.object(admission.ProviderQualificationReceipt,'load',return_value=receipt), \
                 patch.object(admission,'load_baseline_bundle',return_value=incumbent), \
                 patch('open_cake_ir.lab.incumbents.TaskIncumbentRegistry.open_if_exists',return_value=registry):
                lock=TaskLab(ROOT).preflight(path)
            package=TaskLab(ROOT).task_package(lock,'open_cake-1')
            self.assertIn('black-box `task_incumbent`',package.task_markdown)
            self.assertIn(run_id,package.task_markdown)

            registry.current=lambda observed: None
            with patch.object(preflight,'resolve_execution_bindings',return_value=(study,executor)), \
                 patch.object(admission.ProviderQualificationReceipt,'load',return_value=receipt), \
                 patch.object(admission,'load_baseline_bundle',return_value=incumbent), \
                 patch('open_cake_ir.lab.incumbents.TaskIncumbentRegistry.open_if_exists',return_value=registry):
                with self.assertRaisesRegex(ValueError,'not the selected current'):
                    TaskLab(ROOT).preflight(path)


if __name__=='__main__':
    unittest.main()
