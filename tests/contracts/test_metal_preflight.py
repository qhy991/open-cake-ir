"""Full preflight policy with explicit CPU authority doubles, never qualification.

Compiler assessment/lowering and task input bindings are real. Only unreleased
Executor/provider resolution and the nonexecuted binary baseline are test doubles.
No live receipt, Executor descriptor or Campaign Lock is published by these tests.
"""
from hashlib import sha256
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.evaluation import LaunchableCandidate, WorkloadContract
from open_cake_ir.evaluation.core import TensorLaunchManifest
from open_cake_ir.evaluation.metal_manifest import MetalTensorLaunchManifest
from open_cake_ir.evaluation.paired import candidate_identity
from open_cake_ir.lab import preflight, admission
from open_cake_ir.lab.incumbents import TaskIncumbentKey
from open_cake_ir.lab.provider_policy import provider_configuration
from open_cake_ir.tasks.normalization.study import (
    METAL_SCAFFOLD, SCAFFOLD, canonical, study_template)
from open_cake_ir.tasks.runtime import TaskLab
from open_cake_ir.tasks.workloads import create_task

ROOT = Path(__file__).resolve().parents[2]


class MetalPreflightTests(unittest.TestCase):
    def test_task_preparation_freezes_an_independent_run_without_loading_a_study(self):
        from open_cake_ir.lab import bindings, StudyContract
        from open_cake_ir.lab.executor import ExecutorRevision
        from open_cake_ir.lab.metal_build import MetalArchiveHost
        from open_cake_ir.tasks.preparation import prepare_task_run
        from open_cake_ir.tasks.normalization.study import task_run_inputs
        from open_cake_ir.tasks.workloads import load_workload
        from tests.contracts._executor_fixture import compiler_reference
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            _,study,executor,receipt,candidate = self.fixture(directory,'claude-code')
            workload = load_workload(directory/'workload.json')
            inputs = task_run_inputs(ROOT,workload,directory/'workload.json',directory/'starter.py',
                harness='claude-code',model='exact-test-model',effort='high',turns=2)
            self.assertEqual(inputs['authoring']['input_format'], 'python_source_v1')
            self.assertEqual(inputs['authoring']['tool_surface'], ['submit_python_bundle'])
            from open_cake_ir.lab.provider_documents import PYTHON_SOURCE_FILE_V1, PYTHON_CANDIDATE_BUNDLE_V1
            self.assertEqual(inputs['authoring']['provider']['submission_contract'], PYTHON_CANDIDATE_BUNDLE_V1)
            source_run = task_run_inputs(ROOT,workload,directory/'workload.json',directory/'starter.py',
                harness='claude-code',model='exact-test-model',effort='high',turns=2,
                maximum_candidates=1,searches_per_turn=1,source_file=True)
            self.assertEqual(source_run['authoring']['provider']['submission_contract'], PYTHON_SOURCE_FILE_V1)
            self.assertEqual(source_run['authoring']['scaffold']['path'],
                             'contracts/scaffolds/python-artifact-optimization-source-file-v1.md')
            with self.assertRaisesRegex(ValueError, 'one candidate and one search'):
                task_run_inputs(ROOT,workload,directory/'workload.json',directory/'starter.py',
                    harness='claude-code',model='exact-test-model',effort='high',source_file=True)
            json_starter = directory/'starter.json'
            json_starter.write_text(json.dumps(frontend.read_schedule(directory/'starter.py').document))
            with self.assertRaisesRegex(ValueError, 'Python starter'):
                task_run_inputs(ROOT,workload,directory/'workload.json',json_starter,
                    harness='claude-code',model='exact-test-model',effort='high',turns=2)
            executable = directory/'provider';executable.write_bytes(b'CPU provider; not executed')
            receipt.executable_sha256 = sha256(executable.read_bytes()).hexdigest()
            runtime = {'schema_version':1,'provider':{'executable':str(executable),'workspace_root':str(directory/'actors')},
                'toolchain':{'output_root':str(directory/'builds')},
                'broker':{'command':[str(executable)],'cwd':str(ROOT),'timeout_seconds':30,
                          'service_user':'fixture','service_group':'fixture'}}
            runtime_path = directory/'runtime.json';runtime_path.write_bytes(canonical(runtime))
            baseline_path = directory/'baseline.json';baseline_path.write_text('{}')
            selection = {'schema_version':1,'policy':'starter_reference','source':'starter_reference',
                         'incumbent_key':None,'promotion_run_id':None,'registry_root':None}
            with patch.object(StudyContract,'load',side_effect=AssertionError('ordinary task loaded a Study')), \
                 patch.object(admission.ProviderQualificationReceipt,'load',return_value=receipt), \
                 patch.object(ExecutorRevision,'load_reference',return_value=executor), \
                 patch.object(MetalArchiveHost,'from_executor',return_value=SimpleNamespace(canonical_sha256='1'*64)), \
                 patch.object(bindings,'broker_execution_sha256',return_value='2'*64), \
                 patch.object(bindings,'load_baseline_bundle',return_value=candidate), \
                 patch.object(admission,'load_baseline_bundle',return_value=candidate):
                def prepare():
                    return prepare_task_run(ROOT,inputs,compiler_reference=compiler_reference(ROOT),executor=executor,
                        qualification_path=directory/'receipt-double.json',qualification_anchor_path=directory/'anchor-double.json',
                        runtime_config_path=runtime_path,baseline_path=baseline_path,baseline_selection=selection)
                specification = prepare()
                self.assertIsNone(specification.document['assignment'])
                self.assertEqual(specification.document['workload']['workload_id'],workload.workload_id)
                self.assertEqual(specification.document['execution']['fixed_baseline']['selection'],selection)
                self.assertEqual(specification.document['authoring']['toolchain_sha256'],'1'*64)
                self.assertNotIn('analysis_plan',specification.document)
                original_attribution = inputs['evaluation_protocol']['attribution_evaluation']
                inputs['evaluation_protocol']['attribution_evaluation'] = None
                with self.assertRaisesRegex(ValueError,'Metal optimization'):
                    prepare()
                inputs['evaluation_protocol']['attribution_evaluation'] = original_attribution
                from open_cake_ir.tasks.normalization.study import evaluation_policy
                cuda_workload,_ = create_task('silu',backend='triton-b200',rows=2,columns=8)
                original_evaluation = inputs['evaluation_protocol']
                inputs['evaluation_protocol'] = evaluation_policy(WorkloadContract(cuda_workload))
                with self.assertRaisesRegex(ValueError,'Metal optimization'):
                    prepare()
                inputs['evaluation_protocol'] = original_evaluation
                payloads = {**candidate.artifact_payloads,'lowered_source':candidate.artifact_payloads['lowered_source']+b'\n// other source'}
                changed = LaunchableCandidate(candidate.candidate_sha256,candidate.target,candidate.entry_point,
                    {role:sha256(value).hexdigest() for role,value in payloads.items()},candidate.launch_spec_sha256,payloads)
                with patch.object(bindings,'load_baseline_bundle',return_value=changed), \
                     patch.object(admission,'load_baseline_bundle',return_value=changed):
                    with self.assertRaisesRegex(ValueError,'fixed baseline differs from the frozen Compiler'):
                        prepare()
                inputs['evaluation_protocol']['validation_case_ids'].pop()
                with self.assertRaisesRegex(ValueError,'omits Workload validation'):
                    prepare()

    def fixture(self, directory, harness, *, backend='metal-m1-pro'):
        cuda = backend.startswith('triton-')
        document, source = create_task('silu' if cuda else 'rmsnorm', backend=backend,
                                       rows=2, columns=8 if cuda else 7)
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
        compiler = Compiler.load(ROOT, ROOT/'compiler/revision.json')
        lowering = compiler.lower(compiler.assess(frontend.parse(source).document))
        requirements = lowering.toolchain_requirements
        if cuda:
            from open_cake_ir.compiler.target import Target
            from open_cake_ir.lab.pairing import native_block
            manifest = TensorLaunchManifest.for_workload(workload, 'primary', target=workload.target,
                kernel_name=lowering.route.entry_point, grid=requirements['grid'], block=native_block(requirements, warp_size=Target.load(
                    ROOT / f'compiler/targets/{workload.target}.json').warp_size),
                dynamic_shared_memory_bytes=0, hidden_null_pointer_parameters=2)
        else:
            manifest = MetalTensorLaunchManifest.for_workload(workload, 'primary', target=workload.target,
                kernel_name=lowering.route.entry_point, grid=requirements['threadgroups_per_grid'],
                block=requirements['threads_per_threadgroup'])
        payloads = {'lowered_source':lowering.source.encode(),
                    ('cubin' if cuda else 'metal_binary_archive'):b'CPU test only, nonexecutable',
                    'launch_manifest':canonical(manifest.as_dict())}
        candidate = LaunchableCandidate(lowering.schedule_sha256, workload.target, lowering.route.entry_point,
            {key:sha256(value).hexdigest() for key,value in payloads.items()}, manifest.canonical_sha256, payloads)
        executor = SimpleNamespace(reference={'executor_id':'cpu-test-double','path':'runtime/not-released.json'})
        study['arms']['open_cake']['toolchain_sha256']='1'*64
        study['execution'].update(executor_revision=executor.reference, broker_execution_sha256='2'*64,
            fixed_baseline={'bundle_path':str(directory/'baseline-double.json'),'candidate':candidate_identity(candidate)},
            runtime_config={'path':str(directory/'runtime-double.json'),'sha256':'3'*64})
        return study_path, study, executor, receipt, candidate

    def test_full_cuda_task_preflight_uses_existing_abi_policy_and_all_case_gate(self):
        for backend in ('triton-b200','triton-b300'):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as temporary:
                path,study,executor,receipt,candidate = self.fixture(Path(temporary).resolve(),'claude-code',backend=backend)
                with patch.object(preflight,'resolve_execution_bindings',return_value=(study,executor)), \
                     patch.object(admission.ProviderQualificationReceipt,'load',return_value=receipt), \
                     patch.object(admission,'load_baseline_bundle',return_value=candidate):
                    lock=TaskLab(ROOT).preflight(path)
                    self.assertEqual(lock.document['execution']['gpu']['mode'],'exclusive')
                    self.assertEqual(lock.document['evaluation_protocol']['paired_timing']['kind'],'fixed_baseline_paired_cupti_v1')
                    self.assertEqual(len(lock.document['evaluation_protocol']['validation_case_ids']),5)
                    study['evaluation_protocol']['validation_case_ids'].pop()
                    with self.assertRaisesRegex(ValueError,'case projection'):
                        TaskLab(ROOT).preflight(path)
                    del study['evaluation_protocol']['validation_case_ids']
                    with self.assertRaisesRegex(ValueError,'validation_case_ids'):
                        TaskLab(ROOT).preflight(path)

    def test_python_only_study_refuses_json_starter_during_preflight(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            path, study, executor, receipt, candidate = self.fixture(directory, 'claude-code')
            arm = study['arms']['open_cake']
            source_path = Path(arm['schedule_skeleton']['path'])
            document = frontend.read_schedule(source_path).document
            json_starter = directory/'starter.json'
            json_starter.write_bytes(canonical(document))
            arm['schedule_skeleton'] = {'path': str(json_starter),
                                        'canonical_sha256': sha256(canonical(document)).hexdigest()}
            path.write_bytes(canonical(study))
            with patch.object(preflight, 'resolve_execution_bindings', return_value=(study, executor)), \
                 patch.object(admission.ProviderQualificationReceipt, 'load', return_value=receipt), \
                 patch.object(admission, 'load_baseline_bundle', return_value=candidate):
                with self.assertRaisesRegex(ValueError, 'Python-only Study requires a .py Schedule starter'):
                    TaskLab(ROOT).preflight(path)

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
                                 'contracts/scaffolds/python-artifact-optimization-metal-v4.md')
                # The package owner delivers the frozen scaffold in AGENTS.md and
                # references it from TASK.md; do not require a second body copy.
                self.assertIn((ROOT/METAL_SCAFFOLD).read_text().strip(), package.agents_markdown)
                self.assertNotIn((ROOT/METAL_SCAFFOLD).read_text().strip(), package.task_markdown)
                self.assertIn('AGENTS.md', package.task_markdown)
                self.assertIn('execution_groups=[0]', package.task_markdown)
                self.assertIn('execution_groups=[0, 1]', package.agents_markdown)
                self.assertIn('tile=1', package.task_markdown)
                self.assertIn('coalesced=False', package.task_markdown)
                self.assertFalse((directory/'campaign-lock.json').exists())

    def test_default_scaffold_follows_route_and_metal_example_lowers(self):
        compiler = Compiler.load(ROOT, ROOT/'compiler/revision.json')
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            for backend, expected in (
                ('metal-m2', METAL_SCAFFOLD),
                ('triton-b200', SCAFFOLD),
            ):
                with self.subTest(backend=backend):
                    document, source = create_task('silu', backend=backend, rows=2, columns=8)
                    workload = WorkloadContract(document)
                    workload_path = directory/f'{backend}-workload.json'
                    starter = directory/f'{backend}-starter.py'
                    workload_path.write_bytes(canonical(document)); starter.write_text(source)
                    study = study_template(ROOT, workload, workload_path, starter,
                                           harness='codex', model='exact-test-model',
                                           effort='high')
                    self.assertEqual(study['arms']['open_cake']['scaffold']['path'], expected)

            override = directory/'AGENTS.md'
            override.write_text('explicit task rules')
            document, source = create_task('silu', backend='metal-m2', rows=2, columns=8)
            workload = WorkloadContract(document)
            workload_path = directory/'override-workload.json'
            starter = directory/'override-starter.py'
            workload_path.write_bytes(canonical(document)); starter.write_text(source)
            study = study_template(ROOT, workload, workload_path, starter, harness='codex',
                                   model='exact-test-model', effort='high', agents_md=override)
            self.assertEqual(study['arms']['open_cake']['scaffold']['path'], str(override))

        scaffold = (ROOT/METAL_SCAFFOLD).read_text()
        scaffold_words = ' '.join(scaffold.split())
        self.assertIn('group count alone does not satisfy the required structural alternative',
                      scaffold_words)
        self.assertIn('timestamp-only profiling does not measure physical registers, spills or '
                      'occupancy', scaffold_words)
        examples = re.findall(r'```python\n(.*?)\n```', scaffold, flags=re.DOTALL)
        self.assertEqual(len(examples), 1)
        assessment = compiler.assess(frontend.parse(examples[0]).document)
        self.assertTrue(assessment.lowering_eligible, assessment.findings)
        lowering = compiler.lower(assessment)
        self.assertIn('fma(', lowering.source)
        self.assertEqual(assessment.analysis['total_execution_groups'], 2)

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
