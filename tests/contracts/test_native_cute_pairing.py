"""CuTe Lab contract tests; CPU fixtures never establish GPU qualification."""
from __future__ import annotations

import ast
import contextlib
import copy
import dataclasses
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from open_cake_ir.compiler.frontend import parse
from open_cake_ir.compiler.cute_toolchain import validate_cute_kernel
from open_cake_ir.evaluation.core import EvaluationProtocol, TensorLaunchManifest
from open_cake_ir.lab.environments import CandidateSubmission, NativeCuTeEnvironment
from open_cake_ir.lab.cute_build import CuTeToolchainBuilder
from open_cake_ir.lab.faults import CandidateCompileRejected, RunProtocolFault
from open_cake_ir.lab.endpoints import NORMAL_BUDGET_TERMINAL
from open_cake_ir.lab.pairing import (backend_policy, bind_baseline, comparison_arm, native_backend,
                                     native_baseline, native_optimization_analysis_plan)
from open_cake_ir.lab.providers import _project_candidate_submission, CANDIDATE_SET_ENVELOPE_V1
from open_cake_ir.tasks.environments import TaskOpenCakeEnvironment as OpenCakeEnvironment
from open_cake_ir.tasks.runtime import TaskLab
from open_cake_ir.tasks.tiles.evaluation import evaluate_tile_workload
from open_cake_ir.tasks.tiles.workload import reference_outputs
from open_cake_ir.tasks.workloads import load_workload
from tests.contracts.test_cute_toolchain import compilation_fixture
from tests.contracts.test_cutedsl_register import register_schedule
from tests.contracts.test_lab import FakeProvider, FakeEvaluator, _submission_envelope
from tests.contracts.test_native_triton_pairing import DraftCompilerFixture, encoded

ROOT = Path(__file__).resolve().parents[2]


class CompilationFixture:
    def __init__(self):
        self.requests = []

    def compile(self, source, requirements):
        validate_cute_kernel(source, requirements)
        self.requests.append((source, requirements))
        compiled = compilation_fixture(source, requirements)
        # Parameterize the CPU receipt's observed SDK symbol for this authored name.
        name = compiled.entry_point.replace('fixture', requirements['kernel_entry_point'])
        payloads = {role: payload.replace(compiled.entry_point.encode(), name.encode())
                    for role, payload in compiled.artifacts.items()}
        return dataclasses.replace(compiled, entry_point=name, artifacts=payloads)


def baseline(workload, case_id='primary'):
    abi = workload.tensor_abi(case_id)
    schedule = register_schedule(shape=(abi[0].shape[0], abi[1].shape[0], abi[0].shape[1]))
    schedule['metadata'] = {}
    return bind_baseline(schedule, workload, case_id)


class NativeCuTePairingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = DraftCompilerFixture()
        cls.workload = load_workload(ROOT / 'contracts/workloads/gemm-bias-bf16-fp32-v2.json')
        cls.schedule = baseline(cls.workload)
        cls.lowering = cls.compiler.lower(cls.compiler.assess(cls.schedule))
        cls.native = native_baseline(cls.lowering)

    def environments(self, case_id='primary'):
        schedule = baseline(self.workload, case_id)
        lowering = self.compiler.lower(self.compiler.assess(schedule))
        fixture = CompilationFixture()
        policy = native_backend('native_cute_dsl')
        builder = policy.builder(workload=self.workload, case_id=case_id, isolated_compiler=fixture)
        open_env = OpenCakeEnvironment(self.compiler, builder,
            authority_document={'lowering_route': schedule['lowering'], 'input_format': 'schedule_or_python_v1'},
            workload=self.workload, case_id=case_id)
        native_env = policy.environment(builder, toolchain_requirements=lowering.toolchain_requirements,
            authority_document={'environment_kind': policy.arm}, workload=self.workload, case_id=case_id)
        return schedule, lowering, open_env, native_env, fixture

    def test_both_arms_keep_same_source_abi_and_launch_for_all_five_cases(self):
        for case_id in ('primary', 'tail', 'tiny', 'zeros', 'cancellation'):
            with self.subTest(case=case_id):
                schedule, lowering, open_env, native_env, fixture = self.environments(case_id)
                self.assertEqual(schedule['metadata']['workload_contract_sha256'], self.workload.canonical_sha256)
                ir = open_env.build(CandidateSubmission.seal(open_env.media_type, encoded(schedule)))
                native = native_env.build(CandidateSubmission.seal(native_env.media_type, encoded(native_baseline(lowering))))
                self.assertEqual((ir.disposition, native.disposition), ('launchable', 'launchable'))
                self.assertEqual(fixture.requests[0], fixture.requests[1])
                self.assertEqual(ir.launchable.launch_spec_sha256, native.launchable.launch_spec_sha256)
                manifest = TensorLaunchManifest.from_dict(json.loads(native.launchable.artifact_payloads['launch_manifest']))
                manifest.check_workload(self.workload, case_id)
                self.assertEqual(manifest.hidden_null_pointer_parameters, 0)
                self.assertEqual(list(manifest.block), [32, 1, 1])
                self.assertNotEqual(manifest.kernel_name, lowering.route.entry_point)

    def test_python_example_preserves_real_workload_binding_and_source_locations(self):
        source = (ROOT / 'examples/python/b300_cute_gemm_bias.py').read_text()
        parsed = parse(source)
        fixture = CompilationFixture()
        builder = CuTeToolchainBuilder(workload=self.workload, case_id='primary', isolated_compiler=fixture)
        env = OpenCakeEnvironment(self.compiler, builder, authority_document={
            'lowering_route': parsed.document['lowering'], 'input_format':'schedule_or_python_v1'},
            workload=self.workload, case_id='primary')
        result = env.build(CandidateSubmission.seal(env.media_type, encoded({'python_source': source})))
        self.assertEqual(result.disposition, 'launchable', result.feedback)
        bad = source.replace('tile=16', 'tile=open("forbidden")', 1)
        rejected = env.build(CandidateSubmission.seal(env.media_type, encoded({'python_source': bad})))
        self.assertEqual(rejected.disposition, 'rejected')
        self.assertEqual(rejected.feedback['source_location']['filename'], 'candidate.ir.py')
        self.assertEqual(len(fixture.requests), 1)

    def test_native_rejects_host_effects_pointer_reordering_and_unsupported_launch_controls(self):
        _, _, _, env, fixture = self.environments()
        variants = [
            {**self.native, 'kernel_source': self.native['kernel_source'] + '\nimport os\n'},
            {**self.native, 'kernel_source': self.native['kernel_source'] + '\ncute.compile()\n'},
            {**self.native, 'kernel_source': self.native['kernel_source'].replace('a: cute.Pointer, b:', 'b: cute.Pointer, a:')},
            {**self.native, 'compile_options': {'num_warps': 2}},
            {**self.native, 'signature': []}, {**self.native, 'target': 'sm_100a'},
            {**self.native, 'block': [64, 1, 1]}, {**self.native, 'block': [32, True, 1]},
            {**self.native, 'grid': [True, 1, 1]}, {**self.native, 'grid': [0, 1, 1]},
            {**self.native, 'dynamic_shared_memory_bytes': 16},
            {**self.native, 'dynamic_shared_memory_bytes': False},
        ]
        for member in variants:
            with self.subTest(member=member.keys()):
                result = env.build(CandidateSubmission.seal(env.media_type, encoded(member)))
                self.assertEqual(result.disposition, 'rejected', result.feedback)
                self.assertEqual(result.feedback['stage'], 'source_admission')
        self.assertFalse(fixture.requests)
        with self.assertRaisesRegex(ValueError, 'Workload ABI'):
            requirements = copy.deepcopy(dict(self.lowering.toolchain_requirements))
            requirements['signature'][0], requirements['signature'][1] = requirements['signature'][1], requirements['signature'][0]
            NativeCuTeEnvironment(mock.Mock(), toolchain_requirements=requirements,
                authority_document={}, workload=self.workload, case_id='primary')

    def test_grid_is_explicit_but_backend_and_pointer_authority_cannot_change(self):
        _, _, _, env, fixture = self.environments()
        member = {**self.native, 'grid':[64, 8, 1]}
        result = env.build(CandidateSubmission.seal(env.media_type, encoded(member)))
        self.assertEqual(result.disposition, 'launchable')
        self.assertEqual(fixture.requests[0][1]['grid'], [64, 8, 1])
        self.assertEqual(comparison_arm({'open_cake':{}, 'native_cute_dsl':{}}), 'native_cute_dsl')
        with self.assertRaises(ValueError): comparison_arm({'open_cake':{}, 'native_cute_dsl':{}, 'native_triton':{}})
        with self.assertRaises(ValueError): backend_policy('cute')
        with self.assertRaises(ValueError): bind_baseline(self.schedule, self.workload, 'tiny')

    def test_compile_rejection_and_infrastructure_failure_remain_distinct_for_both_arms(self):
        schedule, lowering, open_env, native_env, fixture = self.environments()
        for env, member in ((open_env, schedule), (native_env, native_baseline(lowering))):
            with self.subTest(environment=type(env).__name__):
                submission = CandidateSubmission.seal(env.media_type, encoded(member))
                with mock.patch.object(fixture, 'compile', side_effect=CandidateCompileRejected('bad candidate', artifact_payloads={'toolchain_stderr':b'bad candidate'})):
                    result = env.build(submission)
                    self.assertEqual(result.feedback['stage'], 'compile')
                    self.assertEqual(result.artifact_payloads['toolchain_stderr'], b'bad candidate')
                with mock.patch.object(fixture, 'compile', side_effect=RunProtocolFault('harness_fault', 'missing dependency')):
                    with self.assertRaises(RunProtocolFault): env.build(submission)

    def test_provider_projection_and_common_oracle_retain_native_boundary(self):
        _, lowering, _, env, _ = self.environments('tiny')
        member = native_baseline(lowering)
        payload = encoded({'schema_version':1, 'arm':'native_cute_dsl', 'candidates':[member]})
        self.assertEqual(_project_candidate_submission(payload, submission_contract=CANDIDATE_SET_ENVELOPE_V1,
            arm='native_cute_dsl', maximum_candidates_per_turn=1), (encoded(member),))
        candidate = env.build(CandidateSubmission.seal(env.media_type, encoded(member))).launchable
        workload = self.workload
        class Launcher:
            wrong = False
            mutate = False
            def launch_tensors(self, candidate, manifest, inputs):
                outputs = reference_outputs(workload, 'tiny', inputs)
                after = copy.deepcopy(inputs)
                if self.wrong: outputs['c'][-1] += 1
                if self.mutate: after['a'][0] += 1
                return outputs, after, {'candidate_sha256':candidate.candidate_sha256, 'kernel_calls':1, 'fallback_calls':0}
        launcher = Launcher()
        protocol = EvaluationProtocol('fixture', 'search', workload.canonical_sha256, 'tiny', 'none')
        self.assertTrue(evaluate_tile_workload(candidate, workload, protocol, launcher).correctness_passed)
        launcher.wrong = True
        self.assertFalse(evaluate_tile_workload(candidate, workload, protocol, launcher).correctness_passed)
        launcher.wrong = False; launcher.mutate = True
        self.assertFalse(evaluate_tile_workload(candidate, workload, protocol, launcher).correctness_passed)


class CuTePairedLabFixtureTests(unittest.TestCase):
    def test_real_matched_loop_preflight_submission_confirmation_replay_and_threshold_view(self):
        draft = DraftCompilerFixture()
        workload = load_workload(ROOT / 'contracts/workloads/gemm-bias-bf16-fp32-v2.json')
        schedule = baseline(workload)
        lowering = draft.lower(draft.assess(schedule))
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory) / 'project'; root.mkdir()
            for name in ('contracts', 'corpus', 'compiler', 'src', 'docs', 'examples/python'):
                shutil.copytree(ROOT / name, root / name)
            document = json.loads((root / 'contracts/studies/matched-search-cute-b300-gemm-optimization-template.json').read_text())
            # This fixture binds real source and opaque CPU-only runtime references;
            # full released-source and live qualification remain integration gates.
            historical = json.loads((root / 'contracts/studies/matched-search-triton-optimization-template.json').read_text())
            for name, arm in document['arms'].items():
                old = historical['arms']['open_cake' if name == 'open_cake' else 'native_triton']
                schema = arm['provider']['output_schema']
                arm['provider'] = {**old['provider'], 'output_schema':schema}
                arm['toolchain_sha256'] = old['toolchain_sha256']
            target, gpu = document['execution']['target'], document['execution']['gpu']
            document['execution'] = {**historical['execution'], 'target':target, 'gpu':gpu}
            skeleton_path = root / document['arms']['open_cake']['schedule_skeleton']['path']
            skeleton_path.write_bytes(encoded(schedule))
            document['arms']['open_cake']['lowering_route'] = schedule['lowering']
            document['arms']['open_cake']['schedule_skeleton']['canonical_sha256'] = sha256(encoded(schedule)).hexdigest()
            document['budget'].update({'unit':'provider_tokens', 'limit':80000, 'checkpoints':[80000], 'maximum_turns':1, 'maximum_candidates_per_turn':1})
            document['evaluation_protocol'] = {'case_id':'primary', 'search_evaluation':'correctness_then_paired_cupti',
                'confirmatory_evaluation':'fresh_fixed_candidate_correctness_then_paired_cupti'}
            configuration = {**FakeProvider.configuration, 'output_schema_sha256': document['arms']['open_cake']['provider']['output_schema']['sha256']}
            qualification = json.loads((root / 'contracts/providers/fixture-provider-candidate-set-ralph-v1.json').read_text())
            qualification['configuration_sha256'] = sha256(encoded(configuration)).hexdigest()
            qualification['provider_revision'] = 'native-ralph-pairing-cpu-fixture'
            qp = root / 'contracts/providers/native-fixture.json'; qp.write_bytes(encoded(qualification))
            for arm, value in document['arms'].items():
                value['provider']['disabled_features'] = configuration['disabled_features']
                value['provider']['revision'] = qualification['provider_revision']
                value['provider']['qualification'] = {'path':'contracts/providers/native-fixture.json', 'canonical_sha256':sha256(encoded(qualification)).hexdigest()}
                value['feedback'] = (['findings'] if arm == 'open_cake' else ['compile']) + ['correctness','qualified_timing']
            study = root / 'fixture-study.json'; study.write_bytes(encoded(document))
            gate = SimpleNamespace(compiler_revision_id='compiler-fixture', compiler_revision_sha256='a'*64, passed=True)
            reference = {'revision_id':'compiler-fixture','path':'compiler/revision.lock.json','canonical_sha256':'a'*64}
            executor = {'executor_id':'open-cake-ir-b200-v9000','path':'runtime/executors/fixture.json','canonical_sha256':'e'*64}
            stack.enter_context(mock.patch('open_cake_ir.lab.preflight._resolve_compiler_reference', return_value=(gate, reference['path'], reference)))
            bound_executor = SimpleNamespace(reference=executor)
            stack.enter_context(mock.patch('open_cake_ir.lab.preflight.resolve_executor', return_value=bound_executor))
            stack.enter_context(mock.patch('open_cake_ir.lab.executor.ExecutorRevision.load_reference', return_value=bound_executor))
            stack.enter_context(mock.patch('open_cake_ir.compiler.Compiler.load', return_value=draft))
            # Match the existing draft Compiler fixture at the new replay boundary.
            def admit_fixture_compiler(project_root, value, context):
                self.assertEqual(Path(project_root).resolve(), root.resolve())
                self.assertEqual(value, reference, context)
                return SimpleNamespace(revision_id=reference['revision_id'], canonical_sha256=reference['canonical_sha256'])
            stack.enter_context(mock.patch('open_cake_ir.lab.bindings.load_compiler_reference', side_effect=admit_fixture_compiler))
            lab = TaskLab(root)
            lock = lab.preflight(study)
            self.assertEqual(lock.analysis_plan, {**native_optimization_analysis_plan("native_cute_dsl"),
                "endpoint_policy": NORMAL_BUDGET_TERMINAL})
            fixture = CompilationFixture()
            builder = CuTeToolchainBuilder(workload=workload, case_id='primary', isolated_compiler=fixture)
            arms = lock.document['resolved_inputs']['arm_environments']
            environments = {
                'open_cake':OpenCakeEnvironment(draft,builder,authority_document=arms['open_cake'],workload=workload,case_id='primary'),
                'native_cute_dsl':NativeCuTeEnvironment(builder,toolchain_requirements=lowering.toolchain_requirements,
                    authority_document=arms['native_cute_dsl'],workload=workload,case_id='primary')}
            class Provider(FakeProvider):
                def turn(self, request):
                    original = super().turn(request)
                    member = schedule if request.arm == 'open_cake' else native_baseline(lowering)
                    # Retain one real candidate rejection as an observed failure.
                    if request.run_id == 'native_cute_dsl-3': member = {**member, 'host_launcher':'forbidden'}
                    payload = encoded(member)
                    return dataclasses.replace(original,candidates=(payload,),raw_submission=_submission_envelope(request.arm, (payload,)),candidate_sha256s=(sha256(payload).hexdigest(),))
            provider = Provider(); provider.configuration = configuration
            provider.provider_revision = qualification['provider_revision']
            provider.packages = {run_id: TaskLab(root).task_package(lock, run_id) for run_id in lock.run_order}
            provider.qualification_sha256 = sha256(encoded(qualification)).hexdigest()
            class Evaluator(FakeEvaluator):
                def evaluate(self,candidate,*,case_id,purpose):
                    # Existing sealed broker-receipt fixture; these are never live measurements.
                    arm = 'open_cake' if 'lowered_source' in candidate.artifact_roles else 'direct_cuda'
                    virtual_name = 'open_cake_turn_10' if purpose == 'search' else arm + '_turn_1'
                    return super().evaluate(dataclasses.replace(candidate,entry_point=virtual_name),case_id=case_id,purpose=purpose)
            evaluator = Evaluator(lock.document['evaluation_protocol'],sha256(encoded(lock.document['evaluation_protocol'])).hexdigest(),workload.canonical_sha256,
                compiler_revision_reference=lock.document['compiler_revision'])
            task = provider.packages['native_cute_dsl-1']
            self.assertIn('candidate-baseline.cute.json', task.task_markdown)
            self.assertIn('candidate.schema.json', task.task_markdown)
            self.assertIn('Generated-source comments', task.task_markdown)
            self.assertNotIn('import triton', task.task_markdown)
            campaign = lab.execute(lock, Path(directory) / 'evidence',provider=provider,environments=environments,evaluator=evaluator)
            report = lab.audit(campaign)
            self.assertTrue(report.archive_integrity_passed)
            self.assertTrue(report.semantic_replay_passed, report)
            self.assertEqual(len(report.descriptive['paired_runs']),3)
            self.assertEqual(report.descriptive['paired_runs'][2]['native_cute_dsl_outcome'],'no_qualified_candidate')
            view = lab.threshold_view(campaign,0.5)
            # Search fixture latencies are 0.1ms, but fresh confirmations are 1/2ms.
            self.assertEqual(len(view['runs']),6)
            self.assertTrue(all(row['first_confirmation_turn'] is None for row in view['runs']))
            self.assertTrue(report.filesystem_custody_verified, 'new Evidence fixture requires a custody-capable temporary filesystem')
            reached = lab.threshold_view(campaign,1.5)
            self.assertEqual(sum(row['status']=='reached_by_fresh_confirmation' for row in reached['runs']),3)
            self.assertTrue(all(row['elapsed_wall_seconds'] is not None for row in reached['runs'] if row['first_confirmation_turn']))
