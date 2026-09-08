"""CPU boundary probes; fixture CUBINs and drivers are not GPU qualification."""
from __future__ import annotations
from open_cake_ir.tasks.workloads import load_workload

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from open_cake_ir.compiler import Compiler, Schedule, profile_envelope
from open_cake_ir.compiler.frontend import parse, FrontendError
from open_cake_ir.compiler.target import cuda_architecture, cuda_target
from open_cake_ir.evaluation import CudaDeviceAdmission, LaunchableCandidate, LoadedCudaCandidate, WorkloadContract
from open_cake_ir.evaluation.core import TensorLaunchManifest
from open_cake_ir.evaluation.cuda_manifest import CudaKernelSpec
from open_cake_ir.tasks.flash_kmeans.cuda_manifest import CudaLaunchManifest
from open_cake_ir.lab import CandidateSubmission, NativeTritonEnvironment, TritonToolchainBuilder
from open_cake_ir.tasks.environments import TaskOpenCakeEnvironment as OpenCakeEnvironment
from open_cake_ir.lab.pairing import bind_baseline, native_baseline
from examples.paired_triton.prepare import baseline_schedule
from tests.contracts.test_cuda_driver import CUBIN, FakeDriver, FakeTensor
from tests.contracts.test_native_triton_pairing import CompilationFixture, DraftCompilerFixture, encoded, python_rms

ROOT = Path(__file__).resolve().parents[2]


class TargetCompilationFixture(CompilationFixture):
    def compile(self, source, requirements):
        compiled = super().compile(source, requirements)
        return replace(compiled, target=requirements['target'])


class B300ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')
        cls.workload = load_workload(ROOT / 'contracts/workloads/rmsnorm-fp32-v2.json')

    def test_three_b300_ralph_templates_are_stable_and_require_external_bindings(self):
        from open_cake_ir.tasks.runtime import TaskLab as Lab
        from open_cake_ir.lab.contracts import StudyContract
        from open_cake_ir.lab.providers import CODEX_DISABLED_FEATURES
        for suffix in ('', '-gemm', '-gather'):
            with self.subTest(operator=suffix):
                path = ROOT / f'contracts/studies/matched-search-triton-b300{suffix}-optimization-template.json'
                study = StudyContract.load(path)
                self.assertEqual(study.schema_version, 2)
                self.assertEqual(study.document['agent_interface']['kind'], 'task_agents_ralph_v1')
                self.assertEqual(study.document['budget']['limit'], 150000)
                self.assertEqual(study.document['budget']['maximum_turns'], 4)
                self.assertEqual(study.document['budget']['maximum_candidates_per_turn'], 3)
                self.assertEqual(study.document['execution']['target'], 'sm_103a')
                self.assertEqual(study.document['execution']['fixed_baseline'], {'binding':'campaign_lock'})
                for arm in study.document['arms'].values():
                    self.assertNotIn('prompt_template', arm)
                    self.assertEqual(arm['provider']['cwd_policy'], 'independent_task_workspace')
                    self.assertEqual(arm['provider']['reference_visibility'], 'workspace_task_files')
                    self.assertEqual(arm['provider']['disabled_features'], list(CODEX_DISABLED_FEATURES))
                # This test deliberately has no runtime binding; it must stop before
                # Compiler/Executor host admission or any author workspace is created.
                with self.assertRaisesRegex(ValueError, 'require external execution bindings'):
                    Lab(ROOT).preflight(path)

    def test_fifteen_successor_cases_preserve_math_and_lower_to_b300(self):
        count = 0
        for name in ('rmsnorm-fp32', 'gemm-bias-bf16-fp32', 'indexed-gather-bf16'):
            old = load_workload(ROOT / f'contracts/workloads/{name}-v1.json')
            new = load_workload(ROOT / f'contracts/workloads/{name}-v2.json')
            for field in ('cases', 'tensors', 'oracle'):
                self.assertEqual(old.document[field], new.document[field])
            self.assertEqual({k: v for k, v in old.document['semantics'].items() if k != 'target'},
                             {k: v for k, v in new.document['semantics'].items() if k != 'target'})
            for field in ('atol', 'rtol', 'comparison', 'all_cases_required'):
                self.assertEqual(old.document['validation'][field], new.document['validation'][field])
            for case in new.case_ids:
                with self.subTest(workload=name, case=case):
                    assessment = self.compiler.assess(baseline_schedule(new, case))
                    self.assertTrue(assessment.accepted)
                    self.assertTrue(assessment.lowering_eligible)
                    self.assertFalse(assessment.calibration_available)
                    lowering = self.compiler.lower(assessment)
                    self.assertEqual(lowering.target, 'sm_103a')
                    self.assertEqual(lowering.toolchain_requirements['target'], 'sm_103a')
                    count += 1
        self.assertEqual(count, 15)

    def test_three_python_starters_have_the_same_kernel_body_as_json_baselines(self):
        for file, name in (('b300_rmsnorm', 'rmsnorm-fp32'), ('b300_gemm_bias', 'gemm-bias-bf16-fp32'),
                           ('b300_indexed_gather', 'indexed-gather-bf16')):
            with self.subTest(example=file):
                workload = load_workload(ROOT / f'contracts/workloads/{name}-v2.json')
                document = parse((ROOT / f'examples/python/{file}.py').read_text()).document
                authored = self.compiler.lower(self.compiler.assess(bind_baseline(document, workload, 'primary')))
                baseline = self.compiler.lower(self.compiler.assess(baseline_schedule(workload, 'primary')))
                self.assertEqual(dict(authored.toolchain_requirements), dict(baseline.toolchain_requirements))
                # Only the two provenance comments can differ between representations.
                self.assertEqual(authored.source.splitlines()[2:], baseline.source.splitlines()[2:])

    def test_target_identity_is_exact_and_calibration_is_not_inherited(self):
        self.assertEqual(cuda_architecture('sm_100a'), 100)
        self.assertEqual(cuda_architecture('sm_103a'), 103)
        for value in ('sm_103', 'sm_100f', 'sm_103f', 'sm_104a', '../sm_103a', None, 103):
            with self.subTest(target=value), self.assertRaises(ValueError):
                cuda_architecture(value)
        target = cuda_target('sm_103a')
        self.assertIsNone(target.occupancy)
        self.assertIsNone(target.peak)
        document = baseline_schedule(self.workload, 'primary')
        assessment = self.compiler.assess(document)
        profile = profile_envelope(Schedule.from_dict(document), target,
                                   lowered_source=self.compiler.lower(assessment).source).as_dict()
        self.assertNotIn('B200', json.dumps(profile))

    def test_schedule_and_manifest_cannot_cross_workload_targets(self):
        document = baseline_schedule(self.workload, 'primary')
        document['target'] = 'sm_100a'
        with self.assertRaisesRegex(ValueError, 'target'):
            bind_baseline(document, self.workload, 'primary')
        with self.assertRaisesRegex(ValueError, 'Workload'):
            TensorLaunchManifest.for_workload(self.workload, 'primary',
                target='sm_100a', kernel_name='kernel', grid=[1, 1, 1], block=[128, 1, 1],
                dynamic_shared_memory_bytes=0, hidden_null_pointer_parameters=2)

    def test_generic_tensor_structure_does_not_widen_historical_flash_abi(self):
        launch = dict(target='sm_103a', kernel_name='kernel', grid=[1, 1, 1],
                      block=[128, 1, 1], dynamic_shared_memory_bytes=0)
        self.assertEqual(CudaKernelSpec.from_dict(launch).target, 'sm_103a')
        with self.assertRaises(ValueError):
            CudaLaunchManifest.from_dict(dict(schema_version=1, abi='flash_kmeans_assign_v1', **launch))
        for key, value in (('block', [1024, 2, 1]), ('grid', [1, 65536, 1]),
                           ('dynamic_shared_memory_bytes', 232449)):
            with self.subTest(field=key), self.assertRaises(ValueError):
                CudaKernelSpec.from_dict({**launch, key: value})

    def _pair(self, compilation=None):
        fixture = compilation or TargetCompilationFixture()
        builder = TritonToolchainBuilder(workload=self.workload, case_id='primary', isolated_compiler=fixture)
        document = baseline_schedule(self.workload, 'primary')
        lowering = self.compiler.lower(self.compiler.assess(document))
        authority = {'input_format': 'schedule_or_python_v1', 'lowering_route': document['lowering']}
        cake = OpenCakeEnvironment(DraftCompilerFixture(), builder, workload=self.workload,
                                   case_id='primary', authority_document=authority)
        native = NativeTritonEnvironment(builder, workload=self.workload, case_id='primary',
                    authority_document={}, toolchain_requirements=lowering.toolchain_requirements)
        return cake, native, lowering

    def test_python_and_native_build_paths_keep_b300_target(self):
        cake, native, lowering = self._pair()
        source = python_rms(self.workload).replace('target="sm_100a"', 'target="sm_103a"')
        result = cake.build(CandidateSubmission.seal(cake.media_type, encoded({'python_source': source})))
        self.assertIsNotNone(result.launchable)
        self.assertEqual(result.launchable.target, 'sm_103a')
        result = native.build(CandidateSubmission.seal(native.media_type, encoded(native_baseline(lowering))))
        self.assertIsNotNone(result.launchable)
        self.assertEqual(result.launchable.target, 'sm_103a')

    def test_builder_cannot_relabel_b200_compilation_as_b300(self):
        _, native, lowering = self._pair(CompilationFixture())
        with self.assertRaisesRegex(ValueError, 'compilation target'):
            native.build(CandidateSubmission.seal(native.media_type, encoded(native_baseline(lowering))))

    def test_missing_sm_facts_refuse_persistent_grid_before_emission(self):
        document = json.loads((ROOT / 'corpus/schedules/rmsnorm-b128-persistent.json').read_text())
        document['target'] = 'sm_103a'
        assessment = self.compiler.assess(document)
        self.assertFalse(assessment.lowering_eligible)
        self.assertIn('TRITON_PERSISTENT_TARGET_FACTS_MISSING', [f.code for f in assessment.findings])

    def test_missing_occupancy_does_not_disable_register_budget_legality(self):
        document = baseline_schedule(self.workload, 'primary')
        document['roles'][0]['registers_per_thread'] = 32
        assessment = self.compiler.assess(document)
        self.assertFalse(assessment.accepted)
        self.assertIn('ROLE_REGISTERS_NOT_CONSERVED', [f.code for f in assessment.findings])

    def test_json_indexed_tile_reuse_refuses_impossible_reduction_axis(self):
        document = json.loads((ROOT / 'corpus/schedules/b300-indexed-tile-reuse-refusal.json').read_text())
        for target in ('sm_100a', 'sm_103a'):
            with self.subTest(target=target):
                document['target'] = target
                assessment = self.compiler.assess(document)
                self.assertFalse(assessment.lowering_eligible)
                self.assertIn('TRITON_INDEXED_TILE_REUSE', [f.code for f in assessment.findings])

    def _launch_authority(self):
        manifest = TensorLaunchManifest.for_workload(self.workload, 'primary', target='sm_103a',
            kernel_name='kernel', grid=[1, 1, 1], block=[128, 1, 1],
            dynamic_shared_memory_bytes=0, hidden_null_pointer_parameters=2)
        candidate = LaunchableCandidate('a' * 64, 'sm_103a', 'kernel',
            {'cubin': sha256(CUBIN).hexdigest()}, manifest.canonical_sha256)
        admission = CudaDeviceAdmission('NVIDIA B300 SXM6 AC', (10, 3), 'GPU-fixture', 'gpuq-fixture', 'exclusive')
        return manifest, candidate, admission

    def test_mixed_device_admission_refused_before_module_load(self):
        manifest, candidate, _ = self._launch_authority()
        driver = FakeDriver()
        admission = CudaDeviceAdmission('NVIDIA B200', (10, 0), 'GPU-fixture', 'gpuq-fixture', 'exclusive')
        with self.assertRaisesRegex(ValueError, 'launch authority'):
            LoadedCudaCandidate.load(candidate, CUBIN, manifest, admission, driver=driver)
        self.assertEqual(driver.calls, [])
        for name, cc in (('NVIDIA B200', (10, 3)), ('NVIDIA B300 SXM6 AC', (10, 0)),
                         ('NVIDIA B300 SXM6 AC', (10, True))):
            with self.subTest(name=name, cc=cc), self.assertRaises(ValueError):
                CudaDeviceAdmission(name, cc, 'GPU-fixture', 'gpuq-fixture', 'exclusive')

    def test_actual_binary_version_must_match_b300_and_failure_unloads(self):
        manifest, candidate, admission = self._launch_authority()
        driver = FakeDriver()
        with self.assertRaisesRegex(ValueError, 'binary version'):
            LoadedCudaCandidate.load(candidate, CUBIN, manifest, admission, driver=driver)
        self.assertIn(('cuModuleUnload', 42), driver.calls)
        self.assertFalse(any(call[0] == 'cuLaunchKernel' for call in driver.calls))

    def test_b300_loaded_function_launch_and_tensor_target_guard(self):
        manifest, candidate, admission = self._launch_authority()
        driver = FakeDriver()
        driver.attributes[6] = 103
        loaded = LoadedCudaCandidate.load(candidate, CUBIN, manifest, admission, driver=driver)
        arguments = [FakeTensor(shape, dtype, 1000 * (index + 1))
                     for index, (_, shape, dtype) in enumerate(manifest.tensors)]
        try:
            with self.assertRaisesRegex(ValueError, 'tensor Target'):
                loaded.launch(arguments, tensor_contract=SimpleNamespace(target='sm_100a'), stream=0)
            self.assertEqual(loaded.launch_calls, 0)
            loaded.launch(arguments, tensor_contract=manifest, stream=0)
            self.assertEqual(loaded.launch_calls, 1)
        finally:
            loaded.close(synchronize=lambda: None)


if __name__ == '__main__':
    unittest.main()
