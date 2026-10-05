"""Fixed opponent admission uses real sealed bundles, without device qualification.

The MetaX ELF fixture has real inspected argument metadata but no executable GPU
body. These tests prove software admission and rejection, never measured readiness.
"""
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.compiler.target import Target
from open_cake_ir.evaluation import LaunchableCandidate, WorkloadContract
from open_cake_ir.evaluation.core import TensorLaunchManifest
from open_cake_ir.evaluation.paired import candidate_identity
from open_cake_ir.lab.admission import admit_paired_baseline_artifact, validate_paired_baseline
from open_cake_ir.lab.bindings import bind_fixed_baseline
from open_cake_ir.serialization import canonical_json_bytes
from open_cake_ir.tasks.contraction.authoring import starter_source
from open_cake_ir.tasks.contraction.workload import workload_document
from tests.contracts.test_metax_binary import bundle, native_fixture

ROOT = Path(__file__).resolve().parents[2]


def selection(policy):
    return {'schema_version': 1, 'policy': policy,
            'source': 'explicit' if policy == 'explicit_fixed_bundle' else 'starter_reference',
            'incumbent_key': None, 'promotion_run_id': None, 'registry_root': None}


class FixedEvolutionBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')
        self.workload = WorkloadContract(workload_document(
            'pairwise_sqdist', rows=64, depth=256, columns=32, backend='triton-metax'))
        self.schedule = frontend.parse(starter_source(self.workload)).document
        self.lowering = self.compiler.lower(self.compiler.assess(self.schedule))
        target = Target.load(ROOT / 'compiler/targets/xcore1002.json')
        requirements = self.lowering.toolchain_requirements
        manifest = TensorLaunchManifest.for_workload(self.workload, 'primary', target=target.target_id,
            kernel_name=self.lowering.route.entry_point, grid=requirements['grid'],
            block=[target.warp_size * requirements['compile_options']['num_warps'], 1, 1],
            dynamic_shared_memory_bytes=0, hidden_null_pointer_parameters=0)
        count = len(manifest.tensor_abi)
        arguments = ', '.join(f'%p{index}: !tt.ptr<f32>' for index in range(count))
        self.payloads = {
            'lowered_source': self.lowering.source.encode(),
            'launch_manifest': canonical_json_bytes(manifest.as_dict()),
            'ttgir': f'tt.func public @{manifest.kernel_name}({arguments}) attributes {{}}'.encode(),
            'mcfatbin': bundle(native=native_fixture(count, manifest.kernel_name))[0],
        }
        self.candidate = self.seal(self.payloads)
        self.path = self.publish('c0', self.candidate)

    def seal(self, payloads):
        manifest = TensorLaunchManifest.from_dict(json.loads(payloads['launch_manifest']))
        return LaunchableCandidate(self.lowering.schedule_sha256, manifest.target, manifest.kernel_name,
            {role: sha256(value).hexdigest() for role, value in payloads.items()},
            manifest.canonical_sha256, payloads)

    def publish(self, name, candidate):
        directory = self.directory / name
        directory.mkdir()
        paths = {}
        for role, payload in candidate.artifact_payloads.items():
            filename = role + '.artifact'
            (directory / filename).write_bytes(payload)
            paths[role] = filename
        path = directory / 'baseline.json'
        path.write_bytes(canonical_json_bytes({'candidate': candidate_identity(candidate), 'artifact_paths': paths}))
        return path

    def execution(self, *, policy='explicit_fixed_bundle', path=None):
        return {'fixed_baseline': bind_fixed_baseline(ROOT, path or self.path, selection(policy))}

    def validate(self, execution, lowering, *, workload=None):
        return validate_paired_baseline(project_root=ROOT, workload=workload or self.workload,
            evaluation={'case_id': 'primary'}, execution=execution, route={'backend': 'triton'},
            baseline_lowering=lowering, manifest_parser=TensorLaunchManifest.from_dict)

    def test_c0_bundle_survives_changed_emission_only_under_explicit_selection(self):
        # Two actual legal emissions model the changed starter output of a successor
        # Compiler. The old executable bundle and its identity remain unchanged.
        rewritten = self.compiler.tile_squared_difference_outputs(
            self.schedule, output_tile=4, schedule_id='successor_mapping',
            entry_point=self.lowering.route.entry_point)
        self.assertTrue(rewritten.applied, rewritten.message)
        successor = self.compiler.lower(rewritten.assessment)
        self.assertNotEqual(successor.source, self.lowering.source)
        self.assertNotEqual(successor.toolchain_requirements['grid'], self.lowering.toolchain_requirements['grid'])
        before = {path: path.read_bytes() for path in self.path.parent.iterdir()}
        explicit = self.execution()
        self.validate(explicit, successor)
        self.validate(explicit, None)
        admitted, independent = admit_paired_baseline_artifact(project_root=ROOT, workload=self.workload,
            evaluation={'case_id': 'primary'}, execution=explicit, route={'backend': 'triton'})
        self.assertTrue(independent)
        self.assertEqual(candidate_identity(admitted), candidate_identity(self.candidate))
        for policy in ('starter_reference', 'explicit_fixed_bundle'):
            self.validate(self.execution(policy=policy), self.lowering)
        with self.assertRaisesRegex(ValueError, 'fixed baseline differs from the frozen Compiler'):
            self.validate(self.execution(policy='starter_reference'), successor)
        with self.assertRaisesRegex(ValueError, 'requires the frozen Compiler lowering'):
            self.validate(self.execution(policy='starter_reference'), None)
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_explicit_bundle_keeps_target_and_workload_abi_checks(self):
        other_target = WorkloadContract(workload_document(
            'pairwise_sqdist', rows=64, depth=256, columns=32, backend='triton-b200'))
        with self.assertRaises(ValueError):
            self.validate(self.execution(), None, workload=other_target)
        manifest = json.loads(self.payloads['launch_manifest'])
        manifest['tensor_abi'][0]['shape'][0] += 1
        changed = self.seal({**self.payloads, 'launch_manifest': canonical_json_bytes(manifest)})
        changed_path = self.publish('wrong-abi', changed)
        with self.assertRaisesRegex(ValueError, 'ABI differs'):
            self.validate(self.execution(path=changed_path), None)

    def test_explicit_bundle_keeps_native_family_and_argument_checks(self):
        count = len(self.workload.tensor_abi('primary'))
        wrong_family = self.seal({**self.payloads, 'mcfatbin': bundle(architecture='xcore1001',
            native=native_fixture(count, self.lowering.route.entry_point))[0]})
        wrong_pointers = self.seal({**self.payloads, 'mcfatbin': bundle(
            native=native_fixture(count + 2, self.lowering.route.entry_point))[0]})
        for name, candidate, expected in (
            ('wrong-family', wrong_family, 'only'),
            ('wrong-pointers', wrong_pointers, 'hidden pointer commitments')):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, expected):
                self.validate(self.execution(path=self.publish(name, candidate)), None)
        missing_metadata = self.seal({role: value for role, value in self.payloads.items() if role != 'ttgir'})
        with self.assertRaisesRegex(ValueError, 'lacks native argument inspection artifacts'):
            self.validate(self.execution(path=self.publish('missing-metadata', missing_metadata)), None)

    def test_explicit_bundle_rejects_artifact_swap_and_selection_identity_substitution(self):
        execution = self.execution()
        execution['fixed_baseline']['candidate'] = {**execution['fixed_baseline']['candidate'],
                                                   'entry_point': 'different_kernel'}
        with self.assertRaisesRegex(ValueError, 'identity differs from its sealed artifact'):
            self.validate(execution, None)
        execution = self.execution()
        (self.path.parent / 'mcfatbin.artifact').write_bytes(b'replaced after binding')
        with self.assertRaisesRegex(ValueError, 'artifact bytes differ from their seals'):
            self.validate(execution, None)

    def test_explicit_policy_does_not_accept_malformed_selection_or_symlink(self):
        execution = self.execution()
        execution['fixed_baseline']['selection']['source'] = 'starter_reference'
        with self.assertRaisesRegex(ValueError, 'selection policy differs'):
            self.validate(execution, None)
        alias = self.directory / 'alias.json'
        alias.symlink_to(self.path)
        execution = self.execution()
        execution['fixed_baseline']['bundle_path'] = str(alias)
        with self.assertRaisesRegex(ValueError, 'custody differs'):
            self.validate(execution, None)
