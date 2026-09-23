"""A paired Study binds Python Cake and native JSON transport separately."""
import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest

from open_cake_ir.lab.bindings import resolve_execution_bindings
from open_cake_ir.lab.admission import validate_provider_binding
from open_cake_ir.lab.contracts import StudyContract, _matched_study_shape
from open_cake_ir.lab.provider_documents import ProviderQualificationReceipt
from open_cake_ir.lab.provider_policy import execution_configuration
from open_cake_ir.lab.python_reference import read_skeleton_reference
from open_cake_ir.serialization import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[2]
STUDIES = ROOT / 'contracts/studies'


class PythonPairedStudyTests(unittest.TestCase):
    def test_b300_successors_use_python_starters_and_per_arm_transport(self):
        successors = sorted(STUDIES.glob('*optimization-python-template.json'))
        self.assertEqual(len(successors), 4)
        for path in successors:
            with self.subTest(path=path.name):
                study = StudyContract.load(path)
                cake = study.arms['open_cake']
                native = study.arms[study.comparison]
                self.assertEqual(cake['input_format'], 'python_source_v1')
                self.assertEqual(cake['tool_surface'], ['submit_python_bundle'])
                self.assertEqual(cake['provider']['submission_contract'],
                                 'python_candidate_bundle_v1')
                self.assertNotIn('submission_contract', native['provider'])
                self.assertTrue(cake['schedule_skeleton']['path'].endswith('.py'))
                read_skeleton_reference(ROOT, cake['schedule_skeleton'])

    def test_matched_controls_remain_equal_across_transports(self):
        path = STUDIES/'matched-search-triton-b300-optimization-python-template.json'
        document = json.loads(path.read_text())
        document['arms']['native_triton']['provider']['model'] = 'different-model'
        with self.assertRaisesRegex(ValueError, 'differ in provider'):
            _matched_study_shape(document)

    def test_v3_execution_binding_preserves_each_arm_qualification(self):
        study = StudyContract.load(
            STUDIES/'matched-search-triton-b300-optimization-python-template.json')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            paths = {}
            for name in ('cake-receipt', 'native-receipt', 'cake-anchor', 'native-anchor',
                         'runtime', 'baseline'):
                paths[name] = root/f'{name}.json'
                paths[name].write_text('{}')
            bindings = root/'bindings.json'
            bindings.write_text(json.dumps({
                'schema_version': 3,
                'qualification_paths': {'open_cake': str(paths['cake-receipt']),
                                        'native_triton': str(paths['native-receipt'])},
                'qualification_anchor_paths': {'open_cake': str(paths['cake-anchor']),
                                               'native_triton': str(paths['native-anchor'])},
                'runtime_config_path': str(paths['runtime']),
                'fixed_baseline_bundle_path': str(paths['baseline']),
            }))
            seen = []
            def bind(_root, provider, _row, *, runtime_path, receipt_path, anchor_path):
                seen.append((provider.get('submission_contract'), receipt_path, anchor_path))
                return {**provider, 'qualification': {'path': str(receipt_path)}}, {'same': True}
            with (patch('open_cake_ir.lab.bindings.bind_cli_provider', side_effect=bind),
                  patch('open_cake_ir.lab.bindings.resolve_executor', return_value=SimpleNamespace(reference={})),
                  patch('open_cake_ir.lab.bindings.bind_runtime_execution', return_value=('a'*64, {})),
                  patch('open_cake_ir.lab.bindings.bind_fixed_baseline', return_value={'candidate': 'fixture'})):
                resolved, _ = resolve_execution_bindings(ROOT, study, bindings)
            self.assertCountEqual([item[0] for item in seen],
                                  ['python_candidate_bundle_v1', None])
            self.assertNotEqual(resolved['arms']['open_cake']['provider']['qualification'],
                                resolved['arms']['native_triton']['provider']['qualification'])
            bindings.write_text(json.dumps({
                'schema_version': 1,
                'qualification_path': str(paths['cake-receipt']),
                'qualification_anchor_path': str(paths['cake-anchor']),
                'runtime_config_path': str(paths['runtime']),
                'fixed_baseline_bundle_path': str(paths['baseline']),
            }))
            with self.assertRaisesRegex(ValueError, 'per-arm execution bindings v3'):
                resolve_execution_bindings(ROOT, study, bindings)

    def test_old_template_retains_replay_contract(self):
        old = StudyContract.load(
            STUDIES/'matched-search-triton-b300-optimization-template.json')
        self.assertEqual(old.arms['open_cake']['input_format'], 'schedule_or_python_v1')

    def test_a_native_receipt_cannot_qualify_the_cake_python_transport(self):
        study = StudyContract.load(
            STUDIES/'matched-search-triton-b300-optimization-python-template.json')
        providers = {name: dict(arm['provider']) for name, arm in study.arms.items()}
        for provider in providers.values():
            provider.update(revision='fixture-provider', executable_sha256='a'*64,
                            qualification_anchor=None,
                            code_mode_host={'path': '/tmp/fixture-host', 'sha256': 'b'*64})
        native_config = execution_configuration(providers['native_triton'])
        native_receipt = ProviderQualificationReceipt(
            provider_revision='fixture-provider', executable_sha256='a'*64,
            configuration_sha256=sha256(canonical_json_bytes(native_config)).hexdigest(),
            initial_and_resume_equivalent=True, file_lifecycle_observed=True,
            usage_observed=True, qualified=True, scope='zero_gpu_contract_fixture_only')
        with tempfile.TemporaryDirectory() as directory:
            receipt_path = Path(directory).resolve()/'receipt.json'
            receipt_path.write_bytes(canonical_json_bytes(native_receipt.document))
            cake = providers['open_cake']
            cake['qualification'] = {'path': str(receipt_path),
                                     'canonical_sha256': native_receipt.canonical_sha256}
            with self.assertRaisesRegex(ValueError, 'qualification bytes or capability'):
                validate_provider_binding(
                    provider=cake, project_root=ROOT,
                    expected_provider_configuration=execution_configuration(cake),
                    admitted_scopes={'zero_gpu_contract_fixture_only'})
