"""Engineering and research assignments execute through one Run lifecycle."""
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import tempfile
from unittest.mock import patch

from open_cake_ir.lab import RunSpecification, ProviderQualificationReceipt
from open_cake_ir.lab.contracts import StudyContract
from open_cake_ir.serialization import canonical_json_bytes
from open_cake_ir.tasks.runtime import TaskLab
from tests.contracts.test_lab import SemanticLabTestCase, RalphFakeProvider, FakeEnvironment, FakeEvaluator

ROOT = Path(__file__).resolve().parents[2]


class IndependentRunTests(SemanticLabTestCase):
    def fixture(self, *, condition=None):
        lab = TaskLab(ROOT)
        campaign = lab.preflight(ROOT/'contracts/studies/matched-search-system-qualification-ralph-template.json')
        document = campaign.run_specification('open_cake-1').document
        document['run_id'] = 'arbitrary-run-identity'
        document['assignment'] = (None if condition is None else
            {'study_id': 'transfer-study', 'study_sha256': '1'*64, 'condition_id': condition})
        # Bind the generic condition schema and a clearly labeled CPU-only receipt.
        # A provider qualified against an enum-limited schema cannot author new labels.
        from open_cake_ir.lab.provider_policy import execution_configuration
        declared = document['authoring']['provider']
        schema_path = 'contracts/providers/run-turn-output-schema-v1.json'
        declared['output_schema'] = {'path': schema_path, 'sha256': sha256((ROOT/schema_path).read_bytes()).hexdigest()}
        qualification = ProviderQualificationReceipt.load(ROOT/declared['qualification']['path'])
        qualification = replace(qualification, configuration_sha256=sha256(canonical_json_bytes(execution_configuration(declared))).hexdigest())
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        receipt_path = Path(temporary.name).resolve()/'qualification.json'
        receipt_path.write_bytes(canonical_json_bytes(qualification.document))
        declared['qualification'] = {'path': str(receipt_path), 'canonical_sha256': qualification.canonical_sha256}
        return lab, RunSpecification.from_dict(document)

    def run_fixture(self, *, condition=None):
        lab, specification = self.fixture(condition=condition)
        document = specification.document
        provider = RalphFakeProvider({specification.run_id: lab.task_package(specification, specification.run_id)})
        provider.qualification_sha256 = document['authoring']['provider']['qualification']['canonical_sha256']
        from open_cake_ir.lab.provider_policy import execution_configuration
        provider.configuration = execution_configuration(document['authoring']['provider'])
        evaluator = FakeEvaluator(document['evaluation_protocol'],
            sha256(canonical_json_bytes(document['evaluation_protocol'])).hexdigest(),
            document['workload']['canonical_sha256'])
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'run.json'
            source.write_bytes(canonical_json_bytes(document))
            # No fake Study is loaded or assigned to the engineering execution.
            with patch.object(StudyContract, 'load', side_effect=AssertionError('Study required at runtime')):
                specification = lab.preflight_run(source)
                run = lab.execute_run(specification, Path(directory)/'evidence', provider=provider,
                    environment=FakeEnvironment('open_cake', document['authoring']), evaluator=evaluator)
                audit, replay = lab.audit_run(run)
            self.assertTrue(audit.archive_integrity)
            self.assertTrue(replay, replay.refusals)
            self.assertEqual(audit.protocol_adherence, 'adhered')
            self.assertEqual(audit.authority_sha256, specification.canonical_sha256)
            self.assertTrue(provider.requests)
            self.assertTrue(all(request.environment_kind == 'open_cake' for request in provider.requests))
            self.assertTrue(all(request.arm == specification.condition_id for request in provider.requests))
            return audit.endpoint_observation

    def test_ordinary_run_executes_and_replays_without_study_or_estimand(self):
        self.assertEqual(self.run_fixture(), 'qualified')

    def test_condition_rename_does_not_change_candidate_representation(self):
        self.assertEqual(self.run_fixture(condition='E0P0'), self.run_fixture(condition='named-treatment'))

    def test_frozen_run_cannot_be_mutated_via_input_or_projection(self):
        _, specification = self.fixture()
        document = specification.document
        document['budget']['limit'] = 1
        self.assertNotEqual(specification.document['budget']['limit'], 1)
        self.assertIsNone(specification.document['assignment'])
        self.assertNotIn('analysis_plan', specification.document)
        self.assertNotIn('study', specification.document)

    def test_invalid_budget_or_unresolved_dependency_refuses_before_execution(self):
        _, specification = self.fixture()
        for mutate in (lambda d:d['budget'].update(checkpoints=[1]),
                       lambda d:d['execution'].update(executor_revision={'binding':'current_release'})):
            document = specification.document
            mutate(document)
            with self.assertRaises(ValueError):
                RunSpecification.from_dict(document)
