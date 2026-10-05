"""Qualification receipt/anchor admission using synthetic CPU archives, never a live model.

The executable fixture stays fixture-only. Synthetic live-scope archives below test
admission mechanics; they are ephemeral counterfactuals, not provider qualifications.
"""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from open_cake_ir.evidence import EvidenceStore
from open_cake_ir.lab.admission import validate_provider_binding
from open_cake_ir.lab.native_skill_qualification import verify_qualification_evidence
from open_cake_ir.lab.provider_documents import ProviderQualificationReceipt, NATIVE_SKILL_QUALIFICATION_V1
from open_cake_ir.lab.provider_policy import execution_configuration
from open_cake_ir.serialization import canonical_json_bytes
from tests.contracts import test_provider_qualification as fixtures


class NativeQualificationReceiptTests(unittest.TestCase):
    def test_old_receipts_never_infer_native_input_capability(self):
        base = ProviderQualificationReceipt('fixture', 'a'*64, 'b'*64, True, True, True,
                                            True, 'zero_gpu_contract_fixture_only')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'receipt.json'
            for version, receipt in ((1, base), (2, replace(base, system_skills_sha256='c'*64)),
                (3, replace(base, system_skills_sha256='c'*64,
                            native_skill_input_contract=NATIVE_SKILL_QUALIFICATION_V1))):
                path.write_bytes(canonical_json_bytes(receipt.document))
                loaded = ProviderQualificationReceipt.load(path)
                self.assertEqual(loaded.document, receipt.document)
                self.assertEqual(loaded.document['schema_version'], version)
                self.assertEqual(loaded.native_skill_input_contract,
                                 NATIVE_SKILL_QUALIFICATION_V1 if version == 3 else None)
            for capability in (None, 'unknown', True):
                document = dict(receipt.document, native_skill_input_contract=capability)
                path.write_bytes(canonical_json_bytes(document))
                with self.assertRaises(ValueError): ProviderQualificationReceipt.load(path)
            with self.assertRaises(ValueError):
                replace(base, native_skill_input_contract=NATIVE_SKILL_QUALIFICATION_V1)


class NativeQualificationAdmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        executable = self.root/'codex'
        helper = fixtures.ProviderQualificationContractTests(methodName='runTest')
        helper._write_provider(executable)
        package = self.root/'skills.tar'
        with tarfile.open(package, 'w') as archive:
            for name, data in (
                ('SKILL.md', b'---\nname: cake\ndescription: fixture\n---\nRead references.\n'),
                ('scripts/check.py', b"print('fixture')\n"),
                ('assets/table.bin', bytes(range(256))),
            ):
                item = tarfile.TarInfo('skills/cake/' + name)
                item.size, item.mode = len(data), 0o644
                archive.addfile(item, io.BytesIO(data))
        auth = self.root/'auth.json'
        auth.write_bytes(b'fixture credential')
        auth.chmod(0o600)
        completed, receipt, anchor, evidence_root = helper._run_qualification(
            self.root, executable, provider_revision='native-admission-fixture', run_id='qualification',
            author_skill_package=package, auth_source=auth)
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        self.receipt = ProviderQualificationReceipt.load(receipt)
        self.anchor = json.loads(anchor.read_bytes())
        self.store = EvidenceStore.open(evidence_root)
        self.authority = self.store.replay_authority('qualification')
        self.events = self.store.replay_events('qualification')
        self.counter = 0

    def synthetic_archive(self, *, capability=NATIVE_SKILL_QUALIFICATION_V1,
                          authority_scope='live_two_turn_current_provider', mutate=None):
        """A newly sealed counterfactual, never relabel or repair the original fixture."""
        self.counter += 1
        directory = self.root/f'synthetic-{self.counter}'
        directory.mkdir()
        receipt = replace(self.receipt, scope='live_two_turn_current_provider',
                          native_skill_input_contract=capability)
        authority = deepcopy(self.authority)
        authority['qualification_scope'] = authority_scope
        events = deepcopy(list(self.events))
        endpoint = deepcopy(self.store.audit_run('qualification').endpoint)
        endpoint.update(qualification_scope=receipt.scope, qualification_receipt_sha256=receipt.canonical_sha256)
        if mutate: mutate(authority, events, endpoint)
        store = EvidenceStore.create(directory/'evidence')
        identity = sha256(canonical_json_bytes(authority)).hexdigest()
        ledger = store.start_run('qualification', authority_sha256=identity, authority=authority)
        for event in events:
            if event['kind'] == 'run_terminal': continue
            payload = event['payload']
            if 'objects' in payload:
                payload['objects'] = [store.put(
                    canonical_json_bytes(receipt.document) if ref['role'] == 'qualification_receipt'
                    else self.store.read_object(ref), media_type=ref['media_type']).reference(ref['role'])
                    for ref in payload['objects']]
            ledger.append(event['kind'], payload)
        ledger.seal(protocol_adherence='adhered', endpoint_observation='qualified', endpoint=endpoint)
        audit = store.audit_run('qualification')
        self.assertTrue(audit.archive_integrity)
        self.assertTrue(audit.filesystem_custody_verified)
        anchor = dict(self.anchor, evidence_root=str(store.root), authority_sha256=identity,
                      qualification_receipt_sha256=receipt.canonical_sha256,
                      terminal_seal_sha256=audit.terminal_seal_sha256)
        receipt_path, anchor_path = directory/'receipt.json', directory/'anchor.json'
        receipt_path.write_bytes(canonical_json_bytes(receipt.document))
        anchor_path.write_bytes(canonical_json_bytes(anchor))
        provider = {key: authority[key] for key in (
            'model', 'reasoning_effort', 'service_tier', 'removed_environment', 'sandbox',
            'reference_visibility', 'disabled_features', 'code_mode_host', 'submission_contract',
            'author_home_policy', 'native_skill_package', 'web_search')}
        provider.update(revision=receipt.provider_revision, executable_sha256=receipt.executable_sha256,
            system_skills_sha256=receipt.system_skills_sha256, cwd_policy='independent_task_workspace',
            output_schema={'path': authority['native_skill_context']['output_schema'],
                           'sha256': authority['output_schema_sha256']},
            qualification={'path': str(receipt_path), 'canonical_sha256': receipt.canonical_sha256},
            qualification_anchor={'path': str(anchor_path),
                                  'canonical_sha256': sha256(canonical_json_bytes(anchor)).hexdigest()})
        return receipt, anchor, provider

    def admit(self, provider):
        return validate_provider_binding(provider=provider, project_root=fixtures.ROOT,
            expected_provider_configuration=execution_configuration(provider),
            admitted_scopes={'live_two_turn_current_provider'})

    def test_fixture_scope_is_preserved_and_retained_inputs_reconstruct(self):
        self.assertEqual(self.receipt.native_skill_input_contract, NATIVE_SKILL_QUALIFICATION_V1)
        self.assertEqual(self.receipt.scope, 'zero_gpu_contract_fixture_only')
        with patch('pathlib.Path.open', side_effect=AssertionError('reopened original qualification files')):
            result = verify_qualification_evidence(qualification=self.receipt, anchor=self.anchor)
        self.assertEqual(set(result), {'open_cake', 'direct_cuda'})
        self.assertEqual([len(turns) for turns in result.values()], [2, 2])

    def test_admission_reopens_evidence_and_checks_semantics(self):
        receipt, anchor, provider = self.synthetic_archive()
        self.assertEqual(self.admit(provider), receipt)
        for field, value in (('authority_sha256', 'a'*64), ('terminal_seal_sha256', 'b'*64)):
            changed = dict(anchor, **{field: value})
            path = self.root/(field + '.json')
            path.write_bytes(canonical_json_bytes(changed))
            provider2 = dict(provider, qualification_anchor={'path': str(path),
                'canonical_sha256': sha256(canonical_json_bytes(changed)).hexdigest()})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'unverified'):
                self.admit(provider2)

    def test_old_capability_and_scope_substitution_refuse_at_their_own_rules(self):
        _, _, provider = self.synthetic_archive(capability=None)
        with self.assertRaisesRegex(ValueError, 'capability is not qualified'):
            self.admit(provider)
        _, _, provider = self.synthetic_archive(authority_scope='zero_gpu_contract_fixture_only')
        with self.assertRaisesRegex(ValueError, 'receipt authority differs'):
            self.admit(provider)

    def test_integrity_without_historical_custody_cannot_qualify(self):
        receipt, anchor, provider = self.synthetic_archive()
        copied = self.root/'copied-evidence'
        shutil.copytree(anchor['evidence_root'], copied)
        audit = EvidenceStore.open(copied).audit_run('qualification')
        self.assertTrue(audit.archive_integrity)
        self.assertFalse(audit.filesystem_custody_verified)
        changed = dict(anchor, evidence_root=str(copied))
        path = self.root/'copied-anchor.json'
        path.write_bytes(canonical_json_bytes(changed))
        provider['qualification_anchor'] = {'path': str(path),
            'canonical_sha256': sha256(canonical_json_bytes(changed)).hexdigest()}
        with self.assertRaisesRegex(ValueError, 'custody.*unverified'):
            self.admit(provider)

    def test_sealed_archive_with_wrong_semantics_is_refused(self):
        for variant in ('missing_observation', 'duplicate_observation', 'endpoint', 'foreign_context'):
            def mutate(authority, events, endpoint):
                observation = next(event for event in events if event['kind'] == 'provider_qualification_observed')
                if variant == 'missing_observation': events.remove(observation)
                elif variant == 'duplicate_observation': events.insert(-1, deepcopy(observation))
                elif variant == 'endpoint': endpoint['update_observed'] = False
                else: authority['native_skill_context']['arms']['open_cake']['cwd'] = '/foreign/task'
            _, _, provider = self.synthetic_archive(mutate=mutate)
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                self.admit(provider)

    def test_authority_reader_refuses_foreign_id_and_symlink(self):
        self.assertEqual(self.store.replay_authority('qualification'), self.authority)
        with self.assertRaises(ValueError): self.store.replay_authority('../qualification')
        copied = self.root/'untrusted-copy'
        shutil.copytree(self.store.root, copied)
        path = copied/'runs/qualification/authority.json'
        # Deliberately corrupt this new negative fixture; never repair source custody.
        path.parent.chmod(0o750)
        path.unlink()
        path.symlink_to(self.store.root/'runs/qualification/authority.json')
        with self.assertRaises((ValueError, OSError)):
            EvidenceStore.open(copied).replay_authority('qualification')
