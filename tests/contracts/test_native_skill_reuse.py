"""Reuse is checked against real retained CPU-fixture inputs, never a live model."""
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.lab.native_skill_qualification import verify_qualification_evidence
from open_cake_ir.lab.provider_documents import ProviderQualificationReceipt
from tests.contracts import test_provider_qualification as fixtures
from tools import launch_task


class NativeSkillQualificationReuseTests(unittest.TestCase):
    def qualification(self, root, missing=None):
        helper = fixtures.ProviderQualificationContractTests(methodName='runTest')
        executable = root/'codex'
        helper._write_provider(executable)
        # The fixture always requests cake. A second body can be observed without
        # inventing another declaration of qualification coverage. Its directory
        # deliberately differs from its native name.
        source = executable.read_text()
        old = 'append_rollout(thread_id, arguments,'
        self.assertEqual(source.count(old), 1)
        executable.write_text(source.replace(old, old +
            f" extra_skill=('cake:metal', 'metal-folder', (arm, 2 if resumed else 1) != {missing!r}),"))
        package = root/'skills.tar'
        with tarfile.open(package, 'w') as archive:
            for name, data in (
                ('cake/SKILL.md', b'---\nname: cake\ndescription: fixture\n---\nRead references.\n'),
                ('cake/scripts/check.py', b"print('fixture')\n"),
                ('cake/assets/table.bin', bytes(range(256))),
                ('metal-folder/SKILL.md', b'---\nname: cake:metal\ndescription: fixture\n---\nExtra body.\n'),
            ):
                item = tarfile.TarInfo('skills/' + name)
                item.size, item.mode = len(data), 0o644
                archive.addfile(item, io.BytesIO(data))
        auth = root/'auth.json'
        auth.write_bytes(b'fixture credential')
        auth.chmod(0o600)
        completed, receipt, anchor, _ = helper._run_qualification(root, executable,
            provider_revision='native-reuse-fixture', run_id='qualification',
            author_skill_package=package, auth_source=auth)
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        return receipt, anchor, package

    def test_reuse_covers_observed_names_without_reopening_original_homes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            receipt_path, anchor_path, package = self.qualification(root)
            receipt = ProviderQualificationReceipt.load(receipt_path)
            anchor = json.loads(anchor_path.read_bytes())
            with patch('pathlib.Path.open', side_effect=AssertionError('reopened original files')):
                result = verify_qualification_evidence(qualification=receipt, anchor=anchor,
                                                       requested_names=['cake:metal', 'cake'])
            self.assertEqual(set(result), {'open_cake', 'direct_cuda'})
            self.assertTrue(all(len(turns) == 2 for turns in result.values()))
            self.assertEqual(receipt.scope, 'zero_gpu_contract_fixture_only')
            args = SimpleNamespace(author_skill_package=package, native_skill_name=['cake:metal'],
                qualification=receipt_path, qualification_anchor=anchor_path)
            with patch.object(launch_task.subprocess, 'run') as process:
                self.assertEqual(launch_task._qualify(fixtures.ROOT, root/'unwritten', args, None, None),
                                 (receipt_path, anchor_path))
                args.native_skill_name = ['not-loaded']
                with self.assertRaisesRegex(ValueError, 'open_cake/initial/not-loaded'):
                    launch_task._qualify(fixtures.ROOT, root/'unwritten', args, None, None)
                process.assert_not_called()
            self.assertFalse((root/'unwritten').exists())

    def test_every_arm_and_current_turn_must_cover_the_reused_selection(self):
        for arm in ('open_cake', 'direct_cuda'):
            for turn, phase in ((1, 'initial'), (2, 'resumed')):
                with self.subTest(arm=arm, turn=turn), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory).resolve()
                    receipt_path, anchor_path, _ = self.qualification(root, missing=(arm, turn))
                    receipt = ProviderQualificationReceipt.load(receipt_path)
                    anchor = json.loads(anchor_path.read_bytes())
                    # Original selection remains valid and sealed. Only the extra
                    # request is refused, at this rule rather than another gate.
                    verify_qualification_evidence(qualification=receipt, anchor=anchor,
                                                  requested_names=['cake'])
                    with self.assertRaisesRegex(ValueError, f'{arm}/{phase}/cake:metal'):
                        verify_qualification_evidence(qualification=receipt, anchor=anchor,
                                                      requested_names=['cake:metal'])

    def test_invalid_requested_names_refuse_before_opening_evidence(self):
        for names in ([], ['cake', 'cake'], ['bad name'], ['x']*33):
            with self.subTest(names=names), patch('open_cake_ir.evidence.EvidenceStore.open') as opened:
                with self.assertRaisesRegex(ValueError, 'native skill selection differs'):
                    verify_qualification_evidence(qualification=None, anchor=None, requested_names=names)
                opened.assert_not_called()


if __name__ == '__main__':
    unittest.main()
