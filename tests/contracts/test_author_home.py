"""A new author Run cannot inherit user skills or another Run's session home."""
from pathlib import Path
import tempfile
import unittest

from open_cake_ir.lab.author_home import (provision_codex_home, system_skills_snapshot,
                                         verify_codex_home)


class IsolatedCodexHomeTests(unittest.TestCase):
    def test_new_home_contains_only_the_private_credential(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root/'source-auth.json'
            source.write_bytes(b'fixture credential')
            source.chmod(0o600)
            home = provision_codex_home(source, root/'author-home')
            self.assertEqual({item.name for item in home.iterdir()}, {'auth.json'})
            self.assertEqual((home/'auth.json').stat().st_mode & 0o777, 0o600)
            self.assertEqual(home.stat().st_mode & 0o777, 0o700)
            self.assertEqual(verify_codex_home(home), home)
            with self.assertRaisesRegex(ValueError, 'new canonical'):
                provision_codex_home(source, home)

    def test_user_skill_and_plugin_injection_is_refused_between_turns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root/'source-auth.json'
            source.write_bytes(b'fixture credential')
            source.chmod(0o600)
            home = provision_codex_home(source, root/'author-home')
            (home/'skills').mkdir()
            (home/'skills'/'.system').mkdir()
            with self.assertRaisesRegex(ValueError, 'prior skill state'):
                verify_codex_home(home, fresh=True)
            verify_codex_home(home)
            baseline = system_skills_snapshot(home)
            (home/'skills'/'.system'/'injected').mkdir()
            (home/'skills'/'.system'/'injected'/'SKILL.md').write_text('unreviewed')
            with self.assertRaisesRegex(ValueError, 'changed between Turns'):
                verify_codex_home(home, expected_system_skills=baseline)
            (home/'skills'/'.system'/'injected'/'SKILL.md').unlink()
            (home/'skills'/'.system'/'injected').rmdir()
            (home/'skills'/'user-skill').mkdir()
            with self.assertRaisesRegex(ValueError, 'user skills'):
                verify_codex_home(home)
            (home/'skills'/'user-skill').rmdir()
            (home/'skills'/'.system'/'link').symlink_to(source)
            with self.assertRaisesRegex(ValueError, 'link or special'):
                verify_codex_home(home)
            (home/'skills'/'.system'/'link').unlink()
            (home/'plugins').mkdir()
            with self.assertRaisesRegex(ValueError, 'plugins'):
                verify_codex_home(home)

    def test_symlinked_credential_source_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root/'source-auth.json'
            source.write_bytes(b'fixture credential')
            source.chmod(0o600)
            alias = root/'alias.json'
            alias.symlink_to(source)
            with self.assertRaisesRegex(ValueError, 'source path'):
                provision_codex_home(alias, root/'author-home')
            self.assertFalse((root/'author-home').exists())
