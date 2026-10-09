"""CPU material/projection fixtures, not native skill discovery qualification."""
from __future__ import annotations

import io
import os
from pathlib import Path
import stat
import tarfile
import tempfile
import unittest
from unittest import mock

from open_cake_ir.lab import native_skills
from open_cake_ir.lab.author_home import (
    provision_codex_home,
    provision_user_home,
    verify_codex_home,
    verify_user_home,
)
from open_cake_ir.lab.native_skills import NativeSkillPackage


SKILL = (b'---\r\nname: "native-example"\r\ndescription: >-\r\n'
         b'  Native package fixture.\r\nmetadata:\r\n  purpose: cpu-only\r\n'
         b'---\r\nRead scripts/check.py and assets/table.bin.\r\n')
SCRIPT = b'#!/usr/bin/env python3\nprint("fixture only; never executed")\n'
BINARY = b'\x00\xff\x80\r\n\x00\x01'


def member(name, payload=b'', *, kind=tarfile.REGTYPE, mode=0o644, linkname=''):
    info = tarfile.TarInfo(name)
    info.type, info.mode, info.linkname = kind, mode, linkname
    info.size = len(payload) if info.isfile() else 0
    return info, payload


def archive_bytes(entries):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w', format=tarfile.PAX_FORMAT) as archive:
        for info, payload in entries:
            archive.addfile(info, io.BytesIO(payload) if info.isfile() else None)
    return output.getvalue()


def complete_entries():
    # The directory label need not equal the native front-matter name. The Lab
    # preserves these bytes; only native discovery can establish skill validity.
    return [member('skills/example/SKILL.md', SKILL),
            member('skills/example/scripts/check.py', SCRIPT, mode=0o755),
            member('skills/example/assets/table.bin', BINARY),
            member('skills/example/assets/empty.bin'),
            member('skills/example/examples/empty/', kind=tarfile.DIRTYPE, mode=0o755)]


class NativeSkillMaterialTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.project = self.root / 'project'
        self.project.mkdir()
        self.sequence = 0

    def source(self, entries=None, *, raw=None):
        self.sequence += 1
        path = self.root / f'skills-{self.sequence}.tar'
        path.write_bytes(raw if raw is not None else archive_bytes(
            complete_entries() if entries is None else entries))
        return path

    def package(self, entries=None):
        path = self.source(entries)
        reference = NativeSkillPackage.bind(self.project, path)
        return NativeSkillPackage.load(self.project, reference)

    def test_complete_archive_preserves_native_header_scripts_binary_and_empty_directories(self):
        path = self.source()
        reference = NativeSkillPackage.bind(self.project, path)
        package = NativeSkillPackage.load(self.project, reference)
        self.assertEqual(package.raw_bytes, path.read_bytes())
        self.assertEqual(package.entry_names, ('example',))
        files = {item.path: item for item in package.files}
        self.assertEqual(files['skills/example/SKILL.md'].payload, SKILL)
        self.assertEqual(files['skills/example/scripts/check.py'].payload, SCRIPT)
        self.assertTrue(files['skills/example/scripts/check.py'].executable)
        self.assertEqual(files['skills/example/assets/table.bin'].payload, BINARY)
        self.assertFalse(files['skills/example/assets/table.bin'].executable)
        self.assertEqual(files['skills/example/assets/empty.bin'].payload, b'')
        self.assertIn('skills/example/examples/empty', package.directories)
        detached = package.reference
        detached['path'] = '/different/package.tar'
        self.assertEqual(package.reference, reference)
        inventory = package.inventory()
        self.assertEqual(inventory['native_discovery_and_delivery'], 'unverified')
        self.assertEqual(inventory['dependency_availability'], 'unverified')
        self.assertEqual(inventory['additional_tool_grants'], [])

    def test_complete_package_can_contain_multiple_native_entries_without_a_dependency_manifest(self):
        entries = complete_entries() + [
            member('skills/second/SKILL.md', b'Native instructions are opaque to this Lab.\n'),
            member('skills/second/references/note.md', b'An ordinary bundled reference.\n')]
        package = self.package(entries)
        self.assertEqual(package.entry_names, ('example', 'second'))
        self.assertEqual(len(package.files), 6)

    def test_nested_empty_directory_materializes_without_explicit_parent_entries(self):
        package = self.package([
            member('skills/cake/SKILL.md', b'CPU fixture instructions.\n'),
            member('skills/cake/cache/empty/', kind=tarfile.DIRTYPE, mode=0o755),
        ])
        home = provision_user_home(self.root / 'nested-empty-home', package)
        cache = home / '.agents/skills/cake/cache'
        self.assertEqual({path.name for path in cache.iterdir()}, {'empty'})
        self.assertTrue((cache / 'empty').is_dir())
        self.assertEqual(list((cache / 'empty').iterdir()), [])
        self.assertEqual(stat.S_IMODE(cache.stat().st_mode), 0o500)
        self.assertEqual(verify_user_home(home, package), home)

    def test_file_cannot_shadow_an_implicit_parent_of_an_empty_directory(self):
        path = self.source([
            member('skills/cake/SKILL.md', b'CPU fixture instructions.\n'),
            member('skills/cake/cache', b'A file cannot contain the empty directory.'),
            member('skills/cake/cache/empty/', kind=tarfile.DIRTYPE, mode=0o755),
        ])
        with self.assertRaisesRegex(ValueError, 'shadowed'):
            NativeSkillPackage.bind(self.project, path)

    def test_replacing_the_bound_archive_is_refused_at_the_material_handoff(self):
        path = self.source()
        reference = NativeSkillPackage.bind(self.project, path)
        replacement = complete_entries()
        replacement[2] = member('skills/example/assets/table.bin', b'changed')
        path.write_bytes(archive_bytes(replacement))
        with self.assertRaisesRegex(ValueError, 'frozen material'):
            NativeSkillPackage.load(self.project, reference)

    def test_links_and_nonregular_source_materials_are_not_admitted(self):
        source = self.source()
        symlink = self.root / 'symlink.tar'
        symlink.symlink_to(source)
        hardlink = self.root / 'hardlink.tar'
        os.link(source, hardlink)
        directory = self.root / 'directory.tar'
        directory.mkdir()
        for path in (symlink, hardlink, directory):
            with self.subTest(path=path.name), self.assertRaises((OSError, ValueError)):
                NativeSkillPackage.bind(self.project, path)

    def test_unsafe_archive_entries_are_refused_without_extracting(self):
        cases = {
            'traversal': member('skills/example/../../outside', b'x'),
            'absolute': member('/skills/example/outside', b'x'),
            'noncanonical': member('skills/example/./outside', b'x'),
            'backslash': member('skills/example/other\\outside', b'x'),
            'wrong-root': member('plugins/example/data', b'x'),
            'duplicate': member('skills/example/SKILL.md', SKILL),
            'symlink': member('skills/example/link', kind=tarfile.SYMTYPE, linkname='/outside'),
            'hardlink': member('skills/example/link', kind=tarfile.LNKTYPE,
                               linkname='skills/example/SKILL.md'),
            'fifo': member('skills/example/pipe', kind=tarfile.FIFOTYPE),
            'character-device': member('skills/example/device', kind=tarfile.CHRTYPE),
            'block-device': member('skills/example/block', kind=tarfile.BLKTYPE),
            'privileged-mode': member('skills/example/script', b'x', mode=0o4755),
            'directory-shadows-file': member('skills/example/SKILL.md/', kind=tarfile.DIRTYPE),
            'file-shadows-parent': member('skills/example/assets', b'x'),
            'undeclared-root-file': member('skills/loose-file', b'x'),
        }
        for name, unsafe in cases.items():
            with self.subTest(case=name):
                path = self.source(complete_entries() + [unsafe])
                with self.assertRaises(ValueError):
                    NativeSkillPackage.bind(self.project, path)
        self.assertFalse((self.root / 'outside').exists())
        self.assertFalse((self.root / 'skills').exists())

    def test_missing_empty_or_non_utf8_entry_documents_are_refused(self):
        cases = {
            'missing': [member('skills/example/assets/table.bin', BINARY)],
            'directory': [member('skills/example/SKILL.md/', kind=tarfile.DIRTYPE)],
            'empty': [member('skills/example/SKILL.md')],
            'non-utf8': [member('skills/example/SKILL.md', b'\xff\x80')],
            'empty-package': [],
        }
        for name, entries in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                NativeSkillPackage.bind(self.project, self.source(entries))

    def test_corrupt_and_compressed_archives_are_not_supported(self):
        compressed = io.BytesIO()
        with tarfile.open(fileobj=compressed, mode='w:gz') as archive:
            info, payload = member('skills/example/SKILL.md', SKILL)
            archive.addfile(info, io.BytesIO(payload))
        for raw in (b'not a tar archive', compressed.getvalue()):
            with self.subTest(compressed=raw.startswith(b'\x1f\x8b')):
                with self.assertRaises(ValueError):
                    NativeSkillPackage.bind(self.project, self.source(raw=raw))

    def test_input_entry_file_content_and_entry_count_budgets_are_enforced(self):
        # Small fixture limits exercise boundary refusal without large allocations.
        skill = member('skills/example/SKILL.md', SKILL)
        cases = (
            ('MAX_ARCHIVE_BYTES', 1, [skill]),
            ('MAX_FILES', 1, [skill, member('skills/example/resource', b'x')]),
            ('MAX_FILE_BYTES', len(SKILL), [skill, member('skills/example/resource', b'x' * (len(SKILL) + 1))]),
            ('MAX_CONTENT_BYTES', len(SKILL), [skill, member('skills/example/resource', b'x')]),
            ('MAX_SKILLS', 1, [skill, member('skills/second/SKILL.md', b'Other native instructions.\n')]),
        )
        for constant, limit, entries in cases:
            path = self.source(entries)
            with (self.subTest(bound=constant), mock.patch.object(native_skills, constant, limit),
                  self.assertRaises(ValueError)):
                NativeSkillPackage.bind(self.project, path)

    def test_file_and_content_budgets_accept_exact_limit(self):
        entries = [member('skills/example/SKILL.md', SKILL)]
        with (mock.patch.object(native_skills, 'MAX_FILE_BYTES', len(SKILL)),
              mock.patch.object(native_skills, 'MAX_CONTENT_BYTES', len(SKILL)),
              mock.patch.object(native_skills, 'MAX_FILES', 1),
              mock.patch.object(native_skills, 'MAX_SKILLS', 1)):
            package = self.package(entries)
        self.assertEqual(package.files[0].payload, SKILL)

    def projection(self):
        package = self.package()
        home = provision_user_home(self.root / f'home-{self.sequence}', package)
        return home, package

    def test_projection_preserves_complete_bytes_and_executable_intent(self):
        home, package = self.projection()
        self.assertEqual(stat.S_IMODE(home.stat().st_mode), 0o700)
        self.assertEqual({path.name for path in home.iterdir()}, {'.agents'})
        agents = home / '.agents'
        self.assertEqual(stat.S_IMODE(agents.stat().st_mode), 0o500)
        for directory in package.directories:
            self.assertTrue((agents / directory).is_dir())
            self.assertEqual(stat.S_IMODE((agents / directory).stat().st_mode), 0o500)
        for item in package.files:
            path = agents / item.path
            self.assertEqual(path.read_bytes(), item.payload)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o500 if item.executable else 0o400)
        self.assertEqual(verify_user_home(home, package), home)
        with self.assertRaisesRegex(ValueError, 'new canonical'):
            provision_user_home(home, package)

    def test_distinct_runs_have_separate_user_and_codex_homes(self):
        package = self.package()
        auth = self.root / 'auth-source.json'
        auth.write_bytes(b'cpu fixture credential')
        auth.chmod(0o600)
        user_homes, codex_homes = [], []
        for run in ('first', 'second'):
            user_homes.append(provision_user_home(self.root / f'{run}-user-home', package))
            codex_homes.append(provision_codex_home(auth, self.root / f'{run}-codex-home'))
        self.assertEqual(len(set(user_homes + codex_homes)), 4)
        for home in codex_homes:
            self.assertEqual(verify_codex_home(home, fresh=True), home)
            self.assertEqual({item.name for item in home.iterdir()}, {'auth.json'})
        first_file = user_homes[0] / '.agents/skills/example/assets/table.bin'
        second_file = user_homes[1] / '.agents/skills/example/assets/table.bin'
        self.assertNotEqual((first_file.stat().st_dev, first_file.stat().st_ino),
                            (second_file.stat().st_dev, second_file.stat().st_ino))
        first_file.chmod(0o600)
        first_file.write_bytes(b'changed')
        first_file.chmod(0o400)
        with self.assertRaises(ValueError):
            verify_user_home(user_homes[0], package)
        self.assertEqual(second_file.read_bytes(), BINARY)
        self.assertEqual(verify_user_home(user_homes[1], package), user_homes[1])

    def test_projection_drift_is_refused_before_a_later_turn(self):
        cases = ('bytes', 'missing-file', 'extra-file', 'extra-directory', 'file-mode',
                 'directory-mode', 'symlink', 'hardlink', 'missing-empty-directory',
                 'directory-symlink', 'home-mode')
        for case in cases:
            with self.subTest(case=case):
                home, package = self.projection()
                agents = home / '.agents'
                path = agents / 'skills/example/assets/table.bin'
                # Only these disposable fixture files are modified. Restoring a
                # test directory's mode isolates content/link drift from mode drift.
                if case == 'bytes':
                    path.chmod(0o600)
                    path.write_bytes(b'X' * len(BINARY))
                    path.chmod(0o400)
                elif case in {'missing-file', 'extra-file', 'extra-directory', 'symlink', 'hardlink'}:
                    parent = path.parent
                    parent.chmod(0o700)
                    if case in {'missing-file', 'symlink', 'hardlink'}:
                        path.unlink()
                    if case == 'extra-file':
                        (parent / 'undeclared').write_bytes(b'not in the package')
                        (parent / 'undeclared').chmod(0o400)
                    elif case == 'extra-directory':
                        (parent / 'undeclared').mkdir(mode=0o500)
                    elif case in {'symlink', 'hardlink'}:
                        outside = self.root / f'outside-{self.sequence}'
                        outside.write_bytes(BINARY)
                        outside.chmod(0o400)
                        if case == 'symlink':
                            path.symlink_to(outside)
                        else:
                            os.link(outside, path)
                    parent.chmod(0o500)
                elif case == 'file-mode':
                    path.chmod(0o600)
                elif case == 'directory-mode':
                    path.parent.chmod(0o700)
                elif case in {'missing-empty-directory', 'directory-symlink'}:
                    empty = agents / 'skills/example/examples/empty'
                    empty.parent.chmod(0o700)
                    empty.rmdir()
                    if case == 'directory-symlink':
                        outside = self.root / f'outside-directory-{self.sequence}'
                        outside.mkdir(mode=0o500)
                        empty.symlink_to(outside, target_is_directory=True)
                    empty.parent.chmod(0o500)
                elif case == 'home-mode':
                    home.chmod(0o755)
                with self.assertRaises((OSError, ValueError)):
                    verify_user_home(home, package)

    def test_other_package_or_symlinked_home_is_not_the_frozen_projection(self):
        home, package = self.projection()
        entries = complete_entries()
        entries[2] = member('skills/example/assets/table.bin', b'changed')
        other = self.package(entries)
        with self.assertRaises(ValueError):
            verify_user_home(home, other)
        alias = self.root / 'home-alias'
        alias.symlink_to(home, target_is_directory=True)
        with self.assertRaises(ValueError):
            verify_user_home(alias, package)


if __name__ == '__main__':
    unittest.main()
