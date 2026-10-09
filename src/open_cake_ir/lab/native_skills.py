"""One frozen native-skill material, projected without installing its dependencies.

The archive is the authority; its extracted files and human-readable inventory are
derived views. File preparation does not qualify native discovery or model delivery.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile

from open_cake_ir.serialization import canonical_json_bytes

MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_CONTENT_BYTES = 32 * 1024 * 1024
MAX_FILES = 2048
MAX_SKILLS = 32


def _regular_bytes(path: Path, limit: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        before = os.fstat(descriptor)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or not 0 < before.st_size <= limit):
            raise ValueError('native skill material is not a bounded regular file')
        chunks, remaining = [], before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError('native skill material changed during read')
            chunks.append(chunk)
            remaining -= len(chunk)
        identity = lambda item: (item.st_dev, item.st_ino, item.st_size,
                                 item.st_mtime_ns, item.st_ctime_ns, item.st_mode)
        if (os.read(descriptor, 1) or identity(before) != identity(os.fstat(descriptor))
            or identity(before) != identity(path.lstat())):
            raise ValueError('native skill material changed during read')
        return b''.join(chunks)
    finally:
        os.close(descriptor)


def validate_skill_reference(reference) -> dict:
    if (not isinstance(reference, Mapping) or set(reference) != {'path', 'sha256'}
        or not isinstance(reference['path'], str) or not reference['path']
        or '\x00' in reference['path'] or '\\' in reference['path']
        or '..' in PurePosixPath(reference['path']).parts
        or not isinstance(reference['sha256'], str)
        or re.fullmatch('[a-f0-9]{64}', reference['sha256']) is None):
        raise ValueError('native skill package reference differs')
    return dict(reference)


def author_skill_reference(authoring) -> dict | None:
    """The Run's existing authoring authority owns the one material selection."""
    from .author_home import ISOLATED_SKILL_PACKAGE_V1
    provider = authoring.get('provider', {})
    if not isinstance(provider, Mapping):
        raise ValueError('native skill provider declaration differs')
    selected = provider.get('author_home_policy') == ISOLATED_SKILL_PACKAGE_V1
    if selected != ('native_skill_package' in provider):
        raise ValueError('native skill material and author home policy differ')
    if not selected:
        return None
    if (provider.get('harness', 'codex') != 'codex'
        or authoring.get('reference_access') != 'known_kernel_reproduction'):
        raise ValueError('native skill packages require Codex known-kernel authoring')
    return validate_skill_reference(provider['native_skill_package'])


@dataclass(frozen=True)
class NativeSkillFile:
    path: str
    payload: bytes
    executable: bool


@dataclass(frozen=True, init=False)
class NativeSkillPackage:
    """Validated complete archive snapshot, shared by preparation and invocation."""
    _reference: bytes
    raw_bytes: bytes
    entry_names: tuple[str, ...]
    files: tuple[NativeSkillFile, ...]
    directories: tuple[str, ...]

    @property
    def reference(self) -> dict:
        return json.loads(self._reference)

    @classmethod
    def bind(cls, project_root, path) -> dict:
        return cls.read(project_root, path).reference

    @classmethod
    def read(cls, project_root, path) -> 'NativeSkillPackage':
        """Read an explicit input once before snapshotting or freezing its reference."""
        from .bindings import source_reference_path
        location, resolved = source_reference_path(project_root, str(path), 'native skill package')
        raw = _regular_bytes(resolved, MAX_ARCHIVE_BYTES)
        # Establish the one external-material handoff identity, not per-file hashes.
        reference = {'path': location, 'sha256': sha256(raw).hexdigest()}
        return cls._parse(raw, reference)

    @classmethod
    def load(cls, project_root, reference) -> 'NativeSkillPackage':
        from .bindings import source_reference_path
        reference = validate_skill_reference(reference)
        _, path = source_reference_path(project_root, reference['path'], 'native skill package')
        return cls.from_bytes(_regular_bytes(path, MAX_ARCHIVE_BYTES), reference)

    @classmethod
    def from_bytes(cls, raw: bytes, reference) -> 'NativeSkillPackage':
        reference = validate_skill_reference(reference)
        if (not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_ARCHIVE_BYTES
            or sha256(raw).hexdigest() != reference['sha256']):
            raise ValueError('native skill package differs from its frozen material')
        return cls._parse(raw, reference)

    @classmethod
    def _parse(cls, raw: bytes, reference: dict) -> 'NativeSkillPackage':
        files, explicit_directories, names, total = [], set(), set(), 0
        try:
            # Uncompressed tar only: bounded input and no archive extraction API.
            with tarfile.open(fileobj=io.BytesIO(raw), mode='r:') as archive:
                for index, member in enumerate(archive):
                    if index >= MAX_FILES:
                        raise ValueError('native skill package has too many entries')
                    name = member.name.rstrip('/') if member.isdir() else member.name
                    path = PurePosixPath(name)
                    if (not name or path.is_absolute() or path.as_posix() != name
                        or any(part in {'.', '..'} for part in path.parts)
                        or '\\' in name or any(ord(char) < 32 or ord(char) == 127 for char in name)
                        or len(name.encode()) > 1024
                        or name in names or member.mode & 0o7000
                        or member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE}):
                        raise ValueError('native skill package contains an unsafe or duplicate entry')
                    names.add(name)
                    if path.parts[0] != 'skills':
                        raise ValueError('native skill package entries must belong to declared skills')
                    if member.isdir():
                        explicit_directories.add(name)
                        continue
                    if not 0 <= member.size <= MAX_FILE_BYTES:
                        raise ValueError('native skill package file exceeds its bound')
                    total += member.size
                    if total > MAX_CONTENT_BYTES:
                        raise ValueError('native skill package exceeds its content bound')
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError('native skill package file is unreadable')
                    payload = stream.read(MAX_FILE_BYTES + 1)
                    if len(payload) != member.size:
                        raise ValueError('native skill package file size differs')
                    files.append(NativeSkillFile(name, payload, bool(member.mode & 0o111)))
                if any(raw[archive.offset:]):
                    raise ValueError('native skill package contains data after its tar end')
        except (tarfile.TarError, OSError) as error:
            raise ValueError('native skill package is not a supported tar material') from error
        skills = sorted({PurePosixPath(name).parts[1] for name in names
                         if len(PurePosixPath(name).parts) > 1})
        if (not 0 < len(skills) <= MAX_SKILLS
            or any(not name.strip() or len(name.encode()) > 128 for name in skills)):
            raise ValueError('native skill package entry labels differ')
        files_by_name = {item.path: item for item in files}
        for name in skills:
            entry = files_by_name.get(f'skills/{name}/SKILL.md')
            if entry is None:
                raise ValueError('native skill package lacks a declared SKILL.md')
            try:
                text = entry.payload.decode('utf-8')
            except UnicodeError as error:
                raise ValueError('native SKILL.md is not UTF-8') from error
            if not text.strip():
                raise ValueError('native SKILL.md is empty')
            # YAML semantics and canonical skill names belong to the native loader.
            # A package entry label is not evidence that Codex accepts that skill.
        directories = {'skills'} | explicit_directories
        for name in explicit_directories:
            directories.update(str(parent) for parent in PurePosixPath(name).parents if str(parent) != '.')
        for item in files:
            parts = PurePosixPath(item.path).parts
            if len(parts) < 3 or parts[1] not in skills:
                raise ValueError('native skill package has undeclared skill files')
            directories.update(str(parent) for parent in PurePosixPath(item.path).parents if str(parent) != '.')
        for name in directories:
            parts = PurePosixPath(name).parts
            if (name in files_by_name or parts[0] != 'skills'
                or len(parts) > 1 and parts[1] not in skills):
                raise ValueError('native skill package has undeclared or shadowed directories')
        result = object.__new__(cls)
        for field, value in (
            ('_reference', canonical_json_bytes(reference)), ('raw_bytes', raw),
            ('entry_names', tuple(skills)), ('files', tuple(sorted(files, key=lambda item: item.path))),
            ('directories', tuple(sorted(directories))),
        ):
            object.__setattr__(result, field, value)
        return result

    def inventory(self) -> dict:
        """Non-secret derived metadata; no second file identity catalogue."""
        return {'package_entries': list(self.entry_names),
                'files': [{'path': item.path, 'bytes': len(item.payload), 'executable': item.executable}
                          for item in self.files],
                'native_discovery_and_delivery': 'unverified',
                'dependency_availability': 'unverified', 'additional_tool_grants': []}
