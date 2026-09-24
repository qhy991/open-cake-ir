"""Fresh Codex author state, separate from task files and other Runs.

This controls CLI skills and persistent session state. It is not a filesystem read
jail: clean-start still needs its independent provider read-isolation qualification.
"""
from __future__ import annotations

import os
import stat
from hashlib import sha256
from pathlib import Path

from open_cake_ir.serialization import canonical_json_bytes

ISOLATED_AUTH_ONLY_V1 = 'isolated_auth_only_v1'


def _regular_private(path: Path, *, maximum_bytes: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        info = os.fstat(descriptor)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
            or info.st_nlink != 1 or info.st_mode & 0o077
            or not 0 < info.st_size <= maximum_bytes):
            raise ValueError('Codex credential custody differs')
        value = os.read(descriptor, info.st_size + 1)
        if len(value) != info.st_size:
            raise ValueError('Codex credential changed while provisioning')
        return value
    finally:
        os.close(descriptor)


def verify_auth_source(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError('Codex credential source path differs')
    _regular_private(path, maximum_bytes=1024 * 1024)
    return path


def provision_codex_home(auth_source: Path, destination: Path) -> Path:
    """Copy only the credential into one new private author home."""
    credential = _regular_private(verify_auth_source(auth_source), maximum_bytes=1024 * 1024)
    destination = Path(destination)
    if (not destination.is_absolute() or '..' in destination.parts
        or destination.exists() or destination.is_symlink()
        or destination.parent.resolve(strict=True) != destination.parent):
        raise ValueError('isolated Codex home must be a new canonical external directory')
    destination.mkdir(mode=0o700)
    auth = destination/'auth.json'
    descriptor = os.open(auth, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(credential):
            offset += os.write(descriptor, credential[offset:])
    finally:
        os.close(descriptor)
    return verify_codex_home(destination)


def _system_skills_snapshot(system: Path) -> tuple[tuple[str, int, str], ...]:
    """Capture the CLI's system-skill tree and refuse links or writable entries."""
    root_info = system.lstat()
    if (not stat.S_ISDIR(root_info.st_mode) or root_info.st_uid != os.geteuid()
        or root_info.st_mode & 0o022):
        raise ValueError('isolated Codex system skills custody differs')
    rows = []
    total_bytes = 0
    pending = [system]
    while pending:
        directory = pending.pop()
        for path in directory.iterdir():
            info = path.lstat()
            relative = path.relative_to(system).as_posix()
            if stat.S_ISLNK(info.st_mode):
                raise ValueError('isolated Codex system skills contain a link or special file')
            if info.st_uid != os.geteuid() or info.st_mode & 0o022:
                raise ValueError('isolated Codex system skills custody differs')
            if stat.S_ISDIR(info.st_mode):
                rows.append((relative + '/', info.st_mode & 0o777, ''))
                pending.append(path)
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                total_bytes += info.st_size
                if info.st_size > 16 * 1024 * 1024 or total_bytes > 64 * 1024 * 1024:
                    raise ValueError('isolated Codex system skills exceed the bounded tree')
                descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
                try:
                    before = os.fstat(descriptor)
                    identity = lambda value: (value.st_dev, value.st_ino, value.st_size,
                                              value.st_mtime_ns, value.st_ctime_ns, value.st_mode)
                    if identity(before) != identity(info):
                        raise ValueError('isolated Codex system skill changed while checking')
                    digest = sha256()
                    while chunk := os.read(descriptor, 1024 * 1024):
                        digest.update(chunk)
                    if (identity(os.fstat(descriptor)) != identity(before)
                        or identity(path.lstat()) != identity(before)):
                        raise ValueError('isolated Codex system skill changed while checking')
                finally:
                    os.close(descriptor)
                rows.append((relative, info.st_mode & 0o777, digest.hexdigest()))
            else:
                raise ValueError('isolated Codex system skills contain a link or special file')
    return tuple(sorted(rows))


def system_skills_snapshot(home: Path) -> tuple[tuple[str, int, str], ...]:
    system = Path(home)/'skills'/'.system'
    return _system_skills_snapshot(system) if system.is_dir() else ()


def system_skills_identity(snapshot: tuple[tuple[str, int, str], ...]) -> str:
    """One identity for the post-first-Turn tree admitted by qualification."""
    return sha256(canonical_json_bytes(snapshot)).hexdigest()


def verify_codex_home(home: Path, *, fresh: bool = False,
                      expected_system_skills: tuple[tuple[str, int, str], ...] | None = None) -> Path:
    """Refuse unsafe credential custody and skill-tree drift before every Turn."""
    home = Path(home)
    info = home.lstat()
    if (not home.is_absolute() or home.resolve(strict=True) != home
        or not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
        or info.st_mode & 0o077):
        raise ValueError('isolated Codex home custody differs')
    _regular_private(home/'auth.json', maximum_bytes=1024 * 1024)
    skills = home/'skills'
    if skills.exists() or skills.is_symlink():
        if fresh:
            raise ValueError('fresh Codex author home contains prior skill state')
        if (skills.is_symlink() or not skills.is_dir()
            or {item.name for item in skills.iterdir()} - {'.system'}):
            raise ValueError('isolated Codex home contains user skills')
        skills_info = skills.lstat()
        if skills_info.st_uid != os.geteuid() or skills_info.st_mode & 0o022:
            raise ValueError('isolated Codex skills directory custody differs')
        system = skills/'.system'
        if system.is_symlink():
            raise ValueError('isolated Codex system skills are symlinked')
        if system.exists() and not system.is_dir():
            raise ValueError('isolated Codex system skills differ')
        observed = system_skills_snapshot(home)
    else:
        observed = ()
    if expected_system_skills is not None and observed != expected_system_skills:
        raise ValueError('isolated Codex system skills changed between Turns')
    if (home/'plugins').exists() or (home/'plugins').is_symlink():
        raise ValueError('isolated Codex home contains plugins')
    return home
