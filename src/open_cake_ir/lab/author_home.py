"""Fresh Codex author state, separate from task files and other Runs.

This controls CLI skills and persistent session state. It is not a filesystem read
jail: clean-start still needs its independent provider read-isolation qualification.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

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


def verify_codex_home(home: Path) -> Path:
    """Refuse a changed credential or user-installed skills before every Turn."""
    home = Path(home)
    info = home.lstat()
    if (not home.is_absolute() or home.resolve(strict=True) != home
        or not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
        or info.st_mode & 0o077):
        raise ValueError('isolated Codex home custody differs')
    _regular_private(home/'auth.json', maximum_bytes=1024 * 1024)
    skills = home/'skills'
    if skills.exists() or skills.is_symlink():
        if (skills.is_symlink() or not skills.is_dir()
            or {item.name for item in skills.iterdir()} - {'.system'}):
            raise ValueError('isolated Codex home contains user skills')
        system = skills/'.system'
        if system.is_symlink():
            raise ValueError('isolated Codex system skills are symlinked')
    if (home/'plugins').exists() or (home/'plugins').is_symlink():
        raise ValueError('isolated Codex home contains plugins')
    return home
