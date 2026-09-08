"""Prospective writer-origin witnesses outside an Evidence archive.

The local OS, registry and writer UID are trusted. These records distinguish native
publication from copies/grafts; they do not attest adversarial same-UID history.
No inspection path creates or restamps a witness. Existing event hashes are referenced,
not recomputed or mirrored into another payload store.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat
import uuid

MARKER = ".writer-custody"
ENVIRONMENT = "OPEN_CAKE_CUSTODY_DIRECTORY"
_IDENTIFIER = re.compile(r"[0-9a-f]{32}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def external_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError("custody path must be absolute and canonical without symlinks")
    if any((parent / ".git").exists() for parent in (path, *path.parents)):
        raise ValueError("custody paths must remain outside every Git checkout")
    return path


def identity(metadata: os.stat_result, *, ctime: bool = True) -> dict:
    result = {"device": metadata.st_dev, "inode": metadata.st_ino, "uid": metadata.st_uid}
    if ctime:
        result["ctime_ns"] = metadata.st_ctime_ns
    return result


def _identity_matches(value, metadata, *, ctime=True):
    expected = identity(metadata, ctime=ctime)
    return isinstance(value, dict) and set(value) == set(expected) and all(
        type(item) is int and item >= 0 for item in value.values()) and value == expected


def _private_dir(path: Path, *, create=False) -> int:
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    metadata = os.fstat(descriptor)
    if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        os.close(descriptor)
        raise ValueError("external custody directory must be private to its owner")
    return descriptor


def _regular(metadata, mode):
    return (stat.S_ISREG(metadata.st_mode) and metadata.st_uid == os.geteuid()
            and metadata.st_nlink == 1 and stat.S_IMODE(metadata.st_mode) == mode)


def _read(directory_fd: int, name: str) -> dict:
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
    try:
        metadata = os.fstat(descriptor)
        if not _regular(metadata, 0o400) or metadata.st_size > 8192:
            raise ValueError("external custody record protection or size differs")
        payload = os.read(descriptor, 8193)
        value = json.loads(payload)
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode() + b"\n"
        if len(payload) != metadata.st_size or payload != canonical or not isinstance(value, dict):
            raise ValueError("external custody record is not a canonical object")
        return value
    finally:
        os.close(descriptor)


def _publish(directory_fd: int, name: str, value: dict) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode() + b"\n"
    if len(payload) > 8192:
        raise ValueError("external custody record exceeds its bound")
    descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400, dir_fd=directory_fd)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short custody record write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(directory_fd)


class WriterCustody:
    def __init__(self, root: Path):
        self.root = root
        self.directory = Path(os.environ.get(ENVIRONMENT,
            str(Path.home() / ".local/share/open-cake-ir/custody-anchors")))

    def _paths(self):
        external_path(self.root)
        registry = external_path(self.directory)
        if registry == self.root or self.root in registry.parents or registry in self.root.parents:
            raise ValueError("custody registry and archive must occupy separate paths")
        return registry

    def _marker(self, root_fd):
        descriptor = os.open(MARKER, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
        try:
            metadata = os.fstat(descriptor)
            payload = os.read(descriptor, 34)
            if not _regular(metadata, 0o600) or len(payload) != 33 or payload[-1:] != b"\n":
                raise ValueError("original private store marker differs")
            identifier = payload[:-1].decode("ascii")
            if _IDENTIFIER.fullmatch(identifier) is None:
                raise ValueError("store marker is not a registry identifier")
            return identifier, metadata
        finally:
            os.close(descriptor)

    def create(self, root_fd):
        registry = self._paths()
        descriptor = _private_dir(registry, create=True)
        try:
            identifier = uuid.uuid4().hex
            os.mkdir(identifier, 0o700, dir_fd=descriptor)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        marker = os.open(MARKER, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=root_fd)
        try:
            if os.write(marker, (identifier + "\n").encode()) != 33:
                raise OSError("short store marker write")
            os.fsync(marker)
            marker_metadata = os.fstat(marker)
        finally:
            os.close(marker)
        os.fsync(root_fd)
        namespace = _private_dir(registry / identifier)
        try:
            os.mkdir("runs", 0o700, dir_fd=namespace)
            _publish(namespace, "origin.json", {"schema_version": 1, "root": str(self.root),
                "store_id": identifier, "root_identity": identity(os.fstat(root_fd)),
                "marker_identity": identity(marker_metadata)})
        finally:
            os.close(namespace)

    def _namespace(self, root_fd):
        registry = self._paths()
        identifier, marker = self._marker(root_fd)
        registry_fd = _private_dir(registry)
        os.close(registry_fd)
        namespace = _private_dir(registry / identifier)
        try:
            origin = _read(namespace, "origin.json")
            if (set(origin) != {"schema_version", "root", "store_id", "root_identity", "marker_identity"}
                    or type(origin["schema_version"]) is not int or origin["schema_version"] != 1
                    or origin["root"] != str(self.root) or origin["store_id"] != identifier
                    or not _identity_matches(origin["root_identity"], os.fstat(root_fd))
                    or not _identity_matches(origin["marker_identity"], marker)):
                raise ValueError("original store custody identity differs")
            return registry / identifier
        finally:
            os.close(namespace)

    def store_verified(self, root_fd):
        try:
            self._namespace(root_fd)
            return True
        except (OSError, ValueError, UnicodeError, RecursionError):
            return False

    def _run_path(self, root_fd, run_id):
        namespace = self._namespace(root_fd)
        runs = _private_dir(namespace / "runs")
        os.close(runs)
        return namespace / "runs" / run_id

    @staticmethod
    def _lock(run_fd):
        descriptor = os.open("writer.lock", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=run_fd)
        try:
            metadata = os.fstat(descriptor)
            if not _regular(metadata, 0o600) or metadata.st_size != 0:
                raise ValueError("original private Run writer lock differs")
            return metadata
        finally:
            os.close(descriptor)

    def register_run(self, root_fd, run_fd, run_id, authority_record_hash):
        path = self._run_path(root_fd, run_id)
        parent = _private_dir(path.parent)
        try:
            os.mkdir(run_id, 0o700, dir_fd=parent)
            os.fsync(parent)
        finally:
            os.close(parent)
        descriptor = _private_dir(path)
        try:
            os.mkdir("frontiers", 0o700, dir_fd=descriptor)
            _publish(descriptor, "origin.json", {"run_id": run_id, "authority_record_hash": authority_record_hash,
                "run_identity": identity(os.fstat(run_fd), ctime=False), "lock_identity": identity(self._lock(run_fd))})
        finally:
            os.close(descriptor)
        self.advance(root_fd, run_fd, run_id, authority_record_hash, 0, authority_record_hash)

    def _run(self, root_fd, run_fd, run_id, authority_record_hash):
        path = self._run_path(root_fd, run_id)
        descriptor = _private_dir(path)
        try:
            origin = _read(descriptor, "origin.json")
            if (set(origin) != {"run_id", "authority_record_hash", "run_identity", "lock_identity"}
                    or origin["run_id"] != run_id or origin["authority_record_hash"] != authority_record_hash
                    or not _identity_matches(origin["run_identity"], os.fstat(run_fd), ctime=False)
                    or not _identity_matches(origin["lock_identity"], self._lock(run_fd))):
                raise ValueError("Run writer origin differs from its custody registration")
            return path
        finally:
            os.close(descriptor)

    def frontier(self, root_fd, run_fd, run_id, authority_record_hash):
        path = self._run(root_fd, run_fd, run_id, authority_record_hash)
        directory = _private_dir(path / "frontiers")
        try:
            names = sorted(os.listdir(directory))
            if not names or names != [f"{n:012d}.json" for n in range(len(names))]:
                raise ValueError("external writer frontiers are missing or unordered")
            value = _read(directory, names[-1])
            if (set(value) != {"event_count", "head_record_hash"} or type(value["event_count"]) is not int
                    or value["event_count"] != len(names) - 1 or not isinstance(value["head_record_hash"], str)
                    or _DIGEST.fullmatch(value["head_record_hash"]) is None):
                raise ValueError("external writer frontier differs")
            return value
        finally:
            os.close(directory)

    def advance(self, root_fd, run_fd, run_id, authority_record_hash, event_count, head_record_hash):
        path = self._run(root_fd, run_fd, run_id, authority_record_hash)
        directory = _private_dir(path / "frontiers")
        try:
            _publish(directory, f"{event_count:012d}.json", {"event_count": event_count, "head_record_hash": head_record_hash})
        finally:
            os.close(directory)

    def sealed(self, root_fd, run_fd, run_id, authority_record_hash):
        directory = _private_dir(self._run(root_fd, run_fd, run_id, authority_record_hash))
        try:
            try:
                return _read(directory, "seal.json")
            except FileNotFoundError:
                return None
        finally:
            os.close(directory)

    def seal(self, root_fd, run_fd, run_id, authority_record_hash, event_count, head_record_hash, terminal_seal):
        directory = _private_dir(self._run(root_fd, run_fd, run_id, authority_record_hash))
        try:
            terminal = os.stat("terminal.json", dir_fd=run_fd, follow_symlinks=False)
            _publish(directory, "seal.json", {"event_count": event_count, "head_record_hash": head_record_hash,
                "terminal_seal": terminal_seal, "terminal_identity": identity(terminal)})
        finally:
            os.close(directory)

    def run_verified(self, root_fd, run_fd, run_id, authority_record_hash, event_count, head_record_hash, terminal_seal, heads):
        try:
            frontier = self.frontier(root_fd, run_fd, run_id, authority_record_hash)
            directory = _private_dir(self._run(root_fd, run_fd, run_id, authority_record_hash) / "frontiers")
            try:
                for count, head in enumerate(heads):
                    witness = _read(directory, f"{count:012d}.json")
                    if (type(witness.get("event_count")) is not int
                            or witness != {"event_count": count, "head_record_hash": head}):
                        return False
            finally:
                os.close(directory)
            seal = self.sealed(root_fd, run_fd, run_id, authority_record_hash)
            terminal = os.stat("terminal.json", dir_fd=run_fd, follow_symlinks=False)
            return (frontier == {"event_count": event_count, "head_record_hash": head_record_hash}
                and isinstance(seal, dict) and set(seal) == {"event_count", "head_record_hash", "terminal_seal", "terminal_identity"}
                and type(seal["event_count"]) is int and seal["event_count"] == event_count
                and seal["head_record_hash"] == head_record_hash and seal["terminal_seal"] == terminal_seal
                and _identity_matches(seal["terminal_identity"], terminal))
        except (OSError, ValueError, UnicodeError, RecursionError):
            return False
