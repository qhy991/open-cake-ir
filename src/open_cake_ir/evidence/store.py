"""Secure content-addressed Evidence v2 with create-only event records."""

from __future__ import annotations

import fcntl
import json
import os
import re
import secrets
import stat
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping, cast

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_EVENT_FILE = re.compile(r"^([0-9]{12})\.json$")
_MEDIA_TYPE = re.compile(r"^[A-Za-z0-9.+-]+/[A-Za-z0-9.+-]+$")
_ROLE = re.compile(r"^[a-z][a-z0-9_]*$")
_PROTOCOL = {
    "adhered",
    "provider_fault",
    "harness_fault",
    "custody_violation",
    "contamination",
    "broker_fault",
}
_ENDPOINT = {"qualified", "no_qualified_candidate", "missing", "observed"}
_MAX_OBJECT_BYTES = 1 << 30
_SECRET_MARKERS = (
    b"BEGIN PRIVATE KEY",
    b"OPENAI_API_KEY=",
    b"CODEX_ACCESS_TOKEN=",
    b"INFINI_API_KEY=",
)
_SECRET_PATTERNS = (
    re.compile(rb"(?i)authorization\s*:\s*bearer\s+[a-z0-9._~+/=-]{8,}"),
    re.compile(rb"(?i)bearer\s+sk-[a-z0-9_-]{8,}"),
    re.compile(
        rb"(?i)(?:openai|anthropic|github|gitlab|azure|aws|codex|infini)"
        rb"[a-z0-9_-]{0,24}(?:key|token|secret)\s*[:=]\s*['\"]?[a-z0-9._~+/=-]{8,}"
    ),
    re.compile(rb"\b(?:ghp|github_pat|sk)-[a-zA-Z0-9_-]{8,}\b"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
)


def _contains_forbidden_secret(payload: bytes) -> bool:
    return any(marker in payload for marker in _SECRET_MARKERS) or any(
        pattern.search(payload) is not None for pattern in _SECRET_PATTERNS
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_line(value: object) -> bytes:
    return _canonical_json_bytes(value) + b"\n"


def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return cast(Mapping[str, object], value)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short evidence write")
        view = view[written:]


def _open_dir(parent_fd: int, name: str, *, create: bool = False) -> int:
    if "/" in name or name in {"", ".", ".."}:
        raise ValueError("evidence directory name is unsafe")
    if create:
        try:
            os.mkdir(name, 0o750, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except FileExistsError:
            pass
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_fd,
    )
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
        os.close(descriptor)
        raise ValueError(f"evidence directory {name!r} owner or type differs")
    if metadata.st_mode & 0o022:
        os.close(descriptor)
        raise ValueError(f"evidence directory {name!r} is group/other writable")
    return descriptor


def _read_regular_at(
    directory_fd: int,
    name: str,
    *,
    maximum_bytes: int = 64 * 1024 * 1024,
) -> bytes:
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size < 0
            or metadata.st_size > maximum_bytes
        ):
            raise ValueError(f"evidence file {name!r} type, link count, or size differs")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError(f"evidence file {name!r} ended early")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError(f"evidence file {name!r} grew during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _publish_new_at(directory_fd: int, name: str, payload: bytes, mode: int = 0o440) -> None:
    temporary = f".tmp-{os.getpid()}-{secrets.token_hex(8)}"
    descriptor = os.open(
        temporary,
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_fd,
    )
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)
    try:
        os.link(
            temporary,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        os.fsync(directory_fd)
    finally:
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _parse_canonical_json(payload: bytes, context: str) -> Mapping[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not valid UTF-8 JSON") from error
    document = _object(value, context)
    if payload != _canonical_line(document):
        raise ValueError(f"{context} bytes are not canonical JSON")
    return document


@dataclass(frozen=True)
class EvidenceObject:
    """One immutable CAS object."""

    sha256: str
    size_bytes: int
    media_type: str
    relative_path: str

    def reference(self, role: str) -> dict[str, object]:
        """Bind this object to one closed semantic role in an event."""

        if _ROLE.fullmatch(role) is None:
            raise ValueError("evidence object role is invalid")
        return {
            "role": role,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "relative_path": self.relative_path,
        }


@dataclass(frozen=True)
class EvidenceFinding:
    """One evidence-integrity finding."""

    code: str
    path: str
    message: str


@dataclass(frozen=True)
class RunAudit:
    """Pure replay of one Evidence v2 Run archive."""

    run_id: str
    authority_sha256: str | None
    integrity: bool
    event_count: int
    protocol_adherence: str | None
    endpoint_observation: str | None
    endpoint: Mapping[str, object] | None
    terminal_seal_sha256: str | None
    findings: tuple[EvidenceFinding, ...]


def _require_sealed_sequence(events: "tuple[Mapping[str, object], ...]") -> None:
    """What makes an event sequence a sealed Run's, rather than a prefix of one.

    Owned here because both the auditor and the reader need it and neither should hold
    its own copy: a reader that admitted a truncated sequence would let a claim be taken
    off a history whose end had been removed, and it would look complete.
    """

    if not events or events[-1].get("kind") != "run_terminal":
        raise ValueError("Run does not end in run_terminal")
    if sum(event.get("kind") == "run_terminal" for event in events) != 1:
        raise ValueError("Run must contain exactly one run_terminal event")


class EvidenceStore:
    """Read-only or writer-capable Evidence v2 root."""

    def __init__(self, root: Path, *, writable: bool) -> None:
        self.root = root
        self._writable = writable

    @classmethod
    def create(cls, root: str | Path) -> "EvidenceStore":
        """Create a new trusted Evidence v2 root."""

        path = Path(root).absolute()
        if path.exists() or path.is_symlink():
            raise ValueError("Evidence root must be new")
        parent = path.parent
        if parent.is_symlink() or parent.resolve(strict=True) != parent:
            raise ValueError("Evidence root ancestors must use canonical non-symlink paths")
        parent_fd = os.open(
            parent,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.mkdir(path.name, 0o750, dir_fd=parent_fd)
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        root_fd = os.open(
            path,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            objects_fd = _open_dir(root_fd, "objects", create=True)
            try:
                sha_fd = _open_dir(objects_fd, "sha256", create=True)
                os.close(sha_fd)
            finally:
                os.close(objects_fd)
            runs_fd = _open_dir(root_fd, "runs", create=True)
            os.close(runs_fd)
        finally:
            os.close(root_fd)
        return cls(path.resolve(strict=True), writable=True)

    @classmethod
    def open(cls, root: str | Path) -> "EvidenceStore":
        """Open an existing Evidence v2 root without creating or mutating it."""

        path = Path(root)
        if path.is_symlink() or not path.is_dir():
            raise ValueError("Evidence root is missing or symlinked")
        resolved = path.resolve(strict=True)
        root_fd = os.open(
            resolved,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            objects_fd = _open_dir(root_fd, "objects")
            try:
                sha_fd = _open_dir(objects_fd, "sha256")
                os.close(sha_fd)
            finally:
                os.close(objects_fd)
            runs_fd = _open_dir(root_fd, "runs")
            os.close(runs_fd)
        finally:
            os.close(root_fd)
        return cls(resolved, writable=False)

    @classmethod
    def writer(cls, root: str | Path) -> "EvidenceStore":
        """Open an existing trusted root for adding new immutable Runs/objects."""

        opened = cls.open(root)
        return cls(opened.root, writable=True)

    def _root_fd(self) -> int:
        descriptor = os.open(
            self.root,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o022
        ):
            os.close(descriptor)
            raise ValueError("Evidence root owner, type, or mode differs")
        return descriptor

    def put(self, payload: bytes, *, media_type: str) -> EvidenceObject:
        """Publish bounded non-secret bytes to CAS."""

        if not self._writable:
            raise ValueError("Evidence store is read-only")
        if (
            not isinstance(payload, bytes)
            or len(payload) > _MAX_OBJECT_BYTES
            or _MEDIA_TYPE.fullmatch(media_type) is None
        ):
            raise ValueError("evidence payload size or media_type differs")
        if _contains_forbidden_secret(payload):
            raise ValueError("evidence payload contains a forbidden secret marker")
        digest = sha256(payload).hexdigest()
        relative = f"objects/sha256/{digest[:2]}/{digest}"
        root_fd = self._root_fd()
        try:
            objects_fd = _open_dir(root_fd, "objects")
            try:
                sha_fd = _open_dir(objects_fd, "sha256")
                try:
                    prefix_fd = _open_dir(sha_fd, digest[:2], create=True)
                    try:
                        try:
                            _publish_new_at(prefix_fd, digest, payload)
                        except FileExistsError:
                            existing = _read_regular_at(
                                prefix_fd, digest, maximum_bytes=_MAX_OBJECT_BYTES
                            )
                            if existing != payload:
                                raise ValueError("existing CAS object bytes differ") from None
                    finally:
                        os.close(prefix_fd)
                finally:
                    os.close(sha_fd)
            finally:
                os.close(objects_fd)
        finally:
            os.close(root_fd)
        return EvidenceObject(digest, len(payload), media_type, relative)

    def start_run(
        self,
        run_id: str,
        *,
        authority_sha256: str,
        authority: Mapping[str, object],
    ) -> "RunLedger":
        """Create one Run with authority as the hash-chain genesis."""

        if not self._writable:
            raise ValueError("Evidence store is read-only")
        if _RUN_ID.fullmatch(run_id) is None or _DIGEST.fullmatch(authority_sha256) is None:
            raise ValueError("Run ID or authority digest is invalid")
        if sha256(_canonical_json_bytes(authority)).hexdigest() != authority_sha256:
            raise ValueError("Run authority bytes differ from authority digest")
        authority_document = {
            "schema_version": 2,
            "run_id": run_id,
            "authority_sha256": authority_sha256,
            "authority": dict(authority),
        }
        authority_record_hash = sha256(_canonical_json_bytes(authority_document)).hexdigest()
        authority_document["record_hash"] = authority_record_hash
        root_fd = self._root_fd()
        try:
            runs_fd = _open_dir(root_fd, "runs")
            try:
                os.mkdir(run_id, 0o750, dir_fd=runs_fd)
                os.fsync(runs_fd)
                run_fd = _open_dir(runs_fd, run_id)
                try:
                    events_fd = _open_dir(run_fd, "events", create=True)
                    os.close(events_fd)
                    _publish_new_at(
                        run_fd, "authority.json", _canonical_line(authority_document)
                    )
                    lock_fd = os.open(
                        "writer.lock",
                        os.O_CREAT
                        | os.O_EXCL
                        | os.O_RDWR
                        | getattr(os, "O_NOFOLLOW", 0),
                        0o600,
                        dir_fd=run_fd,
                    )
                    os.close(lock_fd)
                    os.fsync(run_fd)
                finally:
                    os.close(run_fd)
            finally:
                os.close(runs_fd)
        finally:
            os.close(root_fd)
        return RunLedger(self, run_id, authority_sha256, authority_record_hash)

    def audit_run(self, run_id: str) -> RunAudit:
        """Replay one Run without creating files or normalizing damaged bytes."""

        findings: list[EvidenceFinding] = []
        if _RUN_ID.fullmatch(run_id) is None:
            return RunAudit(
                run_id,
                None,
                False,
                0,
                None,
                None,
                None,
                None,
                (EvidenceFinding("RUN_ID_INVALID", "run_id", "Run ID is invalid"),),
            )
        authority_sha: str | None = None
        terminal_seal: str | None = None
        protocol: str | None = None
        endpoint_state: str | None = None
        endpoint: Mapping[str, object] | None = None
        events: list[Mapping[str, object]] = []
        root_fd: int | None = None
        runs_fd: int | None = None
        run_fd: int | None = None
        events_fd: int | None = None
        try:
            root_fd = self._root_fd()
            runs_fd = _open_dir(root_fd, "runs")
            run_fd = _open_dir(runs_fd, run_id)
            authority = _parse_canonical_json(
                _read_regular_at(run_fd, "authority.json"), "authority.json"
            )
            if set(authority) != {
                "schema_version",
                "run_id",
                "authority_sha256",
                "authority",
                "record_hash",
            } or authority.get("schema_version") != 2 or authority.get("run_id") != run_id:
                raise ValueError("authority fields, schema, or Run ID differ")
            authority_sha_value = authority.get("authority_sha256")
            if not isinstance(authority_sha_value, str) or _DIGEST.fullmatch(
                authority_sha_value
            ) is None:
                raise ValueError("authority digest is invalid")
            authority_preimage = dict(authority)
            authority_record_hash = authority_preimage.pop("record_hash", None)
            if authority_record_hash != sha256(
                _canonical_json_bytes(authority_preimage)
            ).hexdigest():
                raise ValueError("authority record hash differs")
            authority_payload = authority.get("authority")
            if authority_payload is not None and sha256(
                _canonical_json_bytes(authority_payload)
            ).hexdigest() != authority_sha_value:
                raise ValueError("embedded authority bytes differ from authority digest")
            authority_sha = authority_sha_value
            previous_hash = cast(str, authority_record_hash)

            events_fd = _open_dir(run_fd, "events")
            names = sorted(os.listdir(events_fd))
            for sequence, name in enumerate(names):
                match = _EVENT_FILE.fullmatch(name)
                if match is None or int(match.group(1)) != sequence:
                    raise ValueError("event filenames are not contiguous")
                event = _parse_canonical_json(
                    _read_regular_at(events_fd, name), f"events/{name}"
                )
                preimage = dict(event)
                record_hash = preimage.pop("record_hash", None)
                if (
                    set(event)
                    != {
                        "schema_version",
                        "run_id",
                        "authority_sha256",
                        "sequence",
                        "kind",
                        "payload",
                        "previous_hash",
                        "record_hash",
                    }
                    or event.get("schema_version") != 2
                    or event.get("run_id") != run_id
                    or event.get("authority_sha256") != authority_sha
                    or event.get("sequence") != sequence
                    or event.get("previous_hash") != previous_hash
                    or record_hash != sha256(_canonical_json_bytes(preimage)).hexdigest()
                ):
                    raise ValueError(f"event {sequence} chain or identity differs")
                payload = _object(event.get("payload"), f"events/{name}.payload")
                self._audit_object_references(payload, name, findings)
                events.append(event)
                previous_hash = cast(str, record_hash)

            terminal = _parse_canonical_json(
                _read_regular_at(run_fd, "terminal.json"), "terminal.json"
            )
            terminal_preimage = dict(terminal)
            terminal_seal_value = terminal_preimage.pop("seal_sha256", None)
            if (
                set(terminal)
                != {
                    "schema_version",
                    "run_id",
                    "authority_sha256",
                    "event_count",
                    "head_record_hash",
                    "seal_sha256",
                }
                or terminal.get("schema_version") != 2
                or terminal.get("run_id") != run_id
                or terminal.get("authority_sha256") != authority_sha
                or terminal.get("event_count") != len(events)
                or terminal.get("head_record_hash") != previous_hash
                or terminal_seal_value
                != sha256(_canonical_json_bytes(terminal_preimage)).hexdigest()
            ):
                raise ValueError("terminal seal identity or hash differs")
            terminal_seal = cast(str, terminal_seal_value)
            _require_sealed_sequence(events)
            terminal_payload = _object(events[-1].get("payload"), "run_terminal.payload")
            if set(terminal_payload) != {
                "protocol_adherence",
                "endpoint_observation",
                "endpoint",
            }:
                raise ValueError("run_terminal payload fields differ")
            protocol_value = terminal_payload.get("protocol_adherence")
            endpoint_value = terminal_payload.get("endpoint_observation")
            if protocol_value not in _PROTOCOL or endpoint_value not in _ENDPOINT:
                raise ValueError("run_terminal protocol or endpoint state differs")
            endpoint_data = terminal_payload.get("endpoint")
            if (
                endpoint_value in {"qualified", "no_qualified_candidate", "observed"}
                and endpoint_data is None
            ) or (endpoint_value == "missing" and endpoint_data is not None):
                raise ValueError("run_terminal endpoint payload differs from its state")
            if endpoint_data is not None:
                endpoint = _object(endpoint_data, "run_terminal.payload.endpoint")
            protocol = cast(str, protocol_value)
            endpoint_state = cast(str, endpoint_value)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
            findings.append(EvidenceFinding("RUN_ARCHIVE_INVALID", run_id, str(error)))
        finally:
            for descriptor in (events_fd, run_fd, runs_fd, root_fd):
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
        return RunAudit(
            run_id=run_id,
            authority_sha256=authority_sha,
            integrity=not findings,
            event_count=len(events),
            protocol_adherence=protocol,
            endpoint_observation=endpoint_state,
            endpoint=endpoint,
            terminal_seal_sha256=terminal_seal,
            findings=tuple(findings),
        )

    def replay_events(self, run_id: str) -> tuple[Mapping[str, object], ...]:
        """Read a sealed Run's canonical ordered event records without mutating the store.

        This is not a substitute for `audit_run`. The chain hashes, the terminal seal and
        the object references are that method's to verify, and a caller that needs to
        trust what it reads has to run it.

        What this does refuse is a sequence that is not a Run's: one that does not end in
        exactly one `run_terminal`. Without that check an archive missing its terminal
        event replayed as a complete, shorter Run -- the reader saw a plausible history
        with no sign that the end of it had been removed.
        """

        if _RUN_ID.fullmatch(run_id) is None:
            raise ValueError("Run ID is invalid")
        root_fd = self._root_fd()
        try:
            runs_fd = _open_dir(root_fd, "runs")
            try:
                run_fd = _open_dir(runs_fd, run_id)
                try:
                    events_fd = _open_dir(run_fd, "events")
                    try:
                        names = sorted(os.listdir(events_fd))
                        if names != [f"{index:012d}.json" for index in range(len(names))]:
                            raise ValueError("event files are not contiguous")
                        events = tuple(
                            _parse_canonical_json(
                                _read_regular_at(events_fd, name), f"events/{name}"
                            )
                            for name in names
                        )
                        _require_sealed_sequence(events)
                        return events
                    finally:
                        os.close(events_fd)
                finally:
                    os.close(run_fd)
            finally:
                os.close(runs_fd)
        finally:
            os.close(root_fd)

    def read_object(self, reference: Mapping[str, object]) -> bytes:
        """Read and rehash one role-bearing CAS reference."""

        if set(reference) != {
            "role",
            "sha256",
            "size_bytes",
            "media_type",
            "relative_path",
        }:
            raise ValueError("Evidence object reference fields differ")
        digest = reference.get("sha256")
        relative = reference.get("relative_path")
        size = reference.get("size_bytes")
        if (
            not isinstance(digest, str)
            or _DIGEST.fullmatch(digest) is None
            or relative != f"objects/sha256/{digest[:2]}/{digest}"
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or size > _MAX_OBJECT_BYTES
        ):
            raise ValueError("Evidence object reference identity differs")
        root_fd = self._root_fd()
        try:
            objects_fd = _open_dir(root_fd, "objects")
            try:
                sha_fd = _open_dir(objects_fd, "sha256")
                try:
                    prefix_fd = _open_dir(sha_fd, digest[:2])
                    try:
                        payload = _read_regular_at(
                            prefix_fd, digest, maximum_bytes=_MAX_OBJECT_BYTES
                        )
                    finally:
                        os.close(prefix_fd)
                finally:
                    os.close(sha_fd)
            finally:
                os.close(objects_fd)
        finally:
            os.close(root_fd)
        if len(payload) != size or sha256(payload).hexdigest() != digest:
            raise ValueError("Evidence object bytes differ")
        return payload

    def _audit_object_references(
        self,
        payload: Mapping[str, object],
        event_name: str,
        findings: list[EvidenceFinding],
    ) -> None:
        objects = payload.get("objects")
        if objects is None:
            return
        if not isinstance(objects, list):
            findings.append(
                EvidenceFinding(
                    "OBJECT_REFS_INVALID",
                    f"events/{event_name}.payload.objects",
                    "object references must be a list",
                )
            )
            return
        roles: set[str] = set()
        for index, value in enumerate(objects):
            path = f"events/{event_name}.payload.objects[{index}]"
            try:
                reference = _object(value, path)
                if set(reference) != {
                    "role",
                    "sha256",
                    "size_bytes",
                    "media_type",
                    "relative_path",
                }:
                    raise ValueError("object reference fields differ")
                role = reference.get("role")
                digest = reference.get("sha256")
                size = reference.get("size_bytes")
                media_type = reference.get("media_type")
                if (
                    not isinstance(role, str)
                    or _ROLE.fullmatch(role) is None
                    or role in roles
                    or not isinstance(digest, str)
                    or _DIGEST.fullmatch(digest) is None
                    or not isinstance(size, int)
                    or isinstance(size, bool)
                    or size < 0
                    or not isinstance(media_type, str)
                    or _MEDIA_TYPE.fullmatch(media_type) is None
                ):
                    raise ValueError("object role, digest, size, or media type differs")
                roles.add(role)
                expected_relative = f"objects/sha256/{digest[:2]}/{digest}"
                if reference.get("relative_path") != expected_relative:
                    raise ValueError("object path is not derived from digest")
                root_fd = self._root_fd()
                try:
                    objects_fd = _open_dir(root_fd, "objects")
                    try:
                        sha_fd = _open_dir(objects_fd, "sha256")
                        try:
                            prefix_fd = _open_dir(sha_fd, digest[:2])
                            try:
                                data = _read_regular_at(
                                    prefix_fd, digest, maximum_bytes=_MAX_OBJECT_BYTES
                                )
                            finally:
                                os.close(prefix_fd)
                        finally:
                            os.close(sha_fd)
                    finally:
                        os.close(objects_fd)
                finally:
                    os.close(root_fd)
                if sha256(data).hexdigest() != digest or len(data) != size:
                    raise ValueError("object bytes differ")
            except (OSError, ValueError) as error:
                findings.append(EvidenceFinding("OBJECT_INVALID", path, str(error)))


class RunLedger:
    """Single-writer append/seal Interface for one Evidence v2 Run."""

    def __init__(
        self,
        store: EvidenceStore,
        run_id: str,
        authority_sha256: str,
        authority_record_hash: str,
    ) -> None:
        self._store = store
        self.run_id = run_id
        self.authority_sha256 = authority_sha256
        self._authority_record_hash = authority_record_hash

    def _run_fds(self) -> tuple[int, int, int]:
        root_fd = self._store._root_fd()
        runs_fd = _open_dir(root_fd, "runs")
        run_fd = _open_dir(runs_fd, self.run_id)
        os.close(runs_fd)
        return root_fd, run_fd, os.open(
            "writer.lock",
            os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=run_fd,
        )

    def append(self, kind: str, payload: Mapping[str, object]) -> str:
        """Append one event atomically under the Run writer lock."""

        if not isinstance(kind, str) or not kind or kind == "run_terminal":
            raise ValueError("use seal() for run_terminal")
        return self._append_locked(kind, payload, terminal=False)

    def _append_locked(
        self,
        kind: str,
        payload: Mapping[str, object],
        *,
        terminal: bool,
    ) -> str:
        root_fd, run_fd, lock_fd = self._run_fds()
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                _read_regular_at(run_fd, "terminal.json")
            except FileNotFoundError:
                pass
            else:
                raise ValueError("Run is already sealed")
            events_fd = _open_dir(run_fd, "events")
            try:
                names = sorted(os.listdir(events_fd))
                sequence = len(names)
                expected_names = [f"{index:012d}.json" for index in range(sequence)]
                if names != expected_names:
                    raise ValueError("event files are not contiguous before append")
                previous_hash = self._authority_record_hash
                if names:
                    previous = _parse_canonical_json(
                        _read_regular_at(events_fd, names[-1]), f"events/{names[-1]}"
                    )
                    previous_hash_value = previous.get("record_hash")
                    if not isinstance(previous_hash_value, str):
                        raise ValueError("event head hash is missing")
                    previous_hash = previous_hash_value
                    if previous.get("kind") == "run_terminal":
                        if (
                            not terminal
                            or kind != "run_terminal"
                            or previous.get("payload") != dict(payload)
                        ):
                            raise ValueError("Run already contains a terminal event")
                        terminal_preimage = {
                            "schema_version": 2,
                            "run_id": self.run_id,
                            "authority_sha256": self.authority_sha256,
                            "event_count": sequence,
                            "head_record_hash": previous_hash,
                        }
                        terminal_document = dict(terminal_preimage)
                        terminal_document["seal_sha256"] = sha256(
                            _canonical_json_bytes(terminal_preimage)
                        ).hexdigest()
                        _publish_new_at(
                            run_fd,
                            "terminal.json",
                            _canonical_line(terminal_document),
                        )
                        return previous_hash
                preimage = {
                    "schema_version": 2,
                    "run_id": self.run_id,
                    "authority_sha256": self.authority_sha256,
                    "sequence": sequence,
                    "kind": kind,
                    "payload": dict(payload),
                    "previous_hash": previous_hash,
                }
                record_hash = sha256(_canonical_json_bytes(preimage)).hexdigest()
                event = dict(preimage)
                event["record_hash"] = record_hash
                _publish_new_at(
                    events_fd, f"{sequence:012d}.json", _canonical_line(event)
                )
                if terminal:
                    terminal_preimage = {
                        "schema_version": 2,
                        "run_id": self.run_id,
                        "authority_sha256": self.authority_sha256,
                        "event_count": sequence + 1,
                        "head_record_hash": record_hash,
                    }
                    seal_sha = sha256(
                        _canonical_json_bytes(terminal_preimage)
                    ).hexdigest()
                    terminal_document = dict(terminal_preimage)
                    terminal_document["seal_sha256"] = seal_sha
                    _publish_new_at(
                        run_fd, "terminal.json", _canonical_line(terminal_document)
                    )
                return record_hash
            finally:
                os.close(events_fd)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
            os.close(run_fd)
            os.close(root_fd)

    def seal(
        self,
        *,
        protocol_adherence: str,
        endpoint_observation: str,
        endpoint: Mapping[str, object] | None = None,
    ) -> str:
        """Append the sole terminal event and atomically publish its seal."""

        if protocol_adherence not in _PROTOCOL or endpoint_observation not in _ENDPOINT:
            raise ValueError("terminal protocol or endpoint state is invalid")
        if endpoint_observation in {"qualified", "no_qualified_candidate", "observed"} and endpoint is None:
            raise ValueError("observed endpoint requires endpoint data")
        if endpoint_observation == "missing" and endpoint is not None:
            raise ValueError("missing endpoint cannot carry endpoint data")
        return self._append_locked(
            "run_terminal",
            {
                "protocol_adherence": protocol_adherence,
                "endpoint_observation": endpoint_observation,
                "endpoint": dict(endpoint) if endpoint is not None else None,
            },
            terminal=True,
        )
