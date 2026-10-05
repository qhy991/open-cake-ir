"""Bind task-owned Python starters without introducing another Schedule language."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from open_cake_ir.compiler import frontend


def read_skeleton(path: Path) -> dict:
    """Use the existing frontend for Python; preserve the canonical JSON replay path."""
    return frontend.read_schedule(path).document


def bind_python_reference(source: str, prepared: Mapping[str, object], *, filename: str) -> bytes:
    """Keep the author's Python untouched while Lab owns its Workload binding.

    Shape or arithmetic specialization belongs in the task-owned Python starter.
    The prepared Schedule carries the Lab's binding; the author does not copy
    its content hash into a decorator. Other metadata must still match.
    """
    original = frontend.parse(source, filename=filename).document
    if {k: v for k, v in original.items() if k != "metadata"} != {
        k: v for k, v in prepared.items() if k != "metadata"
    }:
        raise ValueError("Python starter must already match the prepared Schedule; only metadata may be bound")
    authored_metadata = original["metadata"]
    prepared_metadata = prepared["metadata"]
    if (not isinstance(prepared_metadata, Mapping)
            or {key: value for key, value in authored_metadata.items()
                if key != "workload_contract_sha256"}
            != {key: value for key, value in prepared_metadata.items()
                if key != "workload_contract_sha256"}
            or ("workload_contract_sha256" in authored_metadata
                and authored_metadata["workload_contract_sha256"]
                != prepared_metadata.get("workload_contract_sha256"))):
        raise ValueError("Python starter metadata differs from the prepared Schedule")
    return source.encode("utf-8")


def read_skeleton_reference(project_root, reference, context='Schedule skeleton'):
    """Read and seal the actual delivered source against its frozen reference."""
    import json
    from hashlib import sha256
    from .bindings import source_reference_path
    from ._documents import _object, _digest, _canonical_json_bytes, differs

    reference = _object(reference, context)
    if set(reference) != {'path', 'canonical_sha256'}:
        raise ValueError(f'{context} reference fields differ')
    _, path = source_reference_path(project_root, reference['path'], context)
    payload = path.read_bytes()
    document = (frontend.parse(payload.decode('utf-8'), filename=str(path)).document
                if path.suffix == '.py' else json.loads(payload))
    expected = _digest(reference['canonical_sha256'], context)
    observed = sha256(_canonical_json_bytes(document)).hexdigest()
    if expected != observed:
        raise differs(f'{context} bytes differ', expected=expected, observed=observed)
    return payload, document
