"""Bind task-owned Python starters without introducing another Schedule language."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Mapping

from open_cake_ir.compiler import frontend


def read_skeleton(path: Path) -> dict:
    """Use the existing frontend for Python; preserve the canonical JSON replay path."""
    return frontend.read_schedule(path).document


def bind_python_reference(source: str, prepared: Mapping[str, object], *, filename: str) -> bytes:
    """Only Lab-owned metadata may differ from the task's explicit Python program.

    Shape or arithmetic specialization belongs in the task-owned Python starter.
    This function refuses changes it cannot express by binding metadata alone.
    """
    original = frontend.parse(source, filename=filename).document
    if {k: v for k, v in original.items() if k != "metadata"} != {
        k: v for k, v in prepared.items() if k != "metadata"
    }:
        raise ValueError("Python starter must already match the prepared Schedule; only metadata may be bound")
    tree = ast.parse(source, filename=filename)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    decorator = function.decorator_list[0]
    assert isinstance(decorator, ast.Call)  # frontend.parse already admitted this form
    value = ast.parse(repr(dict(prepared["metadata"])), mode="eval").body
    keyword = next((item for item in decorator.keywords if item.arg == "metadata"), None)
    if keyword is None:
        decorator.keywords.append(ast.keyword(arg="metadata", value=value))
    else:
        keyword.value = value
    bound = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    if frontend.parse(bound, filename=filename).document != prepared:
        raise ValueError("bound Python starter differs from the prepared Schedule")
    return bound.encode("utf-8")


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
