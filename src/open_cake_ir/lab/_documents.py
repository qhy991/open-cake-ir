"""Narrow Lab document primitives; their existing admission semantics are unchanged."""

from __future__ import annotations

from open_cake_ir.serialization import canonical_json_bytes as _canonical_json_bytes

import json
import re
from pathlib import Path, PurePosixPath
from typing import Mapping, cast


_DIGEST = re.compile(r"^[0-9a-f]{64}$")




def _object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return cast(Mapping[str, object], value)


def _name(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _digest(value: object, context: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA256 digest")
    return value


def _project_path(root: Path, value: object, context: str) -> tuple[str, Path]:
    relative = _name(value, context)
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or ".." in parsed.parts or "\\" in relative:
        raise ValueError(f"{context} is unsafe")
    path = (root / relative).resolve(strict=True)
    if root not in path.parents:
        raise ValueError(f"{context} escapes project root")
    return relative, path
