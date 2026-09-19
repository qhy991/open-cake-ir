"""Narrow Lab document primitives; their existing admission semantics are unchanged."""

from __future__ import annotations

from open_cake_ir.serialization import canonical_json_bytes as _canonical_json_bytes

import json
import re
from pathlib import Path, PurePosixPath
from typing import Mapping, cast


_DIGEST = re.compile(r"^[0-9a-f]{64}$")


_VERB = re.compile(r"\bdiffers?\b")


def differs(context: str, *, expected: object, observed: object) -> ValueError:
    """The refusal for a comparison that failed: it says both sides.

    `context` is the leading text of the message and stays the words a reader searches
    for; the two values follow it, so the reader learns what the document carried and
    what the rule wanted without re-running the check. A context that already carries
    its verb ("Executor Nsight Compute bytes differ", "X differs from Y") is kept
    byte-identical; otherwise " differs" is appended. The error is returned, not
    raised, so a call site reads `raise differs(...)`.
    """
    message = context if _VERB.search(context) else f"{context} differs"
    return ValueError(f"{message}: expected {expected!r}, observed {observed!r}")


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
