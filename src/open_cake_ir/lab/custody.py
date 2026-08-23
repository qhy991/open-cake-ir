"""Campaign output custody shared by the Lab and its CLI."""

from __future__ import annotations

import os
from pathlib import Path


def admit_new_campaign_path(
    project_root: str | Path,
    requested_path: str | Path,
    *,
    role: str,
) -> Path:
    """Resolve one create-only Campaign path outside the project checkout."""

    root = Path(project_root).resolve(strict=True)
    absolute = Path(os.path.abspath(requested_path))
    lexical_parent = absolute.parent
    resolved_parent = lexical_parent.resolve(strict=True)
    for parent in (lexical_parent, resolved_parent):
        if parent == root or root in parent.parents:
            raise ValueError(f"{role} must be outside the project checkout")
    target = resolved_parent / absolute.name
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"{role} must be a new path")
    return target
