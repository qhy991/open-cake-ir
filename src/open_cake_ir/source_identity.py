"""Git commit identity of the source tree a process runs from.

A clean commit pins every tracked byte at once. A Campaign records that commit, and each
process boundary checks that HEAD is still that commit with nothing modified or added,
instead of rehashing per-file lists that had to be re-released whenever a listed file
changed (ADR 0065).

The Compiler records the commit when there is one and runs without it; the Lab refuses
to start or continue a Campaign without it.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_COMMIT = re.compile(r"[0-9a-f]{40}")


class SourceIdentityError(ValueError):
    """The source tree has no clean commit identity."""


def _git(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise SourceIdentityError(f"git is unavailable for {root}: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit {completed.returncode}"
        raise SourceIdentityError(f"{root} is not a readable git checkout: {detail}")
    return completed.stdout


def checkout_commit(project_root: str | Path) -> str:
    """Return HEAD of the checkout rooted at ``project_root`` when it has no changes.

    Untracked files count as changes: an unrecorded module on the import path would run
    without being part of the identity. Ignored files, such as caches and virtual
    environments, do not.
    """

    root = Path(project_root).resolve(strict=True)
    top = Path(_git(root, "rev-parse", "--show-toplevel").strip()).resolve()
    if top != root:
        raise SourceIdentityError(f"{root} is not the top of its git checkout {top}")
    commit = _git(root, "rev-parse", "--verify", "HEAD^{commit}").strip()
    if _COMMIT.fullmatch(commit) is None:
        raise SourceIdentityError(f"{root} HEAD is not a full commit id")
    changes = [
        line for line in
        _git(root, "status", "--porcelain=v1", "--untracked-files=all").splitlines()
        if line.strip()
    ]
    if changes:
        shown = ", ".join(line[3:] for line in changes[:5])
        more = f" and {len(changes) - 5} more" if len(changes) > 5 else ""
        raise SourceIdentityError(
            f"{root} has uncommitted or untracked changes ({shown}{more}); commit them "
            "or run from a clean checkout of the pinned commit"
        )
    return commit


def checkout_commit_or_none(project_root: str | Path) -> str | None:
    """Return the clean commit, or None when there is none to record."""

    try:
        return checkout_commit(project_root)
    except SourceIdentityError:
        return None


def untracked_paths(project_root: str | Path, relative_paths: tuple[str, ...]) -> tuple[str, ...]:
    """Return those of ``relative_paths`` (POSIX, root-relative) that the index does not track.

    `checkout_commit` refuses a tree with untracked files, but a file hidden by
    `.git/info/exclude` or a global excludes file is invisible to `git status` and still
    present on disk, so a loader that globs a directory would read it into an identity
    the commit does not cover. A loader that claims the commit as its identity asks here
    for every document it read.
    """

    if not relative_paths:
        return ()
    root = Path(project_root).resolve(strict=True)
    listed = _git(root, "ls-files", "-z", "--", *relative_paths)
    tracked = {name for name in listed.split("\0") if name}
    return tuple(path for path in relative_paths if path not in tracked)
