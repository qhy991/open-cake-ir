"""Explicit CPU semantic dependency; never a released descriptor or live host admission."""
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

if __name__ == "__main__":
    # Fresh-process tests may run from a copied project; bind imports to that copy.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from open_cake_ir.lab.executor import ExecutorRevision
from tests.contracts.test_executor import _synthetic_cuda_host


class SemanticExecutorFixture:
    """Replace only Executor loading/resolution in semantic tests, without a cycle.

    The fixture has its own identity and no source records. It is never serialized
    as a released authority, used for host admission, or used by release tests.
    """
    def revision(self, root):
        return ExecutorRevision(executor_id="CPU-semantic-fixture", canonical_sha256="e" * 64,
            document={"schema_version": 1, "executor_id": "CPU-semantic-fixture",
                      "commit": "0" * 40, "target": "cpu-semantic-fixture",
                      "host_environment": _synthetic_cuda_host()},
            project_root=Path(root).resolve(),
            relative_path="runtime/hosts/cpu-semantic-fixture.json")

    def load(self, root, path):
        return self.revision(root)

    def load_reference(self, root, reference, context):
        revision = self.revision(root)
        if reference != dict(revision.reference):
            raise ValueError(f"{context} differs from the explicitly injected CPU fixture")
        return revision

    def resolve(self, root, value, context, *, template, target=None):
        if template:
            if value != {"binding": "current_release"}:
                raise ValueError("Study template Executor binding differs")
            if not isinstance(target, str) or not target:
                raise ValueError("current Executor resolution requires an exact target")
            return self.revision(root)
        if value == {"binding": "current_release"}:
            raise ValueError("frozen Study cannot follow the current Executor")
        return self.load_reference(root, value, context)

    def __enter__(self):
        self.stack = ExitStack()
        self.stack.enter_context(patch.object(ExecutorRevision, "load", self.load))
        self.stack.enter_context(patch.object(ExecutorRevision, "load_reference", self.load_reference))
        for owner in ("open_cake_ir.lab.bindings", "open_cake_ir.lab.preflight", "open_cake_ir.tasks.flash_kmeans.study"):
            self.stack.enter_context(patch(owner + ".resolve_executor", self.resolve))
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)


def commit_project(root):
    """Make one copied project a clean checkout of itself, and return its commit.

    A copy is never a checkout of itself: a worktree's `.git` is a file naming the
    original, and a copy that omits tracked directories reads as deletions. Both make
    `checkout_commit` refuse, so the copy is given a repository of its own. The commit
    is the copy's source identity (ADR 0065); it publishes nothing.
    """
    from open_cake_ir.source_identity import checkout_commit

    root = Path(root).resolve(strict=True)
    git = root / ".git"
    if git.is_dir():
        shutil.rmtree(git)
    elif git.exists():
        git.unlink()

    def run(*arguments):
        subprocess.run(["git", "-C", str(root), *arguments], check=True, capture_output=True)

    run("init", "-q")
    run("add", "-A")
    run("-c", "user.name=fixture", "-c", "user.email=fixture@invalid",
        "commit", "-q", "-m", "fixture checkout", "--allow-empty")
    # A fixture keeps generating into its project after this point -- a frozen Study,
    # a provider stub, a captured run. Those are outputs, not the source the commit
    # identifies, so they are ignored rather than committed again at every boundary.
    # Editing a file the commit tracks still dirties the checkout, which is the
    # refusal these fixtures exist downstream of.
    exclude = root / ".git/info/exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    exclude.write_text("*\n", encoding="utf-8")
    return checkout_commit(root)


@lru_cache(maxsize=None)
def compiler_reference(root):
    """The exact reference this checkout provides; Compiler admission is tested separately."""
    from open_cake_ir.compiler.revision import load_revision
    revision = load_revision(root, Path(root) / "compiler/revision.json")
    return {"path": "compiler/revision.json", "revision_id": revision.revision_id,
            "canonical_sha256": revision.canonical_sha256}


if __name__ == "__main__":
    # Explicit fresh-process semantic test driver, not a production runtime mode.
    with SemanticExecutorFixture():
        if sys.argv[1:2] == ["--script"]:
            import runpy
            sys.argv = sys.argv[2:]
            runpy.run_path(sys.argv[0], run_name="__main__")
        else:
            from open_cake_ir.cli import main
            raise SystemExit(main(sys.argv[1:]))
