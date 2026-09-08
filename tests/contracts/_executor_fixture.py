"""Explicit CPU semantic dependency; never a released descriptor or live host admission."""
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

from open_cake_ir.lab.executor import ExecutorRevision
from tests.contracts.test_executor import _synthetic_cuda_host


class SemanticExecutorFixture:
    """Replace only Executor loading/resolution in semantic tests, without a cycle.

    The fixture has its own identity and no source records. It is never serialized
    as a released authority, used for host admission, or used by release tests.
    """
    def revision(self, root):
        return ExecutorRevision(executor_id="CPU-semantic-fixture", canonical_sha256="e" * 64,
            document={"schema_version": 1, "state": "fixture", "sources": [],
                      "host_environment": _synthetic_cuda_host()},
            project_root=Path(root).resolve(), relative_path="CPU-semantic-fixture.json")

    def load(self, root, path):
        return self.revision(root)

    def load_reference(self, root, reference, context):
        revision = self.revision(root)
        if reference != dict(revision.reference):
            raise ValueError(f"{context} differs from the explicitly injected CPU fixture")
        return revision

    def resolve(self, root, value, context, *, template):
        if template:
            if value != {"binding": "current_release"}:
                raise ValueError("Study template Executor binding differs")
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


@lru_cache(maxsize=None)
def compiler_reference(root):
    """Read the declared fixture input identity; Compiler admission is tested separately."""
    import json
    from hashlib import sha256
    document = json.loads((Path(root) / "compiler/revision.lock.json").read_bytes())
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return {"path": "compiler/revision.lock.json", "revision_id": document["revision_id"],
            "canonical_sha256": sha256(payload).hexdigest()}


if __name__ == "__main__":
    # Explicit fresh-process semantic test driver, not a production runtime mode.
    import sys
    from open_cake_ir.cli import main
    with SemanticExecutorFixture():
        raise SystemExit(main(sys.argv[1:]))
