"""A Compiler loaded from a temporary root that declares extra Target documents.

The checkout's `compiler/` and `corpus/` trees are copied whole, so the Revision is this
checkout's own plus the documents a test adds beside it. The copy is no git checkout, so
its identity is `open-cake-ir@uncommitted`: `Compiler.load` admits that, and only Lab
binding refuses it, which is what keeps these fixtures from ever becoming bound targets.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import tempfile
from typing import Iterator

from open_cake_ir.compiler import Compiler

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/targets"


@contextmanager
def synthetic_compiler(*documents: Path) -> Iterator[Compiler]:
    """Yield a Compiler whose Revision declares the checkout's Targets plus `documents`."""
    with tempfile.TemporaryDirectory(prefix="cake-synthetic-revision-") as directory:
        root = Path(directory).resolve()
        shutil.copytree(ROOT / "compiler", root / "compiler")
        shutil.copytree(ROOT / "corpus", root / "corpus")
        for document in documents:
            # A declared document is named by its id; the fixture file need not be.
            target_id = json.loads(document.read_text(encoding="utf-8"))["target_id"]
            shutil.copy(document, root / "compiler/targets" / f"{target_id}.json")
        yield Compiler.load(root, root / "compiler/revision.json")
