"""Read the Corpus's public JSON and Python authoring inputs for contract tests."""
from __future__ import annotations

import json
from pathlib import Path

from open_cake_ir.compiler import frontend


def corpus_document(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    return (frontend.parse(source, filename=str(path)).document
            if path.suffix == ".py" else json.loads(source))
