from __future__ import annotations

import re
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

STABLE_ENTRY_DOCUMENTS = (
    ROOT / "README.md",
    ROOT / "CONTEXT-MAP.md",
    ROOT / "docs" / "ARCHITECTURE.md",
    ROOT / "docs" / "TOP_LEVEL_DESIGN.md",
    ROOT / "docs" / "ACCEPTANCE_GATES.md",
    ROOT / "docs" / "RUNBOOK.md",
    ROOT / "docs" / "MIGRATION_PLAN.md",
    ROOT / "migration" / "CAPABILITY_MATRIX.md",
    ROOT / "docs" / "contexts" / "compiler" / "CONTEXT.md",
    ROOT / "docs" / "contexts" / "lab" / "CONTEXT.md",
    ROOT / "docs" / "contexts" / "evaluation" / "CONTEXT.md",
    ROOT / "docs" / "contexts" / "evidence" / "CONTEXT.md",
)


class DocumentationContractTests(unittest.TestCase):
    def test_current_release_view_is_regenerated_from_authorities(self) -> None:
        result = subprocess.run(
            [sys.executable, "tools/render_current_status.py", "--check"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_stable_entry_docs_do_not_expose_legacy_gate_codes(self) -> None:
        opaque_gate = re.compile(r"(?<![A-Za-z0-9])G[0-9]+[A-Z]?(?![A-Za-z0-9])")
        for path in STABLE_ENTRY_DOCUMENTS:
            with self.subTest(path=path.relative_to(ROOT)):
                text = path.read_text(encoding="utf-8")
                self.assertIsNone(opaque_gate.search(text))
                self.assertNotIn("Campaign Lock", text)

    def test_readme_does_not_copy_current_revision_ids(self) -> None:
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotRegex(text, r"open-cake-ir-sm100a-v[0-9]+")
        self.assertNotRegex(text, r"open-cake-ir-b200-v[0-9]+")
        self.assertIn("reports/current/STATUS.md", text)

    def test_contexts_reference_the_glossary_instead_of_redefining_it(self) -> None:
        for path in (ROOT / "docs" / "contexts").glob("*/CONTEXT.md"):
            with self.subTest(path=path.relative_to(ROOT)):
                text = path.read_text(encoding="utf-8")
                self.assertIn("GLOSSARY.md", text)
                self.assertNotIn("## Language", text)

    def test_legacy_current_state_is_explicitly_frozen(self) -> None:
        text = (ROOT / "inventory" / "CURRENT_STATE.md").read_text(encoding="utf-8")
        self.assertIn("Frozen historical snapshot", text)
        self.assertIn("reports/current/STATUS.md", text)

    def test_changed_documentation_has_no_broken_local_links(self) -> None:
        markdown_link = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
        paths = (
            *STABLE_ENTRY_DOCUMENTS,
            ROOT / "docs" / "GLOSSARY.md",
            ROOT / "docs" / "adr" / "README.md",
            ROOT
            / "docs"
            / "adr"
            / "0047-documentation-separates-stable-history-and-current-views.md",
            ROOT / "reports" / "current" / "STATUS.md",
            ROOT / "inventory" / "CURRENT_STATE.md",
            ROOT / "contracts" / "workloads" / "README.md",
        )
        for path in paths:
            for target in markdown_link.findall(path.read_text(encoding="utf-8")):
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                relative = target.split("#", 1)[0]
                if not relative:
                    continue
                with self.subTest(path=path.relative_to(ROOT), target=target):
                    self.assertTrue((path.parent / relative).exists())


if __name__ == "__main__":
    unittest.main()
