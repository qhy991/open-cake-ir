from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT / "docs/data/aka-qualified-ir-v6-review-20260904"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AkaQualifiedReviewExportTests(unittest.TestCase):
    def test_published_projection_passes_its_deterministic_verifier(self) -> None:
        verifier = _module(
            ROOT / "tools/verify_aka_qualified_review_export.py",
            "aka_qualified_review_export_verifier",
        )
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "verification.json"
            argv = [
                "verify_aka_qualified_review_export.py",
                "--export-dir",
                str(EXPORT),
                "--receipt",
                str(receipt),
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(verifier.main(), 0)
            result = json.loads(receipt.read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["phase_a_rows"], 677)
        self.assertEqual(
            result["phase_a_status_counts"],
            {"accepted": 439, "infrastructure_failure": 1, "rejected": 237},
        )
        self.assertEqual(result["lab_rows"], 57)
        self.assertEqual(
            result["lab_status_counts"],
            {"authoring_rejected": 1, "dynamic_valid": 56},
        )
        self.assertEqual(result["ir_gap_cases"], 326)
        self.assertEqual(result["ir_gap_exact_candidate_names"], 296)
        self.assertEqual(result["ir_gap_semantic_clusters"], 202)
        self.assertTrue(result["cluster_partition_exact"])
        self.assertEqual(result["sensitive_findings"], 0)
        self.assertFalse(result["performance_measured"])
        self.assertFalse(result["compiler_change_approved"])
        self.assertFalse(result["release_performed"])


if __name__ == "__main__":
    unittest.main()
