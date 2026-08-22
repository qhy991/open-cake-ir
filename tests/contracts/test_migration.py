from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class MigrationContractTests(unittest.TestCase):
    def test_final_unpushed_history_is_durable_and_manifested(self) -> None:
        source_set = json.loads((ROOT / "migration/source_set.json").read_text())
        bundle = ROOT / source_set["durability"]["bundle_path"]
        payload = bundle.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "verify.git"
            subprocess.run(
                ["git", "init", "--bare", "-q", str(repository)], check=True
            )
            completed = subprocess.run(
                ["git", "-C", str(repository), "bundle", "verify", str(bundle)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

        self.assertEqual(source_set["repository"]["ahead_of_origin"], 0)
        self.assertEqual(
            source_set["repository"]["initial_observed_ahead_of_origin"], 68
        )
        self.assertEqual(
            source_set["repository"]["origin_main"],
            source_set["repository"]["revision"],
        )
        self.assertEqual(sha256(payload).hexdigest(), source_set["durability"]["sha256"])
        self.assertEqual(len(payload), source_set["durability"]["size_bytes"])
        self.assertIn(source_set["repository"]["revision"], completed.stdout + completed.stderr)

    def test_final_manifest_covers_all_five_terminal_archives(self) -> None:
        records = [
            json.loads(line)
            for line in (ROOT / "migration/legacy_manifest.jsonl").read_text().splitlines()
        ]
        by_path = {record.get("path"): record for record in records[1:]}
        expected_trees = {
            "evidence/raw/r41-codex-matched-replication-campaign-v2-invalid": 255,
            "evidence/raw/r42-codex-matched-replication-campaign-v3": 308,
            "evidence/raw/flash-kmeans-r43-heldout-specialist-dispatcher-b200-v1-negative": 3,
            "evidence/raw/flash-kmeans-r44-heldout-specialist-dispatcher-b200-v2-negative": 23,
            "evidence/raw/flash-kmeans-r45-heldout-specialist-dispatcher-b200-v3": 23,
        }

        self.assertEqual(len(records), 146)
        self.assertEqual(records[0]["revision"], "2fa79092c143fd8c2d9caa93fd84ad79a7504836")
        self.assertEqual(records[0]["tree"], "b02d730bb892629f250a20b8c5bd5869262e5c03")
        self.assertEqual(len(by_path), len(records) - 1)
        for path, count in expected_trees.items():
            self.assertEqual(by_path[path]["file_count"], count)

    def test_historical_claim_projection_binds_real_legacy_index_digests(self) -> None:
        catalog = json.loads(
            (ROOT / "evidence/historical/final-stage6-index.json").read_text()
        )
        manifest_payload = (ROOT / "migration/legacy_manifest.jsonl").read_bytes()
        records = [
            json.loads(line)
            for line in (ROOT / "migration/legacy_manifest.jsonl").read_text().splitlines()
        ]
        admitted = {
            record.get("canonical_json_sha256")
            for record in records
            if "canonical_json_sha256" in record
        }

        for observation in catalog["observations"]:
            digest = observation.get("legacy_index_canonical_sha256")
            if digest is not None:
                self.assertIn(digest, admitted)
        self.assertEqual(
            catalog["source"]["manifest_sha256"],
            sha256(manifest_payload).hexdigest(),
        )
        self.assertFalse(catalog["claim_view"]["representation_effect_available"])
        self.assertFalse(
            catalog["claim_view"]["stable_heldout_dispatcher_performance_supported"]
        )
        self.assertFalse(catalog["claim_view"]["serving_supported"])


if __name__ == "__main__":
    unittest.main()
