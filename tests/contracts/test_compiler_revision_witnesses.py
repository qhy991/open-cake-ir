from __future__ import annotations

import json
import sys
import unittest
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.compiler_revision_witnesses import (  # noqa: E402
    _is_compiler_revision_id,
    compiler_revision_witnesses,
    generic_compiler_revision_id,
    plan_compiler_revision_cycle,
)


def _canonical_sha256(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


class CompilerRevisionWitnessTests(unittest.TestCase):
    def test_revision_identity_accepts_historical_and_generic_lineages(self) -> None:
        self.assertTrue(_is_compiler_revision_id("open-cake-ir-sm100a-v24"))
        self.assertTrue(_is_compiler_revision_id("open-cake-ir-v25"))
        self.assertFalse(_is_compiler_revision_id("open-cake-ir-sm100a-v24-draft"))
        self.assertFalse(_is_compiler_revision_id("open-cake-ir-apple-v25"))
        self.assertFalse(_is_compiler_revision_id("open-cake-ir-v0"))

    def test_inventory_observations_are_revision_witnesses(self) -> None:
        witnesses = compiler_revision_witnesses(ROOT)
        identities = {item.revision_id for item in witnesses}

        self.assertIn("open-cake-ir-sm100a-v4", identities)
        # This unresolved historical name is still unavailable for reuse.
        self.assertIn("open-cake-ir-sm100a-v6", identities)
        self.assertTrue(
            any(
                item.revision_id == "open-cake-ir-sm100a-v4"
                and item.path
                == "contracts/studies/matched-search-infrastructure-v4.json"
                for item in witnesses
            )
        )

    def test_amd_branch_base_reserves_both_lineages(self) -> None:
        witnesses = compiler_revision_witnesses(ROOT)
        base_path = "inventory/AMD_BRANCH_BASE_20260825.json"
        observed = {
            (item.revision_id, item.revision_sha256)
            for item in witnesses
            if item.path == base_path
        }

        self.assertIn(
            (
                "open-cake-ir-sm100a-v28",
                "46c6bd0fbc6c7043092c766d1d262126c1e24ec77cb7eb9c0376c0c45a6e4331",
            ),
            observed,
        )
        self.assertIn(
            (
                "open-cake-ir-v17",
                "68cab7d1c2ef426c27499c2dd131164368b62431ece006409f0fd3af186ad023",
            ),
            observed,
        )

    def test_witnessed_v28_requires_the_generic_v29_successor(self) -> None:
        current = json.loads(
            (ROOT / "compiler/revision.lock.json").read_text(encoding="utf-8")
        )
        archived = json.loads(
            (ROOT / "compiler/releases/v27/revision.lock.json").read_text(
                encoding="utf-8"
            )
        )

        plan = plan_compiler_revision_cycle(ROOT, current["revision_id"])

        self.assertEqual(current["revision_id"], "open-cake-ir-sm100a-v28")
        self.assertEqual(archived["revision_id"], "open-cake-ir-sm100a-v27")
        self.assertEqual(plan.next_label, "v29")
        self.assertEqual(plan.archive_label, "v28")
        self.assertEqual(plan.stale_labels, ())
        self.assertEqual(
            generic_compiler_revision_id(plan.next_label, draft=True),
            "open-cake-ir-v29-draft",
        )

    def test_the_collision_incident_accounts_for_every_observed_v4_digest(self) -> None:
        witnesses = compiler_revision_witnesses(ROOT)
        observed = {
            item.revision_sha256
            for item in witnesses
            if item.revision_id == "open-cake-ir-sm100a-v4"
            and item.path.startswith("inventory/")
            and item.revision_sha256 is not None
        }
        incident = json.loads(
            (
                ROOT
                / "inventory/COMPILER_REVISION_IDENTITY_INCIDENT_20260824.json"
            ).read_text(encoding="utf-8")
        )
        recorded = {
            item["revision_sha256"] for item in incident["known_variants"]
        }
        archived_last_v4 = (
            "8a984aef6bbd5fad51f50bd859adc12d281f055ea2eb0c043f5d4fdb785fff5f"
        )

        self.assertEqual(len(observed), 6)
        self.assertEqual(observed, recorded - {archived_last_v4})

    def test_the_incident_successor_is_archived_and_current_has_advanced(self) -> None:
        current = json.loads(
            (ROOT / "compiler/revision.lock.json").read_text(encoding="utf-8")
        )
        archived = json.loads(
            (ROOT / "compiler/releases/v7/revision.lock.json").read_text(
                encoding="utf-8"
            )
        )
        incident = json.loads(
            (
                ROOT
                / "inventory/COMPILER_REVISION_IDENTITY_INCIDENT_20260824.json"
            ).read_text(encoding="utf-8")
        )
        successor = incident["resolution"]["successor_revision"]

        self.assertEqual(successor["revision_id"], archived["revision_id"])
        self.assertEqual(successor["canonical_sha256"], _canonical_sha256(archived))
        current_ordinal = int(current["revision_id"].rsplit("-v", 1)[1])
        successor_ordinal = int(successor["revision_id"].rsplit("-v", 1)[1])
        self.assertGreater(current_ordinal, successor_ordinal)


if __name__ == "__main__":
    unittest.main()
