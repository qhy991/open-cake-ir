from __future__ import annotations

import json
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.executor_revision_witnesses import (  # noqa: E402
    _is_executor_revision_id,
    executor_revision_witnesses,
    parse_executor_revision_id,
    plan_executor_revision_cycle,
    verify_executor_revision_commit,
)


CANONICAL = "a" * 64
DESCRIPTOR = "b" * 64


def _write(root: Path, relative: str, document: object) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _descriptor(root: Path, executor_id: str) -> str:
    relative = f"runtime/executors/{executor_id}.json"
    _write(root, relative, {"executor_id": executor_id})
    return relative


class ExecutorRevisionWitnessTests(unittest.TestCase):
    def test_registered_external_gfx1151_evidence_forces_v2(self) -> None:
        registration = (
            "inventory/AMD_GFX1151_EXECUTOR_V1_BINDINGS_20260826.json"
        )

        witnesses = tuple(
            item
            for item in executor_revision_witnesses(ROOT)
            if item.revision_id == "open-cake-ir-gfx1151-v1"
            and item.path == registration
        )
        plan = plan_executor_revision_cycle(
            ROOT, "open-cake-ir-gfx1151-v1"
        )

        self.assertEqual(len(witnesses), 4)
        self.assertEqual(
            {
                dict(item.digests)["canonical_sha256"]
                for item in witnesses
            },
            {
                "8c6c10cedbe0eb3a7e88a10a6fa80c20316d4249d75ed3f0c0a1a181974b4f45",
                "fcb57b4e1404e3145e5f9929b805e23bd766019d92eefa559c68dbc09e168de8",
                "2ae412eb92c324a4ea46d19f4299b5205ea771bba6db1f9732e84976f02d3012",
                "6a081a224d7626a812861c25f5798b1243090eef19c07e8d0afe3d20157e8817",
            },
        )
        self.assertTrue(plan.current_witnessed)
        self.assertEqual(plan.next_revision_id, "open-cake-ir-gfx1151-v2")
        self.assertEqual(plan.reclaimable_descriptors, ())

    def test_revision_identity_has_two_independent_released_families(self) -> None:
        self.assertTrue(_is_executor_revision_id("open-cake-ir-b200-v1"))
        self.assertTrue(_is_executor_revision_id("open-cake-ir-gfx1151-v29"))
        self.assertFalse(_is_executor_revision_id("open-cake-ir-b200-v0"))
        self.assertFalse(_is_executor_revision_id("open-cake-ir-gfx1151-v2-draft"))
        self.assertFalse(_is_executor_revision_id("open-cake-ir-amd-v2"))
        self.assertFalse(_is_executor_revision_id("open-cake-ir-b200-v2-extra"))
        self.assertIsNone(parse_executor_revision_id(None))

    def test_all_frozen_owners_contribute_digest_bound_references(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(
                root,
                "contracts/studies/study.json",
                {
                    "state": "frozen",
                    "execution": {
                        "executor_revision": {
                            "executor_id": "open-cake-ir-b200-v3",
                            "canonical_sha256": CANONICAL,
                        }
                    },
                },
            )
            _write(
                root,
                "contracts/calibrations/search.json",
                {
                    "state": "frozen",
                    "authority": {
                        "executor": {
                            "executor_id": "open-cake-ir-gfx1151-v2",
                            "descriptor_raw_sha256": DESCRIPTOR,
                        }
                    },
                },
            )
            _write(
                root,
                "evidence/run/result.json",
                {
                    "nested": [
                        {
                            "executor_id": "open-cake-ir-b200-v4",
                            "executor_descriptor_raw_sha256": DESCRIPTOR,
                        }
                    ]
                },
            )
            _write(
                root,
                "inventory/AMD_OBSERVATION.json",
                {
                    "kind": "amd_observation",
                    "state": "frozen",
                    "executor_revision": {
                        "executor_id": "open-cake-ir-gfx1151-v5",
                        "canonical_sha256": CANONICAL,
                    },
                },
            )
            _write(
                root,
                "runtime/amd.campaign.lock.json",
                {
                    "execution": {
                        "executor_revision": {
                            "executor_id": "open-cake-ir-gfx1151-v6",
                            "canonical_sha256": CANONICAL,
                        }
                    }
                },
            )

            witnesses = executor_revision_witnesses(root)

            self.assertEqual(
                {item.revision_id for item in witnesses},
                {
                    "open-cake-ir-b200-v3",
                    "open-cake-ir-b200-v4",
                    "open-cake-ir-gfx1151-v2",
                    "open-cake-ir-gfx1151-v5",
                    "open-cake-ir-gfx1151-v6",
                },
            )

    def test_template_and_unbound_mentions_do_not_witness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(
                root,
                "contracts/studies/template.json",
                {
                    "state": "template",
                    "executor_revision": {
                        "executor_id": "open-cake-ir-b200-v8",
                        "canonical_sha256": CANONICAL,
                    },
                },
            )
            _write(
                root,
                "contracts/calibrations/draft.json",
                {
                    "state": "draft",
                    "executor_revision": {
                        "executor_id": "open-cake-ir-gfx1151-v8",
                        "canonical_sha256": CANONICAL,
                    },
                },
            )
            _write(
                root,
                "evidence/run/unbound.json",
                {"executor_id": "open-cake-ir-b200-v9"},
            )

            self.assertEqual(executor_revision_witnesses(root), ())

    def test_malformed_attempted_digest_binding_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(
                root,
                "contracts/calibrations/malformed.json",
                {
                    "state": "frozen",
                    "executor_revision": {
                        "executor_id": "open-cake-ir-gfx1151-v2",
                        "canonical_sha256": "not-a-digest",
                    },
                },
            )

            with self.assertRaisesRegex(ValueError, "digest differs"):
                executor_revision_witnesses(root)

    def test_frozen_calibration_forces_a_same_family_successor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = "open-cake-ir-gfx1151-v3"
            _write(
                root,
                "contracts/calibrations/gfx-search.json",
                {
                    "state": "frozen",
                    "authority": {
                        "executor_revision": {
                            "executor_id": current,
                            "canonical_sha256": CANONICAL,
                        }
                    },
                },
            )
            _descriptor(root, current)

            plan = plan_executor_revision_cycle(root, current)

            self.assertTrue(plan.current_witnessed)
            self.assertEqual(plan.next_revision_id, "open-cake-ir-gfx1151-v4")
            self.assertEqual(plan.reclaimable_descriptors, ())

    def test_lineages_do_not_consume_each_others_ordinals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(
                root,
                "contracts/studies/b200.json",
                {
                    "state": "frozen",
                    "executor_revision": {
                        "executor_id": "open-cake-ir-b200-v9",
                        "canonical_sha256": CANONICAL,
                    },
                },
            )
            _write(
                root,
                "contracts/calibrations/gfx.json",
                {
                    "state": "frozen",
                    "executor_revision": {
                        "executor_id": "open-cake-ir-gfx1151-v2",
                        "canonical_sha256": CANONICAL,
                    },
                },
            )
            b200_working = _descriptor(root, "open-cake-ir-b200-v10")
            gfx_working = _descriptor(root, "open-cake-ir-gfx1151-v3")

            b200 = plan_executor_revision_cycle(root, "open-cake-ir-b200-v10")
            gfx = plan_executor_revision_cycle(root, "open-cake-ir-gfx1151-v3")

            self.assertEqual(b200.next_revision_id, "open-cake-ir-b200-v10")
            self.assertEqual(b200.reclaimable_descriptors, (b200_working,))
            self.assertNotIn(gfx_working, b200.reclaimable_descriptors)
            self.assertEqual(gfx.next_revision_id, "open-cake-ir-gfx1151-v3")
            self.assertEqual(gfx.reclaimable_descriptors, (gfx_working,))
            self.assertNotIn(b200_working, gfx.reclaimable_descriptors)

    def test_derived_inventory_does_not_witness_but_a_sealed_archive_does(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = "open-cake-ir-b200-v7"
            _descriptor(root, current)
            _write(
                root,
                "inventory/EXECUTOR_REVISIONS.json",
                {
                    "current": {
                        "executor_id": current,
                        "canonical_sha256": CANONICAL,
                        "descriptor_raw_sha256": DESCRIPTOR,
                    }
                },
            )
            _write(
                root,
                "evidence/executors/archive/runtime/executor.json",
                {
                    "schema_version": 1,
                    "executor_id": current,
                    "state": "released",
                    "sources": [{"sha256": CANONICAL}],
                },
            )

            plan = plan_executor_revision_cycle(root, current)

            self.assertTrue(plan.current_witnessed)
            self.assertEqual(plan.next_revision_id, "open-cake-ir-b200-v8")
            self.assertEqual(plan.reclaimable_descriptors, ())
            self.assertTrue(
                any(
                    item.revision_id == current
                    and item.path
                    == "evidence/executors/archive/runtime/executor.json"
                    for item in executor_revision_witnesses(root)
                )
            )

    def test_cycle_plan_rejects_invalid_draft_and_zero_identities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for identity in (
                "open-cake-ir-b200-v0",
                "open-cake-ir-gfx1151-v2-draft",
                "open-cake-ir-amd-v2",
            ):
                with self.subTest(identity=identity):
                    with self.assertRaises(ValueError):
                        plan_executor_revision_cycle(root, identity)

    def test_commit_guard_rechecks_witnesses_and_working_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = "open-cake-ir-gfx1151-v1"
            relative = _descriptor(root, identity)
            final = root / relative
            initial = sha256(final.read_bytes()).hexdigest()

            verify_executor_revision_commit(root, identity, final, initial)

            original = final.read_bytes()
            final.write_text(
                json.dumps({"executor_id": identity, "drift": True}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "working Executor changed"):
                verify_executor_revision_commit(root, identity, final, initial)
            final.write_bytes(original)

            _write(
                root,
                "contracts/calibrations/frozen.json",
                {
                    "state": "frozen",
                    "executor": {
                        "executor_id": identity,
                        "canonical_sha256": CANONICAL,
                    },
                },
            )
            with self.assertRaisesRegex(ValueError, "witness state changed"):
                verify_executor_revision_commit(root, identity, final, initial)

    def test_b200_cycle_admits_and_rechecks_before_atomic_replace(self) -> None:
        source = (ROOT / "tools/release_executor_cycle.sh").read_text(
            encoding="utf-8"
        )
        operations = (
            "released.admit_host()",
            "released.admit_profiler()",
            "verify_executor_revision_commit(",
            "committed = ExecutorRevision.load",
            'mv -f -- "$EXECUTOR_RELEASE_CANDIDATE" "$FINAL_EXECUTOR"',
        )
        positions = [source.index(operation) for operation in operations]

        self.assertEqual(positions, sorted(positions))


if __name__ == "__main__":
    unittest.main()
