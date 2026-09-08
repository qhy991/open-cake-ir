"""Prospective custody tests use newly created external private registries."""
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from open_cake_ir.evidence import EvidenceStore
from open_cake_ir.evidence import custody, store
from open_cake_ir.lab.reporting import _promoted_artifact


class EvidenceCustodyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.registry = self.base / "registry"
        self.environment = patch.dict(os.environ, {custody.ENVIRONMENT: str(self.registry)})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def new(self, name="evidence", *, sealed=True):
        evidence = EvidenceStore.create(self.base / name)
        run = evidence.start_run("run", authority_sha256=sha256(b"{}").hexdigest(), authority={})
        if sealed:
            run.append("observation", {"value": name})
            run.seal(protocol_adherence="adhered", endpoint_observation="observed", endpoint={"value": 1})
        return evidence, run

    def witness(self, evidence):
        return self.registry / (evidence.root / custody.MARKER).read_text().strip() / "runs/run"

    def assert_untrusted(self, evidence):
        audit = EvidenceStore.open(evidence.root).audit_run("run")
        self.assertTrue(audit.archive_integrity, audit.findings)
        self.assertFalse(audit.filesystem_custody_verified)
        self.assertIsNone(_promoted_artifact(evidence, audit))
        return audit

    def test_native_private_positive_and_reopen(self):
        evidence, _ = self.new()
        self.assertTrue(evidence.audit_run("run").filesystem_custody_verified)
        self.assertTrue(EvidenceStore.writer(evidence.root).audit_run("run").filesystem_custody_verified)

    def test_copy_and_move_cannot_inherit_origin(self):
        evidence, _ = self.new()
        copy = self.base / "copy"
        shutil.copytree(evidence.root, copy)
        self.assert_untrusted(EvidenceStore.open(copy))
        moved = self.base / "moved"
        evidence.root.rename(moved)
        self.assert_untrusted(EvidenceStore.open(moved))

    def test_git_checkout_and_registry_paths_refused_without_creation(self):
        checkout = self.base / "checkout"
        checkout.mkdir()
        (checkout / ".git").write_text("gitdir: elsewhere")
        with self.assertRaisesRegex(ValueError, "Git"):
            EvidenceStore.create(checkout / "evidence")
        self.assertFalse((checkout / "evidence").exists())
        evidence, _ = self.new()
        (evidence.root / ".git").mkdir()
        self.assert_untrusted(evidence)
        clean, _ = self.new("clean")
        with patch.dict(os.environ, {custody.ENVIRONMENT: str(checkout / "registry")}):
            self.assert_untrusted(clean)

    def test_missing_registry_is_not_recreated_by_open_or_audit(self):
        evidence, _ = self.new()
        shutil.rmtree(self.registry)
        self.assert_untrusted(evidence)
        self.assertFalse(self.registry.exists())
        with self.assertRaisesRegex(ValueError, "custody"):
            EvidenceStore.writer(evidence.root)

    def test_changed_anchor_and_restored_marker_modes_fail(self):
        evidence, _ = self.new()
        marker = evidence.root / custody.MARKER
        marker.chmod(0o644)
        marker.chmod(0o600)
        self.assert_untrusted(evidence)
        second, _ = self.new("second")
        frontier = self.witness(second) / "frontiers/000000000000.json"
        frontier.chmod(0o600)
        frontier.write_text('{}\n')
        frontier.chmod(0o400)
        self.assert_untrusted(second)

    def test_symlinked_registry_or_anchor_is_not_followed(self):
        evidence, _ = self.new()
        actual = self.base / "actual-registry"
        self.registry.rename(actual)
        self.registry.symlink_to(actual, target_is_directory=True)
        self.assert_untrusted(evidence)
        self.registry.unlink()
        actual.rename(self.registry)
        origin = self.witness(evidence) / "origin.json"
        other = self.base / "other.json"
        origin.rename(other)
        origin.symlink_to(other)
        self.assert_untrusted(evidence)

    def test_old_run_graft_into_fresh_store_and_same_id_directory(self):
        original, _ = self.new("original")
        fresh = EvidenceStore.create(self.base / "fresh")
        shutil.copytree(original.root / "runs/run", fresh.root / "runs/run")
        self.assert_untrusted(fresh)
        started, _ = self.new("started", sealed=False)
        destination = started.root / "runs/run"
        inode = destination.stat().st_ino
        # Preserve registered directory, authority and writer lock; graft only history.
        shutil.copytree(original.root / "runs/run/events", destination / "events", dirs_exist_ok=True)
        shutil.copy2(original.root / "runs/run/terminal.json", destination / "terminal.json")
        self.assertEqual(destination.stat().st_ino, inode)
        self.assert_untrusted(started)

    def test_missing_or_partial_frontier_after_event_cannot_be_adopted(self):
        for partial in (False, True):
            with self.subTest(partial=partial):
                evidence, run = self.new(str(partial), sealed=False)
                publish = custody._publish
                def fail(fd, name, value):
                    if name == "000000000001.json":
                        if partial:
                            descriptor = os.open(name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o400, dir_fd=fd)
                            os.write(descriptor, b'{')
                            os.close(descriptor)
                        raise OSError("injected frontier persistence failure")
                    return publish(fd, name, value)
                with patch.object(custody, "_publish", fail), self.assertRaises(OSError):
                    run.append("observation", {"value": 1})
                with self.assertRaises((ValueError, OSError)):
                    run.seal(protocol_adherence="adhered", endpoint_observation="missing")
                self.assertEqual(len(list((evidence.root / "runs/run/events").iterdir())), 1)
                self.assertFalse(evidence.audit_run("run").archive_integrity)

    def test_failure_before_event_leaves_native_frontier_usable(self):
        evidence, run = self.new(sealed=False)
        with patch.object(store, "_publish_new_at", side_effect=OSError("before event")), self.assertRaises(OSError):
            run.append("observation", {"value": 1})
        run.seal(protocol_adherence="adhered", endpoint_observation="missing")
        self.assertTrue(evidence.audit_run("run").filesystem_custody_verified)

    def test_completed_seal_deletion_is_not_pending_recovery(self):
        evidence, run = self.new()
        (evidence.root / "runs/run/terminal.json").unlink()
        with self.assertRaisesRegex(ValueError, "sealed"):
            run.seal(protocol_adherence="adhered", endpoint_observation="observed", endpoint={"value": 1})
        self.assertFalse((evidence.root / "runs/run/terminal.json").exists())

    def test_failure_after_terminal_before_external_seal_fails_closed(self):
        evidence, run = self.new(sealed=False)
        with patch.object(custody.WriterCustody, "seal", side_effect=OSError("before external seal")), self.assertRaises(OSError):
            run.seal(protocol_adherence="adhered", endpoint_observation="missing")
        self.assert_untrusted(evidence)
        with self.assertRaisesRegex(ValueError, "sealed"):
            run.seal(protocol_adherence="adhered", endpoint_observation="missing")
        self.assertFalse((self.witness(evidence) / "seal.json").exists())

    def test_archive_cannot_supply_registry_locator(self):
        evidence, _ = self.new()
        alternate = self.base / "alternate"
        with patch.dict(os.environ, {custody.ENVIRONMENT: str(alternate)}):
            self.assert_untrusted(evidence)
        self.assertFalse(alternate.exists())

    def test_partial_event_publication_is_retained_and_blocks_further_writes(self):
        evidence, run = self.new(sealed=False)
        def partial(fd, payload):
            os.write(fd, payload[:1])
            raise OSError("injected partial publication write")
        with patch.object(store, "_write_all", partial), self.assertRaises(OSError):
            run.append("observation", {"value": 1})
        directory = evidence.root / "runs/run/events"
        prior = {p.name: p.read_bytes() for p in directory.iterdir()}
        self.assertEqual(len(prior), 1)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            run.seal(protocol_adherence="adhered", endpoint_observation="missing")
        self.assertEqual({p.name: p.read_bytes() for p in directory.iterdir()}, prior)

    def test_custody_gate_blocks_an_otherwise_promotable_projection(self):
        # Exercise only the publication projection with explicit CPU fixture evidence;
        # no runtime receipt qualification or actual artifact promotion is claimed.
        evidence, run = self.new(sealed=False)
        candidate = "a" * 64
        receipt = evidence.put(json.dumps({"candidate_sha256": candidate,
            "correctness_passed": True, "kernel_calls": 1, "fallback_calls": 0,
            "timing": {"measurement_quality_passed": True, "pooled_median_ms": 1.0}}).encode(),
            media_type="application/json")
        run.append("candidate_evaluated", {"purpose": "confirmatory", "turn": 1,
            "candidate_sha256": candidate, "objects": [receipt.reference("evaluation_receipt")]})
        run.seal(protocol_adherence="adhered", endpoint_observation="observed", endpoint={"fixture": True})
        audit = evidence.audit_run("run")
        self.assertEqual(_promoted_artifact(evidence, audit)["candidate_sha256"], candidate)
        shutil.rmtree(self.registry)
        self.assert_untrusted(evidence)
