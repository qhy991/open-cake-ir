from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.evidence import EvidenceStore  # noqa: E402


class EvidenceContractTests(unittest.TestCase):
    def test_authority_terminal_and_event_bytes_are_all_integrity_authority(self) -> None:
        targets = ("authority", "terminal", "event")
        for target in targets:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                evidence = EvidenceStore.create(Path(directory).resolve() / "evidence")
                authority = {"kind": "fixture", "id": target}
                authority_sha = sha256(
                    json.dumps(authority, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                run = evidence.start_run(
                    f"tamper-{target}",
                    authority_sha256=authority_sha,
                    authority=authority,
                )
                run.append("observation", {"value": 1})
                run.seal(
                    protocol_adherence="adhered",
                    endpoint_observation="observed",
                    endpoint={"value": 1},
                )
                run_root = evidence.root / "runs" / f"tamper-{target}"
                path = {
                    "authority": run_root / "authority.json",
                    "terminal": run_root / "terminal.json",
                    "event": run_root / "events" / "000000000000.json",
                }[target]
                path.chmod(0o640)
                if target == "event":
                    path.write_bytes(path.read_bytes() + b"\n")
                else:
                    document = json.loads(path.read_text())
                    document["schema_version"] = 999
                    path.write_text(
                        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
                    )

                audit = EvidenceStore.open(evidence.root).audit_run(f"tamper-{target}")

            self.assertFalse(audit.archive_integrity)
            self.assertTrue(audit.filesystem_custody_verified)
            self.assertEqual([finding.code for finding in audit.findings], ["RUN_ARCHIVE_INVALID"])

    def test_parent_symlink_cannot_redirect_cas_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            evidence = EvidenceStore.create(parent / "evidence")
            external = parent / "external"
            external.mkdir()
            objects = evidence.root / "objects"
            objects.rename(evidence.root / "objects-real")
            objects.symlink_to(external, target_is_directory=True)

            with self.assertRaises((OSError, ValueError)):
                evidence.put(b"must-not-escape", media_type="application/octet-stream")

            self.assertEqual(list(external.iterdir()), [])

    def test_concurrent_appends_are_serialized_into_one_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence = EvidenceStore.create(Path(directory).resolve() / "evidence")
            authority = {"kind": "fixture", "id": "concurrent"}
            authority_sha = sha256(
                json.dumps(authority, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            run = evidence.start_run(
                "concurrent",
                authority_sha256=authority_sha,
                authority=authority,
            )
            with ThreadPoolExecutor(max_workers=2) as executor:
                hashes = list(
                    executor.map(
                        lambda value: run.append("observation", {"value": value}),
                        (1, 2),
                    )
                )
            run.seal(
                protocol_adherence="adhered",
                endpoint_observation="observed",
                endpoint={"values": 2},
            )

            audit = EvidenceStore.open(evidence.root).audit_run("concurrent")

        self.assertEqual(len(set(hashes)), 2)
        self.assertTrue(audit.archive_integrity, audit.findings)
        self.assertEqual(audit.event_count, 3)

    def test_read_only_open_does_not_create_missing_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing"
            with self.assertRaises(ValueError):
                EvidenceStore.open(path)
            self.assertFalse(path.exists())

    def test_read_only_clone_separates_integrity_from_filesystem_custody(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "evidence"
            evidence = EvidenceStore.create(root)
            payload = evidence.put(b"retained bytes", media_type="text/plain")
            authority = {"kind": "fixture", "id": "clone-modes"}
            authority_sha = sha256(
                json.dumps(authority, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            run = evidence.start_run(
                "clone-modes",
                authority_sha256=authority_sha,
                authority=authority,
            )
            run.append("observation", {"objects": [payload.reference("retained")]})
            run.seal(
                protocol_adherence="adhered",
                endpoint_observation="observed",
                endpoint={"value": 1},
            )

            for current, directories, files in os.walk(root):
                os.chmod(current, 0o755)
                for name in directories:
                    os.chmod(Path(current) / name, 0o755)
                for name in files:
                    os.chmod(Path(current) / name, 0o644)
            git_like = EvidenceStore.open(root).audit_run("clone-modes")
            self.assertTrue(git_like.archive_integrity, git_like.findings)
            self.assertFalse(git_like.filesystem_custody_verified)

            for current, directories, files in os.walk(root):
                os.chmod(current, 0o770)
                for name in directories:
                    os.chmod(Path(current) / name, 0o770)
                for name in files:
                    os.chmod(Path(current) / name, 0o660)

            archive = EvidenceStore.open(root)
            audit = archive.audit_run("clone-modes")

            self.assertTrue(audit.archive_integrity, audit.findings)
            self.assertFalse(audit.filesystem_custody_verified)
            self.assertEqual(len(archive.replay_events("clone-modes")), 2)
            with self.assertRaisesRegex(ValueError, "custody"):
                EvidenceStore.writer(root)

    def test_authority_and_secret_admission_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence = EvidenceStore.create(Path(directory).resolve() / "evidence")
            with self.assertRaisesRegex(ValueError, "authority bytes"):
                evidence.start_run(
                    "wrong-authority",
                    authority_sha256="0" * 64,
                    authority={"kind": "fixture"},
                )
            secrets = (
                b"OPENAI_" + b"API_KEY=must-not-enter-cas",
                b"Authorization: Bearer " + b"sk-" + b"abcdefghijklmno",
                b"GITHUB_TOKEN=" + b"ghp_" + b"abcdefghijklmnopqrstuvwxyz",
                b"openai_api_key = " + b"sk-" + b"abcdefghijklmno",
            )
            for payload in secrets:
                with self.subTest(case=secrets.index(payload)), self.assertRaisesRegex(
                    ValueError, "secret marker"
                ):
                    evidence.put(payload, media_type="text/plain")

    def test_object_tampering_invalidates_integrity_without_rewriting_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            evidence = EvidenceStore.create(root / "evidence")
            candidate = evidence.put(b"candidate", media_type="application/octet-stream")
            authority = {"kind": "fixture", "id": "tamper"}
            authority_sha = sha256(
                json.dumps(authority, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            run = evidence.start_run(
                "tamper-case",
                authority_sha256=authority_sha,
                authority=authority,
            )
            run.append("candidate_sealed", {"objects": [candidate.reference("candidate")]})
            run.seal(
                protocol_adherence="adhered",
                endpoint_observation="qualified",
                endpoint={"qualified_by_budget": True, "best_confirmed_speedup": 1.0},
            )
            object_path = evidence.root / candidate.relative_path
            object_path.chmod(0o640)
            object_path.write_bytes(b"tampered")

            audit = evidence.audit_run("tamper-case")

        self.assertFalse(audit.archive_integrity)
        self.assertEqual([finding.code for finding in audit.findings], ["OBJECT_INVALID"])

    def test_protocol_failure_is_an_intact_auditable_terminal_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence = EvidenceStore.create(Path(directory).resolve() / "evidence")
            candidate = evidence.put(b'{"candidate":"bytes"}', media_type="application/json")
            authority = {"kind": "fixture", "id": "cake-1"}
            authority_sha = sha256(
                json.dumps(authority, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            run = evidence.start_run(
                "cake-1",
                authority_sha256=authority_sha,
                authority=authority,
            )
            run.append("candidate_sealed", {"objects": [candidate.reference("candidate")]})
            run.seal(
                protocol_adherence="provider_fault",
                endpoint_observation="missing",
            )

            audit = evidence.audit_run("cake-1")

        self.assertTrue(audit.archive_integrity)
        self.assertTrue(audit.filesystem_custody_verified)
        self.assertEqual(audit.event_count, 2)
        self.assertEqual(audit.protocol_adherence, "provider_fault")
        self.assertEqual(audit.endpoint_observation, "missing")
        self.assertEqual(audit.findings, ())

    def test_fresh_process_replay_is_deterministic_and_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "evidence"
            evidence = EvidenceStore.create(root)
            authority = {"kind": "fixture", "id": "fresh-process"}
            authority_sha = sha256(
                json.dumps(authority, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            run = evidence.start_run(
                "fresh-process",
                authority_sha256=authority_sha,
                authority=authority,
            )
            run.append("observation", {"value": 1})
            run.seal(
                protocol_adherence="adhered",
                endpoint_observation="observed",
                endpoint={"value": 1},
            )
            script = (
                "import json,sys; "
                "from open_cake_ir.evidence import EvidenceStore; "
                "a=EvidenceStore.open(sys.argv[1]).audit_run('fresh-process'); "
                "print(json.dumps({'integrity':a.archive_integrity,'events':a.event_count,"
                "'seal':a.terminal_seal_sha256},sort_keys=True))"
            )
            first = subprocess.run(
                [sys.executable, "-c", script, str(root)],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout
            second = subprocess.run(
                [sys.executable, "-c", script, str(root)],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout

        self.assertEqual(first, second)
        self.assertEqual(json.loads(first)["events"], 2)
        self.assertTrue(json.loads(first)["integrity"])

    def test_terminal_seal_recovers_without_appending_a_second_terminal_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence = EvidenceStore.create(Path(directory).resolve() / "evidence")
            authority = {"kind": "fixture", "id": "seal-recovery"}
            authority_sha = sha256(
                json.dumps(authority, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            run = evidence.start_run(
                "seal-recovery",
                authority_sha256=authority_sha,
                authority=authority,
            )
            from open_cake_ir.evidence import store as store_module
            publish = store_module._publish_new_at
            def interrupted(fd, name, payload, **kwargs):
                if name == "terminal.json":
                    raise OSError("injected before terminal publication")
                return publish(fd, name, payload, **kwargs)
            with patch.object(store_module, "_publish_new_at", interrupted):
                with self.assertRaises(OSError):
                    run.seal(protocol_adherence="adhered", endpoint_observation="observed", endpoint={"value": 1})

            run.seal(
                protocol_adherence="adhered",
                endpoint_observation="observed",
                endpoint={"value": 1},
            )
            audit = evidence.audit_run("seal-recovery")

        self.assertTrue(audit.archive_integrity, audit.findings)
        self.assertEqual(audit.event_count, 1)


if __name__ == "__main__":
    unittest.main()


class CalibrationIndexTest(unittest.TestCase):
    """The calibration index names every measurement, and only measurements that exist.

    A file added without a row is a measurement nobody reading the index knows about; a
    row without a file is an index that lies. Both have happened elsewhere in this
    repository today -- the runbook named superseded contracts, and the compiler source
    set omitted Schedules its own gate read -- which is what a hand-kept list does.
    """

    def test_the_calibration_prose_agrees_with_the_measurements(self) -> None:
        """The doc's summary claims are about the records, so they are checkable.

        `docs/ANALYSIS_CALIBRATION.md` says the original records lay in the then-claimed
        directions and that some binding-resource observations were ties. Each is a field
        in the stored historical records. The later QSA counterexample changes the model,
        not those bytes.
        """

        directory = ROOT / "evidence" / "calibration"
        records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("residency-b200-*.json"))
        ]
        prose = (ROOT / "docs" / "ANALYSIS_CALIBRATION.md").read_text(encoding="utf-8")

        # The prose is written in words and the records count in integers, so the two
        # are compared through one small normalisation rather than by making the doc
        # read like a table.
        spelled = {3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
        self.assertIn(f"on {spelled[len(records)]} kernels", prose)
        for record in records:
            with self.subTest(schedule=record["schedule"]["schedule_id"]):
                verdict = record["verdict"]
                self.assertTrue(verdict["residency_bound_sound"])
                self.assertTrue(verdict["register_floor_sound"])
                self.assertTrue(verdict["binding_resource_correct"])
        # Ties are what the doc downgrades the attribution for, and it says how many.
        # They are not a property of one backend: one is CuTe-DSL and one is Triton, so
        # this counts them rather than predicting which kernel they land on.
        unique = sum(
            record["verdict"]["binding_resource_measured_uniquely"]
            for record in records
        )
        self.assertIn(
            f"On {spelled[unique]} of the {spelled[len(records)]} kernels", prose
        )

    def test_every_calibration_file_has_a_row_and_every_row_a_file(self) -> None:
        import re

        directory = ROOT / "evidence" / "calibration"
        present = {path.name for path in directory.glob("*.json")}
        named = set(
            re.findall(
                r"`([a-z0-9.-]+\.json)`",
                (directory / "README.md").read_text(encoding="utf-8"),
            )
        )
        self.assertEqual(present - named, set(), "measurements with no row")
        self.assertEqual(named - present, set(), "rows with no measurement")


class ArchiveShapeTamperTest(unittest.TestCase):
    """The event sequence's shape is authority too, not only its bytes.

    A tampering test already covers the bytes of an event, the authority and the
    terminal. What it does not cover is the sequence: an archive with a gap, with an
    extra file, or one that does not end where a Run has to end. Fifty-one refusals guard
    this store and seven of them had ever fired in a test -- the rest were arguments.
    """

    def _sealed_run(self, root: Path, run_id: str) -> Path:
        evidence = EvidenceStore.create(root / "evidence")
        authority = {"kind": "fixture", "id": run_id}
        run = evidence.start_run(
            run_id,
            authority_sha256=sha256(
                json.dumps(authority, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            authority=authority,
        )
        for index in range(3):
            run.append("observation", {"value": index})
        run.seal(
            protocol_adherence="adhered",
            endpoint_observation="observed",
            endpoint={"value": 2},
        )
        return evidence.root

    def test_a_gap_or_an_extra_file_breaks_the_archive(self) -> None:
        def remove_middle(events: Path) -> None:
            sorted(events.iterdir())[1].unlink()

        def remove_last(events: Path) -> None:
            sorted(events.iterdir())[-1].unlink()

        def leave_a_gap(events: Path) -> None:
            first = sorted(events.iterdir())[0]
            copy = events / "000000000009.json"
            copy.write_bytes(first.read_bytes())

        for label, tamper in (
            ("a middle event removed", remove_middle),
            ("the terminal removed", remove_last),
            ("an event beyond the end", leave_a_gap),
        ):
            with self.subTest(archive=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                run_id = "shape"
                evidence_root = self._sealed_run(root, run_id)
                events = evidence_root / "runs" / run_id / "events"
                events.chmod(0o750)
                for path in events.iterdir():
                    path.chmod(0o640)
                tamper(events)

                store = EvidenceStore.open(evidence_root)
                audit = store.audit_run(run_id)
                self.assertFalse(audit.archive_integrity)
                # And a reader cannot get a sequence out of it either. An archive that
                # audits as broken but still replays would let a claim be read off it.
                with self.assertRaises(ValueError):
                    store.replay_events(run_id)

    def test_a_sealed_run_refuses_a_later_event(self) -> None:
        """History stops when the claim is made, or it is not history.

        `read_object` rehashes what it returns and `audit_run` verifies the chain, so a
        forged archive is caught on the way out. This is the guard on the way in: once a
        Run is sealed, the writer that holds it must not be able to add to it.
        """

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            evidence = EvidenceStore.create(root / "evidence")
            authority = {"kind": "fixture", "id": "sealed"}
            run = evidence.start_run(
                "sealed",
                authority_sha256=sha256(
                    json.dumps(authority, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                authority=authority,
            )
            run.append("observation", {"value": 0})
            run.seal(
                protocol_adherence="adhered",
                endpoint_observation="observed",
                endpoint={"value": 0},
            )

            with self.assertRaisesRegex(ValueError, "already sealed"):
                run.append("observation", {"value": 1})
            # And a second terminal is refused as a terminal too, not only as an append.
            with self.assertRaises(ValueError):
                run.append("run_terminal", {"value": 1})

            audit = EvidenceStore.open(evidence.root).audit_run("sealed")
            self.assertTrue(audit.archive_integrity)


class SecretDetectionTests(unittest.TestCase):
    def test_shared_detector_refuses_credential_shapes_before_cas_publication(self):
        from open_cake_ir.evidence.secret_detection import contains_forbidden_secret
        from open_cake_ir.evidence import custody
        # Synthetic bytes assembled at runtime; no actual credentials or matching
        # values are printed in failures or stored in the source checkout.
        pem = [b"-----BEGIN " + family + b"PRIVATE KEY-----" for family in
               (b"", b"RSA ", b"OPENSSH ", b"EC ", b"DSA ", b"ENCRYPTED ")]
        jwt = b"eyJ" + b"a" * 20 + b"." + b"b" * 20 + b"." + b"c" * 20
        old_export_jwt = b"eyJ" + b"a" * 10 + b"." + b"b" * 10 + b"." + b"c" * 10
        candidates = pem + [b'{"tokens":{"access_token":"' + jwt + b'"}}',
            old_export_jwt, b"HF_TOKEN=" + b"hf_" + b"a" * 20,
            b"ghp_" + b"a" * 30, b"github_pat_" + b"a" * 30,
            b"Authorization: Bearer " + b"a" * 30]
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            with patch.dict(os.environ, {custody.ENVIRONMENT: str(base / "registry")}):
                evidence = EvidenceStore.create(base / "evidence")
                for index, payload in enumerate(candidates):
                    with self.subTest(case=index):
                        self.assertTrue(contains_forbidden_secret(payload))
                        with self.assertRaisesRegex(ValueError, "forbidden secret marker"):
                            evidence.put(payload, media_type="application/octet-stream")
                self.assertEqual(list((evidence.root / "objects/sha256").iterdir()), [])

    def test_detection_preserves_nonsecret_runtime_and_placeholder_bytes(self):
        from open_cake_ir.evidence.secret_detection import contains_forbidden_secret
        for payload in (b'{"usage":{"input_tokens":123,"output_tokens":456}}',
                b'{"access_token":"REDACTED"}', b"BEGIN PUBLIC KEY", b"hf_short",
                b"eyJshort.short.short", b"operator weight_token_count=32"):
            self.assertFalse(contains_forbidden_secret(payload))
