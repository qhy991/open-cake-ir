from __future__ import annotations

import builtins
import copy
import dataclasses
import importlib.util
import json
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.corpus import CorpusGateReport, check_corpus
from open_cake_ir.compiler.errors import CompilerError
from open_cake_ir.compiler.ir import MemorySpace, OperationKind
from open_cake_ir.compiler.revision import load_revision
from open_cake_ir.compiler.target import Target, TargetParseError


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value))


class RevisionAdmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="cake-revision-admission-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.path = self.root / "revision.json"
        self.target = json.loads((ROOT / "compiler/targets/sm_100a.json").read_text())
        self.draft = {
            "schema_version": 2, "corpus_manifest": "corpus.json",
            "calibration_coverage": [],
        }
        (self.root / "compiler/targets").mkdir(parents=True)
        _write(self.root / "corpus.json", {"fixture": True})
        self.bind_target(self.target)

    def bind_target(self, document):
        # Every document under compiler/targets is a declared Target (ADR 0065), so the
        # fixture declares one by writing it there before testing field admission.
        _write(self.root / "compiler/targets/sm_100a.json", document)
        _write(self.path, self.draft)

    def load(self, document=None):
        if document is not None:
            _write(self.path, document)
        return load_revision(self.root, self.path)

    def test_manifest_holds_typed_targets_and_preserves_source_provenance(self):
        revision = self.load()
        target = revision.targets["sm_100a"]
        self.assertIsInstance(target, Target)
        self.assertIsInstance(next(iter(target.memory_spaces)), MemorySpace)
        self.assertIsInstance(next(iter(target.operation_kinds)), OperationKind)
        self.assertEqual(target, Target.from_dict(self.target))
        self.assertEqual(target.source.document, self.target)
        self.assertEqual(target.source.citations, tuple(self.target["citations"]))
        self.assertEqual(target.source.canonical_sha256, sha256(_canonical(self.target)).hexdigest())
        self.assertEqual(revision.project_root, self.root)
        self.assertEqual(revision.corpus_path, self.root / "corpus.json")
        # The fixture root is no checkout, so it has no commit to be identified by;
        # identity reports that absence instead of substituting one.
        self.assertIsNone(revision.commit)
        self.assertEqual(revision.revision_id, "open-cake-ir@uncommitted")
        with self.assertRaises(TypeError):
            revision.targets["other"] = target
        with self.assertRaises(dataclasses.FrozenInstanceError):
            target.resource_limits.maximum_threads_per_cta = 1
        target.source.document["resource_limits"]["maximum_threads_per_cta"] = 1
        self.assertEqual(target.source.document, self.target)
        self.assertEqual(target.resource_limits.maximum_threads_per_cta,
                         self.target["resource_limits"]["maximum_threads_per_cta"])

    def test_bound_target_admits_specification_peak_without_arithmetic_coverage(self):
        target = self.load().targets["sm_100a"]
        self.assertEqual(target.peak.memory_bandwidth.value, 8e12)
        self.assertFalse(target.peak.arithmetic)
        self.assertEqual(target.source.document["peak"], self.target["peak"])

    def test_current_manifest_retains_all_exact_targets_and_zero_tmem(self):
        """The exact declared set, updated only by a deliberate data addition.

        AGENTS.md keeps this pin so a sixth target is a decision rather than an accident;
        the neutrality invariants it used to stand in for now live in
        `test_vendor_neutrality.py` against a fixture that is bound by no Revision.
        gfx938 is the first AMD entry and carries measured device facts.
        """
        revision = load_revision(ROOT, ROOT / "compiler/revision.json")
        self.assertEqual(set(revision.targets), {
            "sm_100a", "apple_gpu_family7", "apple_gpu_family8", "apple_gpu_family9", "sm_103a",
            "gfx938", "gfx1151",
        })
        for target_id, device in (("apple_gpu_family7", "Apple M1 Pro"),
                                  ("apple_gpu_family8", "Apple M2"),
                                  ("apple_gpu_family9", "Apple M4")):
            with self.subTest(target=target_id):
                target = revision.targets[target_id]
                self.assertEqual(target.device_names, (device,))
                self.assertIsNone(target.compute_capability)
                self.assertIsNone(target.warps_per_warpgroup)
                self.assertEqual(target.resource_limits.maximum_tensor_memory_bytes, 0)
                self.assertNotIn(MemorySpace.TENSOR, target.memory_spaces)
        self.assertEqual(revision.targets["sm_103a"].compute_capability, (10, 3))

    def test_revision_fields_targets_and_calibration_remain_strict(self):
        # The retired schema-1 fields are refused as unexpected rather than half-read:
        # a stale manifest must not load as if it declared today's Compiler.
        mutations = [
            ("schema_version", 1), ("state", "released"), ("revision_id", "v1"),
            ("target_definitions", {}), ("sources", []), ("release_approval", {}),
            ("calibration_coverage", "sm_100a"), ("calibration_coverage", [False]),
            ("calibration_coverage", [""]), ("unexpected", True),
            ("corpus_manifest", {"path": "corpus.json"}), ("corpus_manifest", "../escape.json"),
        ]
        for field, value in mutations:
            with self.subTest(field=field, value=value):
                document = copy.deepcopy(self.draft)
                document[field] = value
                with self.assertRaises(CompilerError):
                    self.load(document)

    def test_declared_target_custody_and_identity_remain_strict(self):
        """Discovery replaces per-target references: the directory is the declaration.

        So identity is checked against the file name that declares it, custody at the
        file itself, and an empty directory is reported rather than read as no targets.
        """
        document = copy.deepcopy(self.target)
        document["target_id"] = "sm_103a"
        self.bind_target(document)
        with self.assertRaisesRegex(CompilerError, "identity differs"):
            self.load()
        self.bind_target(self.target)
        with tempfile.TemporaryDirectory() as outside:
            external = Path(outside) / "target.json"
            _write(external, self.target)
            link = self.root / "compiler/targets/sm_120a.json"
            link.symlink_to(external)
            with self.assertRaisesRegex(CompilerError, "custody differs"):
                self.load()
            link.unlink()
        (self.root / "compiler/targets/sm_100a.json").unlink()
        with self.assertRaisesRegex(CompilerError, "declares no Target"):
            self.load()

    def test_target_root_resource_and_citation_checks_are_not_lost(self):
        mutations = [
            ("schema_version", 2), ("extra", True), ("peak", {}),
            ("citations", []), ("citations", ["unstructured citation"]),
        ]
        for field, value in mutations:
            with self.subTest(field=field):
                document = copy.deepcopy(self.target)
                document[field] = value
                self.bind_target(document)
                with self.assertRaises(CompilerError):
                    self.load()
        for extra in ("extra_limit", "grid"):
            document = copy.deepcopy(self.target)
            if extra == "grid":
                document["resource_limits"]["grid"]["w"] = 1
            else:
                document["resource_limits"][extra] = 1
            self.bind_target(document)
            with self.subTest(extra=extra), self.assertRaises(CompilerError):
                self.load()

    def test_typed_target_failure_keeps_its_cause(self):
        document = copy.deepcopy(self.target)
        document["resource_limits"]["maximum_threads_per_cta"] = True
        self.bind_target(document)
        with self.assertRaisesRegex(CompilerError, "maximum_threads_per_cta") as caught:
            self.load()
        self.assertIsInstance(caught.exception.__cause__, TargetParseError)


class CorpusOwnershipTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="cake-corpus-contract-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.path = self.root / "corpus.json"
        (self.root / "schedule.json").write_text("{}")
        self.expected = {
            "accepted": True, "lowering_eligible": True, "finding_codes": ["FIRST", "SECOND"],
            "schedule_sha256": "1" * 64, "lowering_source_sha256": "2" * 64,
        }
        self.manifest = {"schema_version": 1, "corpus_id": "fixture", "state": "draft", "cases": [
            {"case_id": "one", "schedule": "schedule.json", "expected": self.expected},
        ]}
        self.assessment = SimpleNamespace(
            accepted=True, lowering_eligible=True, findings=[SimpleNamespace(code=x) for x in ("FIRST", "SECOND")],
            schedule_sha256="1" * 64, target="fixture_target",
        )
        self.compiler = SimpleNamespace(
            _revision=SimpleNamespace(project_root=self.root, revision_id="fixture", canonical_sha256="3" * 64,
                                      targets={"fixture_target": object(), "unexamined_target": object()}),
            assess_file=Mock(return_value=self.assessment),
            lower=Mock(return_value=SimpleNamespace(source_sha256="2" * 64)),
        )

    def check(self):
        _write(self.path, self.manifest)
        return check_corpus(self.compiler, self.path)

    def test_report_preserves_expected_counts_and_observed_mismatch(self):
        report = self.check()
        self.assertIsInstance(report, CorpusGateReport)
        self.assertTrue(report.passed)
        self.assertEqual(report.compiler_revision_id, "fixture")
        self.assertEqual(report.case_count, 1)
        self.assertEqual((report.accepted_case_count, report.rejected_case_count), (1, 0))
        self.assertEqual((report.lowerable_case_count, report.nonlowerable_case_count), (1, 0))
        self.compiler.lower.reset_mock()
        self.assessment.accepted = False
        self.assessment.lowering_eligible = False
        report = self.check()
        self.assertFalse(report.passed)
        self.assertEqual(report.accepted_case_count, 1)
        self.assertEqual(report.lowerable_case_count, 1)
        self.assertFalse(report.cases[0].observed_accepted)
        self.assertIsNone(report.cases[0].observed_lowering_source_sha256)
        self.compiler.lower.assert_not_called()

    def test_matching_preserves_finding_order_and_both_artifact_bindings(self):
        for field, changed in (("finding_codes", ["SECOND", "FIRST"]),
                               ("schedule_sha256", "4" * 64),
                               ("lowering_source_sha256", "5" * 64)):
            original = self.expected[field]
            self.expected[field] = changed
            with self.subTest(field=field):
                self.assertFalse(self.check().passed)
            self.expected[field] = original

    def test_malformed_corpus_cases_remain_refused(self):
        original = copy.deepcopy(self.manifest)
        mutations = [
            lambda m: m.update(extra=True), lambda m: m.update(state="unknown"),
            lambda m: m.update(cases=[]), lambda m: m["cases"].append(copy.deepcopy(m["cases"][0])),
            lambda m: m["cases"][0].update(extra=True),
            lambda m: m["cases"][0]["expected"].update(accepted=1),
            lambda m: m["cases"][0]["expected"].update(lowering_eligible="true"),
            lambda m: m["cases"][0]["expected"].update(finding_codes=[None]),
            lambda m: m["cases"][0]["expected"].update(schedule_sha256="invalid"),
            lambda m: m["cases"][0]["expected"].update(lowering_source_sha256="invalid"),
        ]
        for index, mutate in enumerate(mutations):
            self.manifest = copy.deepcopy(original)
            mutate(self.manifest)
            with self.subTest(index=index), self.assertRaises(CompilerError):
                self.check()

    def test_corpus_schedule_symlink_cannot_escape_project_root(self):
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "schedule.json"
            target.write_text("{}")
            (self.root / "link.json").symlink_to(target)
            self.manifest["cases"][0]["schedule"] = "link.json"
            with self.assertRaisesRegex(CompilerError, "escapes project root"):
                self.check()
        self.compiler.assess_file.assert_not_called()

    def test_three_existing_corpus_outcomes_use_real_public_compiler_methods(self):
        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        revision = load_revision(ROOT, ROOT / "compiler/revision.json")
        view = SimpleNamespace(_revision=revision, assess_file=compiler.assess_file, lower=compiler.lower)
        self.manifest = json.loads((ROOT / "corpus/manifest.json").read_text())
        self.manifest["cases"] = self.manifest["cases"][:3]
        _write(self.path, self.manifest)
        report = check_corpus(view, self.path)
        self.assertTrue(report.passed)
        self.assertEqual([(c.observed_accepted, c.observed_lowering_eligible) for c in report.cases],
                         [(True, True), (True, False), (False, False)])

    def test_extracted_modules_do_not_import_core_at_runtime(self):
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name in {"core", "open_cake_ir.compiler.core"}:
                raise AssertionError("runtime Core import creates an admission cycle")
            return original_import(name, *args, **kwargs)
        for stem in ("revision", "corpus", "errors"):
            name = f"open_cake_ir.compiler._isolated_{stem}"
            spec = importlib.util.spec_from_file_location(name, ROOT / "src/open_cake_ir/compiler" / f"{stem}.py")
            module = importlib.util.module_from_spec(spec)
            with self.subTest(module=stem), patch.dict(sys.modules, {name: module}), patch("builtins.__import__", guarded):
                spec.loader.exec_module(module)


if __name__ == "__main__":
    unittest.main()
