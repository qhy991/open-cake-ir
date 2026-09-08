"""Controlled reference handoffs, using CPU semantic dependency fixtures only."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.lab._documents import _canonical_json_bytes
from open_cake_ir.lab.contracts import CampaignLock, StudyContract
from open_cake_ir.lab.reference_access import document_role, validate_reference_handoff
from open_cake_ir.lab.task_package import build_run_reference_documents
from open_cake_ir.tasks.authoring import prepare_schedule
from open_cake_ir.tasks.runtime import TaskLab
from open_cake_ir.tasks.workloads import load_workload
from tests.contracts._executor_fixture import SemanticExecutorFixture

ROOT = Path(__file__).resolve().parents[2]
CLEAN = "matched-search-clean-start-reference-template.json"
KNOWN = "matched-search-infrastructure-template.json"


class ReferenceAccessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.external = Path(self.temporary.name).resolve()
        self.enterContext(SemanticExecutorFixture())
        self.lab = TaskLab(ROOT)

    def document(self, name=CLEAN):
        return json.loads((ROOT / "contracts/studies" / name).read_bytes())

    def preflight(self, document):
        path = self.external / "study.json"
        path.write_bytes(_canonical_json_bytes(document))
        return self.lab.preflight(path)

    def test_declaration_is_required_and_closed_for_study_and_lock(self):
        for access in (None, "all_python_is_high_level", {"kind": "clean_start"}):
            document = self.document()
            document["arms"]["open_cake"]["reference_access"] = access
            path = self.external / "study.json"
            path.write_bytes(_canonical_json_bytes(document))
            with self.assertRaisesRegex(ValueError, "reference_access"):
                StudyContract.load(path)
        lock = deepcopy(self.preflight(self.document()).document)
        arm = lock["resolved_inputs"]["arm_environments"]["open_cake"]
        del arm["reference_access"]
        with self.assertRaisesRegex(ValueError, "reference_access"):
            CampaignLock.from_dict(lock)

    def test_infrastructure_declares_known_kernel_scientific_treatment(self):
        document = self.document(KNOWN)
        self.assertEqual(document["claim_scope"], "scientific_matched_search")
        self.assertEqual({arm["reference_access"] for arm in document["arms"].values()}, {"known_kernel_reproduction"})
        lock = self.preflight(document)
        package = self.lab.task_package(lock, "direct_cuda-1")
        self.assertIn("known_kernel_reproduction", package.task_markdown)
        self.assertIn("wmma::mma_sync", package.task_markdown)
        self.assertIn("Reference role: target_reference", package.task_markdown)

    def test_clean_stubs_math_and_api_material_are_projected_without_complete_examples(self):
        lock = self.preflight(self.document())
        for run_id in ("open_cake-1", "direct_cuda-1"):
            package = self.lab.task_package(lock, run_id)
            self.assertIn("mathematical_specification_and_oracle", package.task_markdown)
            self.assertIn("incomplete_target_stub", package.task_markdown)
            self.assertNotIn("python-example.py", package.task_markdown)
            self.assertNotIn("candidate-baseline", package.task_markdown)
            self.assertNotIn("wmma::mma_sync", package.task_markdown)
        self.assertEqual(document_role("python-frontend.md", "clean_start"), "authoring_api")
        self.assertEqual(document_role("workload.json", "clean_start"), "mathematical_specification_and_oracle")
        with self.assertRaisesRegex(ValueError, "unclassified"):
            document_role("arbitrary-source.py", "clean_start")

    def test_full_target_references_refuse_before_compiler_for_restricted_access(self):
        for arm_name, access in (("open_cake", "clean_start"), ("direct_cuda", "clean_start"), ("direct_cuda", "direct_low_level")):
            document = self.document(KNOWN)
            document["arms"][arm_name]["reference_access"] = access
            with self.subTest(arm=arm_name, access=access), patch.object(Compiler, "load") as compiler:
                with self.assertRaisesRegex(ValueError, "target_implementation.*forbidden"):
                    self.preflight(document)
                compiler.assert_not_called()

    def test_external_identical_stubs_are_admitted_and_external_full_reference_is_refused(self):
        document = self.document()
        for arm_name, slot in (("open_cake", "schedule_skeleton"), ("direct_cuda", "candidate_skeleton")):
            reference = document["arms"][arm_name][slot]
            source = ROOT / reference["path"]
            destination = self.external / source.name
            destination.write_bytes(source.read_bytes())
            reference["path"] = str(destination)
        lock = self.preflight(document)
        self.assertIn("Intentionally empty", self.lab.task_package(lock, "direct_cuda-1").task_markdown)
        full = self.document(KNOWN)["arms"]["direct_cuda"]["candidate_skeleton"]
        destination = self.external / "foreign.cu"
        destination.write_bytes((ROOT / full["path"]).read_bytes())
        document["arms"]["direct_cuda"]["candidate_skeleton"] = {**full, "path": str(destination)}
        with self.assertRaisesRegex(ValueError, "target_implementation.*forbidden"):
            self.preflight(document)

    def test_complete_target_python_is_not_a_high_level_oracle_role(self):
        source = self.external / "complete.py"
        source.write_bytes((ROOT / "examples/python/fma.py").read_bytes())
        document = self.document()
        parsed = frontend.read_schedule(source).document
        document["arms"]["open_cake"]["schedule_skeleton"] = {
            "path": str(source), "canonical_sha256": sha256(_canonical_json_bytes(parsed)).hexdigest()}
        with self.assertRaisesRegex(ValueError, "target_implementation.*forbidden"):
            self.preflight(document)
        self.assertEqual(document_role("python-example.py", "known_kernel_reproduction"), "target_implementation")

    def test_inherited_native_lowering_requires_known_kernel_reproduction(self):
        document = self.document("matched-search-triton-optimization-template.json")
        document["arms"]["native_triton"]["reference_access"] = "clean_start"
        with self.assertRaisesRegex(ValueError, "inherited target_implementation"):
            validate_reference_handoff(ROOT, document["arms"])

    def test_external_scaffold_cannot_smuggle_a_target_implementation(self):
        document = self.document()
        full = self.document(KNOWN)["arms"]["direct_cuda"]["candidate_skeleton"]
        path = self.external / "claimed-instructions.md"
        path.write_bytes((ROOT / full["path"]).read_bytes())
        for arm in document["arms"].values():
            arm["scaffold"] = {"path": str(path), "sha256": full["sha256"]}
        with self.assertRaisesRegex(ValueError, "authoring_instructions.*vetted"):
            self.preflight(document)

    def test_locked_package_construction_cannot_bypass_reference_policy(self):
        original = self.preflight(self.document(KNOWN))
        document = deepcopy(original.document)
        arm = document["resolved_inputs"]["arm_environments"]["direct_cuda"]
        arm["reference_access"] = "direct_low_level"
        document["resolved_inputs"]["arm_environment_sha256"]["direct_cuda"] = sha256(_canonical_json_bytes(arm)).hexdigest()
        lock = CampaignLock.from_dict(document)
        with self.assertRaisesRegex(ValueError, "target_implementation.*forbidden"):
            self.lab.task_package(lock, "direct_cuda-1")
        foreign_arm = deepcopy(original.document["resolved_inputs"]["arm_environments"]["open_cake"])
        foreign_arm["reference_access"] = "clean_start"
        with self.assertRaisesRegex(ValueError, "differs from the frozen arm"):
            build_run_reference_documents(ROOT, original, foreign_arm,
                workload_contract=load_workload(ROOT / original.document["workload"]["path"]), prepare_schedule=prepare_schedule)
