"""Every declared Target document, held to the shape the shared layers now read.

Phase 1 of the reform: the Compiler edges read the facts a Target declares, so a
same-vendor target becomes a document. The first half of this file states, over every
document under `compiler/targets/`, what such a document must and must not say. The
second half is the acceptance test: two documents this checkout does not declare -- an
`sm_120a` and an `apple_gpu_family10`, both synthetic and bound by no Revision -- are added
to a copy of the tree, and a Schedule retargeted to each lowers through the same Compiler
with no edit to shared code. Neither fixture is a device; what they prove is that no
shared table stood between a document and a lowering.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from open_cake_ir.compiler.ir import ContractKind, MemorySpace, contract
from open_cake_ir.compiler.revision import load_revision
from open_cake_ir.compiler.target import CodeObject, Target
from open_cake_ir.compiler.toolchain import triton_route
from tests.contracts._synthetic_revision import FIXTURES, synthetic_compiler

ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = sorted((ROOT / "compiler/targets").glob("*.json"))


def _document(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _schedule(name: str, target: str) -> dict:
    document = json.loads(
        (ROOT / "corpus/schedules" / f"{name}.json").read_text(encoding="utf-8"))
    return {**document, "target": target}


class DeclaredDocumentsTest(unittest.TestCase):
    """What every document under compiler/targets/ says, checked document by document."""

    def test_every_document_loads_on_its_own_and_through_the_revision(self) -> None:
        self.assertTrue(DOCUMENTS)
        revision = load_revision(ROOT, ROOT / "compiler/revision.json")
        self.assertEqual(set(revision.targets), {path.stem for path in DOCUMENTS})
        for path in DOCUMENTS:
            with self.subTest(target=path.stem):
                target = Target.load(path)
                self.assertEqual(target.target_id, path.stem)
                # The Revision's typed facts are the file's; only provenance is added.
                self.assertEqual(revision.targets[path.stem].resource_limits,
                                 target.resource_limits)
                self.assertEqual(revision.targets[path.stem].code_object, target.code_object)

    def test_every_declared_contract_is_a_registered_record_of_its_kind(self) -> None:
        for path in DOCUMENTS:
            target = Target.load(path)
            for name in sorted(target.instruction_contracts):
                with self.subTest(target=path.stem, instruction=name):
                    record = contract(name)
                    self.assertIsNotNone(record)
                    self.assertIsNot(record.kind, ContractKind.SYNCHRONIZATION)
            for name in sorted(target.synchronization_contracts):
                with self.subTest(target=path.stem, synchronization=name):
                    record = contract(name)
                    self.assertIsNotNone(record)
                    self.assertIs(record.kind, ContractKind.SYNCHRONIZATION)

    def test_a_compute_capability_is_declared_exactly_by_cubin_documents(self) -> None:
        """The capability is a fact about what a cubin encodes, keyed on the object and
        not on the vendor; a document producing another object has none to declare."""
        for path in DOCUMENTS:
            with self.subTest(target=path.stem):
                document = _document(path)
                cubin = document["code_object"] == CodeObject.CUBIN.value
                self.assertEqual("compute_capability" in document, cubin)
                self.assertEqual("warps_per_warpgroup" in document, cubin)
                target = Target.load(path)
                self.assertEqual(target.compute_capability is not None, cubin)

    def test_a_tensor_limit_is_declared_exactly_by_documents_with_the_space(self) -> None:
        """A limit for a space the Target does not have is unmodeled, never zero."""
        for path in DOCUMENTS:
            with self.subTest(target=path.stem):
                document = _document(path)
                has_space = "tensor" in document["memory_spaces"]
                self.assertEqual(
                    "maximum_tensor_memory_bytes" in document["resource_limits"], has_space)
                limits = Target.load(path).resource_limits
                self.assertEqual(limits.maximum_tensor_memory_bytes is not None, has_space)
                self.assertEqual(limits.capacity(MemorySpace.TENSOR) is not None, has_space)
                if has_space:
                    self.assertGreater(limits.maximum_tensor_memory_bytes, 0)

    def test_no_document_declares_the_slot_budget_beside_the_width(self) -> None:
        """The slot count is derived from the thread budget and the declared width."""
        for path in DOCUMENTS:
            with self.subTest(target=path.stem):
                document = _document(path)
                self.assertNotIn("maximum_warps_per_cta", document["resource_limits"])
                target = Target.load(path)
                self.assertEqual(
                    target.resource_limits.maximum_warps_per_cta,
                    document["resource_limits"]["maximum_threads_per_cta"] // target.warp_size)


class SameVendorTargetIsADocumentTest(unittest.TestCase):
    """Phase 1 acceptance: a target this checkout does not declare arrives as a file."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.sm_120a = FIXTURES / "sm_120a.json"
        cls.apple10 = FIXTURES / "apple_gpu_family10.json"

    def test_the_fixtures_are_declared_by_no_target_document(self) -> None:
        declared = {path.stem for path in DOCUMENTS}
        for fixture in (self.sm_120a, self.apple10):
            with self.subTest(fixture=fixture.name):
                self.assertNotIn(fixture.stem, declared)
                self.assertEqual(_document(fixture)["target_id"], fixture.stem)

    def test_a_revision_declares_them_beside_the_seven(self) -> None:
        with synthetic_compiler(self.sm_120a) as compiler:
            self.assertEqual(len(compiler._revision.targets), len(DOCUMENTS) + 1)
        with synthetic_compiler(self.sm_120a, self.apple10) as compiler:
            targets = compiler._revision.targets
            self.assertEqual(len(targets), len(DOCUMENTS) + 2)
            self.assertEqual(targets["sm_120a"].compute_capability, (12, 0))
            self.assertIsNone(targets["apple_gpu_family10"].compute_capability)

    def test_an_sm_120a_document_lowers_through_triton_with_no_shared_edit(self) -> None:
        with synthetic_compiler(self.sm_120a) as compiler:
            assessment = compiler.assess(_schedule("rmsnorm-b8-smoke", "sm_120a"))
            codes = [finding.code for finding in assessment.findings]
            self.assertTrue(assessment.lowering_eligible, codes)
            self.assertNotIn("BACKEND_TARGET_UNSUPPORTED", codes)
            self.assertEqual([code for code in codes if code.startswith("TARGET_")], [])
            lowering = compiler.lower(assessment)
        requirements = lowering.toolchain_requirements
        self.assertEqual(requirements["target"], "sm_120a")
        # The route is pure: no GPU and no Triton import, only the contract the emitter
        # wrote from the document it held.
        route = triton_route(requirements)
        self.assertIs(route.code_object, CodeObject.CUBIN)
        self.assertEqual(route.gpu_backend, "cuda")
        self.assertEqual(route.architecture, 120)
        self.assertEqual(route.warp_size, 32)
        self.assertEqual(route.target_pattern, rb"^\.target\s+sm_120a(?:\s|,|$)")

    def test_an_sm_120a_document_refuses_what_it_does_not_declare(self) -> None:
        """No tensor memory and no tcgen05 contract: the Target's own refusals, not a
        backend's table, and not a step down to the architecture that has them."""
        with synthetic_compiler(self.sm_120a) as compiler:
            assessment = compiler.assess(_schedule("flash-kmeans-assignment-full", "sm_120a"))
        codes = {finding.code for finding in assessment.findings}
        self.assertFalse(assessment.accepted)
        self.assertIn("TARGET_INSTRUCTION_UNSUPPORTED", codes)
        self.assertIn("TARGET_MEMORY_SPACE_UNSUPPORTED", codes)
        self.assertNotIn("BACKEND_TARGET_UNSUPPORTED", codes)

    def test_an_apple_family10_document_lowers_through_metal_with_no_shared_edit(self) -> None:
        with synthetic_compiler(self.apple10) as compiler:
            assessment = compiler.assess(
                _schedule("metal-rmsnorm-primary", "apple_gpu_family10"))
            codes = [finding.code for finding in assessment.findings]
            self.assertTrue(assessment.lowering_eligible, codes)
            self.assertNotIn("METAL_TARGET_UNSUPPORTED", codes)
            self.assertNotIn("BACKEND_TARGET_UNSUPPORTED", codes)
            lowering = compiler.lower(assessment)
        self.assertEqual(lowering.toolchain_requirements["target"], "apple_gpu_family10")
        self.assertEqual(lowering.toolchain_requirements["source_language"], "metal")
        self.assertIn("kernel void", lowering.source)


if __name__ == "__main__":
    unittest.main()
