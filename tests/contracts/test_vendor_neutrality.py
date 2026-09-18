"""Shared layers treat vendors as peers, held against a synthetic third vendor.

AGENTS.md asks for this hook by name: no test asserted a vendor-neutrality invariant of
shared code, and the exact five-target pin in `test_compiler_revision_architecture.py`
means a sixth target would fail a test rather than be checked by one. The fixture here is
not a device and is bound by no Revision; it exists so the shared code can be held to
rules that a real target would otherwise be the first to discover.

F-2026-09-13-006 sets the sequence. Step 1 declared `warp_size`; this file covers step 2,
declared vendor identity, and step 3, the fixture and its tests.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
import unittest

from open_cake_ir.compiler.backends import triton
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import CodeObject, Target, TargetParseError, Vendor
from open_cake_ir.compiler.toolchain import triton_route
from open_cake_ir.evaluation.artifacts import executable_role
from tests.contracts._synthetic_revision import synthetic_compiler

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/targets/neutrality_third_vendor.json"


def _document() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class DeclaredVendorTest(unittest.TestCase):
    """The parse no longer infers a vendor from the presence of a CUDA field."""

    def test_a_third_vendor_document_parses_on_its_own_terms(self) -> None:
        target = Target.from_dict(_document())
        self.assertIs(target.vendor, Vendor.AMD)
        self.assertIsNone(target.compute_capability)

    def test_the_silent_thirty_two_lane_path_is_closed(self) -> None:
        """The measured failure this replaces.

        A well-formed third-vendor document used to be refused with
        `target.compute_capability is required for CUDA targets`; giving that same
        document a fabricated capability pair made it parse and then carry a 32-lane
        width and a four-slot warpgroup rule against a 64-lane wavefront.
        """
        target = Target.from_dict(_document())
        self.assertEqual(target.warp_size, 64)
        self.assertIsNone(target.warps_per_warpgroup)
        with self.assertRaises(TargetParseError) as raised:
            Target.from_dict({**_document(), "compute_capability": [9, 4]})
        self.assertIn("no CUDA compute capability", str(raised.exception))

    def test_an_undeclared_vendor_is_refused_rather_than_defaulted(self) -> None:
        for value in (None, "", "intel", 4):
            with self.subTest(vendor=value):
                document = {k: v for k, v in _document().items() if k != "vendor"}
                if value is not None:
                    document["vendor"] = value
                with self.assertRaises(TargetParseError):
                    Target.from_dict(document)

    def test_the_slot_budget_is_derived_from_the_declared_width(self) -> None:
        """16 slots of 64 lanes is the 1024-thread CTA; nothing declares the 16 twice."""
        target = Target.from_dict(_document())
        self.assertEqual(target.resource_limits.maximum_warps_per_cta, 16)
        document = _document()
        document["resource_limits"] = {**document["resource_limits"],
                                       "maximum_warps_per_cta": 16}
        with self.assertRaisesRegex(TargetParseError, "resource_limits fields differ"):
            Target.from_dict(document)

    def test_a_limit_for_a_space_the_target_does_not_declare_is_unmodeled(self) -> None:
        """A zero written for tensor memory that does not exist is a substituted fact."""
        target = Target.from_dict(_document())
        self.assertIsNone(target.resource_limits.maximum_tensor_memory_bytes)
        document = _document()
        document["resource_limits"] = {**document["resource_limits"],
                                       "maximum_tensor_memory_bytes": 0}
        with self.assertRaisesRegex(TargetParseError, "declares no tensor memory space"):
            Target.from_dict(document)


class VendorsAreNotFoldedTogetherTest(unittest.TestCase):
    """Two vendors sharing a code-object family are still two vendors."""

    def test_hygon_and_amd_are_distinct_declared_vendors(self) -> None:
        gfx938 = Target.load(ROOT / "compiler/targets/gfx938.json")
        gfx1151 = Target.load(ROOT / "compiler/targets/gfx1151.json")
        self.assertIs(gfx938.vendor, Vendor.HYGON)
        self.assertIs(gfx1151.vendor, Vendor.AMD)
        # What they share is the object Triton emits, not a manufacturer, and each
        # document says so itself rather than appearing in a set the Evaluation layer keeps.
        from open_cake_ir.evaluation.artifacts import executable_role

        self.assertEqual(gfx938.code_object, gfx1151.code_object)
        self.assertEqual(executable_role(gfx938.target_id), "hsaco")
        self.assertEqual(executable_role(gfx1151.target_id), "hsaco")

    def test_the_route_admits_both_and_neither_inherits_the_other(self) -> None:
        document = json.loads(
            (ROOT / "corpus/schedules/rmsnorm-b8-smoke.json").read_text(encoding="utf-8")
        )
        for name, width in (("gfx938", 64), ("gfx1151", 32)):
            target = Target.load(ROOT / "compiler/targets" / f"{name}.json")
            schedule = Schedule.from_dict({**document, "target": name})
            with self.subTest(target=name):
                self.assertEqual(target.warp_size, width)
                self.assertEqual(
                    [f.code for f in triton.preflight(schedule, target)
                     if f.code == "BACKEND_TARGET_UNSUPPORTED"], [])
                # The route facts the emitter writes are each document's own.
                self.assertEqual(triton.target_route_facts(target),
                                 {"code_object": "hsaco", "triton_arch": name,
                                  "warp_size": width})
        # The vendor is not what the route reads: a document that says Apple but still
        # declares an hsaco is emitted for as an hsaco, and the one fact that decides
        # admission is the code object, refused by its own name.
        borrowed = Target.load(ROOT / "compiler/targets/gfx938.json")
        schedule = Schedule.from_dict({**document, "target": "gfx938"})
        self.assertEqual(
            [f.code for f in triton.preflight(schedule, replace(borrowed, vendor=Vendor.APPLE))
             if f.code == "BACKEND_TARGET_UNSUPPORTED"], [])
        with self.assertRaisesRegex(Exception, "emits no 'metal_binary_archive' code object"):
            triton.target_route_facts(
                replace(borrowed, code_object=CodeObject.METAL_BINARY_ARCHIVE))


class SharedArithmeticTest(unittest.TestCase):
    """Thread counts follow the declared width, not a constant."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.target = Target.from_dict(_document())
        cls.base = json.loads(
            (ROOT / "corpus/schedules/rmsnorm-b8-smoke.json").read_text(encoding="utf-8")
        )

    def _verify(self, slots: int):
        from open_cake_ir.compiler.verifier import verify

        document = json.loads(json.dumps(self.base))
        document["target"] = self.target.target_id
        document.pop("residency", None)
        document["roles"] = [{"name": "compute", "warps": list(range(slots))}]
        return {finding.code for finding in verify(Schedule.from_dict(document), self.target)}

    def test_sixteen_slots_fill_the_cta_and_thirty_two_overrun_it(self) -> None:
        self.assertNotIn("TARGET_THREAD_LIMIT", self._verify(16))
        codes = self._verify(32)
        self.assertIn("TARGET_THREAD_LIMIT", codes)

    def test_the_same_slot_count_is_legal_on_a_thirty_two_lane_target(self) -> None:
        """32 slots is 1024 threads there and 2048 here; only the width differs."""
        from open_cake_ir.compiler.verifier import verify

        cuda = Target.load(ROOT / "compiler/targets/sm_100a.json")
        document = json.loads(json.dumps(self.base))
        document.pop("residency", None)
        document["roles"] = [{"name": "compute", "warps": list(range(32))}]
        codes = {finding.code for finding in verify(Schedule.from_dict(document), cuda)}
        self.assertNotIn("TARGET_THREAD_LIMIT", codes)


class RefusalOwnershipTest(unittest.TestCase):
    """A refusal names the class it owns, and no vendor the caller did not name."""

    def test_the_evaluation_layer_refuses_in_its_own_words(self) -> None:
        for target in ("synthetic_third_vendor", "zz_undeclared_target_for_test"):
            with self.subTest(target=target):
                with self.assertRaises(ValueError) as raised:
                    executable_role(target)
                message = str(raised.exception)
                self.assertIn("no executable role in this Evaluation layer", message)
                self.assertNotIn("CUDA", message)

    def test_the_compiler_refuses_a_backend_by_the_declared_code_object(self) -> None:
        """No backend keeps a table of target ids: each declares the objects it emits,
        the Target declares the one it runs, and the Compiler holds the two together."""
        document = json.loads(
            (ROOT / "corpus/schedules/rmsnorm-b8-smoke.json").read_text(encoding="utf-8")
        )
        document["target"] = "synthetic_third_vendor"
        with synthetic_compiler(FIXTURE) as compiler:
            admitted = compiler.assess(document)
            self.assertNotIn("BACKEND_TARGET_UNSUPPORTED", {f.code for f in admitted.findings})
            cute = compiler.assess(
                {**document, "lowering": {**document["lowering"], "backend": "cutlass_cute_dsl"}})
        refusals = [f for f in cute.findings if f.code == "BACKEND_TARGET_UNSUPPORTED"]
        self.assertEqual(
            [f.message for f in refusals],
            ["the cutlass_cute_dsl backend emits ['cubin'] and the "
             "'synthetic_third_vendor' target runs 'hsaco'"])
        self.assertFalse(cute.lowering_eligible)
        # The mismatch skips that backend's own preflight, so no CuTe rule speaks for a
        # target it never emits for, and the caller named no Apple device, so no
        # refusal here may either.
        for finding in cute.findings:
            self.assertFalse(finding.code.startswith("CUTE_"), finding.code)
            self.assertNotIn("Metal", finding.message)


class OfflineRouteMatchesEveryDeclaredDocumentTest(unittest.TestCase):
    """The offline jail keeps no table of targets at all.

    `compile_triton` must not open a Target document -- offline compilation never reads
    that data in its jail -- so the facts it needs (which code object, which architecture
    string, which lane width) ride the compile contract the emitter wrote from the Target
    it held (F-2026-09-15-004). What is held here is that the contract for every declared
    document is the document's own facts, and that the route it selects is keyed on the
    declared code object rather than on the id or the vendor.
    """

    def test_every_declared_document_routes_by_its_own_facts(self) -> None:
        checked = set()
        for path in sorted((ROOT / "compiler/targets").glob("*.json")):
            target = Target.load(path)
            with self.subTest(target=target.target_id):
                if target.code_object is CodeObject.METAL_BINARY_ARCHIVE:
                    with self.assertRaisesRegex(Exception, "emits no"):
                        triton.target_route_facts(target)
                    continue
                route = triton_route(
                    {"target": target.target_id, **triton.target_route_facts(target)})
                self.assertEqual(route.warp_size, target.warp_size)
                self.assertIs(route.code_object, target.code_object)
                if target.code_object is CodeObject.HSACO:
                    self.assertEqual(route.architecture, target.target_id)
                    self.assertEqual(route.text_role, "amdgcn")
                else:
                    major, minor = target.compute_capability
                    self.assertEqual(route.architecture, major * 10 + minor)
                    self.assertEqual(route.text_role, "ptx")
                checked.add(target.code_object)
        self.assertEqual(checked, {CodeObject.CUBIN, CodeObject.HSACO})

    def test_an_unlisted_target_is_refused_not_stepped_down(self) -> None:
        """A contract missing a route fact is refused; nothing reads 32 or cuda for it."""
        for requirements in ({"target": "gfx942"}, {"target": "sm_120a", "warp_size": 32},
                             {"target": "apple_gpu_family9", "code_object": "cubin"}):
            with self.subTest(requirements=requirements):
                with self.assertRaisesRegex(ValueError, "Triton compile contract differs"):
                    triton_route(requirements)
        with self.assertRaisesRegex(ValueError, "no 'metal_binary_archive' code object"):
            triton_route({"target": "apple_gpu_family9", "code_object": "metal_binary_archive",
                          "triton_arch": "apple9", "warp_size": 32})


class NoVendorInTheElseTest(unittest.TestCase):
    """The reintroduced-literal check AGENTS.md's step 3 asks this file to fail on."""

    SHARED = (
        "src/open_cake_ir/compiler/target.py",
        "src/open_cake_ir/compiler/performance/profile.py",
        "src/open_cake_ir/compiler/backends/triton.py",
        "src/open_cake_ir/compiler/backends/cutedsl.py",
        "src/open_cake_ir/evaluation/artifacts.py",
        "src/open_cake_ir/lab/executor.py",
    )

    def test_no_shared_rule_keys_a_vendor_on_the_presence_of_a_cuda_field(self) -> None:
        pattern = re.compile(
            r"compute_capability\s+is\s+(not\s+)?None"
            r"|compute_capability\s*(!=|==)\s*None")
        for relative in self.SHARED:
            with self.subTest(module=relative):
                source = (ROOT / relative).read_text(encoding="utf-8")
                # target.py owns the field's own validation, which necessarily mentions it;
                # what must not recur is a *rule* reading presence as identity.
                offending = [line.strip() for line in source.splitlines()
                             if pattern.search(line)
                             and "target.compute_capability is required" not in line
                             and "capability is None" not in line
                             and "capability is not None" not in line]
                self.assertEqual(offending, [], offending)

    def test_the_apple_architecture_name_set_is_gone_from_the_target_parse(self) -> None:
        source = (ROOT / "src/open_cake_ir/compiler/target.py").read_text(encoding="utf-8")
        self.assertNotIn("_APPLE_ARCHITECTURES", source)


class FixtureIsNotADeviceTest(unittest.TestCase):
    """Keep the pin: this fixture must never become a sixth bound target."""

    def test_the_fixture_is_declared_by_no_target_document(self) -> None:
        declared = {path.stem for path in (ROOT / "compiler/targets").glob("*.json")}
        self.assertNotIn("synthetic_third_vendor", declared)

    def test_the_fixture_says_it_is_not_a_device(self) -> None:
        kinds = {citation["kind"] for citation in _document()["citations"]}
        self.assertEqual(kinds, {"test_fixture"})


if __name__ == "__main__":
    unittest.main()
