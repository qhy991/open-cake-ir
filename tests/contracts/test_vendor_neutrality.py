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

from open_cake_ir.compiler.backends import cutedsl, triton
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import Target, TargetParseError, Vendor
from open_cake_ir.compiler.toolchain import triton_route
from open_cake_ir.evaluation.artifacts import executable_role

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

    def test_a_declared_width_must_fit_the_declared_cta_budget(self) -> None:
        document = _document()
        document["resource_limits"] = {**document["resource_limits"],
                                       "maximum_warps_per_cta": 32}
        with self.assertRaises(TargetParseError) as raised:
            Target.from_dict(document)
        self.assertIn("warp_size", str(raised.exception))


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
        # Swapping one vendor's declaration onto the other's document is refused: the
        # route is pinned to the architecture and width each Target declares.
        borrowed = Target.load(ROOT / "compiler/targets/gfx938.json")
        drifted = replace(borrowed, vendor=Vendor.APPLE)
        schedule = Schedule.from_dict({**document, "target": "gfx938"})
        self.assertIn("BACKEND_TARGET_UNSUPPORTED",
                      {f.code for f in triton.preflight(schedule, drifted)})


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
        for target in ("synthetic_third_vendor", "sm_120a", "apple_gpu_family10"):
            with self.subTest(target=target):
                with self.assertRaises(ValueError) as raised:
                    executable_role(target)
                message = str(raised.exception)
                self.assertIn("no executable role in this Evaluation layer", message)
                self.assertNotIn("CUDA", message)

    def test_the_cuda_backends_refuse_by_the_declared_field(self) -> None:
        target = Target.from_dict(_document())
        document = json.loads(
            (ROOT / "corpus/schedules/rmsnorm-b8-smoke.json").read_text(encoding="utf-8")
        )
        document["target"] = target.target_id
        schedule = Schedule.from_dict(document)
        for module, code in ((triton, "BACKEND_TARGET_UNSUPPORTED"),
                             # CuTe-DSL routes this shape to its SIMT path, whose own
                             # exact-target refusal owns the class first.
                             (cutedsl, "CUTE_SIMT_TARGET")):
            with self.subTest(backend=module.__name__):
                findings = module.preflight(schedule, target)
                self.assertIn(code, {f.code for f in findings})
                # The caller named no Apple device, so no refusal here may either.
                for finding in findings:
                    self.assertNotIn("Metal", finding.message)


class OfflineRouteMatchesEveryDeclaredDocumentTest(unittest.TestCase):
    """The one duplication the offline jail is allowed to keep, held verified.

    `compile_triton` must not open a Target document -- offline compilation never reads
    that data in its jail -- so it carries its own table of exact AMDGCN targets. That
    makes the lane width two owners' fact, which F-2026-09-15-004 proposes to end by
    having the emitter pass the width it already holds. Until then the duplication is at
    least checked: a target added to the Revision without the table fails here rather
    than compiling at a width nobody declared.
    """

    def test_every_declared_amdgcn_target_matches_the_offline_route(self) -> None:
        checked = 0
        for path in sorted((ROOT / "compiler/targets").glob("*.json")):
            target_id = path.stem
            target = Target.load(path)
            if target.vendor not in (Vendor.AMD, Vendor.HYGON):
                with self.subTest(target=target_id):
                    # A non-AMDGCN target must not be decoded by the AMDGCN branch.
                    route = triton_route(target_id) if target.vendor is Vendor.NVIDIA else None
                    if route is not None:
                        self.assertEqual(route.text_role, "ptx")
                continue
            with self.subTest(target=target_id):
                route = triton_route(target_id)
                self.assertEqual(route.warp_size, target.warp_size)
                self.assertEqual(route.architecture, target.target_id)
                self.assertEqual(route.text_role, "amdgcn")
                checked += 1
        self.assertTrue(checked, "no AMDGCN target was declared to check")


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
