"""A per-role register budget is emitted by one backend, and refused by every other.

`setmaxnreg` redistributes a CTA's launch allocation within itself. `Role.from_dict` used
to enforce the instruction's immediate -- a multiple of eight in [24, 256] -- at
construction, and the authoring schema republished that range to every author of every
target, including targets whose backends refuse the redistribution outright. A parse that
resolves no Target cannot know which ISA it is encoding for.

So the range moved to the one backend that emits the instruction, and the routes that
cannot emit it now say so. What these hold is the resulting invariant: no route accepts a
budget it will not apply.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from open_cake_ir.compiler.backends import cutedsl, metal, native_cuda, triton
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import Target

ROOT = Path(__file__).resolve().parents[2]


def _schedule(name: str, **changes) -> dict:
    document = json.loads((ROOT / "corpus/schedules" / f"{name}.json").read_text(encoding="utf-8"))
    document.update(changes)
    return document


def _target(name: str) -> Target:
    return Target.load(ROOT / "compiler/targets" / f"{name}.json")


def _codes(findings) -> set[str]:
    return {finding.code for finding in findings}


class ConstructionAdmitsStructure(unittest.TestCase):
    def test_an_immediate_no_isa_admits_still_parses(self) -> None:
        document = _schedule("flash-kmeans-assignment-full")
        document["roles"][0]["registers_per_thread"] = 100
        self.assertEqual(
            Schedule.from_dict(document).roles[0].registers_per_thread, 100)

    def test_the_schema_no_longer_republishes_the_isa_range(self) -> None:
        source = (ROOT / "src/open_cake_ir/compiler/schema.py").read_text(encoding="utf-8")
        self.assertNotIn("multipleOf", source)


class TheEmittingBackendOwnsTheImmediate(unittest.TestCase):
    """CuTe-DSL is the only route whose emitter writes the redistribution."""

    def _preflight(self, budget: int):
        # The split divides a declared allocation, so the fixture declares one. Without
        # it the verifier refuses the Schedule before any emitter sees it
        # (ROLE_REGISTERS_WITHOUT_TOTAL), and calling this preflight directly on such a
        # Schedule exercises an input the pipeline never delivers.
        document = _schedule("flash-kmeans-assignment-full")
        document["residency"] = {"registers_per_thread": 192}
        document["roles"][0]["registers_per_thread"] = budget
        return _codes(cutedsl.preflight(Schedule.from_dict(document), _target("sm_100a")))

    def test_an_illegal_immediate_is_refused_by_the_backend_that_encodes_it(self) -> None:
        for bad in (100, 20, 264):
            with self.subTest(registers=bad):
                self.assertIn("CUTE_ROLE_REGISTERS_ILLEGAL_IMMEDIATE", self._preflight(bad))

    def test_a_legal_immediate_beyond_the_declared_capacity_is_refused(self) -> None:
        """The disagreement this move resolved.

        256 is a legal `setmaxnreg` immediate and sm_100a declares a per-thread capacity
        of 255, so the parse admitted a budget the Target itself cannot hold. Both facts
        are now read where the Target is in scope.
        """
        codes = self._preflight(256)
        self.assertNotIn("CUTE_ROLE_REGISTERS_ILLEGAL_IMMEDIATE", codes)
        self.assertIn("CUTE_ROLE_REGISTERS_EXCEED_TARGET", codes)
        self.assertEqual(_target("sm_100a").resource_limits.maximum_registers_per_thread, 255)

    def test_a_legal_budget_within_capacity_draws_neither_refusal(self) -> None:
        codes = self._preflight(192)
        self.assertNotIn("CUTE_ROLE_REGISTERS_ILLEGAL_IMMEDIATE", codes)
        self.assertNotIn("CUTE_ROLE_REGISTERS_EXCEED_TARGET", codes)


class NoRouteSilentlyDropsASplit(unittest.TestCase):
    """Every route either emits the redistribution or refuses it in its own words."""

    def test_triton_refuses_the_split_it_cannot_emit(self) -> None:
        document = _schedule("rmsnorm-b8-smoke")
        document["roles"][0]["registers_per_thread"] = 64
        codes = _codes(triton.preflight(Schedule.from_dict(document), _target("sm_100a")))
        self.assertIn("TRITON_ROLE_REGISTERS_UNSUPPORTED", codes)

    def test_the_routes_that_never_emitted_it_still_refuse(self) -> None:
        """Found by scanning the Corpus, not by naming files.

        A hardcoded name that stops matching would make this pass by checking nothing,
        so the route is discovered and the count is asserted.
        """
        wanted = {"metal": (metal, "METAL_REGISTER_CAP_UNSUPPORTED"),
                  "native_cuda": (native_cuda, "NATIVE_ROLE_REGISTERS_UNSUPPORTED")}
        checked: set[str] = set()
        for path in sorted((ROOT / "corpus/schedules").glob("*.json")):
            document = json.loads(path.read_text(encoding="utf-8"))
            backend = (document.get("lowering") or {}).get("backend")
            if backend not in wanted or backend in checked or not document.get("roles"):
                continue
            module, code = wanted[backend]
            document["roles"][0]["registers_per_thread"] = 64
            try:
                schedule = Schedule.from_dict(document)
            except Exception:
                continue
            with self.subTest(backend=backend, schedule=path.stem):
                self.assertIn(code, _codes(module.preflight(schedule, _target(schedule.target))))
            checked.add(backend)
        self.assertEqual(checked, set(wanted), f"routes left unchecked: {set(wanted) - checked}")


if __name__ == "__main__":
    unittest.main()
