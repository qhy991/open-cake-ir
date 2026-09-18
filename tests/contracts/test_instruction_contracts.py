"""Every contract a Target admits, against the analyses that give it meaning.

This lives beside the other shared-layer contract tests rather than in one target's
module: the invariants below are about all seven declared Targets, and an `sm_100a`
author should not learn about a violation from a red DCU test.
"""

from __future__ import annotations

from pathlib import Path
import unittest

from open_cake_ir.compiler.target import Target

ROOT = Path(__file__).resolve().parents[2]


class AdmittedContractsHaveTheirAnalyses(unittest.TestCase):
    """A declared contract whose analyses did not land with it is a fail-open gate.

    The gap this closes, measured through the public boundary: gfx938 admitted
    `triton.dot.fp16_fp32` and `triton.dot.fp8e4m3_fp32` with no row in
    `_CONTRACT_DTYPES`, so `_CONTRACT_DTYPES.get(...)` returned None, the operand and
    accumulator agreement check never ran, and a contraction whose dtypes disagreed with
    its buffers went from a blocking `TARGET_INSTRUCTION_UNSUPPORTED` to accepted with
    only a non-blocking "was not checked" REPORT. Nothing failed, because the declaration
    and the analysis that gives it meaning live in different files and no test related
    them.

    This is stated across every declared Target rather than for gfx938 alone: the next
    vendor to admit a contract should fail here, not ship a gate that does not run.
    """

    def test_every_contract_the_triton_route_can_emit_is_modelled_by_the_dtype_gate(self) -> None:
        """Nothing emittable may be unmodelled, whichever Target admits it.

        Half of a coupling that was missing, and the weaker half -- stated here because a
        contract the route can emit with no dtype row is a gate that does not run.

        It does not catch the defect that occurred, and an independent reviewer proved
        that by mutation rather than argument: remove both gfx938 contracts from
        `_CONTRACT_DTYPES` *and* `_TRITON_MMA_CONTRACTS`, which is exactly what the
        original defect was, and this test passes, because the set difference of two
        absences is empty. The two tests below are what fail. Do not read this one as the
        guard for that class and delete them as snapshot noise; that reinstates the
        defect for every target except the one whose contracts are pinned by name.
        """
        from open_cake_ir.compiler.backends.triton import _TRITON_MMA_CONTRACTS
        from open_cake_ir.compiler.verifier.hardware_conformance import _CONTRACT_DTYPES
        unmodelled = sorted(set(_TRITON_MMA_CONTRACTS) - set(_CONTRACT_DTYPES))
        self.assertEqual(
            unmodelled, [],
            "the Triton route can emit contracts the dtype gate does not model, so the "
            "operand and accumulator agreement check silently does not run for them")

    def test_the_non_mma_contracts_other_targets_admit_are_not_claimed_here(self) -> None:
        """What this class does not check, said rather than left to be inferred.

        `instruction_contracts` is one open set holding contraction contracts beside
        elementwise ones -- `metal.fma.f32`, `libdevice.tanh.f32`,
        `triton.atomic_add.i32.relaxed.gpu`. The dtype table is about the first kind only,
        so "every admitted contract is modelled" is false for every CUDA and Apple target
        and always was. Nothing in the declaration says which kind a contract is, which is
        why the check above is stated over the route's emittable set instead. Pinning the
        gap keeps it visible until a contract can say what it is.
        """
        from open_cake_ir.compiler.verifier.hardware_conformance import _CONTRACT_DTYPES
        unmodelled = {}
        for path in sorted((ROOT / "compiler/targets").glob("*.json")):
            rest = sorted(set(Target.load(path).instruction_contracts) - set(_CONTRACT_DTYPES))
            if rest:
                unmodelled[path.stem] = rest
        self.assertEqual(unmodelled, {
            "apple_gpu_family7": ["metal.fma.f32", "metal.precise.tanh.f32"],
            "apple_gpu_family8": ["metal.fma.f32", "metal.precise.tanh.f32"],
            "apple_gpu_family9": ["metal.fma.f32", "metal.precise.tanh.f32"],
            # gfx938's tanh, admitted after this test was written. The assertion fired on
            # the addition, which is what it is for: every entry here is a contract whose
            # meaning lives in `_ELEMENTWISE_INSTRUCTIONS` instead, and adding one without
            # looking is the thing being prevented.
            "gfx938": ["ocml.tanh.f32"],
            "sm_100a": ["libdevice.tanh.f32", "ptx.fma.rn.f32",
                        "triton.atomic_add.i32.relaxed.gpu"],
            "sm_103a": ["libdevice.tanh.f32", "ptx.fma.rn.f32",
                        "triton.atomic_add.i32.relaxed.gpu"],
        })
        # Each is modelled by whichever table owns its kind: unmodelled by the dtype gate
        # is not unmodelled. Three tables in three files own three kinds -- contraction,
        # elementwise, atomic -- and a contract does not say which it is, which is the
        # structural gap this class exists to keep visible. Checking the union is the most
        # this can assert without that.
        from open_cake_ir.compiler.verifier.hardware_conformance import (
            _ELEMENTWISE_INSTRUCTIONS)
        from open_cake_ir.compiler.backends.triton import _ATOMIC_RMW_CONTRACT
        modelled_elsewhere = set(_ELEMENTWISE_INSTRUCTIONS) | {_ATOMIC_RMW_CONTRACT}
        for target, contracts in unmodelled.items():
            with self.subTest(target=target):
                self.assertEqual(
                    sorted(set(contracts) - modelled_elsewhere), [],
                    f"{target} admits a contract no table models")
        # gfx1151 admits no contracts at all, so its absence from the dict above says
        # nothing about modelling. Asserted as the empty set it is, because the line this
        # replaced read as though two contracts had been checked.
        self.assertEqual(
            sorted(Target.load(ROOT / "compiler/targets/gfx1151.json").instruction_contracts),
            [])



class DeclaredContractsTheGateCannotSpeakFor(unittest.TestCase):
    """Which declared contracts no Corpus case reaches, stated per target.

    F-2026-09-17-009 is the defect: a Target can declare a contract with device evidence
    and the Gate will pass whether or not it is true, because no case exercises it.

    The measurement that owns this question is a deletion sweep -- remove a contract from
    its Target document, re-run `check_corpus`, see whether it still passes -- and that is
    too slow for a suite at roughly two dozen Gate runs. So this derives the same answer
    cheaply, and the derivation has to match the sweep rather than approximate it. A first
    version did not: it scanned `operations[].parameters.instruction.contract` across JSON
    schedules only, which missed two things and produced nine where the sweep measures
    eleven.

    Both misses are handled here. Python schedules under `examples/` declare their target
    and contracts in a decorator, so they are parsed rather than skipped -- one of them,
    `metal_fma.py`, is the only case reaching `metal.fma.f32` on apple_gpu_family8. And an
    `atomic_rmw` operation binds its contract without naming it in the document, which is
    why sm_100a's atomic looked unreached when it is not.
    """

    #: Contracts an operation kind binds without naming them in its document.
    IMPLICIT = {"atomic_rmw": "triton.atomic_add.i32.relaxed.gpu"}

    def _reached(self) -> dict[str, set[str]]:
        import json
        from open_cake_ir.compiler import frontend as cake
        manifest = json.loads((ROOT / "corpus/manifest.json").read_text())
        reached: dict[str, set[str]] = {}
        for case in manifest["cases"]:
            path = ROOT / case["schedule"]
            if path.suffix == ".json":
                document = json.loads(path.read_text())
            else:
                document = cake.parse(path.read_text()).document
            target = case.get("target") or document.get("target")
            for operation in document["operations"]:
                instruction = (operation.get("parameters") or {}).get("instruction") or {}
                contract = instruction.get("contract") or self.IMPLICIT.get(operation["kind"])
                if contract:
                    reached.setdefault(target, set()).add(contract)
        return reached

    def test_the_unreached_declarations_are_the_ones_recorded(self) -> None:
        reached = self._reached()
        unreached = {}
        for path in sorted((ROOT / "compiler/targets").glob("*.json")):
            gap = sorted(set(Target.load(path).instruction_contracts)
                         - reached.get(path.stem, set()))
            if gap:
                unreached[path.stem] = gap
        # Measured by deletion sweep at open-cake-ir@a8a4885d: eleven, not the nine an
        # earlier derivation produced. apple_gpu_family8 keeps only its tanh because
        # metal_fma.py reaches its fma, and sm_100a has none because its atomic is bound
        # implicitly.
        self.assertEqual(unreached, {
            "apple_gpu_family7": ["metal.fma.f32", "metal.precise.tanh.f32"],
            "apple_gpu_family8": ["metal.precise.tanh.f32"],
            "apple_gpu_family9": ["metal.fma.f32", "metal.precise.tanh.f32"],
            "sm_103a": ["libdevice.tanh.f32", "ptx.fma.rn.f32",
                        "triton.atomic_add.i32.relaxed.gpu", "triton.dot.fp32_ieee",
                        "triton.dot.fp32_tf32", "triton.dot.fp8e4m3_block_scale_fp32"],
        })
        self.assertEqual(sum(len(v) for v in unreached.values()), 11)
        # The two AMDGCN targets and sm_100a are absent because every declaration they
        # carry is reached. Asserted so a regression shows up here too.
        for target in ("gfx938", "gfx1151", "sm_100a"):
            self.assertNotIn(target, unreached)

    def test_a_python_schedule_is_read_rather_than_skipped(self) -> None:
        """The miss that produced the wrong count, pinned as its own fact."""
        reached = self._reached()
        self.assertIn("metal.fma.f32", reached["apple_gpu_family8"])

    def test_an_implicitly_bound_contract_counts_as_reached(self) -> None:
        """The other miss: an atomic_rmw never names the contract it binds."""
        reached = self._reached()
        self.assertIn("triton.atomic_add.i32.relaxed.gpu", reached["sm_100a"])


if __name__ == "__main__":
    unittest.main()
