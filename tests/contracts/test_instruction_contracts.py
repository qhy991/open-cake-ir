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



if __name__ == "__main__":
    unittest.main()
