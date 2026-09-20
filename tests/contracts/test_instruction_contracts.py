"""Every contract a Target admits, against the analyses that give it meaning.

This lives beside the other shared-layer contract tests rather than in one target's
module: the invariants below are about all seven declared Targets, and an `sm_100a`
author should not learn about a violation from a red DCU test.
"""

from __future__ import annotations

from pathlib import Path
import unittest

from open_cake_ir.compiler.ir.instruction_contracts import (
    CONTRACTS, ContractKind, PLACED_CONTRACTS, contract)
from open_cake_ir.compiler.target import Target

ROOT = Path(__file__).resolve().parents[2]


class AdmittedContractsHaveTheirAnalyses(unittest.TestCase):
    """A declared contract whose analyses did not land with it is a fail-open gate.

    The gap this closes, measured through the public boundary: gfx938 admitted
    `triton.dot.fp16_fp32` and `triton.dot.fp8e4m3_fp32` with no row in the verifier's
    private dtype table, so `.get(...)` returned None, the operand and accumulator
    agreement check never ran, and a contraction whose dtypes disagreed with its buffers
    went from a blocking `TARGET_INSTRUCTION_UNSUPPORTED` to accepted with only a
    non-blocking "was not checked" REPORT. Nothing failed, because the declaration and
    the analysis that gives it meaning lived in different files and no test related them.

    `ir/instruction_contracts.py` is now the one owner of what a contract means: the
    Revision loader refuses a declared name with no record, and the verifier refuses an
    admitted name it meets without one. These hold that coupling across every declared
    Target rather than for gfx938 alone: the next vendor to admit a contract should fail
    here, not ship a gate that does not run.
    """

    def test_every_contract_the_triton_route_can_emit_is_a_registered_mma(self) -> None:
        """Nothing emittable may be unmodelled, whichever Target admits it.

        The emitter's set is read off the registry by its own prefix, so the two cannot
        drift apart; what is held is that every name the emitter can realize is a
        registered MMA record with operands and an accumulator, and that the emitter's own
        precision spellings name registered contracts only.
        """
        from open_cake_ir.compiler.backends.triton import (
            _TRITON_DOT_INPUT_PRECISION, _TRITON_MMA_CONTRACTS)
        self.assertTrue(_TRITON_MMA_CONTRACTS)
        for name in sorted(_TRITON_MMA_CONTRACTS):
            with self.subTest(contract=name):
                record = contract(name)
                self.assertIsNotNone(record)
                self.assertIs(record.kind, ContractKind.MMA)
                self.assertTrue(record.operand_dtypes)
                self.assertIsNotNone(record.accumulator)
        self.assertLessEqual(set(_TRITON_DOT_INPUT_PRECISION), _TRITON_MMA_CONTRACTS)

    def test_every_declared_contract_is_a_record_of_the_kind_its_list_says(self) -> None:
        """The check `revision._declared_contracts` makes at load, stated over the files.

        Replaces the pinned "unmodelled" snapshot: no admitted contract is unmodelled any
        more, because a contract says what it is. What each Target admits is pinned here
        by kind instead, so a contract that changes kind or arrives unregistered is a
        visible diff rather than a silent one.
        """
        admitted = {}
        for path in sorted((ROOT / "compiler/targets").glob("*.json")):
            target = Target.load(path)
            kinds = {}
            for name in target.instruction_contracts:
                record = contract(name)
                self.assertIsNotNone(record, f"{path.stem} admits unregistered {name!r}")
                self.assertIsNot(record.kind, ContractKind.SYNCHRONIZATION, name)
                kinds.setdefault(record.kind.value, []).append(name)
            for name in target.synchronization_contracts:
                record = contract(name)
                self.assertIsNotNone(record, f"{path.stem} admits unregistered {name!r}")
                self.assertIs(record.kind, ContractKind.SYNCHRONIZATION, name)
                kinds.setdefault("synchronization", []).append(name)
            admitted[path.stem] = {kind: sorted(names) for kind, names in sorted(kinds.items())}
        self.assertEqual(admitted, {
            "apple_gpu_family7": {"elementwise": ["metal.fma.f32", "metal.precise.tanh.f32"]},
            "apple_gpu_family8": {"elementwise": ["metal.fma.f32", "metal.precise.tanh.f32"]},
            "apple_gpu_family9": {"elementwise": ["metal.fma.f32", "metal.precise.tanh.f32"]},
            # gfx1151 admits nothing yet; gfx938 admits the two measured contractions.
            "gfx1151": {},
            "xcore1002": {},
            "gfx938": {"elementwise": ["ocml.tanh.f32"],
                       "mma": ["triton.dot.fp16_fp32", "triton.dot.fp32_ieee",
                               "triton.dot.fp32_tf32", "triton.dot.fp8e4m3_fp32"]},
            "sm_100a": {"atomic": ["triton.atomic_add.i32.relaxed.gpu"],
                        "elementwise": ["libdevice.tanh.f32", "ptx.fma.rn.f32"],
                        "mma": ["mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
                                "tcgen05.mma.cta_group::1.kind::f16",
                                "triton.dot.bf16_fp32", "triton.dot.fp32_ieee",
                                "triton.dot.fp32_tf32",
                                "triton.dot.fp8e4m3_block_scale_fp32"],
                        "synchronization": ["barrier.sync", "mbarrier",
                                            "triton_program_order"]},
            "sm_103a": {"atomic": ["triton.atomic_add.i32.relaxed.gpu"],
                        "elementwise": ["libdevice.tanh.f32", "ptx.fma.rn.f32"],
                        "mma": ["mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
                                "tcgen05.mma.cta_group::1.kind::f16",
                                "triton.dot.bf16_fp32", "triton.dot.fp32_ieee",
                                "triton.dot.fp32_tf32",
                                "triton.dot.fp8e4m3_block_scale_fp32"],
                        "synchronization": ["barrier.sync", "mbarrier",
                                            "triton_program_order"]},
        })

    def test_the_registry_is_closed_and_every_record_says_what_it_is(self) -> None:
        """Sixteen records; each kind carries exactly the fields its analyses read."""
        self.assertEqual(len(CONTRACTS), 18)
        for name, record in CONTRACTS.items():
            with self.subTest(contract=name):
                self.assertEqual(record.name, name)
                self.assertEqual(bool(record.operand_dtypes), record.kind is ContractKind.MMA)
                self.assertEqual(record.accumulator is not None, record.kind is ContractKind.MMA)
                self.assertEqual(record.elementwise_op is not None,
                                 record.kind is ContractKind.ELEMENTWISE)
                self.assertEqual(record.elementwise_dtype is not None,
                                 record.kind is ContractKind.ELEMENTWISE)
                if record.places_operands:
                    self.assertIs(record.kind, ContractKind.MMA)
                if record.realizes is not None:
                    self.assertIs(record.kind, ContractKind.SYNCHRONIZATION)
        self.assertEqual(PLACED_CONTRACTS, {
            "tcgen05.mma.cta_group::1.kind::f16",
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"})
        with self.assertRaises(TypeError):
            CONTRACTS["vendor.new.mma"] = None


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
