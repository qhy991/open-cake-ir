"""Native PTX work claiming, scoped to one warp and one GPU."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler, Schedule, Target
from open_cake_ir.compiler.backends.native_cuda import emit, preflight
from open_cake_ir.compiler.backends.common import EmitError


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "examples/schedules/native/atomic-work-claim.json"
ATOM = "ptx.atom.relaxed.gpu.global.add.s32"


def work_document() -> dict:
    return json.loads(WORK.read_text())


class NativeAtomicWorkClaim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT)

    def test_persistent_old_value_indexes_the_work_table(self):
        document = work_document()
        assessment = self.compiler.assess(document)
        self.assertTrue(assessment.lowering_eligible,
                        [(f.code, f.path) for f in assessment.findings if f.blocks_lowering])
        lowering = self.compiler.lower(assessment)
        source = lowering.source
        self.assertEqual(lowering.toolchain_requirements["grid"], [148, 1, 1])
        self.assertEqual(lowering.toolchain_requirements["block"], [32, 1, 1])
        self.assertIn("for (int cake_work=int(blockIdx.x); cake_work<256; cake_work+=148)", source)
        self.assertIn("atom.relaxed.gpu.global.add.s32 %0, [%1], %2;", source)
        self.assertIn("if ((b4) >= 0 && (b4) < 4)", source)
        self.assertIn("(b5) >= 0 && (b5) < 2048", source)
        self.assertIn("// CAKE_OP: load_claimed", source)
        self.assertIn("// CAKE_OP: store_positions", source)
        self.assertIn("<<<dim3(148,1,1), 32, 0,", source)
        self.assertNotIn("tcgen05.mma", source)
        for operation in document["operations"]:
            self.assertIn(operation["id"], lowering.source_map)

    def test_static_reservation_has_one_program_per_token(self):
        document = json.loads((ROOT / "corpus/schedules/atomic-reservation-b8-smoke.json").read_text())
        document["lowering"]["backend"] = "native_cuda"
        document["roles"][0]["execution_groups"] = [0]
        document["operations"][0]["parameters"].pop("reuse")
        document.pop("residency")
        assessment = self.compiler.assess(document)
        self.assertTrue(assessment.lowering_eligible,
                        [(f.code, f.path) for f in assessment.findings if f.blocks_lowering])
        lowering = self.compiler.lower(assessment)
        self.assertEqual(lowering.toolchain_requirements["grid"], [8, 1, 1])
        self.assertNotIn("for (int cake_work", lowering.source)
        self.assertIn("const int cake_work = int(blockIdx.x);", lowering.source)

    def test_exact_target_must_declare_ptx_atomic(self):
        target = Target.load(ROOT / "compiler/targets/sm_100a.json")
        undeclared = replace(target, instruction_contracts=target.instruction_contracts - {ATOM})
        schedule = Schedule.from_dict(work_document())
        self.assertIn("NATIVE_ATOMIC_CONTRACT_UNSUPPORTED",
                      [finding.code for finding in preflight(schedule, undeclared)])
        with self.assertRaisesRegex(EmitError, "NATIVE_ATOMIC_CONTRACT_UNSUPPORTED"):
            emit(schedule, undeclared)

    def test_two_warp_roles_and_unproven_residency_are_refused(self):
        document = work_document()
        document["roles"].append({"name": "communication", "execution_groups": [1]})
        self.assertIn("NATIVE_SIMT_ROLE",
                      [finding.code for finding in self.compiler.assess(document).findings])
        document = work_document()
        document.pop("residency")
        self.assertIn("PERSISTENT_WITHOUT_RESIDENCY",
                      [finding.code for finding in self.compiler.assess(document).findings])
        document = work_document()
        document["residency"]["registers_per_thread"] = 64
        self.assertIn("NATIVE_SIMT_RESIDENCY",
                      [finding.code for finding in self.compiler.assess(document).findings])

    def test_unimplemented_cache_commitment_and_missing_b300_sm_count_refuse(self):
        document = work_document()
        next(op for op in document["operations"] if op["kind"] == "load")["parameters"]["reuse"] = "streamed"
        self.assertIn("NATIVE_SIMT_LOAD",
                      [finding.code for finding in self.compiler.assess(document).findings])
        document = work_document()
        document["target"] = "sm_103a"
        self.assertIn("NATIVE_SIMT_PERSISTENCE",
                      [finding.code for finding in self.compiler.assess(document).findings])
        document["program_map"]["persistent"] = False
        document.pop("residency")
        self.assertTrue(self.compiler.assess(document).lowering_eligible)

    def test_relaxed_claim_does_not_admit_release_publication(self):
        document = work_document()
        next(op for op in document["operations"] if op["kind"] == "atomic_rmw")["parameters"]["order"] = "release"
        assessment = self.compiler.assess(document)
        self.assertFalse(assessment.lowering_eligible)
        self.assertTrue(any(finding.blocks_lowering for finding in assessment.findings))

    def test_atomic_cannot_sneak_into_tensor_pipeline(self):
        tensor = json.loads((ROOT / "examples/schedules/native/gemm-bias.json").read_text())
        tensor["operations"].append(next(op for op in work_document()["operations"]
                                         if op["kind"] == "atomic_rmw"))
        target = Target.load(ROOT / "compiler/targets/sm_100a.json")
        self.assertIn("NATIVE_ATOMIC_SIMT_ONLY",
                      [finding.code for finding in preflight(Schedule.from_dict(tensor), target)])


if __name__ == "__main__":
    unittest.main()
