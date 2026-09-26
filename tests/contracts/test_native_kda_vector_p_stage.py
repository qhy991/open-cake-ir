"""Vectorize the exact MMA-owned BF16 P stage with an unaligned fallback."""
from __future__ import annotations

import unittest

from open_cake_ir.compiler import Schedule
from open_cake_ir.compiler.backends import native_cuda
from open_cake_ir.compiler.backends.native_cuda import _vector_p_stage
from open_cake_ir.compiler.verifier import verify
from tests.contracts.test_native_kda_role_pipeline import document as role_document
from tests.contracts.test_native_kda_two_stage_prefetch import document as copy_p_document
from tests.contracts.test_native_two_phase_k128 import target


class NativeKdaVectorPStage(unittest.TestCase):
    def test_aligned_16_byte_copy_and_unaligned_scalar_fallback(self):
        schedule = Schedule.from_dict(role_document())
        self.assertEqual([f for f in verify(schedule, target()) if f.blocks_lowering], [])
        self.assertEqual(native_cuda.preflight(schedule, target()), ())
        self.assertTrue(_vector_p_stage(schedule, target(), schedule.operation("load_p")))
        source = native_cuda.emit(schedule, target()).source
        self.assertIn("CAKE_NATIVE_VECTOR_P_STAGE", source)
        self.assertIn("reinterpret_cast<const uint4*>(p_base + e)", source)
        self.assertIn("reinterpret_cast<uint4*>", source)
        self.assertIn("reinterpret_cast<uintptr_t>(p_base) & 15", source)
        self.assertIn("e=int(threadIdx.x & 31)*8; e<1024; e+=256", source)
        self.assertIn("e=int(threadIdx.x & 31); e<1024; e+=32", source)
        stage = source[source.index("CAKE_NATIVE_VECTOR_P_STAGE"):
                       source.index("cake_arrive(bar3)", source.index("CAKE_NATIVE_VECTOR_P_STAGE"))]
        self.assertIn("__syncwarp();\n        __threadfence_block();\n        __syncwarp();", stage)

    def test_copy_owned_p_stage_keeps_ordinary_scalar_emission(self):
        schedule = Schedule.from_dict(copy_p_document())
        self.assertFalse(_vector_p_stage(schedule, target(), schedule.operation("load_p")))
        self.assertNotIn("CAKE_NATIVE_VECTOR_P_STAGE",
                         native_cuda.emit(schedule, target()).source)

    def test_second_p_reader_disables_vector_mapping(self):
        value = role_document()
        next(op for op in value["operations"] if op["id"] == "cast_v")[
            "reads"] = ["p_stage"]
        schedule = Schedule.from_dict(value)
        self.assertFalse(_vector_p_stage(schedule, target(), schedule.operation("load_p")))
        self.assertIn("NATIVE_ROLE_PIPELINE_DOMAIN",
                      {f.code for f in native_cuda.preflight(schedule, target())})


if __name__ == "__main__":
    unittest.main()
