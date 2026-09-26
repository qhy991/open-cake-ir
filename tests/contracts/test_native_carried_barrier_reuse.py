"""Bounded mbarrier lifetime hoisting for the fixed H64/256-chunk route."""
from __future__ import annotations

import unittest

from open_cake_ir.compiler import Schedule
from open_cake_ir.compiler.backends import native_cuda
from open_cake_ir.compiler.verifier import verify
from tests.contracts.test_native_kda_chunk256 import document as full_document
from tests.contracts.test_native_kda_head_address import document as two_chunk_document
from tests.contracts.test_native_two_phase_k128 import target


class NativeCarriedBarrierReuse(unittest.TestCase):
    def test_full_loop_reuses_ready_free_and_completion_phases(self):
        schedule = Schedule.from_dict(full_document())
        self.assertEqual([f for f in verify(schedule, target()) if f.blocks_lowering], [])
        self.assertEqual(native_cuda.preflight(schedule, target()), ())
        source = native_cuda.emit(schedule, target()).source
        marker = source.index("CAKE_NATIVE_PERSISTENT_CARRIED_MBAR")
        loop = source.index("for (int it0=0; it0<256; ++it0)")
        self.assertLess(marker, loop)
        self.assertEqual(source.count("cake_init(bar1, 2)"), 1)
        self.assertEqual(source.count("cake_init(bar2, 1)"), 1)
        self.assertLess(source.index("cake_init(bar1, 2)"), loop)
        self.assertIn("cake_wait(bar1+stage, (it0&1))", source)
        self.assertIn("cake_wait(free0+0, (it0&1))", source)
        self.assertIn("cake_wait(bar2, (it0&1))", source)
        self.assertIn("cake_wait(bar6, (it0&1))", source)
        self.assertGreater(source.index("cake_inval(bar1)"), loop)

    def test_two_chunk_control_retains_its_original_lifetime(self):
        source = native_cuda.emit(Schedule.from_dict(two_chunk_document()), target()).source
        self.assertNotIn("CAKE_NATIVE_PERSISTENT_CARRIED_MBAR", source)
        loop = source.index("for (int it0=0; it0<2; ++it0)")
        self.assertGreater(source.index("cake_init(bar1+stage, 2)"), loop)
        self.assertIn("cake_wait(bar2, 0)", source)


if __name__ == "__main__":
    unittest.main()
