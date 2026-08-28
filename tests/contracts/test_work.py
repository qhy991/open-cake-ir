"""Contract tests for declared work.

The counting half of a performance report: how much arithmetic a Schedule commits to and
how many unique global bytes it must move. There is still no predicted time here -- these
are counts, and the rate that would turn them into one is not declared anywhere.

What these tests pin is the closed-form agreement. A GEMM's derived contraction has to
come out at exactly 2*M*N*K, because that number is knowable without this module and a
work model that disagrees with it is wrong rather than approximate.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from open_cake_ir.compiler.ir import OperationKind, Schedule
from open_cake_ir.compiler.work import program_tiles, work_bound

ROOT = Path(__file__).resolve().parents[2]
SCHEDULES = ROOT / "corpus" / "schedules"
GEMM = SCHEDULES / "gemm-bias-b1-smoke.json"
RMSNORM = SCHEDULES / "rmsnorm-b8-smoke.json"
PERSISTENT = SCHEDULES / "rmsnorm-b128-persistent.json"
SOFTMAX = SCHEDULES / "softmax-b8-smoke.json"
ASSIGNMENT_FULL = SCHEDULES / "flash-kmeans-assignment-full.json"
TINYGEMM2 = SCHEDULES / "tinygemm2-stage4-split-k.json"
CUMSUM = SCHEDULES / "chunk-cumsum-b8-smoke.json"
GATHER = SCHEDULES / "indexed-gather-b8-smoke.json"
RAGGED = SCHEDULES / "ragged-grouped-gemm-b1-smoke.json"
ROPE_FUSED = SCHEDULES / "rope-b8-fused.json"
ROPE = SCHEDULES / "rope-b8-smoke.json"


def _bound(path: Path):
    bound = work_bound(Schedule.load(path))
    assert bound is not None
    return bound


class ContractionTest(unittest.TestCase):
    """A derived contraction agrees with the closed form, or the model is wrong."""

    def test_the_gemm_contraction_is_exactly_two_m_n_k(self) -> None:
        """512x256x256, tiled 64x64 with a four-trip K loop, from the declarations alone.

        Nothing in the Schedule says 2*M*N*K. The grid comes from `program_map`, the trip
        count from the `k_loop` tiling of `a`, and the per-execution work from the MMA's
        `tile_shape`. Their product reproducing the closed form is what establishes that
        the loop nest and the grid are composed in the right order.
        """

        bound = _bound(GEMM)
        self.assertEqual(bound.program_tiles, 32)
        self.assertEqual(bound.mma_flops, 2 * 512 * 256 * 256)

    def test_the_bias_add_is_counted_beside_the_contraction(self) -> None:
        """One arithmetic operation per element written, so the total exceeds the matmul."""

        bound = _bound(GEMM)
        self.assertEqual(bound.flops - bound.mma_flops, 32 * 64 * 64)
        self.assertTrue(bound.flops_exact)

    def test_a_nested_loop_multiplies_rather_than_replaces(self) -> None:
        """Flash-KMeans nests K inside the centroid walk; 1024/256 * 128/64 is eight.

        A model that read `tile_loops` as a flat list would charge the MMA two
        executions instead of eight, and the error would be invisible in a Schedule with
        one loop -- which is every other Schedule with an MMA in the corpus.
        """

        bound = _bound(ASSIGNMENT_FULL)
        self.assertEqual(bound.mma_flops, 8 * 2 * 128 * 256 * 64)

    def test_a_contraction_without_a_declared_shape_abstains(self) -> None:
        """TinyGEMM2's asset is not generated from the Schedule, so it declares no shape."""

        bound = _bound(TINYGEMM2)
        self.assertEqual(bound.mma_flops, 0)
        self.assertFalse(bound.flops_exact)
        self.assertIn("split_k_mma", bound.uncounted_arithmetic)


class CompulsoryTrafficTest(unittest.TestCase):
    def test_each_global_buffer_is_charged_once_in_each_direction(self) -> None:
        """A, B and the bias are read; C is written. Two loads of one input are one input."""

        bound = _bound(GEMM)
        self.assertEqual(bound.compulsory_read_bytes, 512 * 256 * 2 + 256 * 256 * 2 + 256 * 4)
        self.assertEqual(bound.compulsory_written_bytes, 512 * 256 * 4)
        self.assertTrue(bound.compulsory_bytes_exact)

    def test_a_runtime_coordinate_makes_the_charge_an_upper_bound(self) -> None:
        """A gather addresses rows the Schedule cannot name, so full coverage is unsafe."""

        bound = _bound(GATHER)
        self.assertFalse(bound.compulsory_bytes_exact)
        self.assertTrue(bound.partially_addressed)

    def test_a_valid_extent_makes_the_charge_an_upper_bound(self) -> None:
        """A padded axis with a live prefix moves less than the Buffer it is declared as."""

        self.assertFalse(_bound(RAGGED).compulsory_bytes_exact)

    def test_a_subrange_access_is_narrower_than_the_buffer_it_addresses(self) -> None:
        """The fused RoPE arm owns half of one axis; the unfused pair owns whole buffers.

        Both spellings compute the same thing, which is what makes this the case that
        separates a real narrowing from a Schedule that merely looks complicated.
        """

        self.assertFalse(_bound(ROPE_FUSED).compulsory_bytes_exact)
        self.assertTrue(_bound(ROPE).compulsory_bytes_exact)


class AbstentionTest(unittest.TestCase):
    """What the model refuses to count, and why refusing beats guessing."""

    def test_a_transcendental_primitive_is_named_rather_than_priced(self) -> None:
        """`tanh` has two admitted spellings on this Target that do not cost the same."""

        bound = _bound(SOFTMAX)
        self.assertFalse(bound.flops_exact)
        self.assertEqual(bound.uncounted_arithmetic, ("exponentiate",))

    def test_a_prefix_scan_abstains_because_the_decomposition_is_the_backend_s(self) -> None:
        bound = _bound(CUMSUM)
        self.assertFalse(bound.flops_exact)
        self.assertEqual(bound.flops, 0)

    def test_a_reduction_that_adds_nothing_is_zero_rather_than_unknown(self) -> None:
        """Softmax's max folds an axis by comparing. Abstaining on it would be a claim.

        The distinction matters because it separates the two reasons a count can be
        missing: nothing was computed, or something was computed and could not be
        priced. Only the second one makes `flops` a lower bound.
        """

        schedule = Schedule.load(SOFTMAX)
        maximum = next(
            operation
            for operation in schedule.operations
            if operation.kind is OperationKind.REDUCE
            and operation.parameters.op.value == "max"
        )
        self.assertNotIn(maximum.op_id, _bound(SOFTMAX).uncounted_arithmetic)

    def test_a_summing_reduction_is_the_axis_it_collapses_minus_one(self) -> None:
        """The written shape is the read shape with the axis removed, so the count follows."""

        schedule = Schedule.load(RMSNORM)
        fold = next(
            operation
            for operation in schedule.operations
            if operation.kind is OperationKind.REDUCE
        )
        read = schedule.buffer(fold.reads[0]).elements
        written = schedule.buffer(fold.writes[0]).elements
        self.assertGreater(read - written, 0)
        # rmsnorm's fold sits outside every loop, so one program tile performs it once.
        self.assertEqual(_bound(RMSNORM).program_tiles, 64)


class WorkDomainTest(unittest.TestCase):
    def test_a_persistent_map_counts_tiles_rather_than_launched_ctas(self) -> None:
        """Persistence changes how many CTAs walk the work, not how much work there is.

        `ranking` needs the launched count and this needs the walked one. Sharing a
        single number between them would make one of the two wrong on exactly the
        Schedule that declares persistence.
        """

        persistent = Schedule.load(PERSISTENT)
        assert persistent.program_map is not None
        self.assertTrue(persistent.program_map.persistent)
        self.assertEqual(program_tiles(persistent), 1024)


class NoPredictedTimeTest(unittest.TestCase):
    def test_the_work_model_predicts_no_time(self) -> None:
        """Counts, not a rate. The same refusal `analysis` and `ranking` are built on.

        A peak and a measurement turn these into a utilisation, and both of those come
        from outside this module on purpose: what is derived here is falsifiable against
        a closed form, and what is not derived here is what would have been invented.
        """

        import dataclasses

        from open_cake_ir.compiler import work

        fields = {field.name for field in dataclasses.fields(work.WorkBound)}
        self.assertFalse(
            {name for name in fields if "time" in name or "second" in name or "peak" in name}
        )
        body = (
            ROOT / "src" / "open_cake_ir" / "compiler" / "work.py"
        ).read_text(encoding="utf-8")
        for absent in ("clock", "latency_ms", "tflops", "peak_"):
            with self.subTest(term=absent):
                self.assertNotIn(absent, body.lower().split('"""')[2].lower())


if __name__ == "__main__":
    unittest.main()
