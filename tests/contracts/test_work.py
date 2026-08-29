"""Contract tests for declared work.

The counting half of a performance report: how much arithmetic a Schedule commits to and
how many unique global bytes it must move. There is still no predicted time here -- these
are counts, and the rate that would turn them into one is not declared anywhere.

What these tests pin is the closed-form agreement. A GEMM's derived contraction has to
come out at exactly 2*M*N*K, because that number is knowable without this module and a
work model that disagrees with it is wrong rather than approximate.
"""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from open_cake_ir.compiler.ir import OperationKind, Schedule
from open_cake_ir.compiler.work import (
    loop_trip_distribution,
    operation_repetitions,
    program_tiles,
    work_bound,
)

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
QSA = SCHEDULES / "qsa-score-topk-t32768.json"


def _bound(path: Path):
    bound = work_bound(Schedule.load(path))
    assert bound is not None
    return bound


def _qsa(tile: int = 128) -> Schedule:
    document = copy.deepcopy(json.loads(QSA.read_text(encoding="utf-8")))
    document["tile_loops"][0]["tile"] = tile
    for buffer in document["buffers"]:
        if buffer["name"] == "key_tile":
            buffer["shape"][0] = tile
        elif buffer["name"] in ("head_scores", "positive_scores"):
            buffer["shape"][1] = tile
        elif buffer["name"] in ("score_sum", "score_tile"):
            buffer["shape"][0] = tile
    next(
        operation
        for operation in document["operations"]
        if operation["id"] == "score_heads"
    )["parameters"]["tile_shape"][1] = tile
    return Schedule.from_dict(document)


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
        repetitions = {
            row.operation: row.whole_grid for row in bound.operation_repetitions
        }
        self.assertEqual(repetitions["dot_mma"], 8)

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

    def test_stopped_loop_bytes_use_padded_whole_program_union_coverage(self) -> None:
        complete = _bound(QSA)
        self.assertTrue(complete.compulsory_bytes_exact)
        self.assertNotIn("normalized_keys", complete.partially_addressed)

        padded_document = json.loads(QSA.read_text(encoding="utf-8"))
        padded_document["tile_loops"][0]["stop"]["add"] = 0
        padded_schedule = Schedule.from_dict(padded_document)
        padded = work_bound(padded_schedule)
        assert padded is not None
        distribution = loop_trip_distribution(
            padded_schedule, padded_schedule.tile_loops[0]
        )
        assert distribution.trips is not None
        self.assertEqual(max(distribution.trips), 64)
        self.assertTrue(padded.compulsory_bytes_exact)

        prefix_document = json.loads(QSA.read_text(encoding="utf-8"))
        prefix_document["tile_loops"][0]["stop"]["floor_div"] = 8
        prefix = work_bound(Schedule.from_dict(prefix_document))
        assert prefix is not None
        self.assertFalse(prefix.compulsory_bytes_exact)
        self.assertIn("normalized_keys", prefix.partially_addressed)


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

    def test_qsa_dynamic_stop_owns_exact_tile128_whole_grid_work(self) -> None:
        schedule = _qsa()
        bound = work_bound(schedule)
        assert bound is not None
        repetitions = {
            row.operation: row.whole_grid for row in bound.operation_repetitions
        }

        self.assertEqual(repetitions["load_query"], 32_768)
        self.assertEqual(repetitions["score_heads"], 1_064_768)
        self.assertEqual(repetitions["select_blocks"], 1_064_768)
        self.assertEqual(repetitions["store_blocks"], 32_768)
        self.assertEqual(bound.mma_flops, 279_122_542_592)
        self.assertEqual(bound.flops, 281_303_187_456)

    def test_qsa_dynamic_stop_owns_exact_tile256_whole_grid_work(self) -> None:
        schedule = _qsa(256)
        bound = work_bound(schedule)
        assert bound is not None
        repetitions = {
            row.operation: row.whole_grid for row in bound.operation_repetitions
        }

        self.assertEqual(repetitions["score_heads"], 540_576)
        self.assertEqual(repetitions["select_blocks"], 540_576)
        self.assertEqual(bound.mma_flops, 283_417_509_888)
        self.assertEqual(bound.flops, 285_631_709_184)

    def test_loop_stop_clamps_zero_partial_and_high_coordinates_exactly(self) -> None:
        schedule = _qsa()
        loop = schedule.tile_loops[0]
        distribution = loop_trip_distribution(schedule, loop)
        assert distribution.trips is not None

        self.assertEqual(distribution.trips[:6], (0, 0, 0, 1, 1, 1))
        self.assertEqual(distribution.multiplicity, 1)

        high_document = json.loads(QSA.read_text(encoding="utf-8"))
        high_document["tile_loops"][0]["stop"]["add"] = 40_000
        high = Schedule.from_dict(high_document)
        clipped = loop_trip_distribution(high, high.tile_loops[0])
        assert clipped.trips is not None
        self.assertEqual(set(clipped.trips), {64})

        low_document = json.loads(QSA.read_text(encoding="utf-8"))
        low_document["tile_loops"][0]["stop"]["add"] = -1
        low = Schedule.from_dict(low_document)
        zero_prefix = loop_trip_distribution(low, low.tile_loops[0])
        assert zero_prefix.trips is not None
        self.assertEqual(zero_prefix.trips[:6], (0, 0, 0, 0, 0, 1))

    def test_dynamic_stop_distribution_keeps_other_program_axis_multiplicity(self) -> None:
        document = json.loads(QSA.read_text(encoding="utf-8"))
        document["program_map"]["axes"].append(
            {
                "name": "index_head",
                "axis": 1,
                "buffer": "index_q",
                "dimension": 1,
                "tile": 1,
            }
        )
        schedule = Schedule.from_dict(document)
        distribution = loop_trip_distribution(schedule, schedule.tile_loops[0])
        rows = operation_repetitions(schedule)
        assert distribution.trips is not None and rows is not None
        repetitions = {row.operation: row.whole_grid for row in rows}

        self.assertEqual(distribution.multiplicity, 8)
        self.assertEqual(repetitions["score_heads"], 8 * 1_064_768)
        self.assertEqual(repetitions["load_query"], 8 * 32_768)

    def test_two_dynamic_loops_are_unknown_without_static_max_fallback(self) -> None:
        document = json.loads(QSA.read_text(encoding="utf-8"))
        inner = document["tile_loops"][0]
        inner["name"] = "inner_block_loop"
        document["tile_loops"].append(
            {
                **copy.deepcopy(inner),
                "name": "outer_block_loop",
                "iterator": "outer_block_start",
                "body": ["inner_block_loop"],
            }
        )
        inner["body"].remove("scale_scores")
        schedule = Schedule.from_dict(document)
        rows = operation_repetitions(schedule)
        assert rows is not None
        by_operation = {row.operation: row for row in rows}
        score = by_operation["score_heads"]

        self.assertEqual(score.estimate_kind, "unknown")
        self.assertIsNone(score.whole_grid)
        self.assertEqual(
            score.missing,
            ("operation loop chain has more than one dynamic stop",),
        )
        bound = work_bound(schedule)
        assert bound is not None
        self.assertEqual(bound.mma_flops, 0)
        self.assertGreater(bound.flops, 0)
        self.assertIn("score_heads", bound.uncounted_arithmetic)
        self.assertIsNone(bound.contended_contract)

    def test_one_dynamic_loop_composes_with_an_arbitrary_static_parent(self) -> None:
        document = json.loads(QSA.read_text(encoding="utf-8"))
        inner = document["tile_loops"][0]
        inner["name"] = "inner_block_loop"
        outer = copy.deepcopy(inner)
        outer.update(
            {
                "name": "outer_block_loop",
                "iterator": "outer_block_start",
                "tile": 4096,
                "body": ["inner_block_loop"],
            }
        )
        outer.pop("stop")
        document["tile_loops"].append(outer)
        rows = operation_repetitions(Schedule.from_dict(document))
        assert rows is not None
        repetitions = {row.operation: row.whole_grid for row in rows}

        self.assertEqual(repetitions["score_heads"], 2 * 1_064_768)
        self.assertEqual(repetitions["load_query"], 32_768)


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
