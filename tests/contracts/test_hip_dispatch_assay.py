"""What the AMDGCN assay counts, and what it subtracts.

The count exists so a cohort that timed part of a candidate is distinguishable from one
that timed all of it. It shipped once inert -- written, commented as "reported for the
reader", read by nobody, and provably so: forcing it to zero left the whole suite
byte-identical. These tests fail if it stops being computed or stops being subtracted.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from open_cake_ir.evaluation.hip_benchmark import HipDispatchBenchmark  # noqa: E402

KERNEL = "_cake_thing_kernel"


class _Event:
    def __init__(self, name, device_time=1.0):
        self.name = name
        self.device_time = device_time


class _Session:
    def __init__(self, events):
        self._events = events

    def events(self):
        return self._events

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _run(benchmark, events, *, repeat_iters, cold_l2_cache):
    """Drive the assay against a fixed profiler session, with no device."""

    session = _Session(events)
    torch = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"torch": torch,
                                       "torch.profiler": mock.MagicMock(
                                           ProfilerActivity=mock.MagicMock(),
                                           profile=lambda **kw: session)}):
        return benchmark(lambda: None, dry_run_iters=1, repeat_iters=repeat_iters,
                         cold_l2_cache=cold_l2_cache, use_cuda_graph=False)


class EventSelectionTests(unittest.TestCase):
    """The filter that keeps host-side events out of the count.

    Every fake event carried a device time, so dropping the filter changed nothing and the
    thing keeping `FunctionEvent`s out of a device count was pinned by nothing.
    """

    def test_an_event_with_no_device_time_is_neither_sample_nor_non_target(self):
        events = [_Event(KERNEL), _Event(KERNEL),
                  _Event("aten::mul", device_time=0.0),
                  _Event("cudaLaunchKernel", device_time=0.0)]
        benchmark = HipDispatchBenchmark(KERNEL)
        samples = _run(benchmark, events, repeat_iters=2, cold_l2_cache=False)
        self.assertEqual(len(samples), 2)
        self.assertEqual(benchmark.non_target_dispatches, 0)


class UnaccountedEventsTests(unittest.TestCase):
    """The subtraction assumes one device event per reset, which it cannot see.

    A runtime that stopped surfacing the reset would drive the count negative and write it
    into a hashed receipt. The refusal for that shipped with no test: making it dead left
    the suite byte-identical.
    """

    def test_fewer_events_than_the_resets_it_issued_is_refused(self):
        events = [_Event(KERNEL) for _ in range(4)]
        benchmark = HipDispatchBenchmark(KERNEL)
        with self.assertRaisesRegex(ValueError, "cannot account for the remainder"):
            _run(benchmark, events, repeat_iters=4, cold_l2_cache=True)

    def test_the_refusal_names_what_it_saw(self):
        events = [_Event(KERNEL) for _ in range(4)] + [_Event("reset")]
        benchmark = HipDispatchBenchmark(KERNEL)
        with self.assertRaises(ValueError) as raised:
            _run(benchmark, events, repeat_iters=4, cold_l2_cache=True)
        message = str(raised.exception)
        self.assertIn("5 device events", message)
        self.assertIn("4 timed dispatches", message)
        self.assertIn("4 device resets", message)

    def test_exactly_enough_events_is_admitted(self):
        """The boundary: one reset per iteration and nothing else."""

        events = ([_Event(KERNEL) for _ in range(4)]
                  + [_Event("reset") for _ in range(4)])
        benchmark = HipDispatchBenchmark(KERNEL)
        _run(benchmark, events, repeat_iters=4, cold_l2_cache=True)
        self.assertEqual(benchmark.non_target_dispatches, 0)


class NonTargetDispatchTests(unittest.TestCase):
    def test_the_assay_s_own_reset_is_subtracted_by_construction(self):
        """`reset.zero_()` runs inside the profiled region, once per timed iteration.

        Measured on gfx1151 before this was subtracted: the raw difference was 10 of 10
        on the cold path, so a second kernel would have hidden inside it.
        """

        events = ([_Event(KERNEL) for _ in range(10)]
                  + [_Event("void at::native::vectorized_elementwise_kernel<4, ...>")
                     for _ in range(10)])
        benchmark = HipDispatchBenchmark(KERNEL)
        samples = _run(benchmark, events, repeat_iters=10, cold_l2_cache=True)
        self.assertEqual(len(samples), 10)
        self.assertEqual(benchmark.non_target_dispatches, 0)

    def test_a_second_kernel_is_counted_even_through_the_reset(self):
        """The case the count exists for: work the candidate did elsewhere."""

        events = ([_Event(KERNEL) for _ in range(10)]
                  + [_Event("vectorized_elementwise_kernel") for _ in range(10)]
                  + [_Event("_cake_thing_epilogue_kernel") for _ in range(3)])
        benchmark = HipDispatchBenchmark(KERNEL)
        _run(benchmark, events, repeat_iters=10, cold_l2_cache=True)
        self.assertEqual(benchmark.non_target_dispatches, 3)

    def test_without_the_reset_nothing_is_subtracted(self):
        events = [_Event(KERNEL) for _ in range(4)] + [_Event("someone_else")]
        benchmark = HipDispatchBenchmark(KERNEL)
        _run(benchmark, events, repeat_iters=4, cold_l2_cache=False)
        self.assertEqual(benchmark.non_target_dispatches, 1)

    def test_a_cohort_it_cannot_attribute_is_refused_not_averaged(self):
        benchmark = HipDispatchBenchmark("no_such_kernel")
        with self.assertRaisesRegex(ValueError, "attributed 0 dispatches"):
            _run(benchmark, [_Event(KERNEL)], repeat_iters=1, cold_l2_cache=False)


# Where this count has to arrive is asserted where it arrives: see
# `tests/contracts/test_tile_gpu_worker.py`, which drives `_evaluate_tile_candidate` with
# an assay carrying the attribute and asserts the value in the receipt. A test here that
# grepped evaluate.py's source text for the field name passed against a line the suite
# never executed, which is the defect it was written to prevent.


if __name__ == "__main__":
    unittest.main()
