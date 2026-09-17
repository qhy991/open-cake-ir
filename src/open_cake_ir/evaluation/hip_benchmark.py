"""Per-dispatch device time for one AMDGCN kernel, with no fallback.

The AMDGCN peer of `StrictCuptiBenchmark`, and it takes the same call: a function to run,
how many untimed runs precede the timed ones, how many samples to return, and whether the
device is reset before each. It returns one millisecond value per timed dispatch, which is
what the paired assay consumes -- a cohort, not a mean, because the coefficient of
variation is what decides a cohort is stable enough to compare.

Why this source and not the obvious one. CUPTI is an in-process API, which is what lets
the CUDA assay wrap one callable; `rocprofv2`, the profiler this platform ships, wraps a
whole process and would have changed the assay's shape rather than its source. Three
candidates were measured on a BW1101 instead of argued:

* `torch.cuda.Event`: 8.48 us empty interval against kernels of 3-7 us.
* the raw HIP event pair through ctypes: 5.60 us min, 6.08 us median empty interval, and
  it reads a 1 MiB `mul` at 15.68 us where the profiler reads 3.36.
* `torch.profiler`'s device activity, which is roctracer underneath: the same `mul` at
  3.071 us against `rocprofv2`'s retained 3.36 us for that kernel and shape.

roctracer's in-process activity API is present on this host, which makes the third one
CUPTI's structural peer rather than a convenience; going at roctracer's records directly
would mean pinning a record layout that varies by version, and torch is a package the
Executor host already pins.

**What the interval includes**: one dispatch, begin to end, as roctracer reports it. Host
launch overhead is outside it.

**What resets the device**: a buffer larger than L2, zeroed before each timed dispatch.
HIP offers no flush API. Measured: 244-251 us a sample, and it changes the answer only
below about 4 MiB of working set -- at and above that the kernel evicts its own input
while running and warm and cold are indistinguishable.

**Resolution is coarse and the caller is told.** Twenty-five dispatches of a 3 us kernel
produced three distinct durations, so the reported spread of a short kernel is largely the
timer's quantum rather than the kernel's own variance. `resolution_us` reports the smallest
non-zero difference the cohort actually showed; a coefficient of variation taken over
samples that coarse says more about the timer than the candidate, and whoever reads one
should know which.
"""

from __future__ import annotations

from typing import Callable, Sequence

# The reset buffer, in fp32 elements: 256 MiB. This said "larger than this device's 8 MiB
# L2" when one device routed through this assay. Two do now -- gfx938 on a BW1101 and
# gfx1151 on Strix Halo -- and neither Target document states a cache capacity, so no
# number here can be justified against the device it runs on. What is stated instead is
# what this size is for and what it is not: it is large enough that no cache either device
# is known to have can hold it, chosen once and applied to both, and it is not read from
# the Target. A Target that declares its cache should size this from that declaration; a
# caller that knows better passes `flush_elements`.
_FLUSH_ELEMENTS = 64 * 1024 * 1024


class HipDispatchBenchmark:
    """Time one named kernel's dispatches through the admitted runtime's own profiler."""

    def __init__(self, kernel_name: str, *, flush_elements: int = _FLUSH_ELEMENTS) -> None:
        if not isinstance(kernel_name, str) or not kernel_name:
            raise ValueError("a dispatch benchmark times one named kernel")
        if not isinstance(flush_elements, int) or flush_elements <= 0:
            raise ValueError("the device-state reset needs a positive buffer")
        self.kernel_name = kernel_name
        self._flush_elements = flush_elements
        self._flush = None
        self.resolution_us: float | None = None
        self.non_target_dispatches: int | None = None

    def _reset_buffer(self):
        import torch

        if self._flush is None:
            self._flush = torch.empty(self._flush_elements, dtype=torch.float32,
                                      device="cuda")
        return self._flush

    def __call__(
        self,
        function: Callable[[], None],
        *,
        dry_run_iters: int,
        repeat_iters: int,
        cold_l2_cache: bool,
        use_cuda_graph: bool,
    ) -> Sequence[float]:
        import torch
        from torch.profiler import ProfilerActivity, profile

        if use_cuda_graph:
            raise ValueError(
                "this assay times individual dispatches; nothing here captures a graph, "
                "and reporting a captured replay under the same policy would measure a "
                "different thing"
            )
        if any(type(value) is not int or value <= 0
               for value in (dry_run_iters, repeat_iters)):
            raise ValueError("dispatch benchmark iteration counts must be positive")
        reset = self._reset_buffer() if cold_l2_cache else None
        for _ in range(dry_run_iters):
            function()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as session:
            for _ in range(repeat_iters):
                if reset is not None:
                    reset.zero_()
                function()
            torch.cuda.synchronize()
        device_events = [event for event in session.events()
                         if getattr(event, "device_time", 0.0)]
        samples = [float(getattr(event, "device_time", 0.0)) for event in device_events
                   if self.kernel_name in getattr(event, "name", "")]
        # Counted, not silently dropped. A dispatch this assay does not name contributes
        # no sample and so does not move the count check below -- so a candidate that did
        # part of its work in a second kernel would have that work vanish and read as
        # faster. This count is what reaches the receipt so a reader can tell that apart
        # from a clean cohort; it never widens or narrows a measurement.
        #
        # The device reset is the assay's own dispatch and is subtracted by construction
        # rather than by name: `reset.zero_()` is issued inside the profiled region, once
        # per timed iteration, so the raw difference is exactly `repeat_iters` on the cold
        # path and a second kernel would hide inside it. Measured on gfx1151: raw 10 of 10
        # with the reset, 0 without. Matching on the kernel name torch gives that zeroing
        # (`vectorized_elementwise_kernel<...>`) would be reading how a mnemonic is spelled;
        # the assay knows how many resets it issued.
        own_resets = repeat_iters if cold_l2_cache else 0
        observed = len(device_events) - len(samples) - own_resets
        if observed < 0:
            # One device event per reset is a torch property this assay assumes, not one
            # it can see. If a future runtime spells `zero_()` as a fill the profiler does
            # not surface, the subtraction goes negative -- refused here rather than
            # written into a receipt as a negative count nobody can read.
            raise ValueError(
                f"the profiler surfaced {len(device_events)} device events for "
                f"{repeat_iters} timed dispatches and {own_resets} device resets; this "
                "assay subtracts one event per reset and cannot account for the remainder")
        self.non_target_dispatches = observed
        if len(samples) != repeat_iters:
            raise ValueError(
                f"the profiler attributed {len(samples)} dispatches of "
                f"{self.kernel_name!r}, not the {repeat_iters} this cohort launched"
            )
        # The smallest real step this cohort showed, so a reader can tell a spread from a
        # quantum. Reported, never used to widen or narrow anything.
        steps = sorted({round(abs(a - b), 6) for a in samples for b in samples} - {0.0})
        self.resolution_us = steps[0] if steps else None
        return [value / 1000.0 for value in samples]
