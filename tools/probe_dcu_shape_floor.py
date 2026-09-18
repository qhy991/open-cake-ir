#!/usr/bin/env python3
"""Where does reported time start tracking bytes on this target?

F-2026-09-18-002 rests part of its argument on a footprint sweep, and until this existed
the sweep lived in a scratch script on the host: the numbers could be read in the record
and reproduced nowhere. That is the gap `tools/observe_lowered_kernel.py` was written to
close for a different record, and the same rule applies -- evidence that can only be
renewed by editing it is not evidence.

The kernel is deliberately trivial: a copy with a scale. Anything more would mix the
arithmetic into the question, and the question is only whether duration follows size. It
reads the same source the paired assay reads -- roctracer's per-dispatch device time
through torch.profiler -- so the answer is in the assay's own units rather than a second
timer's.

This is a probe, not a campaign: it seals no evidence, takes no candidate and compares
nothing to a baseline. What it produces is a curve for one synthetic kernel on one device,
which is why a record citing it must say so rather than treating it as a task measurement.

    python3 tools/probe_dcu_shape_floor.py --out sweep.json
"""

from __future__ import annotations

import argparse
import json

#: Footprints to sample, in MiB moved (one array read, one written).
DEFAULT_SIZES = (0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0, 16.0, 64.0, 256.0)


def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def scale(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        tl.store(y_ptr + offsets, tl.load(x_ptr + offsets, mask=mask) * 2.0, mask=mask)

    return scale


def timed(scale, elements, *, repeats=25, warmup=11, block=1024):
    import torch
    import triton
    from torch.profiler import ProfilerActivity, profile

    x = torch.randn(elements, device="cuda")
    y = torch.empty_like(x)
    grid = (triton.cdiv(elements, block),)
    for _ in range(warmup):
        scale[grid](x, y, elements, BLOCK=block)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as session:
        for _ in range(repeats):
            scale[grid](x, y, elements, BLOCK=block)
        torch.cuda.synchronize()
    samples = sorted(float(event.device_time) for event in session.events()
                     if getattr(event, "device_time", 0.0)
                     and "scale" in getattr(event, "name", ""))
    if not samples:
        raise SystemExit("the profiler attributed no dispatch of the probe kernel")
    steps = sorted({round(abs(a - b), 6) for a in samples for b in samples} - {0.0})
    return samples[len(samples) // 2], len(samples), (steps[0] if steps else None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="write here instead of stdout")
    parser.add_argument("--sizes", help="comma-separated MiB moved",
                        default=",".join(str(s) for s in DEFAULT_SIZES))
    arguments = parser.parse_args(argv)

    import torch
    import triton

    scale = _kernel()
    rows = []
    for mib in [float(s) for s in arguments.sizes.split(",")]:
        elements = int(mib * 2 ** 20 / 4 / 2)        # one array read, one written
        median, count, resolution = timed(scale, elements)
        moved = elements * 4 * 2 / 2 ** 20
        rows.append({"mib_moved": round(moved, 3), "median_us": round(median, 3),
                     "samples": count, "resolution_us": resolution,
                     "gbytes_per_second": round((moved * 2 ** 20) / (median * 1e-6) / 1e9, 1)})
    document = {
        "schema_version": 1,
        "kind": "synthetic_footprint_sweep_v1",
        "generated_by": "tools/probe_dcu_shape_floor.py",
        "device": torch.cuda.get_device_name(0),
        "triton": triton.__version__,
        "kernel": "scale: one load, one multiply, one store, BLOCK=1024",
        "is_campaign_evidence": False,
        "sweep": rows,
    }
    rendered = json.dumps(document, indent=1) + "\n"
    if arguments.out:
        with open(arguments.out, "w") as handle:
            handle.write(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
