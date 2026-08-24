#!/usr/bin/env python3
"""Measure whether the ranking's wave term describes how the device spends time.

`Cost.order` used to sort on `waves` first, on the reasoning that a wave is a round of the
whole device and a partial final round is dead time. Every candidate in the ranking
calibration fit in a single wave, so that term carried the whole claim and none of the
evidence. This is the instrument that tested it; what it found, and what the model lost as
a result, is in `docs/ANALYSIS_CALIBRATION.md`.

This sweeps one Schedule's batch extent with the tile fixed, so per-CTA work is constant
and the only thing that changes is how the grid quantises against the device. Two things
are then visible in one plot of latency against CTA count:

* whether a staircase exists at all -- flat within a wave, a step at the boundary. If
  latency instead rises smoothly with CTA count, waves are not how this kernel spends time
  and ordering by wave count is ordering by problem size.
* whether the model's boundary is in the right place. `residency_upper_bound` is an upper
  bound, so the predicted CTAs per multiprocessor is at least the achieved one, and the
  predicted boundary therefore sits at or beyond the real one. A measured step *before*
  the predicted boundary is the bound being loose, which is expected; a predicted step
  with no measured step anywhere is the term being wrong.

The analysis has to be run against every plausible wave size rather than only the model's,
or a loose residency bound and a missing staircase are indistinguishable.

Run under the broker in exclusive mode: this is a benchmark.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler.analysis import residency_upper_bound  # noqa: E402
from open_cake_ir.compiler.core import Compiler  # noqa: E402
from open_cake_ir.compiler.ir import Schedule  # noqa: E402
from open_cake_ir.compiler.target import Target  # noqa: E402


def _residency(assessment, target) -> int | None:
    """The residency ceiling, read straight from the analysis rather than from a Cost.

    Past one round of the device `cost()` declines, and that region is most of this sweep.
    The ceiling is a fact about the Schedule either way, so it is taken from the same
    function the ranking uses rather than reconstructed.
    """

    bound = residency_upper_bound(
        Schedule.from_dict(json.loads(assessment.schedule_bytes)), target
    )
    return None if bound is None else bound.ctas_per_multiprocessor


def _grid(assessment) -> int:
    x, y, z = assessment.analysis["grid"]
    return x * y * z


def _variant(base: dict, batch: int) -> dict:
    document = json.loads(json.dumps(base))
    for buffer in document["buffers"]:
        if buffer["name"] in ("x", "y"):
            buffer["shape"][0] = batch
    document["schedule_id"] = f"{base['schedule_id']}-b{batch:04d}"
    return document


def _time_ms(launch, arguments, *, reps: int, warmup: int, flush) -> float:
    import torch

    for _ in range(warmup):
        launch(*arguments)
    torch.cuda.synchronize()
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        # A partial wave is only dead time if the device is not already saturated, so the
        # cache state has to be the same on every sample or the regime moves under us.
        flush.zero_()
        torch.cuda.synchronize()
        start.record()
        launch(*arguments)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", default="corpus/schedules/rmsnorm-b8-smoke.json")
    parser.add_argument("--revision", default="compiler/revision.lock.json")
    parser.add_argument("--first", type=int, default=60)
    parser.add_argument("--last", type=int, default=300)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--reps", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", default="wave-term.json")
    arguments = parser.parse_args()

    import torch

    compiler = Compiler.load(ROOT, arguments.revision)
    base = json.loads((ROOT / arguments.schedule).read_text(encoding="utf-8"))
    target = Target.from_dict(
        json.loads(
            (ROOT / f"compiler/targets/{base['target']}.json").read_text(encoding="utf-8")
        )
    )
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    workspace = tempfile.mkdtemp(prefix="cake-wave-term-")
    revision: dict[str, str] = {}
    rows = []
    for batch in range(arguments.first, arguments.last + 1, arguments.step):
        assessment = compiler.assess(_variant(base, batch))
        if not assessment.lowering_eligible:
            rows.append(
                {
                    "batch": batch,
                    "refused": [finding.code for finding in assessment.findings],
                }
            )
            continue
        revision = {
            "id": assessment.compiler_revision_id,
            "sha256": assessment.compiler_revision_sha256,
        }
        scored, _ = compiler.rank([assessment])
        # The point of the sweep is the region where the model declines, so a Cost is
        # absent for most of it. The grid and the residency ceiling are facts either way.
        predicted = scored[0] if scored else None
        residency = _residency(assessment, target)
        lowering = compiler.lower(assessment)

        # Triton reads a @jit function's own source back off disk, so a lowering has to
        # reach a real file before it can be launched. Nothing here edits it.
        module_path = Path(workspace) / f"{lowering.entry_point}_b{batch:04d}.py"
        module_path.write_text(lowering.source, encoding="utf-8")
        specification = importlib.util.spec_from_file_location(
            module_path.stem, module_path
        )
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        launch = getattr(module, lowering.entry_point)

        shape = tuple(
            buffer["shape"]
            for buffer in _variant(base, batch)["buffers"]
            if buffer["name"] == "x"
        )[0]
        x = torch.randn(tuple(shape), dtype=torch.float32, device="cuda")
        gamma = torch.randn(shape[2], dtype=torch.float32, device="cuda")
        # Preallocated, so an allocator call is not inside the measurement.
        out = torch.empty_like(x)

        median = _time_ms(launch, (x, gamma, out), reps=arguments.reps,
                          warmup=arguments.warmup, flush=flush)
        ctas = _grid(assessment)
        rows.append(
            {
                "batch": batch,
                "ctas": ctas,
                "ctas_per_multiprocessor_upper_bound": residency,
                "ranked": predicted is not None,
                "median_ms": median,
            }
        )
        print(
            f"batch={batch:4d} ctas={ctas:5d} r<={residency} "
            f"ranked={'y' if predicted else 'n'} median={median * 1000:8.2f}us "
            f"per_cta={median / ctas * 1e6:7.3f}ns",
            flush=True,
        )
        del x, gamma, out

    device = torch.cuda.get_device_properties(0)
    document = {
        "schema_version": 1,
        "schedule_id": base["schedule_id"],
        "compiler_revision": revision,
        "target": base["target"],
        "device": {
            "name": device.name,
            "multiprocessor_count": device.multi_processor_count,
        },
        "reps": arguments.reps,
        "l2_flush_bytes": flush.numel(),
        "rows": rows,
    }
    Path(arguments.out).write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
