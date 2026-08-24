#!/usr/bin/env python3
"""Ask whether any declaration-derived quantity orders candidates past one round.

Removing the wave term left the ranking with nothing to say about a grid that overfills the
device, which is most real workloads (`docs/ANALYSIS_CALIBRATION.md`). Declining was the
honest response to a refuted term, but it is not the end state: the question is whether
some *other* quantity the Compiler already derives orders those candidates, and that is a
measurement rather than an argument.

So this builds one workload's candidate set the way the ranking's real caller does -- same
Schedule semantics, different tilings, register budgets and role widths -- checks each is
correct, times each, and scores every candidate order key against the measurement. A key
that beats a coin decisively is a term the model could carry with evidence behind it. One
that does not is a term that should stay out.

The refuted wave term is scored alongside the others deliberately. If it were to come out
concordant on a same-workload set the earlier refutation would need re-reading, and a
calibration that only tests the hypotheses it likes is not a calibration.

Run under the broker in exclusive mode: this is a benchmark.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
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

_ROW_BUFFERS = ("x_tile", "sq", "normed", "y_tile")
_SCALAR_BUFFERS = ("sumsq", "meansq", "shifted", "inv_rms")


def _variant(base: dict, batch: int, tile: int, registers: int, warps: int) -> dict:
    document = json.loads(json.dumps(base))
    for buffer in document["buffers"]:
        if buffer["name"] in ("x", "y"):
            buffer["shape"][0] = batch
        elif buffer["name"] in _ROW_BUFFERS:
            buffer["shape"][0] = tile
        elif buffer["name"] in _SCALAR_BUFFERS:
            buffer["shape"] = [tile]
    for axis in document["program_map"]["axes"]:
        if axis["name"] == "row_block":
            axis["tile"] = tile
    document["roles"][0]["warps"] = list(range(warps))
    # The residency commitment is checked against the bound rather than shaping the code,
    # so it is pinned at the loosest value every variant can honour instead of swept.
    document["residency"] = {
        "ctas_per_multiprocessor": 1,
        "registers_per_thread": registers,
    }
    document["schedule_id"] = f"rmsnorm-t{tile}-r{registers}-w{warps}"
    return document


def _reference(x, gamma):
    import torch

    scaled = x.to(torch.float32)
    inverse = torch.rsqrt(scaled.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    return scaled * inverse * gamma


def _time_ms(launch, arguments, *, reps: int, warmup: int, flush) -> float:
    import torch

    for _ in range(warmup):
        launch(*arguments)
    torch.cuda.synchronize()
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        flush.zero_()
        torch.cuda.synchronize()
        start.record()
        launch(*arguments)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _keys(row: dict) -> dict[str, float]:
    """Every ordering the model could carry, each derived only from declarations.

    Lower is better in all of them, so a key and the measured latency agree on a pair when
    they put it the same way round.
    """

    resident = row["ctas_per_multiprocessor_upper_bound"]
    per_round = resident * row["multiprocessor_count"]
    return {
        "fewer_ctas": row["ctas"],
        "more_ctas": -row["ctas"],
        "higher_residency": -resident,
        "more_resident_warps": -(resident * row["warps"]),
        "fewer_waves": (row["ctas"] + per_round - 1) // per_round,
        "smaller_tile": row["tile"],
        "larger_tile": -row["tile"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", default="corpus/schedules/rmsnorm-b8-smoke.json")
    parser.add_argument("--revision", default="compiler/revision.lock.json")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--reps", type=int, default=41)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=2e-3)
    parser.add_argument("--out", default="ranking-at-scale.json")
    arguments = parser.parse_args()

    import torch

    compiler = Compiler.load(ROOT, arguments.revision)
    base = json.loads((ROOT / arguments.schedule).read_text(encoding="utf-8"))
    target = Target.from_dict(
        json.loads(
            (ROOT / f"compiler/targets/{base['target']}.json").read_text(encoding="utf-8")
        )
    )
    multiprocessors = target.occupancy.multiprocessor_count
    workspace = tempfile.mkdtemp(prefix="cake-ranking-")
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    x = None
    rows: list[dict] = []
    refused = 0
    revision: dict[str, str] = {}
    for tile, registers, warps in itertools.product(
        (16, 32, 64, 128, 256), (64, 96, 128, 168, 224), (4, 8)
    ):
        document = _variant(base, arguments.batch, tile, registers, warps)
        assessment = compiler.assess(document)
        if not assessment.lowering_eligible:
            refused += 1
            continue
        revision = {
            "id": assessment.compiler_revision_id,
            "sha256": assessment.compiler_revision_sha256,
        }
        bound = residency_upper_bound(
            Schedule.from_dict(json.loads(assessment.schedule_bytes)), target
        )
        scored, _ = compiler.rank([assessment])
        lowering = compiler.lower(assessment)

        module_path = Path(workspace) / f"{document['schedule_id'].replace('-', '_')}.py"
        module_path.write_text(lowering.source, encoding="utf-8")
        specification = importlib.util.spec_from_file_location(
            module_path.stem, module_path
        )
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        launch = getattr(module, lowering.entry_point)

        shape = next(
            tuple(item["shape"]) for item in document["buffers"] if item["name"] == "x"
        )
        if x is None or tuple(x.shape) != shape:
            torch.manual_seed(0)
            x = torch.randn(shape, dtype=torch.float32, device="cuda")
            gamma = torch.randn(shape[2], dtype=torch.float32, device="cuda")
            expected = _reference(x, gamma)
        out = torch.empty_like(x)

        launch(x, gamma, out)
        torch.cuda.synchronize()
        deviation = (out - expected).abs().max().item()
        if deviation > arguments.tolerance:
            # An incorrect candidate has no place in a ranking calibration: the order it
            # would take is not the order of anything the workload asked for.
            print(f"{document['schedule_id']}: INCORRECT, max deviation {deviation:.2e}",
                  flush=True)
            continue

        median = _time_ms(launch, (x, gamma, out), reps=arguments.reps,
                          warmup=arguments.warmup, flush=flush)
        grid_x, grid_y, grid_z = assessment.analysis["grid"]
        rows.append(
            {
                "schedule_id": document["schedule_id"],
                "tile": tile,
                "registers_per_thread": registers,
                "warps": warps,
                "ctas": grid_x * grid_y * grid_z,
                "ctas_per_multiprocessor_upper_bound": bound.ctas_per_multiprocessor,
                "binding_resource": bound.binding.resource,
                "multiprocessor_count": multiprocessors,
                "ranked_by_the_model": bool(scored),
                "max_deviation": deviation,
                "median_ms": median,
            }
        )
        print(f"{document['schedule_id']:24s} ctas={rows[-1]['ctas']:6d} "
              f"r<={bound.ctas_per_multiprocessor:2d} {median * 1000:8.2f}us", flush=True)
        del out

    device = torch.cuda.get_device_properties(0)
    Path(arguments.out).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "schedule_id": base["schedule_id"],
                "target": base["target"],
                "compiler_revision": revision,
                "device": {
                    "name": device.name,
                    "multiprocessor_count": device.multi_processor_count,
                },
                "batch": arguments.batch,
                "reps": arguments.reps,
                "refused_by_the_gates": refused,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\n{len(rows)} correct candidates, {refused} refused by the gates")
    pairs = [(i, j) for i in range(len(rows)) for j in range(i + 1, len(rows))]
    print(f"scoring {len(pairs)} pairs\n")
    print(f"{'order key':24s} concordant   share")
    for name in _keys(rows[0]) if rows else {}:
        agree = 0
        for i, j in pairs:
            left = _keys(rows[i])[name] - _keys(rows[j])[name]
            slower = rows[i]["median_ms"] - rows[j]["median_ms"]
            if left * slower > 0:
                agree += 1
            elif left == 0 and slower == 0:
                agree += 1
        print(f"{name:24s} {agree:5d}/{len(pairs):<6d} {agree / len(pairs):6.1%}")
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
