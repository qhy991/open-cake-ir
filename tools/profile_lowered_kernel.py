#!/usr/bin/env python3
"""Measure what bounds a lowered kernel's residency and what it moved, against the analysis.

`docs/ANALYSIS_CALIBRATION.md` reports that the residency bound held and that the binding
resource was predicted correctly, and until now those numbers came from an Nsight run
outside this repository. So the table could be read and not repeated: when a Schedule
changed, the measured column went stale with no way to renew it. That happened.

This is that instrument. It lowers the Schedule, runs one launch under Nsight Compute, and
puts the analysis's prediction and the profiler's measurement side by side.

The comparison is exact about direction. `residency_upper_bound` is an upper bound, so it
is *sound* when the measured limit is at or below it and refuted when the measurement is
higher. Physical register allocation is reported only from measurement; the logical
register-pressure proxy has no sound lower-bound direction. New reports use schema v2
and omit the obsolete null register-floor fields. The measured register-limit resource
is named `registers`; schema v1 reports and their assertions remain unchanged.

It makes a second comparison of the same shape. `work_bound` counts the unique global
bytes the Schedule commits to moving; Nsight counts the bytes that crossed L2 and DRAM.
Their quotient is traffic amplification, and it is the reuse question no declaration can
answer: a Schedule that re-reads one input per output block declares the input once and
fetches it many times, and only the measurement knows how many.

That comparison is not a utilisation and does not pretend to be one. No Target declares a
peak bandwidth, so nothing here divides by a rate. It reports two byte counts, their
quotient, and the bytes per second this launch achieved -- all of which are measured or
declared, and none of which is a prediction.

Run under the broker in exclusive mode: a profiler serialises the device.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kernel_inputs import build_inputs  # noqa: E402
from kernel_oracles import ORACLE_BY_ENTRY_POINT  # noqa: E402
from open_cake_ir.compiler.performance.residency import residency_upper_bound  # noqa: E402
from open_cake_ir.compiler.core import Compiler  # noqa: E402
from open_cake_ir.compiler.ir import Schedule  # noqa: E402
from open_cake_ir.compiler.target import Target  # noqa: E402
from open_cake_ir.compiler.performance.utilization import Utilization, utilization  # noqa: E402
from open_cake_ir.compiler.performance.work import WorkBound, work_bound  # noqa: E402

# Nsight reports physical resource limits. Preserve their values and name the
# register limit as physical registers, never as the logical pressure proxy.
_LIMITS = {
    "launch__occupancy_limit_registers": "registers",
    "launch__occupancy_limit_shared_mem": "shared_memory",
    "launch__occupancy_limit_blocks": "blocks",
    "launch__occupancy_limit_warps": "threads",
}
# What the memory system actually moved, against what the Schedule says has to move.
# `compulsory_bytes` counts each global Buffer once; these count every byte that crossed
# a level. The quotient is traffic amplification -- how many times the kernel asked for
# the same data -- which is the reuse question a declaration cannot answer on its own.
#
# Both levels are read because they answer different halves of it. L2 traffic is what the
# multiprocessors requested, so it sees a re-read whether or not DRAM did. DRAM traffic
# sees only what the cache could not absorb, and on a Corpus Schedule whose whole working
# set is under a megabyte that can sit *below* the compulsory count -- which is a fact
# about the cache and this device, not a Schedule that moved less than it declared.
_TRAFFIC = {
    "dram__bytes_read.sum": "dram_read_bytes",
    "dram__bytes_write.sum": "dram_written_bytes",
    "lts__t_bytes.sum": "l2_bytes",
}
_DURATION = "gpu__time_duration.sum"
_METRICS = ["launch__registers_per_thread", *_LIMITS, *_TRAFFIC, _DURATION]

# Nsight rescales values for display -- a byte count comes back as "Mbyte" when it is
# large enough -- so a reader that ignored the unit column would silently be off by six
# orders of magnitude on exactly the kernels worth profiling. `--print-units base` turns
# the scaling off; these are the units that must then come back, and a different one is
# refused rather than converted by a guessed factor.
_BASE_UNITS = {
    **{metric: "byte" for metric in _TRAFFIC},
    _DURATION: "second",
}

_DRIVER = '''
import importlib.util, json, sys
from pathlib import Path
sys.path.insert(0, {tools!r})
sys.path.insert(0, {src!r})
from kernel_inputs import build_inputs
import torch

document = json.loads(Path({schedule!r}).read_text())
specification = importlib.util.spec_from_file_location("cake_lowered", {module!r})
module = importlib.util.module_from_spec(specification)
specification.loader.exec_module(module)
launch = getattr(module, "launch_once", None) or getattr(module, {entry!r})

torch.manual_seed(0)
inputs = build_inputs(document, torch)
launch(*inputs)
torch.cuda.synchronize()
'''


def _profile(command: list[str]) -> str:
    finished = subprocess.run(command, capture_output=True, text=True)
    if finished.returncode != 0:
        raise SystemExit(
            f"ncu exited {finished.returncode}\n{finished.stdout}\n{finished.stderr}"
        )
    # Nsight writes its own preamble ahead of the CSV, so the header has to be sought
    # rather than assumed to be line one.
    lines = finished.stdout.splitlines()
    for index, line in enumerate(lines):
        if line.startswith('"ID"') or line.startswith("ID,"):
            return "\n".join(lines[index:])
    raise SystemExit(f"no CSV header in ncu output:\n{finished.stdout}")


def _measured(table: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in csv.DictReader(io.StringIO(table)):
        name = row.get("Metric Name")
        raw = (row.get("Metric Value") or "").replace(",", "")
        if name not in _METRICS or not raw:
            continue
        expected = _BASE_UNITS.get(name)
        unit = (row.get("Metric Unit") or "").strip()
        if expected is not None and unit != expected:
            # A converted value would be indistinguishable from a correct one in the
            # stored record, so the wrong unit stops the run instead of being scaled.
            raise SystemExit(
                f"ncu reported {name} in {unit!r}, not the base unit {expected!r}"
            )
        values[name] = float(raw)
    missing = [metric for metric in _METRICS if metric not in values]
    if missing:
        raise SystemExit(f"ncu reported no value for {', '.join(missing)}")
    return values


def traffic_comparison(bound: WorkBound, values: dict[str, float]) -> dict[str, object]:
    """Put what the Schedule commits to moving beside what the memory system moved.

    Neither amplification is a utilisation: no peak bandwidth is declared on any Target,
    so nothing here divides by one. What they are is a ratio of two byte counts, and the
    achieved rate below is measured bytes over measured seconds -- a fact about this
    launch that needs no second declaration to be true.

    An amplification is only as sound as the byte count under it. When the Schedule
    addresses a Buffer through a runtime coordinate, a valid prefix or a subrange, the
    compulsory count charges it in full and the quotient is therefore an under-estimate
    of the real amplification. That is why `compulsory_bytes_exact` travels with it.
    """

    dram = values["dram__bytes_read.sum"] + values["dram__bytes_write.sum"]
    l2 = values["lts__t_bytes.sum"]
    seconds = values[_DURATION]
    compulsory = bound.compulsory_bytes
    return {
        "compulsory_bytes": compulsory,
        "compulsory_bytes_exact": bound.compulsory_bytes_exact,
        "dram_bytes": dram,
        "l2_bytes": l2,
        "duration_seconds": seconds,
        "dram_amplification": (dram / compulsory) if compulsory else None,
        "l2_amplification": (l2 / compulsory) if compulsory else None,
        # Below one means the cache absorbed traffic the Schedule still had to declare,
        # which is the expected reading of a Corpus Schedule small enough to stay
        # resident. It is recorded rather than judged: this instrument measures one
        # launch, and what a working set does against 126MB of L2 is a property of the
        # size that was profiled.
        "dram_below_compulsory": bool(compulsory) and dram < compulsory,
        "achieved_dram_bytes_per_second": (dram / seconds) if seconds else None,
    }


def _utilization(value: Utilization | None) -> dict[str, object] | None:
    """Project a utilisation for the record, or record that the Target declares no peak.

    None here is a fact about the Target rather than about the kernel, and the record
    says which by carrying the whole key as null instead of a block of nulls.
    """

    if value is None:
        return None
    return {
        "arithmetic": value.arithmetic,
        "arithmetic_exact": value.arithmetic_exact,
        "arithmetic_contract": value.arithmetic_contract,
        "arithmetic_peak_source": (
            None if value.arithmetic_peak is None else value.arithmetic_peak.source.value
        ),
        "bandwidth": value.bandwidth,
        "bandwidth_exact": value.bandwidth_exact,
        "bandwidth_peak_source": (
            None if value.bandwidth_peak is None else value.bandwidth_peak.source.value
        ),
        "roofline_seconds": value.roofline_seconds,
        "roofline_efficiency": value.roofline_efficiency,
        # A ratio above one refutes the work count or the declared peak. It is stored
        # beside the ratios so a reader cannot quote one without seeing it.
        "refuted": value.refuted,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--revision", default="compiler/revision.lock.json")
    parser.add_argument("--ncu", default="/usr/local/cuda/bin/ncu")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    compiler = Compiler.load(ROOT, arguments.revision)
    schedule_path = ROOT / arguments.schedule
    document = json.loads(schedule_path.read_text(encoding="utf-8"))
    assessment = compiler.assess(document)
    if not assessment.lowering_eligible:
        codes = ", ".join(finding.code for finding in assessment.findings)
        raise SystemExit(f"the gates refused this Schedule: {codes}")
    entry_point = assessment.route.entry_point
    if entry_point not in ORACLE_BY_ENTRY_POINT:
        # The instruments share one input builder so that a kernel this profiles is a
        # kernel the other checked. Profiling one with no oracle would measure something
        # nothing has established computes the right answer.
        raise SystemExit(f"no oracle for entry point {entry_point!r}")
    lowering = compiler.lower(assessment)

    target = Target.from_dict(
        json.loads(
            (ROOT / f"compiler/targets/{document['target']}.json").read_text(
                encoding="utf-8"
            )
        )
    )
    # Both halves read the same canonical bytes, so a residency bound and a work count
    # in one record cannot be describing two different Schedules.
    canonical = Schedule.from_dict(json.loads(assessment.schedule_bytes))
    predicted = residency_upper_bound(canonical, target)
    if predicted is None or predicted.binding is None:
        raise SystemExit("the analysis says nothing about this Schedule's residency")
    declared = work_bound(canonical)
    if declared is None:
        raise SystemExit("the work model says nothing about this Schedule")

    with tempfile.TemporaryDirectory(prefix="cake-profile-") as directory:
        module_path = Path(directory) / f"{entry_point}.py"
        module_path.write_text(lowering.source, encoding="utf-8")
        driver = Path(directory) / "driver.py"
        driver.write_text(
            _DRIVER.format(
                tools=str(Path(__file__).resolve().parent),
                src=str(ROOT / "src"),
                schedule=str(schedule_path),
                module=str(module_path),
                entry=entry_point,
            ),
            encoding="utf-8",
        )
        table = _profile(
            [
                arguments.ncu,
                "--csv",
                "--print-units",
                "base",
                "--metrics",
                ",".join(_METRICS),
                "--target-processes",
                "all",
                sys.executable,
                str(driver),
            ]
        )

    values = _measured(table)
    limits = {name: values[metric] for metric, name in _LIMITS.items()}
    resident = min(limits.values())
    binding = sorted(name for name, value in limits.items() if value == resident)

    record = {
        "schema_version": 2,
        "observed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "purpose": "residency_attribution",
        "host": socket.gethostname(),
        "compiler_revision": {
            "revision_id": assessment.compiler_revision_id,
            "revision_sha256": assessment.compiler_revision_sha256,
        },
        "schedule": {
            "path": arguments.schedule,
            "schedule_id": assessment.schedule_id,
            "canonical_sha256": assessment.schedule_sha256,
        },
        "lowering": {
            "generated": lowering.generated,
            "entry_point": entry_point,
            "source_sha256": lowering.source_sha256,
        },
        "declared_work": {
            "program_tiles": declared.program_tiles,
            "flops": declared.flops,
            "flops_exact": declared.flops_exact,
            "mma_flops": declared.mma_flops,
            "compulsory_read_bytes": declared.compulsory_read_bytes,
            "compulsory_written_bytes": declared.compulsory_written_bytes,
            "compulsory_bytes_exact": declared.compulsory_bytes_exact,
            "arithmetic_intensity": declared.arithmetic_intensity,
            "uncounted_arithmetic": list(declared.uncounted_arithmetic),
            "partially_addressed": list(declared.partially_addressed),
        },
        "predicted": {
            "ctas_per_multiprocessor_upper_bound": predicted.ctas_per_multiprocessor,
            "binding_resource": predicted.binding.resource,
        },
        "measured": {
            "registers_per_thread": values["launch__registers_per_thread"],
            "occupancy_limits": limits,
            "ctas_per_multiprocessor": resident,
            "binding_resource": binding,
            # Retained per metric as well as summed, because reads and writes are not
            # interchangeable: a Schedule that writes more than it reads and one that
            # reads more than it writes have the same total and different problems.
            **{name: values[metric] for metric, name in _TRAFFIC.items()},
            "duration_seconds": values[_DURATION],
        },
        "traffic": traffic_comparison(declared, values),
        "utilization": _utilization(
            utilization(declared, target, values[_DURATION])
        ),
        "verdict": {
            # The bound is sound while the measurement sits at or under it. A measurement
            # above it is a refutation, not a loose bound, and has to read as one.
            "residency_bound_sound": resident <= predicted.ctas_per_multiprocessor,
            "binding_resource_correct": predicted.binding.resource in binding,
            # A tie means the measurement did not single one resource out, so a correct
            # prediction here discriminated less than the word "correct" suggests.
            "binding_resource_measured_uniquely": len(binding) == 1,
        },
    }
    out = Path(arguments.out)
    if out.exists():
        raise SystemExit(f"{out} already exists; a new measurement gets a new file")
    out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(
        {
            key: record[key]
            for key in (
                "declared_work",
                "predicted",
                "measured",
                "traffic",
                "utilization",
                "verdict",
            )
        },
        indent=1,
    ))
    print(f"wrote {out}")
    return 0 if record["verdict"]["residency_bound_sound"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
