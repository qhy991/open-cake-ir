#!/usr/bin/env python3
"""Measure what actually bounds a lowered kernel's residency, and compare the analysis.

`docs/ANALYSIS_CALIBRATION.md` reports that the residency bound held and that the binding
resource was predicted correctly, and until now those numbers came from an Nsight run
outside this repository. So the table could be read and not repeated: when a Schedule
changed, the measured column went stale with no way to renew it. That happened.

This is that instrument. It lowers the Schedule, runs one launch under Nsight Compute, and
puts the analysis's prediction and the profiler's measurement side by side.

The comparison is exact about direction. `residency_upper_bound` is an upper bound, so it
is *sound* when the measured limit is at or below it and refuted when the measurement is
higher. The register figure is a lower bound on storage and is sound the other way round.
Reporting them without their direction is how a bound gets read as an estimate.

Run under the broker in exclusive mode: a profiler serialises the device.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kernel_inputs import build_inputs  # noqa: E402
from kernel_oracles import ORACLES  # noqa: E402
from open_cake_ir.compiler.analysis import residency_upper_bound  # noqa: E402
from open_cake_ir.compiler.core import Compiler  # noqa: E402
from open_cake_ir.compiler.ir import Schedule  # noqa: E402
from open_cake_ir.compiler.target import Target  # noqa: E402

# The four limits Nsight reports are the same four resources the analysis reasons about,
# which is why this comparison is possible at all: the smallest one is what bounds
# residency, and which one it is answers the attribution half of the report.
_LIMITS = {
    "launch__occupancy_limit_registers": "logical_register_storage",
    "launch__occupancy_limit_shared_mem": "shared_memory",
    "launch__occupancy_limit_blocks": "blocks",
    "launch__occupancy_limit_warps": "threads",
}
_METRICS = ["launch__registers_per_thread", *_LIMITS]

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


def _register_floor(bound) -> int | None:
    """Registers per thread the declared buffers alone already need.

    Derived from the same per-CTA figures the residency bound is built from rather than
    recomputed, so the two halves of the report cannot disagree about the same Schedule.
    """

    registers = next(
        (b for b in bound.bounds if b.resource == "logical_register_storage"), None
    )
    threads = next((b for b in bound.bounds if b.resource == "threads"), None)
    if registers is None or threads is None or not threads.per_cta:
        return None
    return -(-registers.per_cta // threads.per_cta)


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
        if name in _METRICS and raw:
            values[name] = float(raw)
    missing = [metric for metric in _METRICS if metric not in values]
    if missing:
        raise SystemExit(f"ncu reported no value for {', '.join(missing)}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--revision", default="compiler/revision.lock.json")
    parser.add_argument("--ncu", default="/usr/local/cuda/bin/ncu")
    parser.add_argument("--observed-at", required=True, help="ISO 8601 UTC timestamp")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    compiler = Compiler.load(ROOT, arguments.revision)
    schedule_path = ROOT / arguments.schedule
    document = json.loads(schedule_path.read_text(encoding="utf-8"))
    assessment = compiler.assess(document)
    if not assessment.lowering_eligible:
        codes = ", ".join(finding.code for finding in assessment.findings)
        raise SystemExit(f"the gates refused this Schedule: {codes}")
    if assessment.profile not in ORACLES:
        # The instruments share one input builder so that a kernel this profiles is a
        # kernel the other checked. Profiling one with no oracle would measure something
        # nothing has established computes the right answer.
        raise SystemExit(f"no oracle for profile {assessment.profile!r}")
    lowering = compiler.lower(assessment)

    target = Target.from_dict(
        json.loads(
            (ROOT / f"compiler/targets/{document['target']}.json").read_text(
                encoding="utf-8"
            )
        )
    )
    predicted = residency_upper_bound(
        Schedule.from_dict(json.loads(assessment.schedule_bytes)), target
    )
    if predicted is None or predicted.binding is None:
        raise SystemExit("the analysis says nothing about this Schedule's residency")

    with tempfile.TemporaryDirectory(prefix="cake-profile-") as directory:
        module_path = Path(directory) / f"{lowering.entry_point}.py"
        module_path.write_text(lowering.source, encoding="utf-8")
        driver = Path(directory) / "driver.py"
        driver.write_text(
            _DRIVER.format(
                tools=str(Path(__file__).resolve().parent),
                src=str(ROOT / "src"),
                schedule=str(schedule_path),
                module=str(module_path),
                entry=lowering.entry_point,
            ),
            encoding="utf-8",
        )
        table = _profile(
            [
                arguments.ncu,
                "--csv",
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
        "schema_version": 1,
        "observed_at": arguments.observed_at,
        "purpose": "residency_attribution",
        "host": "verda-b200x4",
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
            "entry_point": lowering.entry_point,
            "source_sha256": lowering.source_sha256,
        },
        "predicted": {
            "ctas_per_multiprocessor_upper_bound": predicted.ctas_per_multiprocessor,
            "binding_resource": predicted.binding.resource,
            # The other half of the report, and the one that runs the other way: this is
            # a lower bound on storage, so it is sound while the measurement is above it.
            "registers_per_thread_lower_bound": _register_floor(predicted),
        },
        "measured": {
            "registers_per_thread": values["launch__registers_per_thread"],
            "occupancy_limits": limits,
            "ctas_per_multiprocessor": resident,
            "binding_resource": binding,
        },
        "verdict": {
            # The bound is sound while the measurement sits at or under it. A measurement
            # above it is a refutation, not a loose bound, and has to read as one.
            "residency_bound_sound": resident <= predicted.ctas_per_multiprocessor,
            "binding_resource_correct": predicted.binding.resource in binding,
            # A tie means the measurement did not single one resource out, so a correct
            # prediction here discriminated less than the word "correct" suggests.
            "binding_resource_measured_uniquely": len(binding) == 1,
            "register_floor_sound": (
                None
                if _register_floor(predicted) is None
                else _register_floor(predicted)
                <= values["launch__registers_per_thread"]
            ),
        },
    }
    out = Path(arguments.out)
    if out.exists():
        raise SystemExit(f"{out} already exists; a new measurement gets a new file")
    out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(
        {key: record[key] for key in ("predicted", "measured", "verdict")}, indent=1
    ))
    print(f"wrote {out}")
    return 0 if record["verdict"]["residency_bound_sound"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
