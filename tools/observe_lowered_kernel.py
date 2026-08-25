#!/usr/bin/env python3
"""Compile and run what the Compiler lowers, and write down what happened.

`inventory/EMITTED_KERNEL_OBSERVATION_*.json` records that a lowered kernel compiled and
matched an independent oracle on a B200, and a contract test holds the current lowering to
the artifact that record names. Until now the record had no instrument in the repository:
it was produced by a script outside it, so the observation could be read but not repeated,
and any change to the lowering left the evidence stranded with no way to renew it except
editing the record -- which is the one thing evidence must never allow.

This is that instrument. It lowers the Schedule through the released Compiler, runs the
result against a float32 oracle, and writes a new record. It does not edit an existing one.

Correctness only. No timing is taken here, so nothing this writes supports a performance
claim, and the record says so in a field the contract test asserts.

Run under the broker; shared mode is enough because nothing here is timed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import socket
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kernel_inputs import build_inputs  # noqa: E402
from kernel_oracles import ORACLE_BY_ENTRY_POINT  # noqa: E402
from open_cake_ir.compiler.core import Compiler  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schedule", default="corpus/schedules/flash-kmeans-assignment-full.json"
    )
    parser.add_argument("--revision", default="compiler/revision.lock.json")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="max absolute deviation admitted for a float-valued kernel",
    )
    arguments = parser.parse_args()

    import torch

    compiler = Compiler.load(ROOT, arguments.revision)
    schedule_path = ROOT / arguments.schedule
    document = json.loads(schedule_path.read_text(encoding="utf-8"))
    assessment = compiler.assess(document)
    if not assessment.lowering_eligible:
        codes = ", ".join(finding.code for finding in assessment.findings)
        raise SystemExit(f"the gates refused this Schedule: {codes}")
    lowering = compiler.lower(assessment)

    entry_point = lowering.route.entry_point
    oracle = ORACLE_BY_ENTRY_POINT.get(entry_point)
    if oracle is None:
        raise SystemExit(
            f"no oracle for entry point {entry_point!r}; this tool observes "
            f"{', '.join(sorted(ORACLE_BY_ENTRY_POINT))}"
        )
    torch.manual_seed(0)
    inputs = build_inputs(document, torch)
    reference, distance = oracle(inputs, torch)

    with tempfile.TemporaryDirectory(prefix="cake-observe-") as directory:
        module_path = Path(directory) / f"{entry_point}.py"
        module_path.write_text(lowering.source, encoding="utf-8")
        specification = importlib.util.spec_from_file_location(
            module_path.stem, module_path
        )
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        launch = getattr(module, "launch_once", None) or getattr(
            module, entry_point
        )
        observed = launch(*inputs)
        torch.cuda.synchronize()

    if reference.dtype.is_floating_point:
        # A float-valued kernel is judged by how far it is, not by whether it matches:
        # an exact-equality rule would call float32 softmax wrong for reassociating a
        # sum, which is not a defect and is not something the Schedule chose.
        deviation = (observed - reference).abs()
        mismatch = int(deviation.gt(arguments.tolerance).sum().item())
        measured = {
            "max_deviation": float(deviation.max().item()),
            "tolerance": arguments.tolerance,
        }
        passed = measured["max_deviation"] <= arguments.tolerance
    elif distance is None:
        # A deterministic index-valued operation such as top-k owns the exact source
        # positions and their order. Unlike nearest-neighbour assignment, there is no
        # separate distance oracle under which a different index may be equally legal.
        mismatch = int((observed != reference).sum().item())
        measured = {"exact_match": mismatch == 0}
        passed = mismatch == 0
    else:
        mismatch = int((observed != reference).sum().item())
        rows = torch.arange(observed.shape[0], device=observed.device)
        chosen = distance[rows, observed.long().clamp(0, distance.shape[1] - 1)]
        # A mismatch can still be a legal tie, so the claim is about the distance chosen
        # rather than about the index: an equal distance is an equally correct answer.
        excess = float((chosen - distance[rows, reference.long()]).max().item())
        # Two kernels, two questions. An index kernel is asked whether the answer it
        # chose is as good; a float kernel is asked how far off it is. One field name
        # for both would have made the record say something it did not measure.
        measured = {"max_chosen_distance_excess": excess}
        passed = mismatch == 0 or excess == 0.0

    device = torch.cuda.get_device_properties(0)
    import cutlass

    record = {
        "schema_version": 2,
        "observed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "purpose": "lowered_kernel_correctness",
        "scientific_claim_authorized": False,
        "performance_measured": False,
        "host": socket.gethostname(),
        "device": device.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "toolchain": {
            "cutlass": getattr(cutlass, "__version__", "?"),
            "torch": torch.__version__,
        },
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
            "source_lines": len(lowering.source.splitlines()),
        },
        "result": {
            "compiled": True,
            "launched": True,
            "total_elements": int(reference.numel()),
            "mismatch_count": mismatch,
            **measured,
            "passed": passed,
        },
        "note": (
            (
                "Compiler.lower generated this source from the Schedule; there is no "
                "checked-in template for this lowering route. "
            )
            if lowering.generated
            else (
                "Compiler.lower materialized this source from a closed checked-in asset; "
                "the Schedule did not generate its operation bodies. "
            )
        )
        + "Correctness only -- no timing was taken. Reproduce with "
        "tools/observe_lowered_kernel.py.",
    }
    out = Path(arguments.out)
    if out.exists():
        # A record is what happened once. Renewing one by overwriting it would let a
        # later run quietly inherit an earlier run's authority.
        raise SystemExit(f"{out} already exists; a new observation gets a new file")
    out.write_text(json.dumps(record, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    print(json.dumps(record["result"], indent=1))
    print(f"wrote {out}")
    return 0 if record["result"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
