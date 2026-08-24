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
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler.core import Compiler  # noqa: E402


def _global_shapes(document: dict) -> dict[str, tuple[int, ...]]:
    return {
        buffer["name"]: tuple(buffer["shape"])
        for buffer in document["buffers"]
        if buffer["space"] == "global"
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schedule", default="corpus/schedules/flash-kmeans-assignment-full.json"
    )
    parser.add_argument("--revision", default="compiler/revision.lock.json")
    parser.add_argument("--observed-at", required=True, help="ISO 8601 UTC timestamp")
    parser.add_argument("--out", required=True)
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

    shapes = _global_shapes(document)
    tokens_n, dimension = shapes["tokens"]
    centroids_k, _ = shapes["centroids"]

    torch.manual_seed(0)
    tokens = torch.randn(tokens_n, dimension, device="cuda", dtype=torch.bfloat16)
    centroids = torch.randn(centroids_k, dimension, device="cuda", dtype=torch.bfloat16)
    centroid_sq = (centroids.to(torch.float32) ** 2).sum(dim=1).contiguous()
    distance_scratch = torch.empty(
        tokens_n, centroids_k, device="cuda", dtype=torch.float32
    )
    assignments = torch.full((tokens_n,), -1, device="cuda", dtype=torch.int32)

    # Independent of the kernel: float32 argmin of squared euclidean distance with the
    # token norm elided, which is constant per row and cannot change the argmin.
    left = tokens.to(torch.float32)
    right = centroids.to(torch.float32)
    distance = (right * right).sum(dim=1)[None, :] - 2.0 * (left @ right.t())
    reference = torch.argmin(distance, dim=1).to(torch.int32)

    with tempfile.TemporaryDirectory(prefix="cake-observe-") as directory:
        module_path = Path(directory) / f"{lowering.entry_point}.py"
        module_path.write_text(lowering.source, encoding="utf-8")
        specification = importlib.util.spec_from_file_location(
            module_path.stem, module_path
        )
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        observed = module.launch_once(
            tokens, centroids, centroid_sq, distance_scratch, assignments
        )
        torch.cuda.synchronize()

    mismatch = int((observed != reference).sum().item())
    rows = torch.arange(tokens_n, device=tokens.device)
    chosen = distance[rows, observed.long().clamp(0, centroids_k - 1)]
    # A mismatch can still be a legal tie, so the claim is about the distance chosen
    # rather than about the index: an equal distance is an equally correct answer.
    excess = float((chosen - distance[rows, reference.long()]).max().item())

    device = torch.cuda.get_device_properties(0)
    import cutlass

    record = {
        "schema_version": 2,
        "observed_at": arguments.observed_at,
        "purpose": "lowered_kernel_correctness",
        "scientific_claim_authorized": False,
        "performance_measured": False,
        "host": "verda-b200x4",
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
            "generated": True,
            "entry_point": lowering.entry_point,
            "source_sha256": lowering.source_sha256,
            "source_lines": len(lowering.source.splitlines()),
        },
        "result": {
            "compiled": True,
            "launched": True,
            "total_assignments": tokens_n,
            "mismatch_count": mismatch,
            "max_chosen_distance_excess": excess,
            "passed": mismatch == 0 or excess == 0.0,
        },
        "note": (
            "Compiler.lower generated this source from the Schedule; there is no "
            "checked-in template for this profile. Correctness only -- no timing was "
            "taken. Reproduce with tools/observe_lowered_kernel.py."
        ),
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
