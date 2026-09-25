"""Create-only CPU input and FP64 oracle for the fan-in-scaled successor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from data import load_fanin_contract, make_rank, reference
from runner import load_rank_snapshot


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Fan-in CPU oracle output already exists")
    document = load_fanin_contract(HERE / "model_scale_inputs_fanin_v2.json")
    args.output.mkdir(parents=True)
    began = time.perf_counter_ns()
    ranks = [make_rank(document, rank) for rank in range(4)]
    for rank, inputs in enumerate(ranks):
        np.savez(args.output / f"rank{rank}-input.npz", **inputs)
        load_rank_snapshot(document, args.output, rank, check_values=True)
    expected = reference(document, ranks)
    if not np.all(np.isfinite(expected)):
        raise ValueError("Non-finite fan-in FP64 CPU oracle")
    tolerance = document["oracle"]["atol"] + document["oracle"]["rtol"] * np.abs(expected)
    zero_failures = int(np.count_nonzero(np.abs(expected) > tolerance))
    if zero_failures <= expected.size // 10:
        raise ValueError("All-zero negative control is too weak for this oracle")
    np.save(args.output / "oracle-expected.npy", expected)
    observation = {
        "experiment_id": document["experiment_id"],
        "numpy_version": np.__version__,
        "shape": list(expected.shape),
        "all_finite": True,
        "nonzero_elements": int(np.count_nonzero(expected)),
        "mean_absolute_expected": float(np.abs(expected).mean()),
        "zero_output_failing_elements": zero_failures,
        "cpu_oracle_wall_ns": time.perf_counter_ns() - began,
        "note": "Independent CPU input/oracle preparation, not device latency",
    }
    (args.output / "cpu-oracle-observation.json").write_text(
        json.dumps(observation, indent=2) + "\n")
    print(json.dumps(observation, indent=2))


if __name__ == "__main__":
    main()
