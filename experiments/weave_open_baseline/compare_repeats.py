"""CPU-only comparison of two independently brokered EP4 result sets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compare(first: Path, second: Path) -> dict:
    first_observation = json.loads((first / "device-observation.json").read_text())
    second_observation = json.loads((second / "device-observation.json").read_text())
    if (first_observation["experiment_id"] != second_observation["experiment_id"]
            or first_observation["broker_job_id"] == second_observation["broker_job_id"]):
        raise ValueError("Repeat results need one contract and distinct broker jobs")
    ranks = []
    for rank in range(4):
        left = np.load(first / f"rank{rank}-output.npy", allow_pickle=False)
        right = np.load(second / f"rank{rank}-output.npy", allow_pickle=False)
        if left.shape != right.shape:
            raise ValueError(f"Rank {rank} output shape differs")
        ranks.append({"rank": rank, "bitwise_equal": bool(np.array_equal(left, right)),
                      "max_absolute_difference": float(np.max(np.abs(left - right)))})
    return {"experiment_id": first_observation["experiment_id"],
            "broker_jobs": [first_observation["broker_job_id"],
                            second_observation["broker_job_id"]],
            "all_ranks_bitwise_equal": all(item["bitwise_equal"] for item in ranks),
            "ranks": ranks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Repeat comparison output already exists")
    result = compare(args.first, args.second)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
