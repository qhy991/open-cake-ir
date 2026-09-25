"""CPU-only error localization for a retained full-layer device run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(input_dir: Path, output_dir: Path) -> dict:
    expected = np.load(input_dir / "oracle-expected.npy", allow_pickle=False).astype(np.float64)
    actual = np.stack([np.load(output_dir / f"rank{rank}-output.npy", allow_pickle=False)
                       for rank in range(4)]).astype(np.float64)
    if actual.shape != expected.shape:
        raise ValueError("Device and CPU oracle shapes differ")
    result = {"shape": list(actual.shape), "ranks": []}
    for rank in range(4):
        observed = actual[rank]
        reference = expected[rank]
        difference = observed - reference
        absolute = np.abs(difference)
        tolerance = 0.01 + 0.01 * np.abs(reference)
        numerator = np.sum(observed * reference)
        denominator = np.sum(reference * reference)
        normalized = float(numerator / denominator) if denominator else None
        token_mae = absolute.mean(axis=1)
        result["ranks"].append({
            "rank": rank,
            "failing_elements": int(np.count_nonzero(absolute > tolerance)),
            "mean_absolute_error": float(absolute.mean()),
            "median_absolute_error": float(np.median(absolute)),
            "p99_absolute_error": float(np.quantile(absolute, 0.99)),
            "max_absolute_error": float(absolute.max()),
            "mean_reference_magnitude": float(np.abs(reference).mean()),
            "mean_observed_magnitude": float(np.abs(observed).mean()),
            "observed_over_reference_least_squares": normalized,
            "worst_token_by_mae": int(np.argmax(token_mae)),
            "worst_token_mae": float(token_mae.max()),
            "first_token_expected": reference[0, :8].tolist(),
            "first_token_observed": observed[0, :8].tolist(),
        })
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.inputs, args.output), indent=2))
