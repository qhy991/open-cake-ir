"""Retain a separate CPU BF16-stage oracle and compare a completed device run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data import compare_outputs, load_contract, reference_bf16_stages
from runner import load_rank_snapshot


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--device-output", type=Path, required=True)
    parser.add_argument("--analysis-output", type=Path, required=True)
    parser.add_argument("--dot-accumulation", choices=("fp32", "fp64"), default="fp64")
    args = parser.parse_args()
    document = load_contract(HERE / "contract.json")
    if args.analysis_output.exists():
        raise ValueError("BF16-stage analysis output already exists")
    args.analysis_output.mkdir(parents=True)
    ranks = [load_rank_snapshot(document, args.inputs, rank, check_values=True)
             for rank in range(4)]
    dot_dtype = np.float32 if args.dot_accumulation == "fp32" else np.float64
    expected = reference_bf16_stages(document, ranks, dot_dtype=dot_dtype)
    np.save(args.analysis_output / "cpu-bf16-stage-expected.npy", expected)
    comparison = compare_outputs(document, args.device_output, expected=expected)
    comparison.update({
        "reference": f"CPU_{args.dot_accumulation}_dot_with_BF16_rounding_after_gate_up_silu_mul_down_and_final_combine",
        "original_fp64_oracle_result": str(args.device_output / "oracle-result.json"),
        "note": "Separate arithmetic diagnosis; does not overwrite the original FP64 contract result",
    })
    (args.analysis_output / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
