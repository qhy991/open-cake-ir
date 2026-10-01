"""CPU BF16 Torch operation reference for retained SGLang device outputs.

This is a diagnostic successor reference; it never replaces the original FP64
oracle or its failure record.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import compare_outputs, load_contract, round_bf16
from runner import load_rank_snapshot


HERE = Path(__file__).resolve().parent


def reference_torch_cpu(document: dict, ranks: list[dict[str, np.ndarray]]) -> np.ndarray:
    torch.set_num_threads(8)
    shape = document["geometry"]
    tokens, hidden, experts = (shape[key] for key in ("tokens_per_rank", "hidden", "experts"))
    ids = np.stack([rank["ids"] for rank in ranks])
    output = np.zeros((4, tokens, hidden), dtype=np.float64)
    with torch.inference_mode():
        for expert in range(experts):
            source, token, slot = np.nonzero(ids == expert)
            if not source.size:
                continue
            owner, local = divmod(expert, experts // 4)
            x = np.stack([ranks[int(s)]["hidden"][int(tok)]
                          for s, tok in zip(source, token, strict=True)])
            x = torch.from_numpy(x).to(torch.bfloat16)
            gate_w = torch.from_numpy(ranks[owner]["gate"][local]).to(torch.bfloat16)
            up_w = torch.from_numpy(ranks[owner]["up"][local]).to(torch.bfloat16)
            down_w = torch.from_numpy(ranks[owner]["down"][local]).to(torch.bfloat16)
            gate = F.linear(x, gate_w)
            up = F.linear(x, up_w)
            activated = F.silu(gate) * up
            projected = F.linear(activated, down_w).float().numpy()
            route_weight = np.array([ranks[int(s)]["weights"][int(tok), int(sl)]
                                     for s, tok, sl in zip(source, token, slot, strict=True)])
            np.add.at(output, (source, token), projected.astype(np.float64)
                      * route_weight[:, None])
    return round_bf16(output.astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--device-output", type=Path, required=True)
    parser.add_argument("--analysis-output", type=Path, required=True)
    args = parser.parse_args()
    if args.analysis_output.exists():
        raise ValueError("Torch CPU BF16 analysis output already exists")
    args.analysis_output.mkdir(parents=True)
    document = load_contract(HERE / "contract.json")
    ranks = [load_rank_snapshot(document, args.inputs, rank, check_values=True)
             for rank in range(4)]
    expected = reference_torch_cpu(document, ranks)
    np.save(args.analysis_output / "cpu-torch-bf16-expected.npy", expected)
    comparison = compare_outputs(document, args.device_output, expected=expected)
    comparison.update({
        "reference": "CPU_Torch_BF16_linear_SiLU_mul_linear_with_FP64_weighted_combine",
        "torch_version": torch.__version__,
        "original_fp64_oracle_result": str(args.device_output / "oracle-result.json"),
        "note": "Separate CPU arithmetic diagnosis; original FP64 contract remains unchanged",
    })
    (args.analysis_output / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
