"""CPU-only audit that a retained model-scale input uses real EP4 routes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def audit(input_dir: Path) -> dict:
    placement = np.zeros((4, 4), dtype=np.int64)
    unique_remote_tokens = []
    for source in range(4):
        with np.load(input_dir / f"rank{source}-input.npz", allow_pickle=False) as data:
            ids = data["ids"]
        if ids.shape != (512, 8):
            raise ValueError("Model-scale EP4 route shape differs")
        owners = ids // 32
        placement[source] = np.bincount(owners.ravel(), minlength=4)
        unique_remote_tokens.append(int(np.count_nonzero(np.any(owners != source, axis=1))))
    total = int(placement.sum())
    local = int(np.trace(placement))
    remote = total - local
    if total != 4 * 512 * 8 or local == 0 or remote == 0:
        raise ValueError("Retained input does not exercise both local and remote EP routes")
    return {"total_routes": total, "local_routes": local, "remote_routes": remote,
            "source_to_owner_routes": placement.tolist(),
            "unique_remote_tokens_by_source_rank": unique_remote_tokens}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Routing audit output already exists")
    result = audit(args.inputs)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
