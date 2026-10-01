"""CPU-only audit that retained Chrome traces include actual CUDA work."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def audit(root: Path) -> dict:
    ranks = []
    for rank in range(4):
        data = json.loads((root / f"rank{rank}-trace.json").read_text())
        events = data.get("traceEvents", [])
        categories = Counter(event.get("cat", "") for event in events)
        kernels = [event.get("name", "") for event in events
                   if event.get("cat") == "kernel"]
        cpu_ops = [event.get("name", "") for event in events
                   if event.get("cat") == "cpu_op"]
        if not kernels:
            raise ValueError(f"Rank {rank} trace contains no CUDA kernel events")
        ranks.append({
            "rank": rank,
            "total_events": len(events),
            "categories": dict(categories),
            "cuda_kernel_events": len(kernels),
            "deep_ep_dispatch_kernel_events": sum("deep_ep" in name and "dispatch" in name
                                                   for name in kernels),
            "deep_ep_combine_kernel_events": sum("deep_ep" in name and "combine" in name
                                                  for name in kernels),
            "cpu_bmm_ops": sum("aten::bmm" in name for name in cpu_ops),
            "cpu_silu_ops": sum("aten::silu" in name for name in cpu_ops),
            "kernel_names": kernels,
        })
    return {"trace_type": "CPU/CUDA Chrome trace; no SM/NVLink hardware counters",
            "ranks": ranks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Trace audit output already exists")
    result = audit(args.dir)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"trace_type": result["trace_type"],
                      "ranks": [{key: value for key, value in rank.items()
                                 if key != "kernel_names"} for rank in result["ranks"]]},
                     indent=2))
