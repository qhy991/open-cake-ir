"""Pinned Triton-Distributed EP4 MoE experiment adapter.

`preflight` and `check` are CPU only. `run` must be launched by the B300-M4
GPU broker with torchrun --nproc_per_node=4, never directly over SSH.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

from data import compare_outputs, load_contract, make_rank, reference


HERE = Path(__file__).resolve().parent
UPSTREAM_FILES = (
    "python/triton_dist/function/nvidia/ep_moe_fused.py",
    "python/triton_dist/function/nvidia/common.py",
    "python/triton_dist/kernels/nvidia/swiglu.py",
)


def preflight(document: dict, upstream: Path) -> dict:
    if not upstream.is_dir():
        raise ValueError(f"Upstream checkout absent: {upstream}")
    head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if head != document["upstream"]["commit"]:
        raise ValueError(f"Upstream commit {head} differs from contract")
    changed = subprocess.check_output(["git", "-C", str(upstream), "diff", "--name-only", "HEAD"], text=True)
    if changed.strip():
        raise ValueError(f"Upstream tracked source modified: {changed.strip()}")
    for relative in UPSTREAM_FILES:
        if not (upstream / relative).is_file():
            raise ValueError(f"Upstream entry absent: {relative}")
    source = (upstream / UPSTREAM_FILES[0]).read_text()
    for stage in ("mega_dispatch_group_gemm", "swiglu_forward", "mega_group_gemm_combine"):
        if stage not in source:
            raise ValueError(f"Full-layer stage absent in pinned upstream: {stage}")
    for relative in UPSTREAM_FILES:
        source_path = upstream / relative
        compile(source_path.read_text(), str(source_path), "exec")
    return {"pass": True, "upstream_commit": head, "source_files": list(UPSTREAM_FILES),
            "numpy_version": np.__version__, "mode": "cpu_only_static"}


def _run_on_broker(document: dict, upstream: Path, output_dir: Path) -> None:
    import torch
    import torch.distributed as dist

    shape = document["geometry"]
    if not os.environ.get("GPUQ_JOB_ID") or os.environ.get("GPUQ_MODE") != "exclusive":
        raise RuntimeError("GPU execution requires a broker-issued exclusive lease")
    if int(os.environ.get("WORLD_SIZE", "0")) != 4 or torch.cuda.device_count() != 4:
        raise RuntimeError("EP4 requires exactly four broker-mapped CUDA devices")
    if any(torch.cuda.get_device_capability(i) != (10, 3) for i in range(4)):
        raise RuntimeError("Target mismatch: this experiment requires four sm_103a GPUs")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if rank != local_rank:
        raise RuntimeError("This single-node experiment requires rank == local_rank")
    sys.path.insert(0, str(upstream / "python"))
    from triton_dist.function.nvidia.common import init_triton_dist_ep_op, deinit_triton_dist_ep_op
    from triton_dist.function.nvidia.ep_moe_fused import TritonDistFusedEpMoeFunction
    from triton_dist.utils import init_nvshmem_by_torch_process_group

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    group = dist.new_group(ranks=list(range(4)), backend="nccl")
    inputs = make_rank(document, rank)
    def tensor(name: str, dtype):
        return torch.from_numpy(inputs[name]).to(device="cuda", dtype=dtype)
    hidden = tensor("hidden", torch.bfloat16)
    ids = tensor("ids", torch.int32)
    route_weights = tensor("weights", torch.float32)
    fc1 = torch.cat((tensor("gate", torch.bfloat16), tensor("up", torch.bfloat16)), dim=1)
    fc2 = tensor("down", torch.bfloat16)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        init_nvshmem_by_torch_process_group(group)
        init_triton_dist_ep_op(group, shape["tokens_per_rank"], shape["hidden"],
                               shape["top_k"], rank, shape["experts"], 4,
                               dtype=torch.bfloat16, weight_dtype=torch.float32,
                               num_sm=64, num_buffers=1, capacity=4.0)

        def forward():
            with torch.inference_mode():
                return TritonDistFusedEpMoeFunction.apply(
                    shape["experts"], route_weights, ids, hidden, fc1, None, fc2, group)

        for _ in range(document["measurement"]["warmups"]):
            dist.barrier(group)
            forward()
            torch.cuda.synchronize()
        samples = []
        last = None
        for _ in range(document["measurement"]["iterations"]):
            dist.barrier(group)
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            last = forward()
            torch.cuda.synchronize()
            samples.append((start, time.perf_counter_ns()))
        dist.barrier(group)
        assert last is not None
        np.save(output_dir / f"rank{rank}-output.npy", last.detach().float().cpu().numpy())
        all_samples = [None] * 4
        dist.all_gather_object(all_samples, samples, group=group)
        if rank == 0:
            complete_ns = [max(all_samples[r][i][1] for r in range(4))
                           - min(all_samples[r][i][0] for r in range(4))
                           for i in range(len(samples))]
            (output_dir / "device-observation.json").write_text(json.dumps({
                "experiment_id": document["experiment_id"],
                "upstream_commit": document["upstream"]["commit"],
                "broker_job_id": os.environ["GPUQ_JOB_ID"],
                "broker_device_ids": os.environ["GPUQ_DEVICE_IDS"],
                "geometry": shape,
                "configuration": {"num_sm": 64, "capacity": 4.0, "num_buffers": 1,
                                  "warmups": document["measurement"]["warmups"],
                                  "iterations": document["measurement"]["iterations"]},
                "rank_start_end_ns": all_samples,
                "complete_layer_ns": complete_ns,
                "median_complete_layer_ns": statistics.median(complete_ns),
                "timer": document["measurement"]["timer"],
                "device_state_reset": document["measurement"]["device_state_reset"],
                "software": {"torch": torch.__version__, "numpy": np.__version__,
                             "cuda_runtime": torch.version.cuda,
                             "nccl": torch.cuda.nccl.version()},
                "devices": [torch.cuda.get_device_name(i) for i in range(4)]
            }, indent=2, default=str) + "\n")
        dist.barrier(group)
    finally:
        deinit_triton_dist_ep_op()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "oracle", "run", "check"))
    parser.add_argument("--contract", type=Path, default=HERE / "contract.json")
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    document = load_contract(args.contract)
    if args.mode in ("preflight", "run") and args.upstream is None:
        parser.error("--upstream is required for preflight and run")
    if args.mode in ("oracle", "run", "check") and args.output is None:
        parser.error("--output is required for oracle, run and check")
    if args.mode == "preflight":
        print(json.dumps(preflight(document, args.upstream), indent=2))
    elif args.mode == "oracle":
        if os.environ.get("GPUQ_JOB_ID"):
            raise RuntimeError("CPU oracle must run outside a GPU lease")
        args.output.mkdir(parents=True, exist_ok=False)
        start = time.perf_counter_ns()
        expected = reference(document)
        elapsed_ns = time.perf_counter_ns() - start
        np.save(args.output / "oracle-expected.npy", expected)
        observation = {"experiment_id": document["experiment_id"],
                       "numpy_version": np.__version__, "shape": list(expected.shape),
                       "all_finite": bool(np.all(np.isfinite(expected))),
                       "nonzero_elements": int(np.count_nonzero(expected)),
                       "cpu_oracle_wall_ns": elapsed_ns,
                       "note": "CPU generation evidence; not a device latency"}
        (args.output / "cpu-oracle-observation.json").write_text(
            json.dumps(observation, indent=2) + "\n")
        print(json.dumps(observation, indent=2))
    elif args.mode == "run":
        preflight(document, args.upstream)
        _run_on_broker(document, args.upstream, args.output)
    else:
        observation = json.loads((args.output / "device-observation.json").read_text())
        if observation["experiment_id"] != document["experiment_id"]:
            raise ValueError("Device result belongs to another experiment")
        if observation["upstream_commit"] != document["upstream"]["commit"]:
            raise ValueError("Device result used another upstream commit")
        if observation["software"]["numpy"] != np.__version__:
            raise ValueError("NumPy version differs from device input generator")
        result = compare_outputs(document, args.output)
        (args.output / "oracle-result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        if not result["pass"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
