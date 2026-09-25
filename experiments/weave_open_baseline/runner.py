"""Pinned Triton-Distributed EP4 MoE experiment adapter.

`preflight` and `check` are CPU only. `run` must be launched by the B300-M4
GPU broker with torchrun --nproc_per_node=4, never directly over SSH.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
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
    submodule = upstream / "3rdparty/triton"
    submodule_head = subprocess.check_output(
        ["git", "-C", str(submodule), "rev-parse", "HEAD"], text=True).strip()
    if submodule_head != "f53694a72a1e4f464fa245df2c7305ccda7cb2a9":
        raise ValueError("Upstream Triton submodule differs from reviewed commit")
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
    return {"pass": True, "upstream_commit": head, "triton_submodule_commit": submodule_head,
            "source_files": list(UPSTREAM_FILES),
            "numpy_version": np.__version__, "mode": "cpu_only_static"}


def input_observation(document: dict, input_dir: Path) -> dict:
    observation = json.loads((input_dir / "cpu-oracle-observation.json").read_text())
    if observation["experiment_id"] != document["experiment_id"]:
        raise ValueError("CPU input snapshot belongs to another experiment")
    if observation["numpy_version"] != np.__version__:
        raise ValueError("NumPy version differs from CPU input snapshot")
    if not observation["all_finite"] or not observation["nonzero_elements"]:
        raise ValueError("CPU oracle is empty or non-finite")
    return observation


def load_rank_snapshot(document: dict, input_dir: Path, rank: int,
                       check_values: bool = False) -> dict[str, np.ndarray]:
    shape = document["geometry"]
    t, h, f, k, local_e = (shape["tokens_per_rank"], shape["hidden"],
                          shape["intermediate"], shape["top_k"], shape["experts"] // 4)
    expected = {"hidden": ((t, h), np.float32), "ids": ((t, k), np.int32),
                "weights": ((t, k), np.float32),
                "gate": ((local_e, f, h), np.float32),
                "up": ((local_e, f, h), np.float32),
                "down": ((local_e, h, f), np.float32)}
    with np.load(input_dir / f"rank{rank}-input.npz", allow_pickle=False) as saved:
        if set(saved.files) != set(expected):
            raise ValueError("Rank input ABI differs")
        result = {name: saved[name] for name in expected}
    for name, (dims, dtype) in expected.items():
        value = result[name]
        if value.shape != dims or value.dtype != dtype:
            raise ValueError(f"Invalid rank input: {name}")
        if check_values and not np.all(np.isfinite(value)):
            raise ValueError(f"Non-finite rank input: {name}")
    ids = result["ids"]
    if (np.any(ids < 0) or np.any(ids >= shape["experts"])
            or np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0)):
        raise ValueError("Invalid or duplicate expert route")
    weights = result["weights"]
    if np.any(weights < 0) or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Invalid route weights")
    return result


def _run_on_broker(document: dict, upstream: Path, input_dir: Path,
                   output_dir: Path) -> None:
    import torch
    import torch.distributed as dist

    shape = document["geometry"]
    if not os.environ.get("WEAVE_BROKER_JOB_ID") or not os.environ.get("WEAVE_BROKER_DEVICE_IDS"):
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
    input_observation(document, input_dir)
    inputs = load_rank_snapshot(document, input_dir, rank)
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
            diagnostic_span_ns = [max(all_samples[r][i][1] for r in range(4))
                                  - min(all_samples[r][i][0] for r in range(4))
                                  for i in range(len(samples))]
            (output_dir / "device-observation.json").write_text(json.dumps({
                "experiment_id": document["experiment_id"],
                "upstream_commit": document["upstream"]["commit"],
                "broker_job_id": os.environ["WEAVE_BROKER_JOB_ID"],
                "broker_device_ids": os.environ["WEAVE_BROKER_DEVICE_IDS"],
                "geometry": shape,
                "cpu_input_numpy_version": np.__version__,
                "configuration": {"num_sm": 64, "capacity": 4.0, "num_buffers": 1,
                                  "warmups": document["measurement"]["warmups"],
                                  "iterations": document["measurement"]["iterations"]},
                "rank_start_end_ns": all_samples,
                "diagnostic_global_span_ns": diagnostic_span_ns,
                "qualified_latency_ns": None,
                "measurement_coverage_limitation":
                    "Target CUPTI/L2-reset timer unavailable; host spans are diagnostic only",
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
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    document = load_contract(args.contract)
    if args.mode in ("preflight", "run") and args.upstream is None:
        parser.error("--upstream is required for preflight and run")
    if args.mode in ("oracle", "run", "check") and args.output is None:
        parser.error("--output is required for oracle, run and check")
    if args.mode in ("run", "check") and args.inputs is None:
        parser.error("--inputs is required for run and check")
    if args.mode == "preflight":
        print(json.dumps(preflight(document, args.upstream), indent=2))
    elif args.mode == "oracle":
        if os.environ.get("GPUQ_JOB_ID"):
            raise RuntimeError("CPU oracle must run outside a GPU lease")
        if args.output.exists():
            if any(args.output.iterdir()):
                raise ValueError("CPU oracle output is not empty")
        else:
            args.output.mkdir(parents=True)
        start = time.perf_counter_ns()
        ranks = [make_rank(document, rank) for rank in range(4)]
        for rank, inputs in enumerate(ranks):
            np.savez(args.output / f"rank{rank}-input.npz", **inputs)
            load_rank_snapshot(document, args.output, rank, check_values=True)
        expected = reference(document, ranks)
        if not np.all(np.isfinite(expected)):
            raise ValueError("Non-finite CPU oracle")
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
        _run_on_broker(document, args.upstream, args.inputs, args.output)
    else:
        input_observation(document, args.inputs)
        observation = json.loads((args.output / "device-observation.json").read_text())
        if observation["experiment_id"] != document["experiment_id"]:
            raise ValueError("Device result belongs to another experiment")
        if observation["upstream_commit"] != document["upstream"]["commit"]:
            raise ValueError("Device result used another upstream commit")
        if observation["software"]["numpy"] != np.__version__:
            raise ValueError("NumPy version differs from device input generator")
        expected = np.load(args.inputs / "oracle-expected.npy", allow_pickle=False)
        result = compare_outputs(document, args.output, expected=expected)
        (args.output / "oracle-result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        if not result["pass"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
