"""Complete BF16 EP4 layer using pinned DeepEP and SGLang expert FFN code.

The GPU mode is only invoked as a broker child. CPU preflight and output check
are separate phases. This later SGLang path is not the paper's v0.5.9 setup.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from data import compare_outputs, load_contract
from runner import load_rank_snapshot


HERE = Path(__file__).resolve().parent
CONTRACT = HERE / "contract_sglang_deepep.json"
SGLANG_ROOT = Path("/sgl-workspace/sglang")
SGLANG_LAYER = SGLANG_ROOT / "python/sglang/srt/layers/moe/ep_moe/layer.py"
SGLANG_DISPATCHER = SGLANG_ROOT / "python/sglang/srt/layers/moe/token_dispatcher/deepep.py"
DEEP_EP_BUFFER = Path("/usr/local/lib/python3.12/dist-packages/deep_ep/buffer.py")


def load_experiment() -> tuple[dict, dict]:
    experiment = json.loads(CONTRACT.read_text())
    workload = load_contract(HERE / "contract.json")
    geometry = experiment["geometry"]
    if (experiment["target"] != "sm_103a"
            or geometry != workload["geometry"]
            or experiment["input_snapshot"]["experiment_id"] != workload["experiment_id"]
            or geometry["tokens_per_rank"] != 4 * experiment["execution"]["chunk_tokens_per_rank"]
            or experiment["source"]["sglang_commit"] !=
            "5a15cde858ea09b77116212a39356f2fc51b8584"
            or experiment["source"]["deep_ep_version"] != "1.2.1"):
        raise ValueError("SGLang/DeepEP experiment contract differs from reviewed boundary")
    return experiment, workload


def inspect_inputs(experiment: dict, workload: dict, input_dir: Path,
                   all_ranks: bool = False) -> None:
    observation = json.loads((input_dir / "cpu-oracle-observation.json").read_text())
    if (observation["experiment_id"] != workload["experiment_id"]
            or observation["shape"] != [4, 512, 2048]
            or not observation["all_finite"]
            or not observation["nonzero_elements"]
            or not (input_dir / "oracle-expected.npy").is_file()):
        raise ValueError("Independent CPU input/oracle snapshot is incomplete")
    if all_ranks:
        for rank in range(4):
            load_rank_snapshot(workload, input_dir, rank)


def preflight(experiment: dict, workload: dict, input_dir: Path) -> dict:
    inspect_inputs(experiment, workload, input_dir, all_ranks=True)
    if subprocess.check_output(["git", "-c", f"safe.directory={SGLANG_ROOT}",
                                "-C", str(SGLANG_ROOT), "rev-parse", "HEAD"],
                               text=True).strip() != experiment["source"]["sglang_commit"]:
        raise ValueError("Installed SGLang source differs from pinned commit")
    if importlib.metadata.version("sglang") != experiment["source"]["sglang_version"]:
        raise ValueError("Installed SGLang package version differs")
    if importlib.metadata.version("deep-ep") != experiment["source"]["deep_ep_version"]:
        raise ValueError("Installed DeepEP package version differs")
    for source in (SGLANG_LAYER, SGLANG_DISPATCHER, DEEP_EP_BUFFER):
        compile(source.read_text(), str(source), "exec")
    source = SGLANG_LAYER.read_text()
    for stage in ("def forward_unquantized_deepep_ll", "torch.bmm", "F.silu"):
        if stage not in source:
            raise ValueError(f"Full BF16 expert FFN stage absent: {stage}")
    from deep_ep import Buffer
    rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
        experiment["execution"]["chunk_tokens_per_rank"],
        experiment["geometry"]["hidden"], 4, experiment["geometry"]["experts"])
    return {"pass": True, "mode": "cpu_only_static", "sglang_commit":
            experiment["source"]["sglang_commit"], "sglang_version":
            importlib.metadata.version("sglang"), "deep_ep_version":
            importlib.metadata.version("deep-ep"), "rdma_bytes_per_rank": rdma_bytes,
            "numpy_version": np.__version__, "input_snapshot": str(input_dir)}


def run(experiment: dict, workload: dict, input_dir: Path, output_dir: Path) -> None:
    if (not os.environ.get("GPUQ_JOB_ID") or os.environ.get("GPUQ_MODE") != "exclusive"
            or os.environ.get("GPUQ_BACKEND") != "nvidia"):
        raise RuntimeError("GPU execution requires a broker-issued exclusive NVIDIA lease")
    import torch
    import torch.distributed as dist

    if int(os.environ.get("WORLD_SIZE", "0")) != 4 or torch.cuda.device_count() != 4:
        raise RuntimeError("EP4 requires exactly four broker-mapped CUDA devices")
    if any(torch.cuda.get_device_capability(i) != (10, 3) for i in range(4)):
        raise RuntimeError("Target mismatch: four sm_103a GPUs required")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if rank != local_rank:
        raise RuntimeError("Single-node EP4 rank mapping differs")
    inspect_inputs(experiment, workload, input_dir)
    from deep_ep import Buffer
    from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLDispatchOutput

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    group = dist.new_group(ranks=list(range(4)), backend="nccl")
    snapshot = load_rank_snapshot(workload, input_dir, rank)
    def device_tensor(name: str, dtype):
        return torch.from_numpy(snapshot[name]).to(device="cuda", dtype=dtype)
    hidden = device_tensor("hidden", torch.bfloat16)
    ids = device_tensor("ids", torch.int64)
    weights = device_tensor("weights", torch.float32)
    gate = device_tensor("gate", torch.bfloat16)
    up = device_tensor("up", torch.bfloat16)
    layer = SimpleNamespace(
        moe_runner_config=SimpleNamespace(activation="silu", is_gated=True),
        w13_weight=torch.cat((gate, up), dim=1),
        w2_weight=device_tensor("down", torch.bfloat16))
    shape = experiment["geometry"]
    chunk_tokens = experiment["execution"]["chunk_tokens_per_rank"]
    rdma_bytes = Buffer.get_low_latency_rdma_size_hint(
        chunk_tokens, shape["hidden"], 4, shape["experts"])
    buffer = None
    try:
        buffer = Buffer(group, num_rdma_bytes=rdma_bytes, low_latency_mode=True,
                        num_qps_per_rank=shape["experts"] // 4,
                        allow_nvlink_for_low_latency_mode=True,
                        explicitly_destroy=True)

        def forward():
            outputs = []
            with torch.inference_mode():
                for chunk in range(experiment["execution"]["chunks_per_rank"]):
                    begin = chunk * chunk_tokens
                    end = begin + chunk_tokens
                    chunk_ids = ids[begin:end].contiguous()
                    chunk_weights = weights[begin:end].contiguous()
                    received, counts, handle, event, _ = buffer.low_latency_dispatch(
                        hidden[begin:end].contiguous(), chunk_ids, chunk_tokens,
                        shape["experts"], use_fp8=False,
                        async_finish=True, return_recv_hook=False)
                    event.current_stream_wait()
                    dispatched = DeepEPLLDispatchOutput(
                        received, None, chunk_ids, chunk_weights, counts,
                        (chunk_tokens * 4 * shape["top_k"] + shape["experts"])
                        // shape["experts"])
                    computed = DeepEPMoE.forward_unquantized_deepep_ll(layer, dispatched)
                    combined, event, _ = buffer.low_latency_combine(
                        computed, chunk_ids, chunk_weights, handle,
                        async_finish=True, return_recv_hook=False)
                    event.current_stream_wait()
                    # DeepEP has two reusable buffers. Preserve every earlier chunk.
                    outputs.append(combined.clone())
                return torch.cat(outputs, dim=0)

        dist.barrier(group)
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        output = forward()
        torch.cuda.synchronize()
        finished = time.perf_counter_ns()
        np.save(output_dir / f"rank{rank}-output.npy", output.float().cpu().numpy())
        bounds = [None] * 4
        dist.all_gather_object(bounds, (started, finished), group=group)
        if rank == 0:
            observation = {
                "experiment_id": experiment["experiment_id"],
                "sglang_commit": experiment["source"]["sglang_commit"],
                "deep_ep_version": experiment["source"]["deep_ep_version"],
                "broker_job_id": os.environ["GPUQ_JOB_ID"],
                "broker_device_ids": os.environ["GPUQ_DEVICE_IDS"],
                "input_snapshot_experiment_id": workload["experiment_id"],
                "chunk_tokens_per_rank": chunk_tokens,
                "chunks_per_rank": experiment["execution"]["chunks_per_rank"],
                "rank_start_end_ns": bounds,
                "diagnostic_global_span_ns": max(item[1] for item in bounds)
                - min(item[0] for item in bounds),
                "qualified_latency_ns": None,
                "measurement_coverage_limitation": experiment["measurement"]["coverage_limitation"],
                "software": {"torch": torch.__version__, "numpy": np.__version__,
                             "cuda_runtime": torch.version.cuda},
                "devices": [torch.cuda.get_device_name(i) for i in range(4)],
            }
            (output_dir / "device-observation.json").write_text(
                json.dumps(observation, indent=2, default=str) + "\n")
        dist.barrier(group)
    finally:
        if buffer is not None:
            buffer.destroy()
        dist.barrier(group)
        dist.destroy_process_group()


def check(experiment: dict, workload: dict, input_dir: Path, output_dir: Path) -> dict:
    if os.environ.get("GPUQ_JOB_ID"):
        raise RuntimeError("CPU comparison must run after the broker lease")
    inspect_inputs(experiment, workload, input_dir)
    observation = json.loads((output_dir / "device-observation.json").read_text())
    if (observation["experiment_id"] != experiment["experiment_id"]
            or observation["sglang_commit"] != experiment["source"]["sglang_commit"]
            or observation["input_snapshot_experiment_id"] != workload["experiment_id"]):
        raise ValueError("Device result identity differs from contract")
    expected = np.load(input_dir / "oracle-expected.npy", allow_pickle=False)
    result = compare_outputs(workload, output_dir, expected=expected)
    (output_dir / "oracle-result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run", "check"))
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    experiment, workload = load_experiment()
    if args.mode == "preflight":
        print(json.dumps(preflight(experiment, workload, args.inputs), indent=2))
    else:
        if args.output is None:
            parser.error("--output is required for run and check")
        if args.mode == "run":
            run(experiment, workload, args.inputs, args.output)
        else:
            result = check(experiment, workload, args.inputs, args.output)
            print(json.dumps(result, indent=2))
            if not result["pass"]:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
