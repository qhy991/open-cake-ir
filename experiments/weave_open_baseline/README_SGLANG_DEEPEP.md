# Complete BF16 EP4 fallback: SGLang + DeepEP

The fixed Triton-Distributed source build needs a 1.24 GB prebuilt LLVM
dependency that was not available at a usable rate on B300-M4. This fallback
uses the already present SGLang `v0.5.12.post1` image. Its checked-in source is
[commit `5a15cde`](https://github.com/sgl-project/sglang/tree/5a15cde858ea09b77116212a39356f2fc51b8584),
and the installed DeepEP package reports `1.2.1` ([public tag](https://github.com/deepseek-ai/DeepEP/tree/v1.2.1)).
`contract_sglang_deepep.json` pins the local Docker image identity and states
the remaining binary-provenance limit. The SGLang and DeepEP versions are
later than or distinct from the paper's exact §5.1 SGLang configuration; no
paper-speedup claim follows from this fallback.

This adapter runs a **complete** layer: DeepEP
[`low_latency_dispatch`](https://github.com/deepseek-ai/DeepEP/blob/v1.2.1/deep_ep/buffer.py)
in BF16, SGLang's
[`forward_unquantized_deepep_ll`](https://github.com/sgl-project/sglang/blob/5a15cde858ea09b77116212a39356f2fc51b8584/python/sglang/srt/layers/moe/ep_moe/layer.py)
(gate/up BF16 batched GEMM, SiLU and down BF16 batched GEMM), then DeepEP
weighted `low_latency_combine`. DeepGEMM `0.1.0` is installed in the image,
but its BF16 grouped masked entry is `None`; this path uses SGLang's BF16
expert fallback and is **not** labeled the paper's DeepEP+DeepGEMM baseline.

The upstream [DeepEP low-latency test](https://github.com/deepseek-ai/DeepEP/blob/v1.2.1/tests/test_low_latency.py)
marks a 512-token call buggy and defaults to 128. Each of four ranks therefore
processes its existing 512 tokens as four sequential 128-token chunks. The
complete 2,048-token model-scale input and independent FP64 CPU oracle from
`contract.json` remain unchanged; all four chunk results are concatenated in
original token order. Chunking is an explicit implementation choice included
inside the layer boundary, not an equivalence claim about the frozen Cake
T7/T8 workload or the paper's original SGLang v0.5.9 path.

## Reproduce

1. Run `preflight_sglang_cpu.sh <cpu-input-dir>` on B300-M4, without a broker
   lease. It checks the SGLang source commit, package versions, input snapshots,
   source syntax and DeepEP's RDMA buffer size for this geometry.
2. Create a fresh device output directory and its `runtime-cache` child **before**
   asking for GPUs. Submit `run_sglang_under_broker.sh <cpu-input-dir>
   <new-output-dir>` as the child of `gpu-run --mode exclusive --gpu-count 4`,
   with `--receipt-out <new-output-dir>/admission.json` and a bounded run
   timeout. The B300-M4 broker v0.6 supplies `CUDA_VISIBLE_DEVICES` but no
   `GPUQ_JOB_ID` environment field. The wrapper compares its broker-issued
   admission receipt with live broker status and visibility, then exposes only
   those four physical device IDs to Docker.
3. After the broker has released the lease, run
   `python3.12 runner_sglang_deepep.py check --inputs <cpu-input-dir>
   --output <device-output-dir>` using a CPU-only host Python with NumPy.
   Retain the admission receipt, all rank outputs, device observation, oracle
   result and stdout/stderr. A process exit without all-rank oracle pass is not
   a correct baseline.

The device observation records one raw four-rank host span from first dispatch
through final weighted combine and synchronization, with qualified latency
left null. The `sm_103a` CUPTI/L2-reset timer inputs currently return ENODEV,
so no Cake-vs-fallback latency or speedup can be published. A profiler trace
and a qualified common timer remain separate gates.
