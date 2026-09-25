# Open EP4 MoE baseline candidate for the Weave/Cake study

This is a **separate, unqualified model-scale experiment**, not an edit to
`weave-ep4-bf16-moe-b300-v1`. The selected baseline is the public
[Triton-Distributed 3.4 branch](https://github.com/ByteDance-Seed/Triton-distributed/tree/triton-v3.4)
at commit `63de69e48dde17f32b0ee80ba83901c6950404cd` (2026-09-18).
There is no public `v3.4.0` Git tag in that repository. Its
[`v0.0.2-rc` release](https://github.com/ByteDance-Seed/Triton-distributed/releases/tag/v0.0.2-rc)
ships a wheel named `triton_dist-3.4.0`, but the release source predates the
full `ep_moe_fused.py` entry. Thus this pinned 3.4-branch commit is a later,
explicitly disclosed source baseline, not a byte-identical reconstruction of
the paper's unspecified `v3.4.0` checkout.
The selected [forward implementation](https://github.com/ByteDance-Seed/Triton-distributed/blob/63de69e48dde17f32b0ee80ba83901c6950404cd/python/triton_dist/function/nvidia/ep_moe_fused.py)
calls `mega_dispatch_group_gemm`, `swiglu_forward`, then
`mega_group_gemm_combine`. Its
[upstream EP test](https://github.com/ByteDance-Seed/Triton-distributed/blob/63de69e48dde17f32b0ee80ba83901c6950404cd/python/triton_dist/test/nvidia/test_ep_moe_fused.py)
supports BF16 and four ranks. The test itself also runs backward and smaller
cases, so `runner.py` invokes only the full forward path and captures its exact
window. The [tuning defaults](https://github.com/ByteDance-Seed/Triton-distributed/blob/63de69e48dde17f32b0ee80ba83901c6950404cd/python/triton_dist/function/nvidia/common.py)
are H800 oriented. Source comments mention Hopper/Blackwell, but B300-M4
execution and accuracy remain **unverified** until a broker run succeeds.

## Source audit against Weave §5.1

The [paper](https://arxiv.org/html/2609.21483#S5.SS1) used 4×H100 and named
five baselines. Their public source boundaries are:

| Paper label | Checked public identity | Five-stage forward and B300 status |
| --- | --- | --- |
| SGLang v0.5.9 | [tag `bbe9c7e`](https://github.com/sgl-project/sglang/tree/v0.5.9) | Serving EP MoE path exists. It needs a model/runtime configuration; paper does not identify one. No B300-M4 run here. |
| Triton-Distributed v3.4.0 | [3.4 branch `63de69e`](https://github.com/ByteDance-Seed/Triton-distributed/tree/63de69e48dde17f32b0ee80ba83901c6950404cd) | Public full forward entry above; chosen. The historical wheel version has no corresponding full EP MoE source; B300-M4 unverified. |
| DeepEP v1.2.1 + DeepGEMM | [DeepEP tag `9af0e0d`](https://github.com/deepseek-ai/DeepEP/tree/v1.2.1) | DeepEP provides dispatch/combine, not expert FFN. The paper does not pin DeepGEMM, its composition, or the exact BF16 path. No complete layer was assembled here. |
| Comet/Flux v1.1.2 | [Flux tags](https://github.com/bytedance/flux/tags) | Public Flux has `v1.1.1`, but no `v1.1.2` tag; its `v1.1.1` tree does not include a named Comet/MoE runner. Exact paper source cannot be pinned. |
| ParallelKittens `a8f63a9` | [ThunderKittens public MoE benchmark](https://github.com/HazyResearch/ThunderKittens/tree/main/kernels/parallel/moe_dispatch_gemm) | The public sample is dispatch + first expert GEMM, not a complete MoE layer. The cited short commit is not resolvable in the public repository. |

The paper's §5.1 describes dispatch, two expert GEMMs, activation and combine.
Its Table 1 gives Qwen3-30B `(E=128, top-k=8, H=2048, I=768)`, EP4, BF16.
`contract.json` fixes that geometry with 2,048 total tokens (512 per rank),
independent seeded synthetic routing and a CPU FP64 oracle. The paper samples
ShareGPT and studies 2k/4k/8k; these inputs are **not** the paper's routes or
its measured hardware. Model name here identifies geometry, not model weights
or end-to-end serving.

The frozen Cake development Workload has T7/T8, E8, top-k2, H16, I32 and
distinct balanced/skew/local/remote/tail cases. The upstream TD fused kernels
use 64-wide K tiles, 256-wide N tiles and the public test starts at 1,024
tokens; that code provides no evidence for these tiny cases. The independent
contract does not claim semantic equivalence. Cake's existing
[`RESULTS.md`](/Users/haiyan/Documents/Infinity/Agent4Kernel/open-cake-ir-workspaces/evidence/weave-b300-m4-20260925/cake-ranked-ep4-b300-device-369b8cf3/RESULTS.md)
records one passing plan for each frozen case and unresolved low-communication
CTA / mixed-K timeouts. It contains neither CUPTI/L2-reset latency nor a
matched open baseline. Do not compare its functional outcomes to this
experiment's provisional times.

## Reproduction stages

1. Clone the upstream 3.4 branch at the exact commit above into a **separate**
   checkout, and initialize its `3rdparty/triton` submodule at
   `f53694a72a1e4f464fa245df2c7305ccda7cb2a9`. On B300-M4, run
   `prepare_cpu.sh init`, `prepare_cpu.sh deps`, `prepare_cpu.sh build`, then
   `prepare_cpu.sh probe`, passing the upstream checkout and a new isolated
   build directory to each command. This script launches a container with no
   GPU devices, installs into a private venv and leaves build logs under the
   caller's chosen path. Keep the complete commands, logs and `git status`.
2. Run `python runner.py preflight --upstream /path/to/Triton-distributed`.
   This checks the exact upstream commit, clean tracked source, all three
   forward stages and Python syntax. It is CPU only.
3. Create a new output directory and submit one `gpu-run --mode exclusive
   --gpu-count 4 --receipt-out /path/to/new-output/admission.json` job through
   the B300-M4 broker, with `run_under_broker.sh <upstream> <build> <output>`
   as its child command. The script checks the broker-owned allocation and
   passes exactly those four physical devices into the GPU container.
   Preserve broker stdout/stderr and the admission receipt. The broker sets
   `CUDA_VISIBLE_DEVICES`; do not set it in the launcher.
4. After the broker's lease has ended, run `check_after_release.sh <build>
   <output>` on the CPU with the **same NumPy version** used for
   input generation. Retain `rank*-output.npy`, `device-observation.json`,
   `oracle-result.json`, commands and logs. A failing/missing oracle is not a
   successful baseline.

The measured function includes route preprocessing, dispatch, gate/up GEMM,
SwiGLU, down GEMM and combine, ending only after each rank synchronizes. The
per-iteration layer time is latest rank completion minus earliest rank start
on the same host's monotonic clock. Input
generation, weight transfer, initialization and oracle run outside the window.
The current timer is a host steady clock without a verified L2 flush or CUPTI
target timing contract. Its output is diagnostic only; it cannot support a
Cake-vs-TD latency or overlap claim. A profiler trace and target-aligned
device-state reset are still required for a comparable performance result.
