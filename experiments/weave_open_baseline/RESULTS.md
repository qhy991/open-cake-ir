# EP4 open baseline execution record

Status: **A complete SGLang/DeepEP BF16 EP4 layer passed the independent FP64
CPU oracle in two separate B300 broker jobs and one CUDA trace job for the
fan-in-scaled v2 synthetic contract.** The original fixed-scale v1 input
failed and remains retained. No qualified cross-implementation latency or
speedup exists.

## Fixed sources

- Adapter branch: `task/nvidia-weave-open-baseline`, initially at
  `df63fbd7` for the CPU input-snapshot implementation. Later record commits
  in this task branch supersede this snapshot without modifying the frozen
  Cake Workload.
- Upstream source: ByteDance-Seed/Triton-distributed `triton-v3.4`, commit
  `63de69e48dde17f32b0ee80ba83901c6950404cd`; its
  `3rdparty/triton` submodule is
  `f53694a72a1e4f464fa245df2c7305ccda7cb2a9`.
- Source checkout on B300-M4:
  `/home/qinhaiyan/weave-td-open-baseline-20260925` (separate from all Cake
  task worktrees). Build directory:
  `/home/qinhaiyan/weave-td-build-20260925`.

## Retained CPU evidence

- The selected upstream source was cloned on B300-M4 without a GPU lease.
  `runner.py preflight --upstream
  /home/qinhaiyan/weave-td-open-baseline-20260925` passed with the source
  commit and three forward-stage files checked. Record:
  `/home/qinhaiyan/weave-td-build-20260925/preflight.json`.
- Five CPU tests pass on B300-M4 with host Python 3.12 / NumPy 2.5.3, including
  BF16 tie rounding, owner-specific expert weights, rejection of one corrupted
  rank output, creation/reuse of all four input snapshots, and refusal to run
  the device launcher without a broker lease. Latest log:
  `/home/qinhaiyan/weave-td-build-20260925/cpu-tests-694c84f2.log`.
- The original four tests, Python syntax, shell syntax and contract JSON passed
  in detached local worktree `/tmp/cake-weave-baseline-gate-9d05b6e9` at
  `df63fbd7` (Mac Python 3.11 / NumPy 1.24.2). The fifth broker-refusal
  check also passed in the task checkout and remote CPU test above.
- A later detached local gate at `0fa108e6` passed all five CPU tests plus
  Python syntax, all shell syntax and contract JSON, with a clean fixed-commit
  worktree. This gate covers the current adapter and resource scripts.
- The isolated CPU container environment was initialized by
  `prepare_cpu.sh init <source> <build>`. Its dependency stage is retained as
  `deps.log` (initial DNS failure) and `deps-retry1.log` (host proxy forwarded).
  The latter log reports installed NumPy 1.26.4, CUDA Python 12.4 and NVSHMEM
  3.3.9. SSH timed out before returning the command exit; a subsequent
  no-GPU container imported Torch 2.11.0+cu130, NumPy 1.26.4 and Cython
  0.29.24 from the private venv. pip reported conflicts with some unrelated
  packages visible from the base SGLang image; the upstream source build and
  `triton_dist` import remain unverified. The build container has no GPU
  mapping; its startup message reports no driver.
  A later audit of the selected commit's `python/setup.py` found that its
  Blackwell/CUDA 13 dependency pins are NVSHMEM 3.6.5 and nvshmem4py 0.3.0,
  unlike the older CUDA 12 README example used in the first dependency pass.
  The successor `prepare_cpu.sh` pins CUDA 13 packages; this venv must run its
  `cu13` correction stage and pass an import probe before any device request.
- First CPU-only editable build: `prepare_cpu.sh build <source> <build>` exited
  1. `build.log` shows the direct cause: CMake was absent in the base image,
  while upstream requires CMake >=3.20 for its Triton C++ extensions. This is
  a build-environment failure, not a device compatibility result. The private
  venv will be given pinned `cmake==3.31.10` before retrying; the original log
  remains unchanged.
- `prepare_cpu.sh cmake <source> <build>` completed and
  `cmake-install.log` records `cmake==3.31.10` in the isolated venv.
- A second CPU-only editable build reached `running build_ext` and began
  fetching the selected Triton submodule's prebuilt LLVM archive from
  `oaitriton.blob.core.windows.net/public/llvm-builds/llvm-8957e64a-ubuntu-x64.tar.gz`.
  The server advertises 1,242,831,658 bytes. With no cached copy and a slow
  remote path, the bounded attempt was stopped after several minutes; its
  `build-retry1.log` and exit 137 reflect this deliberate stop, not a compiler
  error or B300 incompatibility. The exact CPU build command remains runnable.
  A successor `prepare_cpu.sh` keeps future toolchain downloads in the isolated
  build directory instead of an ephemeral container home.
  The upstream checkout has no tracked edits; its Triton submodule retains an
  untracked generated `third_party/nvidia/backend/bin/` from the incomplete
  build. Keep it as build evidence, not as a qualified executable.
- Full-scale CPU inputs and oracle were generated **before any baseline GPU
  lease** by `create_oracle_cpu.sh
  /home/qinhaiyan/weave-td-build-20260925
  /home/qinhaiyan/weave-td-cpu-oracle-694c84f2-20260925`. The create-only
  directory holds four `rank*-input.npz` files, `oracle-expected.npy` and
  `cpu-oracle-observation.json`; the command log is
  `/home/qinhaiyan/weave-td-build-20260925/cpu-oracle-run.log`.
  NumPy 1.26.4 produced all 4,194,304 finite, nonzero BF16-representable
  expected values. CPU generation took 14.09 s; that wall time is host
  preparation evidence, not an MoE-layer latency.

The first environment inspection used the existing SGLang Python shim, which
was found to launch a Docker container with `--gpus all`. It imported only
Torch/NumPy and ran no CUDA operation, but it is excluded as a CPU gate. A
second inspection likewise used a GPU-capable Docker launch with visibility
masked and no kernel call. All later preparation uses `prepare_cpu.sh`, which
has no `--gpus` option and sets `NVIDIA_VISIBLE_DEVICES=void`.

## Comparability boundary

`contract.json` is an **independent** synthetic-route Qwen3-30B geometry
(EP4, E128, top-k8, H2048, I768, BF16, 2,048 total tokens). It is not the
paper's ShareGPT distribution. The frozen Cake Workload
`weave-ep4-bf16-moe-b300-v1` uses T7/T8, E8, top-k2, H16, I32 and an
independent CPU oracle; its existing [Cake device record](/Users/haiyan/Documents/Infinity/Agent4Kernel/open-cake-ir-workspaces/evidence/weave-b300-m4-20260925/cake-ranked-ep4-b300-device-369b8cf3/RESULTS.md)
does not contain comparable timing. TD's public test starts at 1,024 tokens,
and its fused GEMM tiles use K64/N256, so the frozen small cases have no
upstream compatibility evidence. No result from one geometry is labeled as an
equivalent result for the other.

The TD adapter was designed to observe the complete upstream forward window across four ranks;
its host monotonic spans are diagnostic. The declared `sm_103a` CUPTI/FlashInfer
timer inputs under `/mnt/b300-shared` currently return ENODEV on B300-M4
(read-only observation from the parallel Cake task). The adapter therefore
reports `qualified_latency_ns: null` and a measurement-coverage limitation.
For the TD candidate, a completed source build, CUDA 13 dependency
correction/import probe, four-card broker admission, device output,
after-release oracle comparison and profiler remain open gates. Its
create-only CPU model-scale oracle completed as recorded above, but TD never
requested a GPU lease. Its bounded CPU build container was stopped and
verified absent before the later fallback work began.

## Complete-layer fallback progression

The TD source build remains pending on the external LLVM dependency. A second
open path has passed its CPU gate using the existing full-scale input/oracle:
SGLang `v0.5.12.post1` at source commit `5a15cde8` plus installed DeepEP
`1.2.1`. `contract_sglang_deepep.json` pins the image identity and states its
binary-provenance and paper-version differences. This is SGLang's unquantized
BF16 DeepEP low-latency path, **not** DeepEP+DeepGEMM or the paper's exact
SGLang v0.5.9 configuration. Each rank's 512 tokens will be processed as four
128-token chunks because DeepEP's upstream test marks a 512-token single call
buggy. Dispatch, SGLang BF16 gate/up-SiLU-down expert computation and weighted
combine are all inside the layer window.

The no-GPU preflight in the pinned image passed for all retained inputs and
reported the exact SGLang source commit, package versions, source syntax and
272,630,912 bytes of DeepEP RDMA buffer per rank. Its retained log is
`/home/qinhaiyan/weave-td-build-20260925/sglang-preflight-image-pinned.log`.
At that preflight checkpoint, device correctness and profiler were pending;
see [fallback reproduction steps](README_SGLANG_DEEPEP.md).

The first fallback broker request, `gpuq-fbf6458ba421`, was admitted for four
physical GPUs and then failed in the launcher **before Docker/CUDA**. The
launcher expected `GPUQ_JOB_ID/GPUQ_MODE/GPUQ_BACKEND`; the running B300-M4
broker v0.6 injects only `CUDA_VISIBLE_DEVICES` into child commands. The
receipt and `broker.log` are retained under
`/home/qinhaiyan/weave-sglang-deepep-ep4-run-d78c69c9-20260925/`.
Live broker status confirmed terminal failure and released devices. A
successor launcher validates the saved broker-issued receipt against live
broker status, including job identity, exclusive four-GPU allocation and
visibility, before passing only those devices into Docker. This is an
admission-boundary correction, not device correctness evidence.

The second fallback request, `gpuq-c328d93fc4fd`, passed the receipt/live
broker check and started its pinned container on four GPUs. All four ranks
failed while importing SGLang: the image has no passwd entry for the mapped
host UID 1010, and TorchDynamo's `getpass.getuser()` raised `KeyError`.
No dispatch or expert computation occurred. The retained `broker.log` and
admission receipt are under
`/home/qinhaiyan/weave-sglang-deepep-ep4-run-828c2c54-20260925/`.
The broker reported terminal failure and released all four GPUs. A no-GPU
container probe then showed `USER`/`LOGNAME` resolves this exact import gate;
the successor wrapper supplies them and the CPU preflight exercises the
TorchDynamo import.

The third request, `gpuq-027816b4d71c`, used commit `4a11da6a` and completed
the full four-rank path (four sequential 128-token chunks per rank) on physical
GPUs `0,1,2,3`. The broker reported exit 0 and released its allocation. All
four rank outputs, admission receipt, raw diagnostic window and full log live
under `/home/qinhaiyan/weave-sglang-deepep-ep4-run-4a11da6a-20260925/`.
DeepEP logged failed IBGDA transport probes before completing this intra-node
execution; a successful process exit by itself is not correctness.

The post-release independent FP64 CPU oracle comparison **failed**:
353,538 / 4,194,304 elements exceeded `atol=rtol=0.01`, with maximum absolute
error 0.125. Its `oracle-result.json` remains unchanged. CPU-only localization
found similar errors on all four ranks, mean absolute error about 0.011 and
observed/reference least-squares scale about 0.9999. Independent diagnostic
references with BF16 rounding at expert tensor boundaries still failed the
same tolerance: 1,923 elements with FP64 dots, 1,910 with FP32 dots and
1,925 with Torch CPU BF16 operations. These are retained in separate analysis
directories and do **not** reclassify the original result as correct.

The fixed 0.1 standard deviation of the original synthetic expert weights is
independent of H and I and produces a mean FP64 oracle magnitude near 3.
`model_scale_inputs_fanin_v2.json` and
`contract_sglang_deepep_fanin_v2.json` define a separate fan-in-scaled
synthetic successor with the same geometry, seed, routing and unchanged FP64
oracle tolerance. It required a new CPU input/oracle snapshot and new broker
jobs; the v1 failure is retained.

## Fan-in-scaled v2: complete EP4 correctness and trace

`create_fanin_oracle.py` generated all four rank input snapshots and the
independent FP64 CPU output at
`/home/qinhaiyan/weave-fanin-cpu-oracle-7a179dc8-20260925/` **before** GPU
time. Its `cpu-oracle-observation.json` records 4,194,304 finite, nonzero
expected values, mean absolute magnitude 0.18587, and a strong negative
control: all-zero output would fail on 4,048,663 elements under the unchanged
`atol=rtol=0.01` rule. The pinned SGLang/DeepEP no-GPU preflight passed for
these exact snapshots; log:
`/home/qinhaiyan/weave-td-build-20260925/sglang-fanin-preflight-7a179dc8.log`.
The retained `routing-volume.json` checks the actual EP traffic: 16,384
expert routes, 4,113 local and 12,271 cross-GPU; every source-rank token has
at least one remote destination. The successful jobs therefore exercised
cross-GPU dispatch/combine, not just local expert math.

The runtime adapter at commit `7a179dc8` used the pinned SGLang source
`5a15cde8`, DeepEP package `1.2.1`, the same v2 inputs, four 128-token chunks
per rank, and the same FP64 CPU oracle in every run. Every broker lease was
exclusive, exactly four GPUs, terminal `completed` with exit 0, then verified
released **before** CPU comparison. The executed command forms and cache
preparation are in [COMMANDS.md](COMMANDS.md).

| Run | Broker job | Evidence directory | Post-release CPU oracle |
| --- | --- | --- | --- |
| First correctness | `gpuq-d085d22e1c15` | `/home/qinhaiyan/weave-sglang-deepep-fanin-ep4-run-7a179dc8-20260925/` | 4,194,304/4,194,304 pass; max absolute error 0.0078125 |
| Independent repeat, fresh JIT cache | `gpuq-af1118ef8413` | `/home/qinhaiyan/weave-sglang-deepep-fanin-ep4-repeat1-7a179dc8-20260925/` | 4,194,304/4,194,304 pass; max absolute error 0.0078125 |
| CPU/CUDA Chrome trace | `gpuq-dbfb2c1adee5` | `/home/qinhaiyan/weave-sglang-deepep-fanin-ep4-profile-7a179dc8-20260925/` | 4,194,304/4,194,304 pass; max absolute error 0.0078125 |

The first two independently brokered outputs are bitwise equal on **all four
ranks**, as retained in the repeat directory's `repeat-comparison.json`. The
profiled output is also bitwise equal to the first run on all four ranks
(`profile-vs-unprofiled.json`). The profile directory retains
`rank0-trace.json` through `rank3-trace.json` and
`trace-audit.json`. Each rank's trace has 49 CUDA kernel events, including four
DeepEP dispatch and four combine kernels; CPU op events include eight `bmm`
and four SiLU operations, consistent with four complete token chunks. This
checks that communication, expert up/gate, activation, down and weighted
combine actually occurred. The trace has no SM/NVLink hardware counters.

`device-observation.json` retains raw four-rank host spans, but the first run
includes JIT compilation, the profile run includes instrumentation, and the
declared `sm_103a` CUPTI/L2-reset timer dependencies remain unavailable.
`qualified_latency_ns` is null in every run. The Cake frozen T7/T8 results,
separate single-GPU H2048/I768 projection tiles and expert-bin pack are not a
matched complete MoE-layer comparator. **No Cake-vs-SGLang latency ratio or
speedup is supported.**
