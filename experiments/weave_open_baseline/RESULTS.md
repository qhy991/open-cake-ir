# EP4 open baseline execution record

Status: **CPU source and adapter gates pass; device qualification pending.**
No GPU lease, oracle comparison, profiler trace or comparable latency has been
produced for this baseline yet.

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
- Four CPU tests pass on B300-M4 with host Python 3.12 / NumPy 2.5.3, including
  BF16 tie rounding, owner-specific expert weights, rejection of one corrupted
  rank output, and creation/reuse of all four input snapshots. Log:
  `/home/qinhaiyan/weave-td-build-20260925/cpu-tests.log`.
- The same four tests, Python syntax, shell syntax and contract JSON passed in
  detached local worktree `/tmp/cake-weave-baseline-gate-9d05b6e9` at the
  adapter commit named above (Mac Python 3.11 / NumPy 1.24.2).
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

The adapter times the complete upstream forward window across four ranks;
its timer is a host monotonic clock with no verified target L2 reset or CUPTI
policy. Any eventual value is diagnostic until the target measurement contract
and profiler overlap evidence are established. A source build, create-only CPU
model-scale oracle, four-card broker admission, device output, after-release
oracle comparison and profiler remain open gates.
