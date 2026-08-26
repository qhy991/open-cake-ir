# ADR 0043: AITER RMSNorm is an external gfx1151 baseline

Status: implemented as a source-pinned, correctness-only candidate; live gfx1151
correctness and all performance claims remain pending.

## Context

The existing `llama-rmsnorm-mul-fp32-independent-v2` Workload owns two FP32
`[8,512,128]` cases, a length-128 FP32 gamma, last-axis reduction,
`epsilon=1e-6`, a CPU-FP64 oracle and fixed tolerance. A useful AMD library baseline
must accept those bytes and semantics directly. Adapting the Workload to a library API
would move the comparison boundary and create a second owner for the operator.

At `ROCm/aiter v0.1.20@fc2e5d57`, `rms_norm_opus` accepts a caller-provided output,
contiguous FP32 input and weight, uses FP32 accumulation, and launches the
`module_rmsnorm` Opus FP32 path. Its prefix dimensions flatten to rows, so the existing
three-dimensional input needs no reshape or copy. The release source admits `gfx1151`
for JIT, while its published wheel does not provide a qualified gfx1151 device closure.

AITER's `silu_and_mul` instead accepts one packed `[...,2D]` input. The frozen SwiGLU
Workload owns separate gate and up tensors. Packing them would add an allocation and
copy or hide producer work outside timing, so it is not a matched baseline.

## Decision

1. Use the frozen llama RMSNorm Workload unchanged and call the pinned source function
   `aiter.ops.rmsnorm.rms_norm_opus` directly as an external baseline candidate.
2. Keep the baseline outside Compiler lowering. It adds no Target, IR primitive,
   Schedule, Corpus case, calibration or Compiler Revision.
3. Prepare-only admits a clean Open Cake checkout, the exact clean AITER tag/commit,
   the fixed source chain and the existing Workload without importing Torch or AITER,
   compiling, or submitting GPU work.
4. Live correctness requires a released schema-v2 gfx1151 Executor that owns the
   runner, one exact gfx1151 wave32 HIP device, a new external JIT directory and a new
   external evidence directory. The JIT environment fixes `GPU_ARCHS=gfx1151`, uses
   the exact source checkout, disables unused CK/HipKittens inputs and permits no
   fallback. The Executor also owns the exact `git`, `rocminfo`, `hipconfig`, `hipcc`,
   C++ compiler, Ninja and POSIX shell commands plus the Packaging, PyBind11, psutil and
   Setuptools Python dependencies that AITER actually imports or executes. It owns the
   explicit `libxml2.so.2` compatibility library required by ROCm 7.2.1 `lld` on hosts
   whose system libxml2 has a newer SONAME.
5. The retained module must be ELF, export `rms_norm_opus`, contain only a gfx1151 code
   object and have a build plan naming only `--offload-arch=gfx1151`. A
   `module_rmsnorm_quant` artifact is a path violation. AITER import also builds the
   host-only `module_aiter_core`; the runner requires its Python initializer, no device
   code and a gfx1151-targeted build plan before explicitly building RMSNorm. Both
   modules and build plans are admitted before it materializes a Workload case or calls
   the operator, then rechecked after correctness.
6. Each Workload case calls the operator once, synchronizes, verifies input immutability,
   output shape/dtype/layout and the existing oracle tolerance. The receipt records the
   source-derived expectation of one launch per call but does not claim an observed
   launch count without an independent activity trace.
7. Correctness failure is retained as `STOP_CORRECTNESS_REJECTED`. It does not authorize
   a new Workload, relaxed tolerance, another dtype or a Torch/Triton fallback.
8. AITER SwiGLU is STOP for the current Workload because its packed public input changes
   the operator ABI and measurement boundary. hipBLASLt TinyGEMM remains a separate
   candidate until it passes both the Workload tolerance and its pinned-parent bitwise
   gate.

## Boundary

This slice prepares a reproducible external correctness candidate; no live result exists
until the successor toolchain-owning gfx1151 Executor is released on infplane. It is not
an accepted incumbent, speedup, llama.cpp integration or serving result. Timing starts
only after the source-pinned module passes both cases. The current Python entry point's
Torch custom-op guard is part of its correctness call path and is not automatically a
matched timing boundary. A later performance study must declare a common direct-C-ABI
or wrapper-inclusive boundary, use the common paired protocol and retain profiler path
evidence before any Open Cake candidate can claim a library-relative improvement.

The Open Cake profiler path is deliberately a post-decision assay. Its input is a sealed
no-profiler `LEAF_TIMING_WIN` root, and its output is a new root; it never edits or
reclassifies the parent result. Candidate and baseline run in separate exact-rocprofv3
children because different Schedule variants can share one physical Triton symbol. Each
child performs one correctness-checked target launch after compile-only artifact replay.
The validator parses the full trace itself (rocprofv3 1.1.0 does not apply
`--kernel-include-regex` to ordinary kernel trace) and requires trace CSV, kernel-stats
CSV and JSON dispatches to agree on exact symbol, count, launch geometry and resources.
Profiler timestamps remain raw diagnostic context; they cannot contribute samples,
speedup, promotion or an end-to-end claim. No gfx1151 occupancy is derived from these
fields without a separately approved target calibration.
