# AMD gfx1151 development path

This branch retains the earlier gfx1151/Q4 lineage and is merged through GitHub
`main@415a1f0f4d090e7f97292adcf3a539be67269550`. It adds exact gfx1151 Target support,
standalone FP32 SwiGLU, llama.cpp FP32 RMSNorm+Mul, a true-one-row hypothesis, a
generated live-Q8_1 producer and the source-pinned AITER RMSNorm baseline below. The
historical 45-case AMD Gate was prepared against the pre-RoPE Compiler and is no longer
the current Gate after this alignment; it is not carried forward or rewritten. The
current 37-case main manifest remains canonical. Any future AMD Compiler promotion must
prepare a new combined Gate and receive the independent approval required by ADR 0030.
Prepare-only commands below submit no GPU work.

## Environment

The previously qualified runtime combination is one visible gfx1151 wave32 device,
ROCm 7.2.1, Python 3.12, ROCm PyTorch 2.9.1 and Triton 3.5.1. PyTorch intentionally uses
the `torch.cuda` namespace for ROCm. Runtime admission checks `torch.version.hip`, the
active Triton target, `gcnArchName=gfx1151` and wave width; another device or a CUDA
build fails before compilation. These observed versions are not yet a released Executor
authority. The B200 v30 descriptor pins CUDA/CUPTI/NCU and cannot authorize this path;
ADR 0038 requires a separately released exact gfx1151 Executor before formal GPU work.

On the admitted infplane host, with exactly one gfx1151 device visible, release that
authority from the final clean candidate checkout:

```bash
OPEN_CAKE_AITER_LIBXML2=/path/to/compat/libxml2.so.2 \
PYTHONPATH=src /path/to/rocm/python \
  tools/release_gfx1151_executor_cycle.py
```

The release command derives the independent `open-cake-ir-gfx1151-vN` identity, captures
the exact Linux/Python/Torch/Triton/HIP facts, AITER JIT Python dependencies,
`git`/ROCm/C++/Ninja commands, the explicit `libxml2.so.2` linker dependency and
`amd-smi`, includes only actually available AMD profilers, builds and live-admits a
temporary descriptor, and only then installs it. Ubuntu 26.04 provides a newer libxml2
SONAME while ROCm 7.2.1 `lld` still requires `.so.2`; use a separately extracted,
versioned compatibility library rather than a global or unrecorded symlink. A failed
capture or validation leaves the previous descriptor untouched. It does not compile a
kernel, freeze a search contract or authorize a performance claim.

## Prepare the two correctness paths

For source construction only, explicitly use the unreleased v29 draft:

```bash
PYTHONPATH=src python3 examples/gpu/swiglu_amd_quickstart.py \
  --revision compiler/revision.json --prepare-only

PYTHONPATH=src python3 examples/gpu/rmsnorm_amd_quickstart.py \
  --revision compiler/revision.json --prepare-only
```

Preparation runs Schedule parsing, exact Target verification, backend preflight and
deterministic lowering. It must report `target=gfx1151`, `backend=triton`, HIP target
metadata and HSACO/AMDGCN roles without submitting GPU work. The current gfx1151 Target
does not declare per-compute-unit occupancy limits, so accepted AMD Schedules also carry
the non-blocking `RESIDENCY_TARGET_UNMODELED` report. That report is a deliberate limit:
the Compiler does not infer residency, and the GPU measurement remains authoritative.

The live-Q8_1 producer has a separate correctness-only successor runner. Its prepare
path uses the draft Compiler by default and imports no ROCm runtime:

```bash
PYTHONPATH=src python3 examples/gpu/llama_q8_1_amd_quickstart.py \
  --prepare-only
```

Preparation must report the seven frozen Workload cases, one `grid=[1,1,1]` Triton
kernel, `num_warps=8`, HIP `gfx1151/wave32`, a UINT8 `[16,36]` output and zero submitted
GPU work. It proves source construction only.

After a new combined AMD Compiler Gate is independently approved and the exact gfx1151
Executor is released, use the qualified ROCm Python and new evidence directories outside
the checkout:

```bash
PYTHONPATH=src /path/to/rocm/python \
  examples/gpu/swiglu_amd_quickstart.py \
  --executor runtime/executors/open-cake-ir-gfx1151-vN.json \
  --artifact-dir /new/external/path/gfx1151-swiglu-v2

PYTHONPATH=src /path/to/rocm/python \
  examples/gpu/rmsnorm_amd_quickstart.py \
  --executor runtime/executors/open-cake-ir-gfx1151-vN.json \
  --artifact-dir /new/external/path/gfx1151-rmsnorm-v2
```

Each quickstart checks two frozen Workload distributions against a CPU-FP64 oracle and
retains generated source, TTIR, TTGIR, LLVM IR, AMDGCN, HSACO, exact hashes, mismatch and
non-finite counts, launch counts and zero-fallback custody. A draft Compiler remains
available only to `--prepare-only`; GPU execution requires the externally released
Compiler and its passing Gate. Each requested live attempt creates an authority receipt
first and retains either `result.json` or a stage-typed `failure.json`, plus a manifest.

## Source-pinned AITER RMSNorm baseline

The first matched AMD library comparison is AITER RMSNorm, not SwiGLU or Q4 MMVQ.
Prepare it from the official `v0.1.20` source checkout at
`fc2e5d57fb5b8ad8e7e23f7103071dde798ea618`:

```bash
PYTHONPATH=src python3 examples/gpu/aiter_rmsnorm_amd_baseline.py \
  --aiter-checkout /path/to/aiter-v0.1.20 \
  --prepare-only
```

Preparation checks the exact clean AITER repository/tag, its Opus FP32 source chain and
the unchanged `llama-rmsnorm-mul-fp32-v2` Workload. It imports neither Torch nor AITER,
does no JIT work and submits no GPU operation. Published AITER wheels are not accepted
for this cell because the release wheel does not own a gfx1151 device closure.

After a successor gfx1151 Executor has been released from a clean checkout that includes
this runner and captures its Git/ROCm/C++/Ninja build tools and Python JIT dependencies,
run correctness with two new, disjoint directories outside both source checkouts:

```bash
PYTHONPATH=src /path/to/rocm/python \
  examples/gpu/aiter_rmsnorm_amd_baseline.py \
  --aiter-checkout /path/to/aiter-v0.1.20 \
  --executor runtime/executors/open-cake-ir-gfx1151-vN.json \
  --jit-dir /new/external/path/aiter-rmsnorm-jit \
  --evidence-root /new/external/path/aiter-rmsnorm-correctness
```

The live path calls `aiter.ops.rmsnorm.rms_norm_opus` directly with caller-owned output,
fixes `GPU_ARCHS=gfx1151`, rejects source-tree import shadows, a wheel or dynamic
dispatch, then builds, validates and retains the JIT ELF before submitting the first
operator call. AITER's import-time, host-only `module_aiter_core` ELF and its gfx1151
build plan are also validated and retained before the RMSNorm module is built. The core
ELF must export its Python initializer and contain no device code; the RMSNorm ELF must
export `rms_norm_opus` and contain only a gfx1151 code object. Neither artifact may
change across the two cases. Both cases must pass the existing CPU-FP64
tolerance without mutating inputs. The result remains an external-baseline candidate:
it performs no timing and authorizes no speedup, promotion, llama.cpp end-to-end or
serving claim. AITER SwiGLU is deliberately excluded because its packed `[gate,up]`
input would change the frozen two-input Workload boundary; see ADR 0043.

After a new combined AMD Compiler successor and `open-cake-ir-gfx1151-v1` are
independently released, run the Q8 producer from a clean checkout with a new evidence
root outside it:

```bash
PYTHONPATH=src /path/to/rocm/python \
  examples/gpu/llama_q8_1_amd_quickstart.py \
  --executor runtime/executors/open-cake-ir-gfx1151-v1.json \
  --evidence-root /new/external/path/gfx1151-q8-producer-v1
```

The live path refuses a draft or non-canonical Compiler path, a non-gfx1151 schema-v2
Executor, an Executor that does not custody this runner, a dirty checkout, non-HIP
runtime, wrong architecture or non-wave32 target before launch. It then launches once
per frozen case, requires the activation to remain unchanged, checks all 576 output
bytes and the fifteen zero padding records, records zero fallback, and retains both
observed/reference workspaces plus generated source, TTIR, TTGIR, LLVM IR, AMDGCN and
HSACO. It performs no timing and cannot authorize MMVQ or promotion.

## True-one-row hypothesis

The historical AMD branch searched row tiles 2–64 and stopped at 1.00453x under a 1.05x
gate. It could not express a one-element row vector because vector addressing was inferred
from tile width. In the proposed v29, `AccessMap source: program_tile` owns that fact directly, so
`row_tile=1` generates `tl.arange(0,1)` while preserving rank-2 `[1,128]` reduction
semantics.

The successor search is intentionally only:

```text
row_tile = 1
num_warps in {1,2,4,8}
baseline = row_tile 64, num_warps 4
```

The llama-faithful point is one row and eight wave32 groups. The search contract is not
frozen until the post-alignment AMD Compiler successor is externally approved; therefore
no GPU timing command or performance result is claimed yet. Once frozen, the runner will
retain correctness for both Workload cases, L2-flushed HIP-event screening, four ABBA
confirmation cohorts, raw samples, a 0.05 CV gate and the unchanged 1.05x materiality
rule.

## Boundary

This is a generated leaf-kernel path. It is not a completed llama.cpp build, model layer,
token trajectory or serving result. No candidate is promoted without the frozen
correctness/noise/materiality gates and profiler evidence. The active AMD direction is
the complete Q4_0/Q8_1 MMVQ path, including activation
quantization and packed scale/sum semantics—not an opaque block-dot primitive.

ADR 0039 now freezes the first dependency-free Q4 conformance boundary: one raw 18-byte
Q4_0 block plus one FP32 activation block, a byte-exact live-produced 576-byte padded
Q8_1 workspace (one consumed record plus 15 zero records), and the source-ordered
two-part stored-s correction oracle. ADR 0040 adds raw UINT8/INT8 storage and exact Q4/Q8
record custody. ADR 0041 adds the first composed Q8 producer: masked 512-value padding,
explicit wave32 XOR reductions, precise FP32 division, half-away rounding, typed casts
and relation-derived little-endian Q8_1 stores. It passes deterministic source lowering,
but the new combined AMD Gate and approval, exact gfx1151 Executor and on-device
576-byte comparison are still pending. The Q4 consumer, dot4 contract, HSACO evidence,
timing and promotion claims do not yet exist. ADR 0042 records the gated consumer design
as typed packed Load plus one
instruction-bound generic Dot and two widening casts; it is not implemented before the
producer passes real gfx1151 correctness.
