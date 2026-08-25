# AMD gfx1151 development path

This branch is based directly on GitHub `main@f02320b17ee1f6ed12d1d858e3439bc16b9da4ed`.
It adds exact gfx1151 Target support, standalone FP32 SwiGLU, llama.cpp FP32
RMSNorm+Mul, and a true-one-row optimization hypothesis. The proposed Compiler v29 has a
passing 39-case Gate but is not released until an external reviewer writes the exact
approval required by ADR 0030. Commands below that use the draft submit no GPU work.

## Environment

The previously qualified runtime combination is one visible gfx1151 wave32 device,
ROCm 7.2.1, Python 3.12, ROCm PyTorch 2.9.1 and Triton 3.5.1. PyTorch intentionally uses
the `torch.cuda` namespace for ROCm. Runtime admission checks `torch.version.hip`, the
active Triton target, `gcnArchName=gfx1151` and wave width; another device or a CUDA
build fails before compilation.

## Prepare the two correctness paths

While v29 approval is pending, explicitly use the draft:

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

After an external approval releases v29, use the qualified ROCm Python and new evidence
directories outside the checkout:

```bash
PYTHONPATH=src /path/to/rocm/python \
  examples/gpu/swiglu_amd_quickstart.py \
  --artifact-dir /new/external/path/gfx1151-swiglu-v2

PYTHONPATH=src /path/to/rocm/python \
  examples/gpu/rmsnorm_amd_quickstart.py \
  --artifact-dir /new/external/path/gfx1151-rmsnorm-v2
```

Each quickstart checks two frozen Workload distributions against a CPU-FP64 oracle and
retains generated source, TTIR, TTGIR, LLVM IR, AMDGCN, HSACO, exact hashes, mismatch and
non-finite counts, launch counts and zero-fallback custody.

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
frozen until Compiler v29 is externally approved; therefore no GPU timing command or
performance result is claimed yet. Once frozen, the runner will retain correctness for
both Workload cases, L2-flushed HIP-event screening, four ABBA confirmation cohorts,
raw samples, a 0.05 CV gate and the unchanged 1.05x materiality rule.

## Boundary

This is a generated leaf-kernel path. It is not a completed llama.cpp build, model layer,
token trajectory or serving result. No candidate is promoted without the frozen
correctness/noise/materiality gates and profiler evidence. The next planned AMD direction
after this bounded result is the complete Q4_0/Q8_1 MMVQ path, including activation
quantization and packed scale/sum semantics—not an opaque block-dot primitive.
