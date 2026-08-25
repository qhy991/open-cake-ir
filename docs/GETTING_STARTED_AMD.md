# AMD gfx1151 development path

This branch is based directly on GitHub `main@f02320b17ee1f6ed12d1d858e3439bc16b9da4ed`.
It adds exact gfx1151 Target support, standalone FP32 SwiGLU, llama.cpp FP32
RMSNorm+Mul, a true-one-row optimization hypothesis and a generated live-Q8_1 producer.
The proposed Compiler v29 has a passing 45-case Gate but is not released until an
external reviewer writes the exact approval required by ADR 0030. Commands below that
use the draft submit no GPU work.

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
PYTHONPATH=src /path/to/rocm/python \
  tools/release_gfx1151_executor_cycle.py
```

The release command derives the independent `open-cake-ir-gfx1151-vN` identity, captures
the exact Linux/Python/Torch/Triton/HIP and `amd-smi` facts, includes only actually
available AMD profilers, builds and live-admits a temporary descriptor, and only then
installs it. A failed capture or validation leaves the previous descriptor untouched.
It does not compile a kernel, freeze a search contract or authorize a performance claim.

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

The live-Q8_1 producer has a separate correctness-only successor runner. Its prepare
path uses the draft Compiler by default and imports no ROCm runtime:

```bash
PYTHONPATH=src python3 examples/gpu/llama_q8_1_amd_quickstart.py \
  --prepare-only
```

Preparation must report the seven frozen Workload cases, one `grid=[1,1,1]` Triton
kernel, `num_warps=8`, HIP `gfx1151/wave32`, a UINT8 `[16,36]` output and zero submitted
GPU work. It proves source construction only.

After an external approval releases v29 and the exact gfx1151 Executor is released, use
the qualified ROCm Python and new evidence directories outside the checkout:

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

After both Compiler v29 and `open-cake-ir-gfx1151-v1` are independently released, run
the Q8 producer from a clean checkout with a new evidence root outside it:

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
frozen until Compiler v29 is externally approved; therefore no GPU timing command or
performance result is claimed yet. Once frozen, the runner will retain correctness for
both Workload cases, L2-flushed HIP-event screening, four ABBA confirmation cohorts,
raw samples, a 0.05 CV gate and the unchanged 1.05x materiality rule.

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
but the v29 approval, exact gfx1151 Executor and on-device 576-byte comparison are still
pending. The Q4 consumer, dot4 contract, HSACO evidence, timing and promotion claims do
not yet exist. ADR 0042 records the gated consumer design as typed packed Load plus one
instruction-bound generic Dot and two widening casts; it is not implemented before the
producer passes real gfx1151 correctness.
