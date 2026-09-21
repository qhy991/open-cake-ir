# gfx1151 task entrypoints

The unified launcher exposes `add_rmsnorm_bf16` and four AKA-derived tasks:
`aka_residual_layernorm`, `aka_gemm_nt_bias`, `aka_row_gather`, and
`aka_momentum_sgd`. They use the existing `triton-gfx1151` device row, tensor
Workload oracle, isolated Triton compiler, HSACO artifacts, and HIP local broker.
The existing explicit `cast` gains one directed typing edge, signed `int32` to
`fp32`, for the dynamic Nesterov flag. Conversion rounds to nearest FP32, ties to
even; the flag's 0/1 values are exact. Reverse and other integer conversions remain
refused. The conversion keeps its explicit operation, shape and effects and uses
the existing backend emission. No Target capability set is widened.

The AKA B200 revision-1 documents remain frozen. A target other than the legacy
B200 gets a revision-2 Workload only for a task with a complete starter, and only
when its Target and lowering route admit the required operations and dtypes.
Historical AKA qualification supplies semantic provenance, not evidence for the
new target. The original oracle, all input distributions, ABI and tolerances remain
part of each task's contract.

| Task | Default shape | Complete outputs |
| --- | --- | --- |
| `add_rmsnorm_bf16` | R=128, C=2560 | normalized value and rounded residual |
| `aka_residual_layernorm` | R=8, C=256 | residual, normalized value, mean, reciprocal standard deviation |
| `aka_gemm_nt_bias` | M=8, N=256, K=32 | FP32 GEMM with N-by-K weight storage plus bias |
| `aka_row_gather` | source/output rows=8, C=256 | complete gathered rows; repeated and boundary indices are required cases |
| `aka_momentum_sgd` | E=8×256 | parameter and momentum, with both Nesterov flag values |

`--rows` and `--columns` override these extents. For the AKA gather they set both
source and output row counts and the row width; its direct Workload factory also
supports distinct source/output counts. For AKA momentum their product is the
one-dimensional element count. Only AKA NT GEMM accepts `--depth`; the matrix
launcher's existing depth flag defaults to 256 and explicitly supplies that value.
Triton row widths and GEMM reduction spans must be powers of two. NT GEMM retains
the predecessor's M/N divisibility by eight. The momentum starter bounds its element
count to signed-int32 coordinates (at most 2**31-1); oversized shapes are refused
before entering the native compiler, including through the legacy B200 starter.

Both `tools/launch_task.py --task` and explicit `tools/launch_task_matrix.py --task`
accept these names. The matrix's default task subset remains the existing portable
matrix; use repeated `--task` arguments to select the new tasks. The supported task
selector is owned by the single-task launcher, rather than duplicated in the matrix.
`--baseline-only` builds and seals a candidate without calling a provider or
executing GPU kernels. A successful build does not establish device correctness or
performance; those require the ordinary task evaluation against every input case.

The gfx1151 GELU tasks use the Target's measured `ocml.tanh.f32` declaration. Its
196,915-input pointwise device probe is cited in `compiler/targets/gfx1151.json`;
full task correctness and performance remain separate evaluations. AKA histogram
and max-pool have CPU contracts but no portable starters. DeepSeek-V4 routing, full
SoL/FIB Attention/MoE and the legacy specialized tasks retain their own execution
contracts. These are remaining adaptation work, not implied support from a renamed
backend or a successful unrelated kernel.
