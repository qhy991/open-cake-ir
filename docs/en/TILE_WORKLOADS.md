# Standalone tile Workloads

The three contracts in `contracts/workloads/` reconstruct the visible Corpus seeds as
standalone operators. They own mathematical semantics, ordered tensor ABI, cases, input
domain and all-element correctness rules. Evaluation owns input materialization and the
independent CPU oracle. The example preparation script specializes the existing seeds;
the Compiler receives complete Schedules and has no Workload dependency.

| Contract | Primary input ABI | Primary output | Definition |
| --- | --- | --- | --- |
| `rmsnorm-fp32-v1.json` | FP32 x `[8,512,128]`, gamma `[128]` | FP32 y `[8,512,128]` | `x * gamma / sqrt(mean(x*x, last_axis) + epsilon)`, epsilon `1e-6` |
| `gemm-bias-bf16-fp32-v1.json` | BF16 a `[512,256]`, b `[256,256]`; FP32 bias `[256]` | FP32 c `[512,256]` | `sum_k a[m,k] * b[n,k] + bias[n]`; b is stored as `[N,K]` |
| `indexed-gather-bf16-v1.json` | BF16 expert_rows `[4,8,16]`; INT32 expert_ids and row_ids `[8,8]` | BF16 gathered_rows `[8,8,16]` | Zipped expert/row selection; either invalid ID produces a row of positive zero |

The domain is contiguous row-major, finite floating inputs with declared magnitude bounds,
unchanged inputs and fresh nonaliasing outputs. This is not framework equivalence, training,
arbitrary-stride support, a dynamic-shape ABI, or serving integration. Old zero metadata
digests are not inherited as Workload authority. A Study must explicitly select the new
frozen contract. Loading checks supported mathematical/structural invariants; the selected
Workload's canonical identity binds its exact cases, scalar and tolerance values.

RMSNorm's seed uses FP32 intermediate buffers, a FP32 reduction, multiplication by `1/D`,
epsilon addition, rsqrt, then two scale multiplications. GEMM accumulates BF16 dot products
in FP32 over K tiles before FP32 bias addition; it does **not** compute `A @ B` with `[K,N]`
storage. Masked K loads contribute zero, and M/N tail stores are suppressed. Neither seed
specifies a hardware reduction tree; rsqrt is also approximate. The independent oracle uses
`math.fsum`/`sqrt` on the mathematical definition and final FP32 rounding, not a simulation
of the emitted instructions. RMSNorm's `atol=2e-6, rtol=2e-5` and GEMM's
`atol=rtol=1e-3` are predeclared allowances for those differences, not observed device
calibrations. Every element must pass. Gather has no arithmetic and requires BF16 bitwise
equality, preserving signed zero for valid selections and positive zero for invalid IDs.
Negative IDs never wrap; repetitions are allowed; expert and row arrays are zipped, not
combined as a Cartesian product. INT32 minimum and maximum are boundary cases.

## Shared interface for pairing and Evaluation

```python
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.evaluation.tile_workloads import materialize_case, reference_outputs

workload = WorkloadContract.load("contracts/workloads/rmsnorm-fp32-v1.json")
abi = workload.tensor_abi("tiny")
inputs = materialize_case(workload, "tiny")
expected = reference_outputs(workload, "tiny", inputs)
```

`tensor_abi(case_id)` returns an immutable tuple of `TensorABI(name, shape, dtype, mode)`:
all inputs in the declared order, then outputs in the declared order. It resolves each
symbolic tensor dimension against the selected case; it never evaluates expressions or
dispatches on an operator name. The descriptor owns shape and dtype; consumers allocate and
reshape with those values and select arguments by `mode`. Historical contracts without this
explicit `{inputs, outputs}` declaration retain admission, but requesting this new helper
fails rather than guessing their ABI. No implicit migration or legacy ABI inference occurs.

`materialize_case` returns an input-name mapping of flat row-major Python lists, using an
explicit local PRNG seed and dtype rounding before return. The frozen Evaluation source and
Executor environment bind its materialization implementation. `reference_outputs` accepts
that mapping, rejects missing/extra tensors, wrong lengths, nonfinite/out-of-domain values,
values not representable in the declared dtype, and non-INT32 index values. It returns an
output-name mapping in output order. It does not mutate inputs. Neither helper imports Torch,
a device runtime, Compiler, Lab, nor candidate code. B should perform generic device
conversion from `tensor_abi`, without its own operator-to-shape table.

## Baseline preparation

`examples/paired_triton/prepare.py:baseline_schedule` verifies the visible seed's primary ABI,
substitutes selected-case global shapes from `tensor_abi`, and makes only the existing seed's
register-width/reduction-constant adjustments. Epsilon comes from the Workload. Performance
choices remain explicit in that Schedule: 64-row RMSNorm blocks, 64×64×64 GEMM tiles and
the seed's existing launch configuration. Small GEMM cases have only 2–15 output elements
but K=65/66, exercising a tail and two loop iterations without a large Python GEMM. The
current Compiler refuses a redundant single-trip TileLoop; this workflow preserves that
rule. All three primary Schedules retain the original shapes, operations and tiling; only
their Schedule IDs and placeholder Workload metadata binding change.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python examples/paired_triton/prepare.py \
  --workload contracts/workloads/rmsnorm-fp32-v1.json --case primary \
  --output-root /absolute/path/outside-checkouts/rmsnorm-primary
```

Use each of the other two Workload paths for GEMM and gather. The output directory must be
new and outside every checkout. The script calls `Compiler.assess` and `Compiler.lower`,
and writes `baseline.schedule.json`, the exact lowering `baseline.triton.py`, and
`preparation.json` with entry point, projected ABI, source map and Compiler toolchain
requirements. The native file is a generated projection of the same IR implementation.
It is not independently authored, and is suitable as an explicitly declared common
optimization baseline, not evidence for a clean-start comparison. Compiler host wrappers
take positional inputs in ABI order and optional `out=`; the lowered kernel signature and
compile constants must be taken from its toolchain requirements.

Preparation checks Python syntax without importing the generated module. It reports
`status=source_only`; no target compilation, launchable-candidate sealing, device equality,
timing or profiler execution occurs. Sealing and common Evaluation remain with the existing
interfaces. CPU tests cover mathematical examples, epsilon, rectangular RHS orientation,
tails/cancellation, index boundaries, ABI refusal and same-implementation source export.
Torch/device compilation, GPU equivalence, performance, profiler and target-framework
end-to-end acceptance are R2 pending. No optional Torch/device path is claimed as tested.

The implementation extends Evaluation as successor work: historical Compiler releases,
Executor descriptors, Corpus expectations and Workload files are preserved. In particular,
old Executor closures pin `evaluation/workload.py`; new runtime use needs a reviewed
Executor successor rather than treating those old descriptors as valid for changed bytes.
