# Original C550 Bench oracle numerics

New C550 Bench Workloads bind the observed Torch matrix policy in
`semantics.oracle_numerics`. The policy contains four effective settings:

- `float32_matmul_precision`
- `allow_tf32`
- `allow_fp16_reduced_precision_reduction`
- `allow_bf16_reduced_precision_reduction`

`initialization.TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` records the observed process
initialization input, or `null` when it is absent. This input is not a Target
capability. The effective settings are read from Torch, not inferred from it.
FP32 tensor storage alone does not require an IEEE FP32 matrix calculation.

## Prepare a successor

Retain the small JSON value returned by
`tasks.c550_bench.binding.observe_oracle_numerics()` in the accepted oracle
environment. For an already retained observation, preserve those observed values;
do not replace them with the preparation host's defaults. Pass that JSON to:

```bash
python tools/prepare_c550_bench.py --bench-root /path/to/fixed-bench \
  --oracle-numerics /path/to/retained-oracle-numerics.json \
  --output /path/to/new-preparation
```

The preparation record references the external observation. The Workload binds
its value and existing TASK material includes the full Workload and its numeric
contract. Preparation does not import Torch, change precision, or launch a Run.
Missing policy data is refused. Old Workloads and their evidence replay at their
original source commit; this change does not fill in historical data.

## Check the reference boundary

`require_oracle_numerics(expected, phase=...)` reads and compares the effective
settings and initialization input. It never changes any of them.

The common Task preparation owns each original call, so it checks immediately
before and after input generation and reference execution. A failed check stops
preparation. The original input factory, reference and comparator are unchanged.

The full-Bench bridge must call this same function after device observation,
before and after the original `check_problem`, and at each candidate entry after
the original reference returns. This covers those boundaries only. It does not
instrument or replace the fixed Bench's internal factory/reference calls.

## Evidence scope

The original RoPE check exposed a concrete distinction: in one retained case,
the candidate matched FP32 elementwise angles and trigonometric output exactly,
while both differed from the original reference's matrix calculation. The
observed environment initialized Torch to `high` precision with TF32 enabled;
its initialization override was `1`. The seed function did not set that policy.
This motivates freezing the actual oracle policy. It does not qualify a TF32
lowering or change the original tolerance.

This is a Workload and evaluation-boundary repair. It adds no Compiler primitive,
Target capability, performance result, Run schema or replacement comparator.
