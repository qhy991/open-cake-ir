# Bounded original RoPE qualification

Related issue: #445. Starter: #444. Original math qualification: #401.

This successor consumes the bounded RNE10 starter through its full original
Workload entry. It does not change the original input factory, reference,
comparator, tolerances or 16 workload IDs. Historical IEEE and TF32 failures stay
at their original commits and evidence paths.

## Source and CPU boundary

Build and evaluate require the same clean source commit and an explicit external
`--oracle-numerics` observation. All 16 Workloads bind that same policy. The
starter checks the fixed original task, factory and HIGH policy. The local math
probe adds only sin/cos to a temporary Target for emission. Production sin/cos
and TF32 declarations must remain closed.

Before a lease, `bind_original_build` reconstructs each original Workload and
current starter, then checks the saved Workload, emitted source, candidate seal,
launch manifest, compile record and native pointer ABI. Old-policy or different
source builds are refused. The stage builder remains the existing isolated
Triton compiler; the tool does not JIT code in the evaluator.

## Original device boundary

The tool passes the bound cases to the existing
`qualify_c550_bench_native.evaluate_original` owner. That owner runs the original
`check_problem` with all 16 workloads and ten rounds, checks the original report
and native trace order, and retains errors. `NativeBench` supplies fresh outputs,
performs only native candidate calls and storage checks, and closes modules.
There is no separate RoPE callback implementation.

The shared oracle-policy owner checks after device observation, before and after
the original whole-task loop, and at each callback entry. These are boundary
checks; the fixed Bench's internal factory/reference calls are not instrumented.

```bash
python tools/benchmarks/c550/qualify_rope.py build \
  --bench-root /path/to/fixed-bench --oracle-numerics /path/to/observed-policy.json \
  --output /path/to/new-build

python tools/benchmarks/c550/qualify_rope.py evaluate \
  --bench-root /path/to/fixed-bench --oracle-numerics /path/to/observed-policy.json \
  --built /path/to/new-build --output /path/to/new-evaluation \
  --physical-device DEVICE --runtime-device RUNTIME --expected-pci PCI
```

Run each phase only after its acceptance gates. Source/CPU passage does not prove
native cast behavior or complete RoPE correctness. This tool records no timing,
provider optimization, or production Target admission. A later successful
original device check still requires independent review before a Target change.
