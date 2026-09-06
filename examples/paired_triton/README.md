# Matched IR/native source baselines

`prepare.py` specializes each formal Workload case from its visible Corpus Schedule and
exports source through the existing Compiler. It never imports or executes generated code.
The outputs are source-only preparation, not a LaunchableCandidate or GPU result.

See [the interface and command](../../docs/en/TILE_WORKLOADS.md) or
[中文说明](../../docs/TILE_WORKLOADS.md). The canonical Workload inputs are:

- [`rmsnorm-fp32-v1.json`](../../contracts/workloads/rmsnorm-fp32-v1.json)
- [`gemm-bias-bf16-fp32-v1.json`](../../contracts/workloads/gemm-bias-bf16-fp32-v1.json)
- [`indexed-gather-bf16-v1.json`](../../contracts/workloads/indexed-gather-bf16-v1.json)

`baseline_schedule(workload, case_id)` returns a complete Schedule document;
`prepare_baseline(workload, case_id, output_root)` writes `baseline.schedule.json`,
`baseline.triton.py` and `preparation.json` to a new external directory. Each native
baseline is exactly the Compiler lowering of its companion IR. Optimization may start
from this declared common implementation; it is not clean-start frontier synthesis.
