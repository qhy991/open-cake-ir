# Weave scheduling reference, before GPU lowering

`scheduler_model.py` implements the workload volumes and joint `(c, K)` search
from [Weave §3](https://arxiv.org/html/2609.21483v1), with all throughput and
chunk-efficiency curves supplied explicitly by the caller. It gives no target
default and reports seconds only when the supplied rates use bytes/s and
FLOPs/s. Its steal count is the paper's continuous estimate, not an integer
work-claim budget.

`ChunkQueue` is a separate immediate-completion model for the execution DAG:
dispatch → GEMM0 → GEMM1 → combine for each chunk. Dispatch tiles are tagged
with their destination chunk so local compute can begin as data arrives, but
the communication workers drain the dispatch queue before entering their one
consolidated steal window. The window closes when any combine begins. Compute
workers may assist the final combine after all GEMM and earlier combines finish.
Its executable checks cover unique tile ownership, dependency order, bounded
steals and absence of a stalled ready queue under fair worker turns.

This is a **control-flow reference**, not a CUDA timing simulation or a Cake
Schedule. It assumes each claimed tile completes immediately. It cannot prove
cross-CTA or cross-GPU memory visibility, physical CTA residency, NVLink
transfer completion, profiler overlap, latency, or liveness under actual GPU
scheduling. Those are precisely the contracts a future Cake primitive and
native CUDA/PTX lowering must state and test. `atomic_rmw` alone supplies only
unique work indices under relaxed device scope; it does not publish a chunk.

Run the CPU checks with:

```sh
PYTHONPATH=src:. python3 -m unittest tests.contracts.test_weave_scheduler_model
```
