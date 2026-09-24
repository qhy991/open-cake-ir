# B300-M4 CTA scheduling mechanism probe

`worker_pipeline.cu` is a direct CUDA/PTX reference for the control mechanisms
Cake does not yet express. It is a synthetic **single-GPU** pipeline: dispatch
copies one 16-element token tile, compute performs a small FP32 matrix-vector
tile, and combine copies the result. These operations deliberately stand in
for MoE communication and grouped GEMM; they do not implement either one.

One cooperative launch creates 148 CTAs, matching the observed SM count of the
exact `sm_103a` Target. Every CTA reads device-resident `(c, K, steal_limit)`.
The host admits the Target's two declared B300 device names; B300-M4 exposed
`NVIDIA B300 SXM6 AC` on a broker-assigned card, which a first prototype had
incorrectly refused before kernel launch.
`blockIdx.x < c` selects communication workers; other CTAs compute. The
runtime values change the worker split and the chunk size without recompiling
the kernel. The host refuses a device without cooperative-launch support or
enough occupancy for this grid. [CUDA's cooperative launch contract](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#when-to-use-cudalaunchcooperativekernel)
is the liveness premise for the cross-CTA waits. Ordinary `ProgramMap.persistent`
currently does not make that promise.

The phase protocol is:

1. Communication CTAs claim dispatch tiles with device-scope relaxed PTX atomics.
   Each tile's payload is published by `st.release.gpu.global.u32` on its own
   ready flag after the whole warp finishes writing.
2. Compute CTAs claim tiles from a second counter and use
   `ld.acquire.gpu.global.u32` on the exact tile flag before reading it.
3. Once all dispatch tiles finish, communication CTAs claim compute tiles until
   the first chunk is complete, the explicit steal budget expires, or the
   compute queue empties. Every communication CTA then reaches one bounded
   phase barrier. Compute CTAs keep processing their claimed work.
4. All CTAs may claim combine tiles after that barrier. A combine waits until
   its entire chunk has completed compute, then acquires its own tile's ready
   flag before reading the result. The release/acquire pair, rather than a
   relaxed completion count, owns payload visibility.

This is a bounded kernel protocol, not an IR vocabulary decision. The test ABI
has 16 separately allocated buffers: input, weights, dispatched values,
computed values, output, two per-tile flags, per-chunk completion counts,
seven work/phase counters, four summary statistics, three per-tile execution
counts, a device-resident three-value plan, status and per-CTA role trace. The standalone runner
must reset flags, counters, execution counts and status before each launch.

The external oracle checks the matrix-vector output and exactly one dispatch,
compute and combine execution per tile. It checks `computed_by_compute +
stolen == tile_count`, the steal budget, the runtime CTA role split and that
each tested plan reaches completion. The first combine records the number of
completed compute tiles, exposing whether K let combine start before all
compute finished. A nonzero stolen count is needed to say the steal path was
exercised. Inputs, compile output, broker receipts and device snapshots stay
outside the source checkout. Correctness and performance are separate phases;
this reference makes no timing or NVLink claim.

Before Cake can admit the same structure, its shared IR and verifier need to
name CTA classes separately from warp `Role`, device-resident runtime split
inputs, bounded work queues, release/acquire payload ownership, cooperative
residency and phase liveness. The native backend then owns PTX emission; the
Workload Contract owns the external oracle and per-rank routing semantics.
