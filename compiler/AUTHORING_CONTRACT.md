# Open Cake Schedule authoring contract

Submit one complete JSON document conforming to `schedule.schema.json`. The Compiler—not the prompt—owns semantic
acceptance. Names are unique within each declaration list; operation dependencies refer only backward; every output
must be written; buffer allocation extents and role warps must fit the exact Target. `program_map` and `grid` are
mutually exclusive. Each role owns one ascending contiguous warp interval, and no warp belongs to two roles.
Findings carry a stable code and path. A blocking Finding is a reason an Assessment is not
lowering-eligible and no candidate reaches the toolchain; a non-blocking Finding accompanies an Assessment that is,
and reports what the declared Schedule implies — such as which declared resource bounds its residency. Unsupported
shapes, operations, instructions, memory spaces or uncalibrated analyses are explicit.
An otherwise-lowerable `mma` in a backend-emitted profile names the Target instruction contract that determines its
lowering; omitting it is a lowering-blocking candidate Finding rather than a late emitter failure. Instruction-free
asset profiles remain valid because they do not emit operation bodies.
Register findings are static bounds over declared logical storage, never a claim about ptxas's physical allocation.
A block scale is an FP32 Buffer with one `scale_of` relation. `granularity` is written in
the FP8 data buffer's axis order; `axis_order` is the full permutation that gives the
scale buffer's physical grouped-axis order. The Compiler derives the scale shape and
requires loads to preserve the relation. The block-scaled MMA read order is data A, data
B, scale(A), scale(B); names and convenient shapes never substitute for those relations.
`reduce_argmin` compares FP32 values and returns INT32 source positions; admitting FP8
storage for another operation does not widen that semantic contract.

For Flash-KMeans, the Workload Contract owns B/N/K/D, BF16/FP32/INT32 semantics, tie handling and oracle. A Study
narrows the public Compiler to one admitted profile and supplies a complete `schedule-skeleton.json`; start from that
skeleton. A Schedule may change admitted block sizes, warps and stages, but must preserve its profile, external tensor
shapes, `metadata.workload_contract_sha256`, operator semantics, and frozen Compiler Revision during a Run.
