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

Lowering selection is exact by `(target, metadata.profile)`. A profile implemented on one
Target never falls back to that Adapter on another Target, and Calibration coverage is
scoped to the same exact pair. The first `apple_gpu_family9` Adapter is deliberately
finite: `flash_kmeans_b32_smoke`, four or eight 32-wide SIMDgroups, one unstaged centroid
loop, BF16 8x8x8 SIMDgroup MMA, canonical access maps and the fixed four-buffer ABI.
Anything outside that subset is a lowering Finding rather than an inferred Metal choice.

For Flash-KMeans, the Workload Contract owns B/N/K/D, BF16/FP32/INT32 semantics, tie handling and oracle. A Study
narrows the public Compiler to one admitted profile and supplies a complete `schedule-skeleton.json`; start from that
skeleton. A Schedule may change admitted block sizes, warps and stages, but must preserve its profile, external tensor
shapes, `metadata.workload_contract_sha256`, operator semantics, and frozen Compiler Revision during a Run.
