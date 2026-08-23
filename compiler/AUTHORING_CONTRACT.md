# Open Cake Schedule authoring contract

Submit one complete JSON document conforming to `schedule.schema.json`. The Compiler—not the prompt—owns semantic
acceptance. Names are unique within each declaration list; operation dependencies refer only backward; every output
must be written; buffer allocation extents and role warps must fit the exact Target. `program_map` and `grid` are
mutually exclusive. Findings carry a stable code and path. A blocking Finding is a reason an Assessment is not
lowering-eligible and no candidate reaches the toolchain; a non-blocking Finding accompanies an Assessment that is,
and reports what the declared Schedule implies — such as which declared resource bounds its residency. Unsupported
shapes, operations, instructions, memory spaces or uncalibrated analyses are explicit.

For Flash-KMeans, the Workload Contract owns B/N/K/D, BF16/FP32/INT32 semantics, tie handling and oracle. A Study
narrows the public Compiler to one admitted profile and supplies a complete `schedule-skeleton.json`; start from that
skeleton. A Schedule may change admitted block sizes, warps and stages, but must preserve its profile, external tensor
shapes, `metadata.workload_contract_sha256`, operator semantics, and frozen Compiler Revision during a Run.
