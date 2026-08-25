# ADR 0037: gfx1151 is an exact Target, not a lowering profile

Status: proposed for Compiler Revision `open-cake-ir-v29`; external Compiler approval
and the one-row gfx1151 evaluation are pending.

## Context

The first AMD development branch proved that generated Triton source can compile to
HSACO and run correctly on gfx1151. It also searched 24 combinations of row tiles 2–64
and one to eight waves for llama.cpp RMSNorm+Mul. All candidates were correct, but the
selected `row_tile=8,num_warps=8` result reached only 1.00453x over the historical
64-row/four-wave baseline and was retained as `STOP_CLOSE_NULL` under a 1.05x gate.

That branch was based on an older Compiler with a Target/profile implementation registry.
Current main has removed that registry: `Schedule.lowering` names a materialization
mechanism and executable symbol, while Workload semantics belong outside the Compiler.
Porting the old profile-specific HIP adapters would reintroduce the duplicate ownership
removed by ADR 0029.

The current Target schema was still CUDA-shaped, however: it required compute capability,
tensor-memory capacity and a four-warp register-budget issue scope from every Target.
Those are not facts about gfx1151. The same IR also inferred vector program addressing
from `ProgramAxis.tile > 1`, which prevented an explicit one-element row vector even
though `AccessMap` already distinguishes `program` from `program_tile`.

The latest audited llama.cpp revision is
`eb25b7263e1604b4382295563f5a924002d6f87c`. Its relevant HIP/CUDA operator files are
byte-identical to the prior `f280b269` audit. For D=128 its RMSNorm launcher uses one row
per workgroup and 256 threads, which is exactly one row and eight wave32 execution groups.

## Decision

1. Add Target schema v2 with explicit `execution_group_width` and architecture-neutral
   workgroup/threadgroup limits. CUDA compute capability, tensor-memory capacity and
   register-budget group width are optional hardware facts; an absent fact is never
   synthesized. Schema v1 remains byte- and behavior-compatible for historical B200
   Targets.
2. Bind exact `gfx1151` as a second Compiler Target. It admits global/shared/register
   memory, load/reduce/elementwise/store, wave32 and the observed workgroup limits. The
   target does not claim CUDA identity, tensor memory or a register-budget instruction.
3. Keep one `lowering.backend: triton` mechanism. For gfx1151, the Target-aware emission
   projects exact HIP runtime admission plus AMDGCN/HSACO artifact roles; no operator name
   or Workload selects a separate backend.
4. Make `AccessMap` the scalar/vector owner. `source: program_tile` always requests an
   offset vector, including when the tile width is one. Existing 32 Corpus lowering
   digests remain unchanged because their access sources already agree with their tiles.
5. Preserve the original llama Workload v1 bytes. Add Workload v2 solely to pin the latest
   upstream revision; it makes no material AMD-kernel-delta claim.
6. Freeze a new four-candidate search over the true-one-row geometry with waves
   `{1,2,4,8}`. The historical 64-row/four-wave Schedule remains the confirmatory
   baseline. Correctness, noise and the 1.05x materiality gate remain unchanged. Old
   2–64-row candidates are not rerun as a new hypothesis.

## Consequences

- The proposed Compiler v29 has a 39-case, 56-source multi-Target Gate. The four new observations are
  gfx1151 SwiGLU positive/instruction-negative and RMSNorm baseline/one-row positives;
  all previous 32 finding and lowering digests are unchanged.
- Per-role register budgets on a Target without a register-budget issue scope fail with
  `ROLE_REGISTERS_TARGET_UNSUPPORTED` rather than borrowing NVIDIA warpgroup semantics.
- A true-one-row leaf timing WIN still does not prove a llama.cpp build, token path or
  serving gain. Missing rocprof/Omniperf evidence blocks automatic promotion.
- The next high-value AMD direction after this bounded result is an honest Q4_0/Q8_1
  MMVQ representation including activation quantization, blocked packing, scale/sum
  correction and the gfx1151 `sudot4` instruction contract—not an opaque block-dot op.
