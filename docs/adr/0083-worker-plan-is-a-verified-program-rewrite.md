# ADR 0083: A CTA worker plan rewrites a complete Program

Status: proposed. The B300 scheduling mechanism has device evidence. A bounded
version-2 Program execution descriptor now checks structure and dataflow;
native worker lowering and Evaluation remain unimplemented and explicitly
refused.

## Problem

`Role.execution_groups` partitions warps **inside** one CTA. A persistent
`ProgramMap` assigns the same Schedule body to each CTA. Neither represents
Weave's per-layer split of CTAs into communication and computation workers,
their distinct task loops, chunk dependencies, or a communication worker
temporarily claiming computation. Reusing `depends_on` across CTAs would be
wrong: it orders one role's operations but publishes no global payload.

The direct B300-M4 prototype at `863eb3fd` validates the mechanism under a
cooperative launch: runtime `c=12/36/120`, `K=1/4/2`, exact-once dispatch,
compute and combine tile ownership, early combine for two chunked plans, and
256/512 compute tiles stolen in the compute-heavy plan. Its independent oracle
and broker receipt are at
`B300-M4:/home/qinhaiyan/cake-weave-worker-b300-m4-863eb3fd/`.
This is single-GPU synthetic work. It supplies no NVLink, grouped-GEMM,
end-to-end MoE or latency evidence.

## Proposed representation and owners

Use one complete Compiler `Program` for the mathematical stages and their
single-assignment global tensors. An explicit **worker-plan rewrite** names
how those existing leaf Schedules execute in one cooperative kernel:

- an exact Target, one device-resident INT32 cutoff `c`, one chunk count `K`
  and one bounded steal allowance;
- two CTA classes, each with an ordered list of phases and a nonempty runtime
  population (`0 < c < N`); the existing `Role` retains intra-CTA warp meaning;
- one typed queue per task domain, with its own INT32 state counter, static
  capacity, returned-old-value claim, tile coordinate and set of eligible CTA
  classes; dispatch, compute and combine are stage names in the Program, not
  IR opcodes;
- per-tile handoffs that name an output payload, an INT32 readiness flag,
  one producing stage and one consuming stage. The producer publishes with
  device-scope **release** after all payload-writing lanes rendezvous; each
  consuming lane uses device-scope **acquire** on that tile's flag before its
  first payload read;
- one consolidated steal window after dispatch and before combine; both CTA
  classes claim the **same compute queue** during it. The final combine phase
  may admit both classes without adding a second combine queue.

This is a Program transformation requested explicitly by the author or Lab
recipe. The Compiler owns its structure, deterministic rewrite and verifier;
the Workload Contract owns MoE mathematics, external oracle and state reset;
the Study controls access and comparison policy. The NVIDIA native backend
owns PTX atomics, release/acquire loads and stores, branch emission and
cooperative launch. A backend name does not multiply for each vendor or
workload. A route that cannot emit all declared effects refuses by name.

The first structural slice is `Program` schema version 2 with
`execution.kind="cooperative_workers"`. Version 1 remains the unchanged
static same-stream authority. Version 2 reuses complete stage Schedules and
single-assignment tensor bindings, and additionally owns three distinct
public INT32 scalar control inputs, two ordered CTA classes, one queue per
stage, one release/acquire device handoff per private intermediate, and one
steal window. The controls count as Program-consumed inputs without a dummy
math stage. Queue and handoff order are canonical. This bounded form is an
internal admission step: `Compiler.lower_program`, existing Program rewrites
and ordered Evaluation all refuse version 2 until one complete native worker
path exists. It claims no device liveness or correctness from type
construction alone.

## Admission and liveness obligations

1. The Program's leaf Schedules and bindings are complete and share the exact
   Target. Queue claims bind one stage and one bounded tile domain; the old
   counter value is a tile coordinate only when it is in that domain. Every
   output tile has one owning claim path, including when the compute queue is
   shared by regular and stealing workers.
2. The verifier proves unique flag and payload ownership, matched
   release/acquire scope, dominance of acquire before any cross-CTA payload
   read, and an acyclic phase-dependency graph. A relaxed atomic may choose a
   work index; it never serves as publication evidence.
3. The chosen `c`, `K` and steal limit are device-resident runtime values with
   declared bounds. Invalid values produce an explicit failure status rather
   than a silent clamp or another Schedule. Chunk extents and tail ownership
   cover every tile exactly once.
4. The exact Target declares cooperative-grid support, and the compiled
   kernel's actual active CTAs per SM meet the requested residency at launch
   (ADR 0082). The plan's waits are bounded by a producer phase whose worker
   class is nonempty and whose queued work cannot be held solely by waiters.
   This is a liveness condition, not an inference from a quiet GPU or the
   upper-bound resource estimator. In particular, a cooperative grid-stride
   Schedule may still deadlock when its first resident work tile waits for a
   flag produced only by a later tile on a blocked CTA. The finite
   `grid_stride_publish_wait` counterexample in
   `experiments/weave/scheduler_model.py` pins two CTAs publishing tiles 0/1
   and each waiting on tile 3/2: all CTAs are resident and neither can reach
   its next tile. Unknown runtime peers require a stronger worker queue
   progress proof or a refusal, not an optimistic static pass.
5. Queue counters and readiness flags begin in the state named by the
   Workload/Executor reset contract on **every** launch. Timing includes the
   declared reset interval; a cost estimate only selects candidates before
   device time and never decides acceptance.

## P1–P8 and counterexamples

P1/P3 keep NumPy/PyTorch-like leaf arithmetic and one canonical Program;
authors state only performance-relevant CTA and queue commitments. P2 exposes
`c`, `K`, phase order, handoffs and cooperative residency. P4 checks typed
INT32 counters, payload bindings, ranges and memory order before emission.
P5 provides queue ownership, phase and liveness analyses. P6 requires the
kernel matrix, adversarial plans, external oracle and B300 profiler evidence.
P7 lands syntax, effects, verifier and lowering together. P8 maps the handoff
to PTX release/acquire and the grid to the qualified cooperative CUDA launch,
without a layout algebra or inherited hardware constants.

Reject a zero-worker class, duplicate queue ownership, an out-of-range claim,
a payload read preceded only by `depends_on`, a relaxed flag publication,
multiple flag producers, a cyclic phase graph, an unbounded steal window, a
compiled occupancy shortfall, a future-iteration wait cycle, and a missing
reset. A local HBM copy is not an
admitted substitute for inter-GPU dispatch: a later EP Program needs an
explicit remote-transfer effect, completion and per-rank oracle before it can
support the paper's claim.
