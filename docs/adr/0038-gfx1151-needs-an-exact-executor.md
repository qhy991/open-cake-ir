# ADR 0038: gfx1151 needs an exact Executor

Status: proposed; schema/runner implementation is prepared, while the exact infplane
release and formal search contract are pending.

## Context

Compiler Revision v29 can describe and lower an exact gfx1151 Schedule, but that is only
the Compiler half of a formal experiment. The current Executor descriptor is
`open-cake-ir-b200-v30`. Its host closure pins a CUDA PyTorch build, CUPTI Python,
FlashInfer and Nsight Compute, and `ExecutorRevision.admit_host()` admits that B200
environment.

The AMD search runner currently loads v30 to verify repository source bytes and then
discards its host authority. The live path calls the HIP-specific runtime admission
directly and discovers `amd-smi` and a profiler through `PATH`. This can prove that the
source closure matches v30 or that the live device looks like gfx1151, but it cannot
prove that one released Executor owns both facts. Naming the B200 descriptor in a frozen
AMD search contract would therefore silently ignore half of the authority it claims to
use.

The one-row preflight also exposed an independent Executor bug: the quickstart accepted a
custom Schedule path but compared its entry point with the default r64-w4 name. The
correct repair makes the Workload own the operator adapter and the Schedule own its
entry point. That source repair intentionally invalidates the current content-bound v30
descriptor until its own B200 release cycle is rerun on a qualified B200 host.

## Decision

1. Keep Executor schema v1 and every B200 descriptor backward compatible. A B200
   descriptor never authorizes an AMD GPU run merely because its source list includes an
   AMD runner.
2. Add architecture-neutral Executor schema v2 and an independent
   `open-cake-ir-gfx1151-vN` ordinal lineage. The first released instance must be created
   on the admitted infplane ROCm environment; it is never synthesized from the B200 host
   document.
3. The gfx1151 Executor owns the exact repository runtime-source closure, Python
   executable bytes and version, required package versions, the PyTorch HIP runtime
   version, and the byte identity of every external command the runner executes.
   `amd-smi` is mandatory because it guards exclusive-process admission. AMD profiler
   tools are optional capabilities, but an absent admitted profiler blocks promotion.
4. The Compiler Target remains the sole owner of gfx1151 instruction, memory and
   execution-group capabilities. The Executor references the Target identity but does
   not copy those capabilities. Runtime admission compares the generated lowering's
   exact Target with the visible HIP backend, gfx architecture and wave width. Evidence
   records the observed device and command outputs; it does not redefine either
   authority.
5. The AMD search runner carries the validated Executor into its live path, admits that
   host before any GPU submission, and invokes only tool paths returned by that
   admission. A `PATH` discovery can be reported during an exploratory preflight but
   cannot enter formal search evidence.
6. A frozen one-row search contract may be created only after both Compiler v29 and the
   exact gfx1151 Executor are released. Compiler approval, Executor release and GPU
   execution remain separate boundaries; none implies another.
7. The quickstart repair uses the default frozen Workload's `operator` as the adapter
   boundary and leaves `Schedule.lowering.entry_point` as the route owner. A custom
   Workload revision is allowed only for the same operator. This removes both the
   hard-coded default entry point and a duplicate operator string.
8. Executor witness discovery is shared by the B200 and gfx1151 release cycles. Frozen
   Studies, calibrations, historical Campaign Locks, sealed Evidence archives and
   digest-bound inventory observations are witnesses; derived inventories and working
   descriptors are not. An external Campaign Lock must have a repository-owned immutable
   registration before dispatch because a release cannot discover external files.
9. Both release cycles build a hidden candidate, perform live host admission, recheck
   witness state and the initial working-descriptor bytes, reload the candidate and its
   source closure, and only then replace atomically. The schema-v2 low-level writer may
   assemble only a hidden candidate; it cannot install a formal gfx1151 release directly.

## Acceptance evidence

- Schema v1 B200 descriptors retain their exact load and host-admission behavior.
- Schema v2 rejects missing or extra host facts, changed Python/package/tool bytes,
  CUDA PyTorch, a non-HIP Triton target, non-gfx1151 architecture and non-wave32
  execution.
- A runner test proves that formal process and profiler probes use Executor-admitted
  paths rather than `PATH`; another proves that a B200 Executor cannot authorize AMD
  execution.
- AMD quickstart tests prove r64-w4, r1-w8 and SwiGLU preparation, Schedule-owned entry
  points, same-operator Workload substitution and cross-operator refusal.
- Live quickstarts require a released Compiler with a passing Corpus Gate, the exact
  Executor, a clean tree and a new external artifact root. Admission or runtime failure
  retains an authority-bound failure receipt rather than disappearing.
- Candidate runtime faults terminate the whole search. Only an observed numerical
  mismatch becomes `CORRECTNESS_REJECTED`; an unclassified launch, HIP or oracle fault is
  never relabeled as a candidate compile result and cannot lead to a timing WIN.
- The exact infplane release receipt records the new descriptor identity and source
  closure. Only after Compiler approval do two-case SwiGLU and RMSNorm correctness runs,
  followed by the frozen four-candidate one-row search, become authorized.

## Consequences

- Current draft lowering and prepare-only results remain useful implementation evidence,
  but they are not formal GPU evidence.
- The B200 Executor release and gfx1151 Executor release advance independently even when
  they bind overlapping source files.
- A missing AMD profiler may still permit correctness and retained leaf timing, but it
  leaves `promotion_authorized=false` and supports no llama.cpp or serving claim.
- Q4_0/Q8_1 MMVQ work does not start as a promoted optimization path until this custody
  boundary and the P0 RMSNorm decision are settled.
