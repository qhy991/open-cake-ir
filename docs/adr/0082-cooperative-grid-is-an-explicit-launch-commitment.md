# ADR 0082: Cooperative grid is an explicit launch commitment

Status: proposed. Shared IR and verifier semantics are implemented in this
task branch; backend and device qualification belong to the NVIDIA successor.

## Problem and evidence

Weave's communication and computation CTA groups wait on data produced by
other CTAs inside one persistent kernel. `ProgramMap.persistent` currently
derives a bounded grid and reuses CTAs, but ordinary CUDA launch does not
authorize a grid-wide synchronization assumption. Its declared
`ctas_per_multiprocessor` is a resource commitment, not proof that the
compiled kernel can co-reside all launched CTAs.

On B300-M4, the direct native prototype at `863eb3fd` used
`cudaLaunchCooperativeKernel`, exact-device admission and per-tile PTX
release/acquire flags. Three synthetic plans completed with unique task
ownership, including an actual 256-tile steal and combine before all compute
finished. Evidence remains at
`B300-M4:/home/qinhaiyan/cake-weave-worker-b300-m4-863eb3fd/`.
This is a single-GPU mechanism result, not MoE or Cake correctness. The
[CUDA cooperative launch contract](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html#when-to-use-cudalaunchcooperativekernel)
requires support on the device and a grid within the compiled kernel's
resident-block capacity.

## Decision

`ProgramMap` gains an optional `cooperative` boolean, default false. True is
legal only with `persistent=true`. It requires the exact Target to declare
`cooperative_grid=true`; an absent declaration is **unmodeled**, a declared
false is **unsupported**, and neither is silently treated as true. The
Target field is optional and generic: vendors may declare the same semantic
capability through their own qualified launch mechanism.

The lowering backend that admits this commitment must use a cooperative
launch and check the compiled kernel's actual resident blocks per SM against
the requested `Residency.ctas_per_multiprocessor` before launch. It refuses a
shortfall; it never reduces the grid or changes the Schedule's role count.
The generated launch and metadata record the commitment. A backend without
such a launch path refuses it by name.

This field asserts a **launch property**, not a synchronization operation.
It does not make a relaxed atomic publish a payload, introduce a cross-CTA
barrier, assign CTAs to exact SM ids, or prove a waiting phase terminates.
Those effects need their own typing, memory-order and liveness rules before
the Weave worker plan can be a Cake Schedule.

## P1–P8 check

- P1/P3: reuse the existing persistent `ProgramMap` and derived grid; no
  destination-passing or second launch count is added.
- P2: cooperative launch is a visible author commitment, not a backend guess.
- P4: construction refuses cooperative without persistence; the verifier
  refuses a Target that lacks the exact capability.
- P5: resource analysis retains the declared CTA budget; backend checks
  compiled occupancy at the execution boundary, where that fact exists.
- P6: accepted and refused contract probes plus the full Corpus Gate gate the
  change; B300 device correctness is a separate successor observation.
- P7: schema, parser, Target, verifier and every affected backend admission
  evolve together. A parser-only field would silently drop the promise.
- P8: NVIDIA's cooperative launch API and B300-M4 attribute observation are
  the concrete hardware behavior; other targets inherit neither support nor
  occupancy numbers.

## Counterexamples and boundary

Reject `cooperative=true` on a nonpersistent map, a Target with absent or
false support, a backend that launches ordinarily, and a compiled kernel
whose actual occupancy is below the requested CTA residency. Exact target
matching and the existing no-layout-algebra rule continue to apply. The
next primitive must separately state CTA worker classes, device-resident
runtime split values, queue ownership, release/acquire handoffs and bounded
progress. A successful source emission or cost estimate cannot accept it.
