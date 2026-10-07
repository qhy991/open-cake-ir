# DCU output ownership and pure pointwise width qualification

The successor refuses a retained partial-reduction Schedule before device execution and lets the existing width action transform two previously refused, otherwise lowerable PReLU schedules. Five accepted emissions pass all25 original Task cases on gfx938. This is a safety/applicability qualification; it makes no latency, community-baseline or equal-budget search claim.

The shared implementation is `4ef5d547f893a2e2ef531fcede703e0e8413b6b9` ([PR372](https://github.com/qhy991/open-cake-ir/pull/372)); the original Compiler, inputs, oracle and comparator remain `d0d0acabf1ac1710745152efd88f5f82e221ac49`. Findings F-2026-10-07-001/002 retain the failure origins. No frozen search was changed.

## Mechanisms

- Shared program safety rejects direct non-loop, non-persistent ordinary OUTPUT stores that omit a varying program axis. Singleton axes, input-anchored coordinates and repeated uses remain valid. STATE, indirect access and loop policies stay with their existing owners. Metal/native non-loop duplicate checks were moved to the shared owner; native loop ownership remains local.
- `specialize_triton_warps` now covers typed pure COMPARE/SELECT and one fixed independent pointwise loop. Arithmetic, accesses, ABI, target admission and cast boundaries are preserved. New schedules still require their original oracle gate. The pass does not select a default width.

## Device replay

Evidence root: `/data3/testuser01/experiments/bw1100-ownership-width-qualification-20261007/run`.

| Retained source / transformation | Cases | Result |
|---|---:|---|
| PReLU `c3-select`, original1 group | 5 | passed, max absolute error0 |
| PReLU `c3-select`, explicit width2 | 5 | passed, max absolute error0 |
| PReLU `c6-loopcols-pipe`, original1 group | 5 | passed, max absolute error0 |
| PReLU `c6-loopcols-pipe`, explicit width2 | 5 | passed, max absolute error0 |
| Moments `coltile16-w1`, positive control | 5 | passed, max absolute error1.1920928955078125e-7 |

The original `primary`, `zeros`, `near_zero`, `alternating`, and `mixed_magnitude` cases,128×1024 shape, dtypes and tolerances were retained. Device execution checks output shape/dtype/device/contiguity/nonaliasing. The original canonical comparator checks all output elements and unchanged inputs after the device worker exits.

The original width action returns `operation_domain` and `loop_domain` respectively. The successor applies both changes while retaining the operation graph, access maps and emitted kernel AST. The negative moments Schedule is rejected at its two stores with `OUTPUT_STORE_PROGRAM_AXIS_COLLISION`; it is not in the device manifest.

## Source and custody locators

- `kit/qualification.py`: separate CPU prepare, admitted device observations, CPU compare.
- `kit/PROVENANCE.json` and `kit/fixtures/`: retained original sources and Workloads.
- `campaign/successor-qualification/old-preparation.json`: original API refusals and oracle owner.
- `campaign/successor-qualification/manifest.json`, `emissions/`: successor and exact emitted schedules/sources.
- `campaign/successor-qualification/negative-assessment.json`: CPU-only safety rejection.
- `campaign/successor-qualification/summary.json`:25/25 canonical acceptance, no timing.
- `campaign/successor-qualification/admission.json` and `admission-terminal.json`: job `bw-e7baa9f3f06f`, HCU5, exit0, after_vram0%, no KFD visibility or running container.

The existing per-user gateway remains `574a6a147eea6998a52c2e102244797dd1705e68`; it does not prove physical exclusivity against unrelated users. DTK image identity is retained by the gateway receipt. Only the positive sources reached the device, under one600-second request with at most300 seconds of queue waiting. CPU preparation and oracle comparison held no GPU allocation.

## Promotion scope

Promote the reviewed ownership gate and explicit action applicability. Do not promote a default width, approximate math, persistent scheduling, a calibrated cost model, or performance claims. The original55-task batch remains pinned to its original Compiler. New compiler-versus-compiler search or independent Bench comparisons require successor Runs with their own fixed budget and confirmation.
