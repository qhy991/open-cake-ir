# ADR 0025: ragged grouped GEMM is primitive composition

Status: accepted in Compiler v20, 2026-08-25.

## Outcome and non-goals

Test whether the existing `ProgramMap`, tiled `mma`, `TileLoop` and
`Buffer.valid_extent` primitives compose into the arithmetic core of KDA MoE v1: one
batched BF16 GEMM per routed group, with padded rows masked by that group's runtime
length.  Success adds a profile and evidence, not a new IR concept.

This slice is not the complete KDA v1 program.  It does not describe top-k routing,
group formation, capacity selection, the second grouped GEMM, quantization, scatter,
persistent work acquisition, CLC, PDL, graph capture or a version portfolio.  Complete
KDA coverage therefore remains 0/57.

## Constraints and ownership

KDA v1 already combines an expert/group coordinate, a padded row coordinate, a K
reduction and a group-specific contraction.  Each fact already has one Cake owner:

| Fact | Canonical owner |
| --- | --- |
| group, M-tile and N-tile assignment | ordinary `ProgramMap` axes |
| group-specific A, B and C coordinates | `AccessMap` expressions |
| valid routed rows | A's `Buffer.valid_extent` relation to `lengths` |
| K traversal | one `TileLoop` |
| contraction and FP32 accumulation | the existing `mma` operation |

Adding `grouped_gemm`, `ragged_mma`, an expert descriptor or a second masking field
would duplicate those facts.  The profile may freeze one observation shape, but it is
not vocabulary and cannot be selected by a production Schedule.

## Smallest complete vertical slice

The smoke profile uses four groups and fixed padded tensors:

- `A[4,16,32]` BF16 with valid rows `lengths[4]`;
- `B[4,16,32]` BF16, interpreted as one `[N,K]` operand per group;
- `C[4,16,16]` FP32;
- one program per group and one 16x16 output tile, reducing K in two 16-wide steps.

Lengths include empty, partial and full groups.  A masked load contributes zeros for
invalid rows, and an ordinary dense store makes all output elements observable.  The
independent oracle first materializes masked A and then evaluates batched matrix
multiplication; it does not reproduce the generated loop.

One companion case changes B's group extent while leaving A, C and `ProgramMap`
unchanged.  The profile must report `PROFILE_SHAPE_MISMATCH` and block lowering.  The
same case revealed a generic safety invariant: a scalar program coordinate may index a
second Buffer dimension only when that dimension covers the complete program range.
Compiler v21 derives that range through `ProgramAxis.tile_count` and rejects the drift
with `ACCESS_PROGRAM_EXTENT_MISMATCH`; it does not invent equality between unrelated
leading dimensions or require unused extra rows to be absent.

## Failure, compatibility and rollback

- The positive composition needs no schema, parser, verifier or emitter special case;
  existing dense GEMMs and standalone valid-extent schedules retain identical generated
  bytes. The successor verifier change is the generic scalar-address safety rule above.
- Unsupported access composition must be reported as missing lowering coverage, not
  silently rewritten into a dense GEMM.
- Static verification proves the authored composition and the profile contract, not
  runtime correctness or performance.
- Failure to emit or match the B200 oracle rejects this proposal.  The failure point then
  becomes evidence for the smallest missing primitive; it is not repaired with a
  profile-specific backend branch.

## Acceptance evidence

1. the positive Schedule parses, verifies and lowers through only existing primitives;
2. emitted source visibly indexes A, B and C by the same group program id, masks A by
   `lengths[group]`, and performs the two-step FP32 contraction;
3. the group-extent drift case fails acceptance and lowering with reviewed profile and
   generic address-safety findings;
4. all 27 prior Corpus schedules and lowering digests remain unchanged;
5. brokered B200 execution matches every output element against the independent oracle;
6. the evidence explicitly makes no performance or complete-KDA claim.

The released Gate matched 29/29 cases over 45 Revision-bound sources. Dry-run review
showed that all prior 27 case expectations and lowering digests were unchanged. The
released generated source compiled and launched on a brokered NVIDIA B200 and matched
1,024/1,024 oracle elements with zero tolerance violations and maximum deviation
`1.9073486328125e-06` under the unchanged `1e-5` boundary. The immutable correctness-only
record is `inventory/RAGGED_GROUPED_GEMM_OBSERVATION_20260825.json`; no performance was
measured.

## References

- [CAKE: Compiler-Agent Co-Design for Frontier Kernel Evolution](https://arxiv.org/html/2608.12629v1#S2)
- local KDA MoE v1 grouped-GEMM `gidx_mapping` under
  `.judge/kernel_versions/moe/v-01`
- [`ADR 0020`](0020-kda-deltas-are-an-external-expressibility-corpus.md)
- [`ADR 0024`](0024-runtime-valid-extents-are-buffer-relations.md)
