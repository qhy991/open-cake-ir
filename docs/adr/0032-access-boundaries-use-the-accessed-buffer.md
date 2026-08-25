# ADR 0032: access boundaries use the accessed Buffer

Status: accepted, 2026-08-25.

## Outcome and non-goals

Lower every `mask_tiled_axes` coordinate against the extent of the global Buffer being
accessed. A ProgramAxis or TileLoop owns work decomposition and tile width; it does not
own the extent of another Buffer that happens to reuse its coordinate. This repairs the
v24 GEMM shape-drift lowering, whose 128-element bias was masked against the 256-element
matrix axis and could therefore read out of bounds.

This change adds no IR primitive, Workload profile, layout algebra, performance claim or
unchecked boundary mode. It does not add a TinyGEMM2 host ABI.

## Invariants and ownership

- `AccessMap.indices[position]` and the accessed Buffer's `shape[position]` are the sole
  authority for that memory bound.
- ProgramAxis and TileLoop extents still own launch and iteration counts. Reusing their
  coordinate does not alias their source Buffer's extent into another access.
- `mask_tiled_axes` keeps its existing zero-fill load semantics. A shorter bias is a
  legal zero-extended Schedule, not an implicit Workload rule.
- Current diagnostic oracles implement those declared semantics independently. Numeric
  tolerance remains an observation-protocol input, not a Compiler fact.
- Frozen v24 observations and source bytes remain unchanged. The repair is a Compiler
  successor because v24 was observed on device before the defect was fixed.

## Smallest complete slice and evidence

The vertical slice is the retained GEMM shape-drift Schedule: `n_block` is derived from
the 256-row B matrix, while its bias access must emit `n_block_offsets < D_BIAS_0` where
`D_BIAS_0=128`. The independent oracle pads the missing bias suffix with zero. A focused
source test proves the mask owner, a CPU test covers batched tie-aware result auditing,
the full Corpus Gate detects every changed lowering digest, and a successor brokered B200
run must compile, launch and compare both GEMM rows. Failure remains evidence and does not
authorize an expectation or tolerance change.
