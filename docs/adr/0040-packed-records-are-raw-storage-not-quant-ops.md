# ADR 0040: packed records are raw storage, not quantized operations

Status: implemented in the Compiler v29 candidate; external Compiler approval and every
GPU observation remain pending. ADR 0041 builds typed Q8_1 production on this raw ABI.

## Context

ADR 0039 freezes the first Q4_0 × live-Q8_1 Workload semantics, but the Compiler could
not name either GGML record mechanically. Q4_0 is not a regular four-bit scalar dtype:
one logical block is an 18-byte record containing FP16 d and 16 packed bytes. Q8_1 is a
36-byte record containing an FP16 d/s union and 32 signed bytes. Reusing the FP8
`ScaleRelation`, floating MMA or an opaque block-dot would assign several unrelated facts
to the wrong owner.

The first Compiler successor needs to establish byte custody without claiming it can
decode, quantize or execute MMVQ. Triton also requires every emitted `tl.arange` extent to
be a power of two, while GGML records are 18 and 36 bytes. A legal raw-copy Schedule must
therefore use a 32/64-element program tile and an explicit tail mask; a direct
`tl.arange(0,36)` must be rejected before lowering.

## Decision

1. Add ordinary one-byte scalar storage types `uint8` and `int8`. They are available to
   generic raw load/store and signatures. Q4_0 itself is not a DType.
2. Add the closed `PackedBlockFormat` vocabulary:
   - `ggml_q4_0_v1`: logical extent 32, record size 18, alignment 2, FP16 d at byte 0,
     UINT8[16] at byte 2, physical low/high nibble mapping `(j,16+j)`;
   - `ggml_q8_1_v1`: logical extent 32, record size 36, alignment 4, FP16 d/s at bytes
     0/2 and INT8[32] at byte 4.
3. A Buffer may declare `packed_block:{format,record_axis}`. The record axis is the
   contiguous last physical dimension and must have the exact record byte extent. The
   whole record uses raw UINT8 storage, one stage, correct base alignment and no FP8
   `scale_of` relation.
4. The immutable IR registry is the single owner of record fields and nibble mapping.
   The generated JSON schema projects its format enum; the Verifier owns dtype, axis,
   extent, stages, alignment and relation-conflict findings.
5. Triton and CuTe dtype tables may name UINT8/INT8 scalar storage. Triton can lower
   generic identity raw copies. CuTe currently refuses every packed relation with
   `CUTE_PACKED_BLOCK_UNSUPPORTED`; naming a byte dtype is not packed decode support.
6. Triton preflight reports `TRITON_ARANGE_EXTENT_UNSUPPORTED` only for actual emitted
   non-power-of-two ProgramTile, Dimension or TileLoop ranges. It is a
   lowering-blocking backend precondition, not IR rejection.
7. Add three positive Corpus cases: Q4 UINT8 record copy through a 32-byte tile, Q8
   UINT8 record copy through a 64-byte tile, and plain INT8 identity copy. Add one
   accepted-but-unlowerable Q8 direct-dimension case with two exact arange findings.
8. Generated positive source must contain only raw address, load, mask and store logic.
   The absence of decode, nibble, quantize, dot4, correction and MMVQ markers is part of
   the contract.

## Acceptance evidence

- Parser/schema round-trip both exact formats and reject relation extra fields.
- Verifier negatives cover Q4 extents 17/19, Q8 extents 35/37, INT8 pretending to be raw
  record storage, unresolved/non-last record axis, multistage storage, Q4 2-byte and Q8
  4-byte alignment, and packed/scale conflicts.
- Triton positive source has `*u8` or `*i8`, power-of-two arange and source/destination
  tail masks. The direct 36-byte Dimension variant remains accepted but unlowerable.
- The prior 39 Corpus expectations do not change. Four independently reviewed
  expectations are appended, producing a 43/43, 60-source draft Gate.
- The new cases prove deterministic source lowering only. No Triton compile, HSACO,
  gfx1151 raw-copy correctness, field decode, live quantization, integer dot, MMVQ or
  performance evidence is claimed.

## Consequences

ADR 0041 uses this registry to encode typed Q8_1 fields without weakening the raw-copy
contract; typed Q4, packed decode and integer dot4 remain absent. Wave search remains
forbidden until two generated K=32 kernels produce the full 576-byte Q8 workspace and
tolerant FP32 output on the exact gfx1151 Executor.
