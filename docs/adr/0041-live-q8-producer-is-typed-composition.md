# ADR 0041: live Q8 producer is typed composition

Status: proposed for Compiler Revision `open-cake-ir-v29`; the 45-case draft Gate is
prepared locally, while independent Compiler approval and gfx1151 execution are pending.

## Context

ADR 0039 freezes `llama.cpp@0a5ac49bce4c42893368585edf3ffb41a38f0108`
Q4_0 × live-Q8_1 semantics. ADR 0040 gives the Compiler raw packed-record custody but
deliberately cannot produce typed Q8_1 fields. The first executable slice must quantize
one FP32 block and its 480 zero-padding values into sixteen exact 36-byte records without
adding an operator-named quantize primitive, a caller-provided Q8 input or MMVQ claims.

The source-visible facts that affect bytes are the 32-lane reduction order, FP32 scale
used by quantization, halfway rounding, FP16 storage rounding, zero-block behavior and
little-endian field layout. Leaving any of them to an unspecified backend default would
make a successful lowering weaker than the Workload contract.

## Decision

1. Add only the reusable primitives the producer needs: register-only `reshape`,
   elementwise `abs`, `divide_no_nan`, `round` and `cast`, plus reduction algorithm
   `xor_tree_32`. Do not add an opaque Q8 quantize operation or compare/select vocabulary.
2. `round` accepts only explicit `nearest_away_from_zero`. `cast` names rounding and
   overflow while its unique write Buffer owns the target dtype. The first subset is
   FP32→FP16 `nearest_even + ieee` and FP32→INT8 `toward_zero + forbid`.
3. `xor_tree_32` accepts only FP32 input/output, a contiguous last axis of extent 32,
   CTA scope and no TileLoop placement. Triton spells the 16/8/4/2/1 pair tree with
   order-preserving reshape, permute, split and explicit add/max operations; it does not
   delegate source order to `tl.sum` or `tl.max`.
4. The packed-record registry also owns little-endian order. A one-read packed Store
   remains a raw UINT8 identity copy. The typed Q8_1 Store reads register FP16 `d[P]`,
   register FP16 `s[P]`, register INT8 `qs[P,32]` in registry order and writes UINT8
   `[P,36]` through one `DIMENSION(0)` record-prefix AccessMap. Typed Q4 remains rejected.
5. The producer pads the 32-element activation to 512 in one masked load, reshapes it to
   `[16,32]`, computes `d=amax/127` and `q=x/d` with precise FP32 `div_rn`, applies
   half-away rounding, and writes all 576 bytes. `divide_no_nan` makes every padding
   record all zero without a second control-flow model.
6. The gfx1151 Target admits the new `reshape` OperationKind. All other additions remain
   parameters of already admitted elementwise or reduce kinds. The sm_100a Target is not
   widened by this AMD slice.
7. The generated program uses one gfx1151 workgroup with eight wave32 groups. This is a
   correctness candidate, not a claim that it reproduces llama.cpp's CTA geometry or is
   faster. Wave search and MMVQ consumer work remain separate successors.

## Acceptance evidence

- Parser/schema tests reject implicit or irrelevant rounding/overflow fields and retain
  historical defaults without changing old Schedule canonical bytes.
- Verifier negatives localize reshape, cast, XOR-tree, typed-field dtype/space/shape,
  AccessMap, Q4 typed-encode and record-count failures. The record-count negative changes
  only the output record prefix and receives `PACKED_BLOCK_STORE_RECORD_COUNT`.
- Generated-source tests require the five XOR stages, two `tl.div_rn` sites, explicit
  half-away rounding, FP16 RTNE casts, four little-endian FP16 byte writes and one
  32-byte INT8 payload write. `tl.arange(0,36)`, generic reduction and fallback markers
  are absent.
- The 45-case expectation review adds exactly one accepted/lowerable producer and one
  rejected record-count neighbor. The previous 43 cases retain identical findings,
  dispositions, Schedule digests and lowering-source digests.
- This evidence proves deterministic parsing, verification and source lowering only.
  It does not prove Triton compilation, HSACO generation, GPU bytes, performance,
  llama.cpp integration or the Q4×Q8 consumer.

## Next gate

After independent Compiler approval and an exact released gfx1151 Executor exist, compile
and launch the producer on infplane for every frozen Workload case through
`examples/gpu/llama_q8_1_amd_quickstart.py`. Retain source, IR, AMDGCN, HSACO, runtime
identity, launch count and both observed/reference 576-byte workspaces. A single byte
mismatch stops the path before ADR 0042's consumer, dot4 or wave tuning.
