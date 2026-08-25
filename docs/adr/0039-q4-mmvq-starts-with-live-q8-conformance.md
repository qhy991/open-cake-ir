# ADR 0039: Q4 MMVQ starts with live-Q8 block conformance

Status: partially implemented; the dependency-free Workload/oracle and generated Q8_1
producer are implemented, while external Compiler approval, the Q4×Q8 consumer and
gfx1151 execution are pending.

## Context

The next high-impact llama.cpp decode direction after one-row RMSNorm is Q4_0 × Q8_1
MMVQ. The current audited upstream authority is
`ggml-org/llama.cpp@1729ed5371cd1ac6f6d6f3226f8803b080042839`. Its Q4/Q8,
quantizer, integer-dot and MMVQ source bytes are unchanged from the prior `eb25b726`
audit.

The native boundary is not a matvec over caller-provided Q8 data. It starts with an FP32
activation, materializes Q8_1 on device, then consumes that block with packed Q4_0
weights. Q4_0 is an 18-byte heterogeneous record containing an FP16 scale and 16 nibble
bytes. Q8_1 is a 36-byte record containing FP16 d/s and 32 signed bytes. The GPU producer
uses FP32 `d=amax/127` to make q, but stores `fp16(d)` and
`fp16(wave_reduce_sum(original_x))`. The dot correction must consume that stored sum; it
cannot replace it with `stored_d * sum(q)`.

Q4_0 records require 2-byte alignment. Q8_1 requires 4-byte alignment because its d/s
union contains `half2` (`uint32_t` in the C declaration); the 36-byte record stride
preserves that alignment.

gfx1151 uses RDNA3 dot4 instruction semantics but inherits the RDNA2 MMVQ parameter
table, which gives one wave and one output row per block. Neighboring RDNA3.0 and RDNA4
tables make eight waves a testable future hypothesis, not a conclusion.

The SubCUDA evidence library supplied three transfer constraints:

- R47 accepted direct quant ownership only after exact payload/scale and no-profiler E2E
  evidence;
- R48 rejected a semantically exact fusion whose real incumbent comparison projected
  only 0.0239% graph saving;
- the Llama FP8 QKV case reduced bytes and profiler duration but lost matched operator
  time, so unpack/conversion cost must remain inside admission timing.

The R47/R48 structured replays remain blocked locally because their exact owning TP2 Git
objects are unavailable. The Llama QKV hash-gated replay passes, but it is a structured
source-report transcription rather than raw AMD evidence.

## Decision

1. Introduce the frozen Workload
   `llama-q4_0-q8_1-mmvq-f32-conformance-v1` before widening Compiler IR. It covers one
   `N=1,M=1,K=32` block and owns seven deterministic cases covering random data,
   stored-s correction, zero Q8/Q4 scales, exact half-away `roundf` ties, XOR-tree sum
   order and two-part FP32 placement.
2. Public Workload inputs are exactly one raw 18-byte Q4_0 block and one 32-element FP32
   activation. The native wrapper pads K to 512, so the named intermediate is a 576-byte
   workspace: one consumed Q8_1 record followed by 15 all-zero padding records. It is
   forbidden as a public input.
3. The independent stdlib oracle preserves source operation order, wave32 XOR-tree sum,
   `roundf` halfway-away semantics, FP16 d/s storage, low-nibble then high-half logical
   order, signed Q8 bytes, two 16-value partials, `-4*s8` per partial and FP32 partial
   combination. Q8 bytes are exact; final FP32 remains tolerant because the future HIP
   compiler may contract floating expressions.
4. Q8 workspace correctness is byte-exact. The final FP32 scalar uses a declared
   tolerance. The conformance contract fixes two required kernel launches, zero fallback,
   no performance measurement and no promotion, llama.cpp-build or E2E claim.
5. One Schedule remains one kernel. The first executable successor must therefore contain
   a Q8 producer Schedule and a Q4×Q8 consumer Schedule; Evaluation owns their ordering,
   shared device intermediate, two launch receipts and combined correctness result.
6. Q4_0 is not added as a scalar DType, existing FP8 `ScaleRelation` is not repurposed,
   existing floating MMA is not called dot4, and no opaque `q4_q8_block_dot` operation is
   introduced.
7. The producer uses regular UINT8/INT8 scalar storage, closed Q4_0/Q8_1 packed-record
   relations, a register-only reshape, abs, divide-no-NaN, explicit rounding/cast and an
   FP32 wave32 XOR-tree reduction. The zero case is expressed by divide-no-NaN rather
   than adding compare/select vocabulary. The future consumer may add only the nibble
   extraction/packing, integer reduction and four-byte integer-dot primitives its first
   executable slice proves necessary, with a required Target instruction contract.
   Correction remains visible composition.
8. The gfx1151 Target may admit only the exact no-clamp dot4 instruction. The Target does
   not own llama.cpp's RDNA2 scheduling table; waves, row ownership, K-loop and partial
   placement remain Schedule facts.

## Acceptance evidence

- Workload loading rejects caller Q8, idealized stored-s, adjacent-nibble order and any
  performance claim.
- Materialization produces an exact 18-byte Q4 input and complete 576-byte padded Q8
  reference workspace without Torch, Triton or GPU dependencies.
- Tests distinguish low nibble positions `0..15` from high nibble positions `16..31`,
  distinguish stored `half(sum(x))` from `half(half(d)*sum(q))`, require zero blocks to be
  all-zero bytes, and reject a one-byte Q8 mutation.
- The producer Schedule passes parser, Verifier and Triton preflight with only the
  non-blocking gfx1151 occupancy report. Its generated source spells the five XOR stages,
  exact FP32 division, half-away rounding, FP16 nearest-even storage and five
  relation-derived little-endian stores; no `tl.arange(0,36)` is emitted.
- Compiler v29's proposed 45-case Gate adds one producer positive and one isolated
  record-count negative with zero expectation drift in the previous 43 cases. It remains
  a draft until an independent reviewer binds the exact Gate; this agent does not write
  that approval.
- No GPU experiment is claimed until the gfx1151 Executor is released and the producer
  produces all 576 reference bytes for every frozen case on infplane. No wave or consumer
  optimization begins until the complete two-kernel K=32 path is correct.

## Stop conditions

Stop this mechanism rather than widening it if the first executable slice requires
caller-provided Q8, a CPU producer, dequantize-to-FP32 fallback, operator-name dispatch,
hidden correction, or an unqualified scalarized substitute for the declared dot4. Do not
begin wave or row search before K=32 two-kernel correctness passes. A later operator win
must time live quantization plus the complete consumer set; bytes or profiler improvement
alone cannot promote it.
