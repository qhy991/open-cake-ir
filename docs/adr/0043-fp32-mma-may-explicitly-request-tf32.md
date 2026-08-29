# ADR 0043: FP32 MMA may explicitly request TF32 input precision

## Status

Proposed. This records the generic instruction contract and Target capability needed for
implementation review. It does not authorize a Compiler Revision, Corpus expectation
adoption, provider run, GPU campaign, or Candidate promotion. Release remains behind the
external repository-owner approval gate.

## Irreducible goal

Let an FP32 `mma` explicitly request Triton's TF32 tensor-core input precision while
retaining FP32 Buffers and FP32 accumulation. The first motivating measurement is a
score/top-k Program whose FP32 IEEE contraction remains material, but the contract is an
SM100a instruction choice rather than a QSA operation or Workload branch.

## Existing vocabulary and missing fact

`mma` already owns arithmetic instruction selection through a Target instruction
contract. `triton.dot.fp32_ieee` expresses FP32 inputs whose dot requests IEEE input
precision, and `triton.dot.bf16_fp32` expresses BF16 inputs with FP32 accumulation.
Neither can represent FP32 storage with TF32 tensor-core input precision. Reusing the
BF16 contract would incorrectly change operand typing; relying on Triton's default would
hide a performance- and accuracy-relevant decision from the Schedule.

Triton's `tl.dot` API documents `"tf32"` as an NVIDIA `input_precision` option for
FP32-by-FP32 tensor-core execution. It also warns that FP32 inputs may be truncated to
TF32 without rounding. The Compiler therefore treats this as an explicit arithmetic
commitment whose numerical acceptance remains external-oracle evidence, not as an
equivalent spelling of IEEE FP32.

## Decision

- Add the canonical Target instruction contract `triton.dot.fp32_tf32`.
- The contract requires FP32 operands and an FP32 accumulator/result. It does not add a
  TF32 `DType`, change Buffer storage, or authorize an implicit cast.
- Triton lowering for this contract must explicitly request `input_precision="tf32"`;
  it must not depend on a backend default or silently fall back to IEEE, TF32x3, BF16, or
  another precision mode.
- `triton.dot.fp32_ieee` remains the canonical explicit IEEE choice. The two contracts
  are distinct hardware commitments under the existing `mma` operation, not separate
  operation kinds.
- Work and effect analysis remain those of the same matrix contraction. Precision must
  be visible in the instruction contract, while latency, throughput, numerical error,
  physical registers, and implicit shared memory remain compile- or device-evidence.
- No QSA-named Compiler branch, precision flag outside the instruction contract, TF32
  Buffer type, fallback path, or layout abstraction is introduced.

The Target declaration by itself does not make a Schedule using the new contract
lowering-eligible. Typing, verifier, profile, lowering, diagnostics, tests, Corpus
coverage, and the full Gate must arrive together in the implementation successor.

## P1-P8 and failure semantics

- P1/P3: an author chooses one existing `mma` instruction contract; there is no new
  operation, cast, dtype, or backend-default spelling.
- P2/P5: the precision choice is explicit and available to analysis, which continues to
  abstain from uncalibrated timing and numerical predictions.
- P4: the implementation successor must admit only FP32 operands with FP32 accumulation
  and reject incompatible dtypes or Targets before emission.
- P6/P7: verifier, profile, lowering, diagnostics, focused tests, and Corpus coverage
  must evolve together before a Compiler Gate is eligible for review.
- P8: the SM100a Target cites the authoritative `tl.dot` API; exact Triton compilation
  and B200 measurement remain required hardware evidence.

Existing instruction-unsupported and MMA dtype Findings remain the canonical failure
owners. A toolchain rejection is `unknown` infrastructure/toolchain evidence; an
external-oracle mismatch is an invalid Candidate; a correct but immaterial result is a
negative performance result.

## Acceptance evidence

1. Focused typing and verifier tests distinguish FP32-TF32 from FP32-IEEE and BF16-FP32,
   including incompatible operand, accumulator, Target, and backend cases.
2. Profile output preserves the exact instruction contract without inventing a latency,
   MFU, error, register, or shared-memory estimate.
3. SM100a lowering explicitly selects Triton `input_precision="tf32"`, and an exact
   toolchain compile proves the option is accepted rather than ignored or rewritten.
4. Generic accepted and deliberate-negative Corpus Schedules join a separately reviewed
   full Corpus Gate; a repository-owner approval binds that exact successor Gate before
   release.
5. Any Workload candidate first passes its unchanged external oracle and tolerance, then
   matched complete-Program timing. No component result establishes checkpoint or
   end-to-end serving performance.
