# ADR 0075: Runtime index values and ordered Cake Schedule launches

Status: accepted implementation scope; compiler and device gates pending.

The owner explicitly requires the remaining FlashInfer GQA, MLA and FP8 MoE tasks
through Cake IR, including compiler extensions, rather than handwritten native kernels.

## Decision

Expose program/loop/range coordinates as INT32 values. Add exact INT32 arithmetic,
typed comparison and selection, and floating log2. A scalar-buffer AccessMap coordinate
removes its indexed dimension; existing BUFFER coordinates retain their zipped vector
domain. Both require dominating register INT32 inputs and masked bounds. Indirect stores
remain refused except for their existing proved reservation ownership.

These are reusable operations, not attention or MoE opcodes. Attention composes runtime
page lookup, contractions, masks, stable normalization and base-2 LSE. MoE composes
sigmoid, group top-two sums, group/expert selection, unbiased normalized routing weights,
explicit FP8 scaling, GEMM, SwiGLU and deterministic weighted combination. Every stage is
a complete Schedule assessed and lowered by the same clean Compiler commit.

Following ADR 0038, Evaluation owns an immutable ordered launch plan: exact target,
public ABI, intermediate shapes/dtypes, producers/consumers and stage bindings. It
allocates buffers and launches kernels on one ordered stream. It performs no routing,
dequantization or other task mathematics. Public inputs are read-only and each global
intermediate has one producer. Whole-plan correctness and later timing include all stages.
There is no task-specific compiler branch or opaque whole-attention/MoE operation.

## P1–P8 review

- P1: coordinate, comparison, selection and arithmetic have ordinary tensor meanings.
- P2: tile choices, masks, stage launches and intermediate storage remain explicit.
- P3: existing arithmetic/AccessMap/selection owners are extended, not shadowed by a DSL.
- P4: dtype, arity, scalar/vector shape and launch bindings are checked before emission.
- P5: runtime addressing and unknown arithmetic work are reported, never guessed.
- P6: new accepted/refused primitive and composition cases accompany full Corpus checks.
- P7: schema, parser, typing, safety, coverage and lowering change together.
- P8: integer arithmetic preserves signed INT32 semantics; log2 uses declared backend
  floating behavior; inter-stage visibility follows ordered same-stream kernel completion.

Only implemented backend paths admit the new operations. Unsupported paths refuse them
by name. Targets are exact; no precision conversion or architecture step-down is implicit.
No layout algebra, automatic acceptance or expectation refresh is introduced.

## Verification requirements

Counterexamples cover scalar/vector confusion, integer-vs-float addresses, missing
index dependencies, unsafe writes, stale intermediates and mismatched stage bindings.
Attention checks include runtime page permutations, empty/masked rows, causal alignment,
nondefault scale and both outputs. MoE checks include local/nonlocal experts, nonzero
local offset, nonunit routing factor, scale blocks and distinct activation halves.
Top-k ties use the existing lowest-index contract and are explicitly distinguished from
an upstream reference whose tie behavior is unspecified. Every device claim names its
fixed-shape/input domain and retained evidence; incomplete gates remain reported.
