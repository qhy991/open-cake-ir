# ADR 0045: Caller-indexed state uses unique indices and concrete strides

Status: proposed, 2026-08-29.

## Outcome and non-goals

Admit the smallest generic Schedule vocabulary needed for a caller to update recurrent
state selected by a runtime index without copying that state into dense storage:

1. a global Buffer may declare exact element strides when its storage is not dense;
2. a global INT32 input may relate its non-sentinel values to one state-buffer axis and
   declare those values unique;
3. an ordinary store may derive its active rows from that relation, with either
   `no_effect` or `write_zero` inactive semantics;
4. multiple stores to one state Buffer are legal only when their AccessMaps prove
   pairwise-disjoint subranges; and
5. `outer` multiplies two resident rank-one vectors into one rank-two FP32 tile.

This adds no KDA-named operation, layout algebra, generic boolean type, comparison or
select language, arbitrary uniqueness assertion, fallback, graph mode, stream edge,
program DAG, CUDA/PTX carrier, or serving claim. The first consumer is the complete
rank-local fused-decode contract, but every new word is independent of that workload.

## Why this is one successor

The released v38 Compiler can read runtime-indexed state but cannot write caller-owned
BF16 or FP32 state, cannot address the envelope slot pitch, cannot derive output-zero and
state-no-write from one padding fact, and cannot form the rank-one update of a delta-rule
state. Shipping any proper subset would leave the first real consumer unconstructable and
would require an approval cycle for a dead intermediate vocabulary. These five pieces are
therefore one minimal vertical slice.

The mathematical body needs no other word. Existing load, cast, elementwise, reduce,
MMA, store and loop composition express causal convolution, Q/K normalization, bounded
decay, beta, state matvec, recurrent output and gated RMSNorm. Runtime per-head floating
scalars use exact zero-stride input views. The relation-bearing cache slot is the one
exception: it is a rank-zero register scratch Buffer so indexed access does not invent a
length-one data axis. No scalar dtype or general scalar global ABI is added.

## Canonical declarations

### Concrete global strides

`Buffer.strides` is an optional tuple of element strides with the same rank as `shape`.
Omission is the sole dense spelling; an explicitly dense tuple is rejected. The first
subset applies only to global input or state Buffers. State strides are non-negative,
non-overlapping row-major-with-padding commitments; input strides may additionally use
zero to describe a read-only broadcast view. Negative strides, permutations and output
allocation layouts remain unsupported.

Shape owns the logical domain and work/traffic counts. Strides own only the address of
that logical domain and the exact host tensor contract. This is a concrete storage and
access commitment, not an agent-manipulated layout algebra.

### Unique runtime indices

`Buffer.unique_index` belongs only to one rank-one global INT32 input. It names a global
state Buffer, one axis of that Buffer, and one negative sentinel. Its precondition is:

- every element is either the sentinel or lies in the named axis;
- all non-sentinel elements are pairwise distinct.

The Workload/evaluator validates that precondition before formal measurement. The
Compiler verifies its type, domain and dataflow, but does not insert `torch.unique`, a
host synchronization or a second authority into the timed path. Loading the index into
a register preserves the relation.

### Guarded stores

`store.valid_if` names that relation-bearing register index. `store.inactive` is either:

- `no_effect`: the destination must be state and the inactive row performs no write; or
- `write_zero`: the destination must be output and the inactive row writes numerical
  zero.

Both fields appear together or not at all. The index relation is the sole owner of the
active predicate. No independently authored predicate Buffer exists.

A state store is admitted only when its destination AccessMap uses the exact `valid_if`
index for the related state axis and every remaining program/loop/vector coordinate is
injective. The relation replaces only the program axis that loaded it; every other
program axis and every enclosing tile-loop iterator must remain visible in the
destination AccessMap. This is a mechanical ownership proof, not a claim that arbitrary
runtime destinations happen to be unique.

Multiple operations may write one state Buffer only when all writers are stores and each
pair has a statically disjoint dimension subrange in its AccessMap. Otherwise the
existing `BUFFER_MULTIPLE_WRITERS` Finding remains.

### Outer product

`outer` reads two rank-one BF16, FP16 or FP32 register Buffers and writes their rank-two
FP32 product. The result shape is `[left_extent, right_extent]`; it performs one multiply
per result element. Triton lowers it as explicit singleton-axis multiplication. A shape,
dtype, space or placement mismatch fails before lowering.

## Lowering and host contract

The Triton SM100a route:

- uses declared strides in direct and runtime-indexed pointer expressions;
- checks exact host tensor shape, dtype, device and stride, retaining the existing
  contiguous requirement for Buffers without a stride declaration;
- combines state-store bounds with the relation-derived active mask;
- emits `tl.where(active, value, 0)` for `write_zero` before the ordinary output store;
- admits guarded state stores inside a tile loop only when verifier ownership includes
  that loop coordinate; and
- emits `left[:, None] * right[None, :]` in FP32 for `outer`.

The CuTe route has no body for these additions and must return localized backend
Findings. There is no fallback to another target or carrier.

## Analysis and cost model

Compulsory traffic remains the logical elements touched, never the padded storage span.
Runtime-indexed or guarded accesses remain upper bounds. The NCU-aligned static profile
lists exact non-dense/broadcast Buffer names and reports them as uncalibrated global
addressing/scoreboard risks. `outer` contributes exactly one counted multiply per output
element. No duration, bandwidth percentage, MFU or BWU is inferred.

## P1--P8 check

- **P1 Ergonomic:** shapes plus optional concrete strides follow tensor APIs; no layout
  language or destination-passing program is introduced.
- **P2 Performance-transparent:** state pitch, broadcast views, inactive-store policy,
  runtime index ownership and outer work are visible in the Schedule and source.
- **P3 Canonical:** dense strides have one spelling; the active predicate comes only
  from `unique_index`; rank-one products have one `outer` spelling.
- **P4 Statically type-checked:** dtype, rank, domain, sentinel, strides, store policy,
  ownership, disjointness and result shape are verifier obligations.
- **P5 Analysis-friendly:** address stride, effects, alias ownership, logical bytes and
  arithmetic work remain directly recoverable.
- **P6 Test-gated:** focused positive/negative contracts and a full Corpus Gate are
  required before a successor release.
- **P7 Analysis-consistent:** work and NCU-aligned profile projections change with the
  IR and lowering in the same commit.
- **P8 Hardware-grounded:** declarations map to Triton pointer arithmetic, masks,
  zero-selects and FP32 vector multiplication on exact SM100a.

## Corpus and diagnostics

The proposal requires two independent positive slices:

1. a padded-stride FP32 state gather/update using random non-contiguous unique indices,
   one sentinel row, output-zero, and post-state verification; and
2. a resident vector outer product.

Focused negative mutations cover dense stride restatement, rank/stride mismatch,
overlapping state storage, wrong unique-index dtype/domain/sentinel, relation loss across
load, unproven state ownership, missing inactive policy, output no-effect, state
write-zero, omitted program or loop coordinates, overlapping state writers, and outer
shape/dtype drift. Findings must name the affected Buffer, store or AccessMap.

The Corpus manifest cannot be changed by this automation. After focused implementation
and tests, the new cases and expected Findings are an expectation-adoption proposal for
`repository_owner`. Only after adoption may the actual full Gate be generated; the
agent must not create or modify `compiler/release-approval.json`.

## Evidence boundary

Static construction, verification and generated source do not prove GPU correctness or
performance. After an exact released successor exists, the KDA judge remains the owner
of output, post-convolution state, post-SSM state and raw CUPTI measurements. GPU Infra
owns only immutable snapshot, broker admission, route and lifecycle. Operator evidence
does not qualify SGLang integration or serving.
