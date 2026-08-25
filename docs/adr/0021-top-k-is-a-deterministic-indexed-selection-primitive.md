# ADR 0021: top-k is a deterministic indexed-selection primitive

Status: accepted, 2026-08-25.

## Outcome and non-goals

Cake admits one `top_k` operation over a resident rank-one tile. It returns both the
selected values and their `int32` source indices, ordered by descending value with
lowest-index tie breaking. `k` is a positive static integer and NaN is outside the input
contract.

This is not a `moe`, `routing` or KDA-version mode. It does not add grouping, reshape,
mask/scatter, weight normalization, grouped GEMM, quantization, dynamic `k`, bottom-k, an
unordered result, or a choice of physical reduction algorithm. Those are separate facts
and remain separate gaps.

## Evidence and authority

The external KDA MoE v1 reference uses the same mathematical primitive three times:

1. take the top two biased scores in each 32-expert group and sum their values;
2. take the top four group scores and consume their indices;
3. take the top eight retained expert scores and consume their indices.

The v1 CuTe implementation resolves every tie to the lowest source index and extracts
winners in descending order. v22 changes only the physical max-reduction implementation,
from a shuffle tree to Blackwell warp reduction; it does not change this selection
contract. The KDA history remains the authority for those observations and measurements.

The paper says its validated corpus covers Top-K, while this reconstruction's current
Target and typed vocabulary reject `top_k` and its Corpus has no Top-K profile. A
pre-change construction probe fails at `operations[*].kind`, before any verifier or
lowering can reason about the operation. This is therefore a live vocabulary gap, not a
request for a speculative abstraction.

`open-cake-ir-experience` contains no `CompilerProposal` and explicitly forbids an
imported episode from authorizing a Compiler Revision. It contributes no semantic fact
to this change. The Compiler's typed IR owns the operation contract, the Target owns
admission, the verifier owns legality, and the Workload/oracle owns live correctness.

## Minimal primitives and invariants

One operation is sufficient:

```text
(values[k], indices[k]) = top_k(source[n], k)
```

- exactly one read and exactly two ordered writes: values first, indices second;
- `source` is a rank-one resident `fp32` register tile;
- `1 <= k <= n`;
- the current SM100 Triton lowering admits power-of-two `k` only;
- values have the source dtype and shape `[k]`;
- indices have dtype `int32` and shape `[k]`;
- results are descending by value; equal values are ordered by lowest source index;
- `reject_input` is the only admitted NaN policy;
- the operation is pure apart from its two declared writes and cannot alias a read.

There is deliberately no `largest` flag: top-k means greatest values. There is no
`sorted` flag: deterministic order is part of the one canonical form. A future bottom-k
or unordered selection needs independent recurring evidence instead of widening this
contract pre-emptively.

The rank-one boundary is intentional. A Schedule maps independent rows to programs and
the operation selects within the resident row tile, matching the KDA one-CTA-per-token
baseline. Group formation and cross-group masks are still unrepresentable and must not be
hidden inside this primitive.

## Lowering and hardware behaviour

The Triton backend lowers the contract as `k` repeated maximum-value and minimum-index
reductions. A Boolean tile tracks selected source positions. Each step takes the maximum
over unselected values, then the minimum source index among unselected positions equal to
that value. This remains correct when a legal score is negative infinity; merely replacing
a selected value with negative infinity would select the same position twice in that case.
The generated source names every step under the one `CAKE_OP` region and constructs both
declared results.

This lowering is inspectable but is not claimed optimal. Its power-of-two `k` restriction
comes from the current Triton vector construction and is a localized hardware-conformance
failure, not a semantic restriction on the typed operation. In particular, the lowering
does not name the KDA v22 warp-reduction mechanism. A second physical algorithm would make
that choice performance-relevant and would require a new Target-backed schedule commitment
plus comparative evidence; it is not smuggled in as an optional flag here.

Static residency remains an optimistic lower bound over declared logical storage. The
source tile and both result tiles are distinct live values and are therefore charged by
the existing lifetime analysis. Backend temporaries may increase ptxas allocation and
remain toolchain evidence.

## Smallest complete slice and failure semantics

The positive Corpus profile maps eight independent 256-score rows to eight programs and
stores eight expert indices per row. Its values output remains declared scratch because
the operation contract always returns both facts. A companion changes the stored index
shape to four; the verifier must reject it locally rather than allowing lowering to drop
or truncate results.

Acceptance requires:

- strict parser/schema coverage for the closed parameters;
- localized arity, rank, `k`, shape and dtype findings;
- generated Triton source whose result matches an independent stable descending sort,
  including injected ties;
- the full Compiler Corpus gate and contract suite;
- one brokered B200 correctness observation with no timing claim.

The change proves deterministic indexed selection, not KDA v1 coverage. Complete KDA
coverage remains 0/57 until grouping/masking, quantization-scale relations,
grouped/ragged work acquisition, atomic/direct scatter and the multi-kernel program DAG
are all represented and validated together.

Historical Compiler releases and observations are immutable. This change ships in a
successor Revision; there is no compatibility alias or rewrite of frozen evidence.
