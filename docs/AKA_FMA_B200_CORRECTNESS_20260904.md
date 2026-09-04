# Released v41 FMA: B200 numerical correctness — 2026-09-04

## Result

One fixed-node run completed with **valid correctness, memcheck and racecheck** for
released Compiler `open-cake-ir-sm100a-v41`. It covers ordinary FMA and nested FMA
with a separately rounded multiply, each at fixed shape [8,128].
All 12 workloads passed in each of the three stages.

The 36 complete-output artifacts were independently recomputed on the node and from
the collected local mirror: **36,864 output elements passed**, comprising 30,333 exact
bitwise comparisons and 6,531 quiet-NaN category comparisons. NaN sign/payload is not
part of the contract. Every input bit pattern was preserved, and the executing judge
validated disjoint pointers, unchanged input/output pointers, the supplied-output
return value, shape and FP32 dtype.

This is fixed-instance target correctness and memory/race safety evidence. It is not
performance, arbitrary-shape coverage, full AKA-parent equivalence, framework end-to-end,
serving or training qualification. `frontier_eligible=false` and
`performance_measured=false` are expected. The 12 workloads below are test cases for
two kernels, not 12 newly completed AKA parents.

## Frozen object and execution

- Compiler release: v41 from `ee32b76870c4166c8143e8774c37c1212a0b6e1c`.
  The released lock, approval, Compiler sources and 72-case Corpus were unchanged.
- Harness commit: `bb8b1dbb94adf3f1a2ebebe61d7a76dad3f6793f`.
  [Task preparation, judge and oracle](../examples/gpu/fma_b200_correctness/README.md)
  are separate from the frozen Compiler; candidate kernels were generated through
  the released Compiler and were not edited.
- GPU Infra: `97a2bff2934562a1e416c753565862d4acc9a784`, version 0.16.0.
- Host/user: `qhy-sol@verda-b200x4`; existing Python runtime
  `/home/qhy-sol/kda-runtime-venv/bin/python`.
- Fixed node: `verda-b200x4-fma-v41-rh7qi8`.
- Unique run: `open-cake-fma-v41-b8x128-b200-correctness-v1-7f305cec31e1`.
- One formal fleet submission; no GPU retry, reroute, predecessor repair or gate change.

| Stage | Resource contract | Broker job | GPU | Outcome |
| --- | --- | --- | ---: | --- |
| correctness | shared, one GPU | gpuq-097fc02cdbcc | 0 | passed / valid, exit 0 |
| sanitize-memcheck | exclusive, one GPU | gpuq-511d89a27aed | 0 | passed / valid, exit 0 |
| sanitize-racecheck | exclusive, one GPU | gpuq-6bae24748d4d | 0 | passed / valid, exit 0 |

The broker was the only allocator. Foreign/shared-control-plane work was not changed.
The memcheck log explicitly reports zero errors; the racecheck log explicitly reports
zero hazards, errors and warnings.

## Numerical contract

Each kernel sees six deterministic 1,024-element input families: cancellation,
signed zero, subnormals, overflow, Inf/quiet-NaN/signaling-NaN combinations, and seeded
arbitrary binary32 bits. Inputs and complete outputs are serialized as unsigned bit
patterns, never JSON NaNs.

The integer oracle performs exact product/add and a single RN-even binary32 rounding.
For the nested kernel it separately rounds `a*b`, evaluates the inner FMA, then
evaluates `fma(inner,c,product)`. It does not call the candidate, CUDA, or Triton to
derive expected results. CPU cross-checks against host fmaf covered 26,144 triples;
another 5,000 random cases cross-checked multiplication and nested evaluation.
Literal tests and negative controls detect unfused cancellation, FTZ, lost zero signs
and invalid/non-quiet NaN outputs. The oracle suite passed 5/5 both locally and remotely.

Output initialization deliberately uses a non-NaN sentinel where NaN is expected and
a NaN sentinel otherwise, so an unwritten output cannot pass simply because NaN payloads
are unconstrained. The evaluator and oracle were frozen in the same immutable input
bundle under GPU Infra's cooperating-agent trust model, not a hidden adversarial judge.

The standalone verification script recomputes numerical values and checks persisted
input arrays; the runtime pointer/return ABI observations remain judge-owned evidence.

## Collection and preservation

The node-owned [run result](data/fma-v41-b200-correctness-20260904/result.json),
three stage receipts and [independent recomputation receipt](data/fma-v41-b200-correctness-20260904/independent-verification.json)
are published as small structured evidence. Raw complete outputs, sanitizer logs,
generated source, route and mirror inventory remain outside Git.

- Local evidence root:
  `/Users/haiyan-infiniai/Agent4Kernel/open-cake-fma-b200-rh7Qi8/`.
- Original node state:
  `/tmp/open-cake-fma-b200-20260904-rh7Qi8/state/`.
- Local successful mirror: `mirror-02/`; it is mirror-only, not lifecycle authority.
- The first `collection-01/` records an SSH fleet-export timeout after 60 seconds.
  It was retained as `fetch_failed`, not repaired or mislabeled successful.
  A later read of the same terminal locator with a 120-second transport limit
  succeeded. No GPU work was repeated.

The isolated GPU Infra deployment passed 76/76 tests; task and fleet checks passed
before submission. Existing home-directory ACLs already allowed the broker user to
traverse the selected runtime. Only fresh task directories received task-owned
permissions; no old run or shared home permissions were changed.

## Cleanup and next action

The task-owned daemon PID 726597 was sent SIGTERM only after its run was terminal and
its three broker jobs were no longer running. Its socket and process disappeared;
node state and all evidence were preserved. The [cleanup receipt](data/fma-v41-b200-correctness-20260904/cleanup.json)
does not claim that unrelated broker jobs or shared daemons were stopped.

Next, re-assess the original FMA-gap AKA parents under v41 and identify which can now
form complete Schedules. Runtime scalar ABI, broadcast, state, indexing or reduction
order gaps can still block those parents. Do not turn this fixed-shape primitive test
into a blanket qualification of the 12 FMA-cluster records. Log and cosine are outside
this run; no further IR or performance claim is made.
