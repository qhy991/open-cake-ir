# AKA FMA parent re-audit against released v41 — 2026-09-06

> Historical v41 report. Its counts and proposed version numbers are retained as
> written. Current repairs and the explicit frozen-Compiler replay command are in
> [the follow-up](ACCESS_DOMAIN_REPAIR_20260906.md); this report is not a current
> Compiler capability or GPU qualification index.

## Result

The twelve fixed-instance parents in semantic cluster
`c050-elementwise-fma-fp32` were re-audited against released
`open-cake-ir-sm100a-v41`. The original absence of single-round FP32 FMA is
resolved for all twelve. It does **not** follow that all twelve complete parents
are now expressible.

| Terminal class | Count | Meaning |
| --- | ---: | --- |
| `candidate_lowered_compiled` | 4 | A semantic-reviewed fixed-instance Schedule is v41 accepted/lowerable and real Triton 3.7.1 sm100 GPU-free compilation produced PTX/cubin. |
| `remaining_schedule_ir_gap` | 5 | FMA is resolved, but runtime by-value FP32 scalar binding and register splat remain the earliest complete-Schedule blocker. |
| `candidate_semantic_review_rejected` | 1 | v41 accepted/lowered the Schedule, but independent review found hidden scalar-to-vector broadcast; real compilation also rejected a non-power-of-two tile. |
| `authoring_rejected` | 2 | The model claimed completeness but submitted an invalid Schedule; no IR conclusion is drawn from those two attempts. |

This establishes a **4/12 fixed-instance static-and-compile survivor lower bound**.
It is not 4 GPU-correct parents, not 33% arbitrary-parent coverage, and not a
performance result. No GPU work was submitted by this re-audit.

## Frozen treatment

- Open-Cake source/result commit:
  `d9d56e835cf96eecf70e0259b65bc1b20c4f6f0d`.
- Compiler: released `open-cake-ir-sm100a-v41`.
- AKA source: `387aa7faf521a0b72c994ff15a7638cd7e6a8583`.
- Model: remote `qhy-sol` ChatGPT subscription,
  `gpt-5.6-sol`, reasoning effort `max`.
- Model tool access, network, GPU and subagents: disabled.
- Filesystem permission profile: source evidence read-only; the credential tree
  was denied before every turn. No credential, token or auth file was read.
- One canary preceded the maximum five-way batch.
- Assessment scope remained `fixed_instance`: only the row's selected frozen
  workload/shape was assessed. Dynamic dimensions, host validation, stream,
  status return, other correctness cases and performance-domain workloads were
  not retroactively required.

All source artifacts referenced by each Phase A row were embedded. The final
prompt sizes remained below the 512 KiB limit. Model output was accepted only
through the strict JSON schema and the released Compiler public API.

## Per-parent disposition

| Parent / case | Final result | Decisive fact |
| --- | --- | --- |
| Momentum SGD pair, copy_vectorized l000002 | remaining IR gap | Runtime by-value `momentum` and `weight_decay` cannot enter/splat into same-shaped FMA registers. |
| Grouped channel dual reduce, gather_scatter l000074 | authoring rejected | Access tile/stage mismatch and scalar reduction output shape mismatch; submitted Schedule is not evidence of a new IR gap. |
| NCHW plane affine, gather_scatter l000075 | compiled survivor | Exact model Schedule; unit output programs load X/scale/bias as shape-[1] values; PTX/cubin produced. |
| AXPBY, gather_scatter l000178 | remaining IR gap | Runtime by-value `alpha` and `beta`, plus scalar-to-register splat, are absent. |
| Gamma/beta reduce, gather_scatter l000217 | authoring rejected | Noncanonical full-range extents and multiple writers for each output; no post-hoc semantic repair. |
| Repeated-segment AXPBY, layout_transform l000036 | remaining IR gap | Runtime by-value `alpha` and `beta` cannot be materialized without changing inputs. |
| NCHW channel affine, memory_cache l000006 | compiled survivor via independent successor | The model's vector Schedule was rejected by semantic review because scalar loads were mislabeled as [128] tiles and Triton broadcast them implicitly. A unit-spatial Schedule preserves original global shapes and removes hidden broadcast; v41 accepted/lowered it and Triton compiled it. |
| NHWC channel affine, memory_cache l000007 | compiled survivor via bounded normalization | Removing only four illegal register Allocations and their Buffer references produced an accepted/lowerable Schedule and successful compile. |
| In-place LAMB scaled subtract, elementwise l000001 | remaining IR gap | Runtime by-value `ratio` and its signed splat remain unavailable; state-store semantics are otherwise present. |
| Grouped NCHW affine combine, normalization_group l000002 | compiled survivor | Exact model Schedule; all operands are unit register values and two ordered FMA nodes compile. |
| GroupNorm backward NHWC, normalization_group l000003 | semantic review rejected | Three tile-one `program_tile` uses normalized to scalar `program`, but scalar x_scale/bias loads were still mislabeled as [1,1,3] tiles; Triton additionally rejected `tl.arange(0,3)`. |
| X-scale/bias group reduce, reduction l000035 | remaining IR gap | Runtime by-value FP32 `alpha` is still a data input and cannot be represented or splatted. |

## New findings

### Runtime scalar plus splat is now the dominant semantic gap

Five independent rows converge on one minimum requirement: typed FP32 values
passed by value at invocation time must appear in the generated executable
signature and be materializable as same-shaped register values. A JSON numeric
`elementwise.scalar` is a compile-time literal, and changing a scalar to a
shape-[1] pointer Buffer changes the ABI. FMA itself must remain three explicit
register reads; the new scalar owner should feed it rather than add a second
scalar spelling to FMA.

This is stronger evidence for the previously deferred runtime-scalar candidate,
but it is still a proposal input. Its typed ABI, scalar/value distinction,
uniformity, splat semantics, verifier rules, analysis effect, host wrapper and
backend lowering require owner review before implementation.

### v41 load/result shape verification is incomplete

The first l000006 model Schedule declared scalar scale/bias loads as [128]
register Buffers. The generated Triton source silently broadcast those values,
so compilation passed despite violating the explicit no-broadcast FMA contract.
The published projection rejects that predecessor. The unit-spatial successor
shows that this parent does not require broadcast IR for its selected instance.

This is a verifier/data-consistency defect: a load's declared register result
shape must agree with its AccessMap vector domain. It does not invalidate the
released v41 FMA B200 test, whose three operands were explicit same-shaped
vectors. It should be repaired in a Compiler successor before treating arbitrary
model-authored FMA Schedules as semantically accepted.

### Triton power-of-two preflight is incomplete

The rejected l000003 Schedule passed v41 assessment and `Compiler.lower` despite
hidden scalar-to-vector load shapes, then real GPU-free compilation also failed
because `tl.arange(0,3)` is unsupported.
The Triton backend should produce a structured preflight Finding for this shape,
or the author should use a padded power-of-two tile with correct masks. This is
a backend diagnostic/authoring problem, not evidence for another arithmetic
primitive.

## Attempts and evidence boundary

The first schema attempt failed before model output. A second canary used a
prompt that accidentally broadened `fixed_instance` to the dynamic ABI and
was discarded; its controller also misclassified a successful output file. A
third canary restored the scope but submitted illegal register Allocations. The
fourth treatment added the existing structural authoring rules, passed one
canary, then ran the remaining cases with at most five workers.

All predecessor attempts, raw model outputs, receipts, evidence files and
failed finalizer output are retained outside Git. The checked-in
[attempt index](data/aka-fma-v41-reaudit-20260906/attempts.json) explains
supersession. Model event trajectories and duplicated source bundles are not
published.

Canonical publication:

- [twelve terminal rows](data/aka-fma-v41-reaudit-20260906/results.jsonl)
- [summary](data/aka-fma-v41-reaudit-20260906/summary.json)
- [deterministic verifier](../tools/verify_aka_fma_v41_reaudit.py)

Every compiled receipt is CPU-only target compilation:
`gpu_test=not_run`, `performance_measured=false`. No result is admitted to
the prior FMA GPU correctness evidence or to training.

## Next actions

1. Independently review the four semantic-survivor Schedules against their
   original fixed workloads; only reviewed survivors enter GPU admission.
2. Add a v42 verifier rule for load-result shape versus AccessMap domain and a
   Triton preflight rule for unsupported non-power-of-two vector extents.
3. Review the five-row runtime FP32 scalar/splat contract as the next minimal IR
   proposal. Do not add a scalar field to FMA or broaden this into arbitrary
   expression/ABI support.
4. Re-author l000074 and l000217 only as fresh successors. Do not repair or
   relabel the rejected model outputs.
5. Keep log/cos implementation and performance measurement separate from this
   attribution.
