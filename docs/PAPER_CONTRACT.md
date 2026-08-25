# Paper contract

## Source freeze

Canonical source: [CAKE: Compiler-Agent Co-Design for Frontier Kernel Evolution,
arXiv:2608.12629v1](https://arxiv.org/html/2608.12629v1), submitted 2026-08-12.
本目录只引用 v1；后续论文版本必须作为新 source revision 显式加入。

论文中的数值统一标为 `paper_reported`。只有本地 immutable raw evidence 支持的陈述才可标为
`local_observed`；公开生成物能独立审计时标为 `public_artifact_auditable`；本地替代实现的结果标为
`independent_reconstruction`；其余为 `unknown`。

## Irreducible paper architecture

```text
External Workload Contract
  -> Authoring Environment
       |-- Cake IR + structured compiler evidence
       `-- direct CUDA/PTX
  -> construction/type checks
  -> verifier hard gates + cost ranking
  -> compile -> external oracle -> GPU measurement
  -> retained evidence
       |-- inner loop: next candidate
       `-- outer loop: IR / verifier / cost-model proposal
                         -> corpus tests + human merge gate
```

This follows the paper's [program representation](https://arxiv.org/html/2608.12629v1#S2),
[compiler harness](https://arxiv.org/html/2608.12629v1#S3), and
[agent workflow](https://arxiv.org/html/2608.12629v1#S4):

- The external workload contract fixes shape, oracle, tolerances, hardware and reference access.
- Cake IR exposes roles, resources, barriers and pipelines; lowering derives mechanical metadata.
- Static analysis is a cheap pre-compile filter; external correctness and GPU measurement remain authoritative.
- Recurring failures may change the compiler only between campaigns and only through corpus-gated review.
- Exact target mismatch or missing calibration is reported explicitly; no silent architecture fallback.

The typed Schedule, Target model, verifier, analysis, feedback and lowering form the paper's reusable system core.
The clean-start campaign is evaluation apparatus for that core, not the product definition. `open-cake-ir`
therefore makes its Compiler independently callable and places provider/campaign logic in a dependent Lab.

## The causal question

The paper's matched clean-start treatment is not syntax alone. It changes the complete **Cake authoring
environment bundle**: typed IR, localized verifier diagnostics, analysis and cost feedback, compared with
direct CUDA/PTX authoring. Calling Table 2 a pure representation effect would require a separate mechanism
ablation that the paper does not report.

The experimental unit is an independent agent Run, not a Turn, Candidate or Checkpoint. A local scientific Study
must define its endpoint, contrast, allocation, missingness and Estimand before execution; offline audit may estimate
that quantity but cannot invent it afterward. A `system_qualification_only` Study instead declares no Estimand and
forbids treatment comparison by contract.

An `artifact_optimization_only` Study may expose provider-default authoring features and select one confirmed
Candidate per Run, but its output is engineering Evidence: it has no scientific inclusion, arm contrast, estimate or
uncertainty and cannot fill a paper Campaign cell.

The paper fixes GPT-5.6-sol, reasoning effort xhigh, task, oracle, benchmark, target shape, B200 and reference
policy; it reports three independent runs per arm at an 80M provider-token budget. For Flash-KMeans `assign`,
the fixed cell is `B=32, N=65536, K=1024, D=128`, BF16 inputs with FP32 accumulation. It reports a tuned
FlashML baseline of 0.938 ms and median best-at-80M results of 1.144x baseline for Cake IR versus 0.928x for
direct CUDA/PTX. These are `paper_reported`, not locally reproduced
([paper section 5 and Table 2](https://arxiv.org/html/2608.12629v1#S5)).

## Claim lanes

| Claim lane | Question | Minimum evidence |
| --- | --- | --- |
| Matched clean-start | Does the Cake authoring environment outperform direct CUDA/PTX under matched conditions? | Independent runs per arm, complete token trajectories, identical frozen factors, contamination audit |
| Frontier synthesis | Can a physical schedule be discovered without a low-level target reference? | Reference-access audit, correctness, fixed endpoint, and target-level validation |
| Known-kernel reproduction | Can a known expert endpoint be matched or improved? | Pinned reference, shapes, correctness, common timing protocol |
| Portfolio generalization | Can fixed-shape seeds become a library family? | Predeclared domain, held-out/boundary/tail cases, guard/fallback coverage, dispatcher-inclusive metric |
| Serving validation | Does the integrated path improve an application? | Framework-level correctness and no-profiler end-to-end measurement |

Single-shape search and portfolio construction are intentionally separate objectives with different failure
modes. The paper requires the latter to declare the shape domain before tuning and to measure dispatch itself
([paper section 6](https://arxiv.org/html/2608.12629v1#S6)).

## Publicly auditable versus unknown

Public material includes the paper protocol and aggregate values plus generated downstream endpoints such as
[FlashInfer KDA prefill PR #4262](https://github.com/flashinfer-ai/flashinfer/pull/4262),
[TinyGEMM2 PR #4274](https://github.com/flashinfer-ai/flashinfer/pull/4274), and
[Alpha-MoE PR #4287](https://github.com/flashinfer-ai/flashinfer/pull/4287). These endpoints may support
remeasurement or known-kernel lanes; they are prohibited clean-start inputs and do not expose the generator.

The currently cited public material does not provide the exact original compiler revision, full IR schema and
semantics, verifier rules and coverage, cost model and calibration, agent scaffold/prompts, raw clean-start
trajectories, complete token receipts and CUPTI samples, formal plateau definition, isolation implementation,
the inclusion boundary of reported active evolve time, the exact tuned FlashML baseline revision, or exact
dispatcher shards. Therefore this project must not claim byte-for-byte CAKE reconstruction or compare any
local artifact-optimization Campaign—including v4's 8M turn-discrete engineering horizon—with the paper's
80M-token result.

Clean-start provenance must include the Compiler Revision, its Corpus, scaffold, memory, diagnostic surface and
the complete frozen reference bundle embedded in each Turn prompt. If a local compiler was evolved from prior observations of the same task, that history is
an explicit treatment prior: the resulting Study can measure package effectiveness but not task-naive discovery.

## Local implementation choices

| Dimension | Paper-reported constraint | Local choice until stronger evidence exists |
| --- | --- | --- |
| Treatment | Cake bundle vs direct CUDA/PTX | Name the local treatment `cake_like`; report it as independent reconstruction |
| Model effort | GPT-5.6-sol with `xhigh`, held fixed across arms | Reasoning effort is an explicit Authoring Environment factor whose exact value must be live-qualified and identical across arms. Codex 0.144.4 now passes the closed two-arm candidate-set qualification at `xhigh`; the receipt proves transport, not a Study. Current frozen local Studies retain the distinct `max` treatment, and a future successor must explicitly freeze the qualified `xhigh` configuration without rewriting them |
| Harness evolution | Evidence-driven outer loop, corpus-gated | Freeze inside a campaign; change only between campaigns |
| Token accounting | 80M-token budget; complete receipts and exact accounting definition unavailable | Define local `provider_tokens` as the provider's Turn-end `input_tokens + output_tokens`; reported cached input remains part of input. Observe boundaries only after a complete Turn, retain the raw usage, and do not compare the resulting count with 80M |
| Active evolve time | Median active evolve time is reported per representation, but the clock's inclusion boundary is unpublished | Do not emit an `active_evolve_time` surrogate. A future local duration must use a distinct name and preregister whether provider wait, compilation, queueing and GPU evaluation are included |
| Scientific endpoint | Table 2 reports best-at-budget plus a prespecified but unpublished plateau rule | Executor v22's successor-only two-part Estimand separates qualification rate from latency conditional on qualification. Adhered candidate failure is observed; external fault or absent archive is missing. The full estimate needs every prescheduled endpoint plus at least one qualified Run per arm and includes `open_cake_rate - direct_cuda_rate`. Frozen v1–v3 plans retain their earlier rule and are not reinterpreted |
| Plateau | Prespecified but definition unavailable | Run the declared budget; compute plateau only as an offline diagnostic |
| Isolation | Isolated clean start and post-run audit | Allowlisted workspace plus complete file/tool/event trace. Executor v25 additionally retains the exact rendered reference bundle for every completed provider Turn; missing bytes create a harness fault rather than passing audit |
| Reference access | No low-level target implementation at clean start | `matched-search-clean-start-reference-v28.json` pairs an implementation-free Schedule interface with an empty CUDA ABI starter and rejects contamination structurally. Combined with v26 bundle retention, an auditor can inspect the actual local input. The fixture still retains 150k/`max` and is not a paper-aligned campaign |
| Timing | B200, CUPTI, cold-L2 samples | Freeze physical GPU and environment evidence; retain every raw sample |
| Dynamic attribution | Benchmark and profiler evidence for evaluated survivors | Executor v18 composes the existing search and no-timing attribution purposes so every correctness-qualified searched survivor retains raw NCU CSV plus a replay-checked projection; only the selected survivor's profile becomes next-Turn feedback. Contract execution, missing-profile rejection and a bounded live two-arm B200 successor cover the complete relation. Scientific v3 executes the relation, but two preregistered missing Runs leave its Estimand unavailable; it is evidence about the local repaired package, not a paper reproduction |
| Evidence semantics | Retained evidence supports post-run audit; the paper does not publish its event schema | Executor v23 successor matched Studies declare the closed `matched_run_v1` vocabulary. Replay rejects unknown events and derives selection and diagnoses from retained filter rows and Receipts. Earlier Studies retain bounded legacy replay; no frozen bytes are reinterpreted |
| Static ranking | Calibrated pre-GPU cost ranking | The structural hypothesis is implemented and measurable, but Compiler v11 coverage is empty. The old total-order 3-to-2 calibration failed at 35.95% and 8.16%. v11 makes tied cuts abstain; a drift-controlled successor evaluated all 1,450 decisive subsets and reported all 850 tied subsets per repeat, but one repeat still exceeded the unchanged 5% limit at 5.73% (the other reached 2.79%). Public ranking therefore reports missing coverage |
| Static feedback | Typed IR verifier and analysis vs compiler diagnostics | One static channel per arm, matched in kind: the Compiler's Findings for a Schedule, ptxas resource output for authored CUDA. A Study that opts into current attribution gives both arms one bounded raw-checked profiler observation per correct search survivor and routes only the selected projection to the next Turn |
| Generalization | Separate portfolio stage | Implement only the frozen r45 three-shape reconstruction; do not call it arbitrary-shape or paper generalization |

The final legacy r42 Campaign completed but its preregistered 150k checkpoint Estimand was unavailable. That does
not authorize replacement Runs, yet it did emit a pre-held-out KernelSeed. A separately frozen r45 Portfolio Study
therefore supplies bounded local correctness and measurement-quality evidence without repairing or pooling r42.
Its two held-out dispatcher timing boundaries are unstable; serving and paper generalization remain unsupported.
