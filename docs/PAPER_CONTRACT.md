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
or exact dispatcher shards. Therefore this project must not claim byte-for-byte CAKE reconstruction or compare
its 150k-token local campaign directly with the paper's 80M-token result.

Clean-start provenance must include the Compiler Revision, its Corpus, scaffold, memory, diagnostic surface and
the complete frozen reference bundle embedded in each Turn prompt. If a local compiler was evolved from prior observations of the same task, that history is
an explicit treatment prior: the resulting Study can measure package effectiveness but not task-naive discovery.

## Local implementation choices

| Dimension | Paper-reported constraint | Local choice until stronger evidence exists |
| --- | --- | --- |
| Treatment | Cake bundle vs direct CUDA/PTX | Name the local treatment `cake_like`; report it as independent reconstruction |
| Harness evolution | Evidence-driven outer loop, corpus-gated | Freeze inside a campaign; change only between campaigns |
| Plateau | Prespecified but definition unavailable | Run the declared budget; compute plateau only as an offline diagnostic |
| Isolation | Isolated clean start and post-run audit | Allowlisted workspace plus complete file/tool/event trace |
| Timing | B200, CUPTI, cold-L2 samples | Freeze physical GPU and environment evidence; retain every raw sample |
| Dynamic attribution | Benchmark and profiler evidence for evaluated survivors | The canonical evaluator now runs a separate correctness-qualified NCU assay, retains raw CSV plus a replay-checked projection, and routes the selected confirmed Candidate's profile without treating profiler duration as timing. This is a complete narrow vertical slice, but not yet profiler coverage for every evaluated survivor or a scientific Campaign |
| Static ranking | Calibrated pre-GPU cost ranking | The structural hypothesis is implemented and measurable, but v8 coverage is empty after declared-domain B200 checks; public ranking reports missing coverage |
| Static feedback | Typed IR verifier and analysis vs compiler diagnostics | One static channel per arm, matched in kind: the Compiler's Findings for a Schedule, ptxas resource output for authored CUDA. A Study that opts into attribution gives both arms the same bounded, raw-checked profiler projection after confirmation |
| Generalization | Separate portfolio stage | Implement only the frozen r45 three-shape reconstruction; do not call it arbitrary-shape or paper generalization |

The final legacy r42 Campaign completed but its preregistered 150k checkpoint Estimand was unavailable. That does
not authorize replacement Runs, yet it did emit a pre-held-out KernelSeed. A separately frozen r45 Portfolio Study
therefore supplies bounded local correctness and measurement-quality evidence without repairing or pooling r42.
Its two held-out dispatcher timing boundaries are unstable; serving and paper generalization remain unsupported.
