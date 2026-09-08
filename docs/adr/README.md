# 设计决策目录

[中文首页](../zh-CN/README.md) · [English](../en/adr/README.md) · [中英文对照](../README.md)

ADR 记录“为什么这样设计”。当前版本看 [发布状态](../../reports/current/STATUS.md)，一次实验的成绩看它自己的报告。
新决策写新记录，并说明取代哪条旧决定；不改写历史证据。

| 想了解什么 | 主要记录 |
| --- | --- |
| 为什么以编译器为核心 | [0001](../zh-CN/adr/0001-compiler-first-with-dependent-lab.md) |
| 为什么单独区分 portfolio | [0002](../zh-CN/adr/0002-portfolio-as-second-study-variant.md) |
| 代码与运行产物放在哪里 | [0005](../zh-CN/adr/0005-forward-compatible-lifecycle-layout.md) |
| 为什么估算需要校准 | [0008](../zh-CN/adr/0008-calibration-coverage-gates-ranking.md) |
| 生成源码与固定源码有什么区别 | [0019](../zh-CN/adr/0019-lowering-generation-is-observable.md) |
| 为什么 Corpus 与 Workload 不同 | [0029](../zh-CN/adr/0029-lowering-route-is-not-a-workload-profile.md)、[0038](../zh-CN/adr/0038-aka-is-a-challenge-corpus-not-a-compiler-corpus.md) |
| 发布如何使用独立人类或模型会话的批准 | [0052](../zh-CN/adr/0052-independent-agent-release-review.md)，背景：[0030](../zh-CN/adr/0030-compiler-release-approval-is-external.md) |
| 为什么完整文件不等于可信的权限历史 | [0031](../zh-CN/adr/0031-archive-integrity-is-not-filesystem-custody.md) |
| 文档怎样避免重复维护事实 | [0047](../zh-CN/adr/0047-documentation-separates-stable-history-and-current-views.md) |
| AI 怎样读取任务并受预算约束 | [0048](../zh-CN/adr/0048-agent-runs-use-task-agents-and-ralph-control.md) |
| 已发布身份怎样保留 | [Executor：0049](../zh-CN/adr/0049-released-executor-descriptors-reserve-their-identities.md)、[Compiler：0050](../zh-CN/adr/0050-released-compiler-locks-reserve-their-identities.md) |
| 为什么读取不能隐式复制成一组数 | [0051](../zh-CN/adr/0051-load-values-follow-the-access-domain.md) |
| Study 怎样显式使用外部模型给候选排序 | [0053](0053-study-bound-advisory-cost-selection.md) |

[中文设计记录目录](../zh-CN/adr/README.md)逐条对应英文原文。历史中重复的编号按完整文件名区分。状态含义：

- **proposed：** 提案，可按授权范围实现和审查，不代表已发布或 GPU 通过。
- **accepted：** 决策已被接受；具体权限范围仍看该记录和当前任务授权。
- **superseded：** 后继记录负责当前决定，旧记录保留背景。
- **rejected：** 保留被拒绝的方案及理由，不据此实施。

- [0054: Lab uses only Ralph](0054-lab-uses-only-ralph.md)
- [0055: Task implementations live outside the common Lab](0055-task-implementations-live-outside-the-common-lab.md)
- [0056: One fixed-baseline paired assay per candidate](0056-fixed-baseline-paired-execution.md)

## 原始决策补充索引 / Additional original decisions

- [ADR 0003: Consider Rust only as a Corpus-equivalent v4 shadow engine](0003-rust-shadow-engine-after-v3.md)
- [ADR 0004: Restore provider-default features only for artifact optimization](0004-tool-rich-artifact-optimization.md)
- [ADR 0006: candidate sets, and a Schedule stays statically shaped](0006-candidate-sets-and-static-shapes.md)
- [ADR 0007: candidate evidence has one identity, and every frozen reference is a witness](0007-candidate-evidence-and-revision-witnesses.md)
- [ADR 0009: live candidate sets use one sealed envelope](0009-live-candidate-set-envelope.md)
- [ADR 0010: Establish feedback before adding agent orchestration](0010-feedback-before-agent-orchestration.md)
- [ADR 0011: calibrate the decision the Lab actually makes](0011-finite-domain-ranking-calibration.md)
- [ADR 0012: Profile each correctness-qualified search survivor](0012-profile-each-correct-search-survivor.md)
- [ADR 0013: Keep candidate failure separate from missing scientific data](0013-two-part-scientific-estimand.md)
- [ADR 0014: Close the matched Run semantic event vocabulary](0014-close-matched-event-vocabulary.md)
- [ADR 0015: Reasoning effort is an explicit treatment factor](0015-reasoning-effort-is-treatment.md)
- [ADR 0016: Clean-start references carry contract, not implementation](0016-clean-start-references-carry-contract-not-implementation.md)
- [ADR 0017: Retain the exact provider reference bundle](0017-retain-the-exact-provider-reference-bundle.md)
- [ADR 0018: ranking is a preorder and calibration controls drift](0018-ranking-is-a-preorder-and-calibration-controls-drift.md)
- [ADR 0020: KDA deltas are an external expressibility corpus](0020-kda-deltas-are-an-external-expressibility-corpus.md)
- [ADR 0021: top-k is a deterministic indexed-selection primitive](0021-top-k-is-a-deterministic-indexed-selection-primitive.md)
- [ADR 0022: tanh names its target implementation contract](0022-tanh-names-its-target-implementation-contract.md)
- [ADR 0023: block scales are axis relations](0023-block-scales-are-axis-relations.md)
- [ADR 0024: runtime-valid extents are buffer relations](0024-runtime-valid-extents-are-buffer-relations.md)
- [ADR 0025: ragged grouped GEMM is primitive composition](0025-ragged-grouped-gemm-is-primitive-composition.md)
- [ADR 0026: runtime-indexed loads are AccessMap composition](0026-runtime-indexed-loads-are-access-map-composition.md)
- [ADR 0027: Backend preconditions are Assessment Findings](0027-backend-preconditions-are-assessment-findings.md)
- [ADR 0028: Study templates defer revision binding to CampaignLock](0028-study-templates-defer-revision-binding-to-campaign-lock.md)
- [ADR 0032: access boundaries use the accessed Buffer](0032-access-boundaries-use-the-accessed-buffer.md)
- [ADR 0033: Workload materialization owns TinyGEMM2 input bytes](0033-workload-materialization-owns-tinygemm2-input-bytes.md)
- [ADR 0034: KDA weighted combine is primitive composition](0034-kda-weighted-combine-is-primitive-composition.md)
- [ADR 0035: atomic slot reservation is state plus RMW](0035-atomic-slot-reservation-is-state-plus-rmw.md)
- [ADR 0036: atomic reservation proves indexed-store ownership](0036-atomic-reservation-proves-indexed-store-ownership.md)
- [ADR 0037: A prefix scan is not a fold with a flag](0037-a-prefix-scan-is-not-a-fold-with-a-flag.md)
- [ADR 0038: QSA needs stateful selection and launch composition](0038-qsa-needs-stateful-selection-and-launch-composition.md)
- [ADR 0039: Logical register pressure is not a physical bound](0039-logical-register-pressure-is-not-a-physical-bound.md)
- [ADR 0039: a single-writer state update is a proven store effect](0039-single-writer-state-store-is-a-store-effect.md)
- [ADR 0040: QSA utilization is a whole-Program roofline claim](0040-qsa-utilization-and-component-attribution.md)
- [ADR 0041: Resident top-k supports signed INT32 values](0041-resident-top-k-supports-signed-int32.md)
- [ADR 0042: Loop-carried top-k may batch two source tiles](0042-loop-carried-top-k-may-batch-two-source-tiles.md)
- [ADR 0043: FP32 MMA may explicitly request TF32 input precision](0043-fp32-mma-may-explicitly-request-tf32.md)
- [ADR 0044: Loop-carried top-k canonical lowering may use exact half-selection](0044-loop-carried-top-k-canonical-lowering-may-use-exact-half-selection.md)
- [ADR 0045: Triton lowering admits multiple explicit MMA DAG nodes](0045-triton-lowering-admits-multiple-explicit-mma-dag-nodes.md)
- [ADR 0046: Dynamic loop stop owns whole-grid work](0046-dynamic-loop-stop-owns-whole-grid-work.md)
- [ADR 0053: B300 is an exact target on the existing Triton path](0053-b300-is-an-exact-target.md)
- [ADR 0057: Metal and CLI harnesses use the existing Lab](0057-metal-and-harnesses-use-the-existing-lab.md)

- [0060: Release review provenance and historical exceptions](0060-release-review-provenance-and-historical-exceptions.md)

- [0058: External writer custody anchors](0058-external-writer-custody-anchors.md)
