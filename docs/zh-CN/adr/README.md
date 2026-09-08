# 设计决策：按问题阅读记录

[中文首页](../README.md) · [English index](../../en/adr/README.md) · [原目录](../../adr/README.md)

ADR 保存为什么作出某个决定。它记录当时的状态，不能仅凭编号认定已实现、已发布或 GPU 通过。中文页保留原文状态并提示有关后继；实际版本查[发布状态](../../../reports/current/STATUS.md)。

目录中有两个 0038、两个 0039 和两个 0053，主题不同，使用完整文件名区分，保留历史编号。

初学者先读 0001（职责分开）、0029（路线与题目）、0047（文档职责）、0048（AI 流程）和 0051（读取形状）。其余遇到具体问题再查。

| 编号 | 中文阅读 | English original |
| --- | --- | --- |
| 0001 | [编译器是核心，Lab 使用它](0001-compiler-first-with-dependent-lab.md) | [Make the compiler the product core and the research lab a dependent application](../../adr/0001-compiler-first-with-dependent-lab.md) |
| 0002 | [portfolio 是同一实验系统里的第二种 Study](0002-portfolio-as-second-study-variant.md) | [Portfolio is the second closed Study variant](../../adr/0002-portfolio-as-second-study-variant.md) |
| 0003 | [如果改用 Rust，先做结果一致的影子实现](0003-rust-shadow-engine-after-v3.md) | [Consider Rust only as a Corpus-equivalent v4 shadow engine](../../adr/0003-rust-shadow-engine-after-v3.md) |
| 0004 | [丰富的 AI 工具用于单产物优化](0004-tool-rich-artifact-optimization.md) | [Restore provider-default features only for artifact optimization](../../adr/0004-tool-rich-artifact-optimization.md) |
| 0005 | [按材料生命周期整理，保留旧路径](0005-forward-compatible-lifecycle-layout.md) | [Adopt lifecycle-first repository governance without moving history](../../adr/0005-forward-compatible-lifecycle-layout.md) |
| 0006 | [一轮可以交多个方案，计划的形状仍然固定](0006-candidate-sets-and-static-shapes.md) | [candidate sets, and a Schedule stays statically shaped](../../adr/0006-candidate-sets-and-static-shapes.md) |
| 0007 | [每个候选和发布版本都要认得准](0007-candidate-evidence-and-revision-witnesses.md) | [candidate evidence has one identity, and every frozen reference is a witness](../../adr/0007-candidate-evidence-and-revision-witnesses.md) |
| 0008 | [没有合格校准，就不能假装会排序](0008-calibration-coverage-gates-ranking.md) | [calibration coverage gates public ranking](../../adr/0008-calibration-coverage-gates-ranking.md) |
| 0009 | [多个候选装进一个固定提交文件](0009-live-candidate-set-envelope.md) | [live candidate sets use one sealed envelope](../../adr/0009-live-candidate-set-envelope.md) |
| 0010 | [先证明反馈有用，再考虑增加管理代理](0010-feedback-before-agent-orchestration.md) | [Establish feedback before adding agent orchestration](../../adr/0010-feedback-before-agent-orchestration.md) |
| 0011 | [校准真实的“三选二”决定](0011-finite-domain-ranking-calibration.md) | [calibrate the decision the Lab actually makes](../../adr/0011-finite-domain-ranking-calibration.md) |
| 0012 | [每个搜索后算对的候选都要留下瓶颈分析](0012-profile-each-correct-search-survivor.md) | [Profile each correctness-qualified search survivor](../../adr/0012-profile-each-correct-search-survivor.md) |
| 0013 | [没找到好方案，与没有拿到有效数据要分开](0013-two-part-scientific-estimand.md) | [Keep candidate failure separate from missing scientific data](../../adr/0013-two-part-scientific-estimand.md) |
| 0014 | [实验事件只能使用约定的词汇](0014-close-matched-event-vocabulary.md) | [Close the matched Run semantic event vocabulary](../../adr/0014-close-matched-event-vocabulary.md) |
| 0015 | [AI 推理强度也是实验条件](0015-reasoning-effort-is-treatment.md) | [Reasoning effort is an explicit treatment factor](../../adr/0015-reasoning-effort-is-treatment.md) |
| 0016 | [从头探索只给题目接口，不给现成答案](0016-clean-start-references-carry-contract-not-implementation.md) | [Clean-start references carry contract, not implementation](../../adr/0016-clean-start-references-carry-contract-not-implementation.md) |
| 0017 | [保存每轮真正给 AI 的参考材料](0017-retain-the-exact-provider-reference-bundle.md) | [Retain the exact provider reference bundle](../../adr/0017-retain-the-exact-provider-reference-bundle.md) |
| 0018 | [并列就不强行淘汰，测量要控制时间漂移](0018-ranking-is-a-preorder-and-calibration-controls-drift.md) | [ranking is a preorder and calibration controls drift](../../adr/0018-ranking-is-a-preorder-and-calibration-controls-drift.md) |
| 0019 | [生成的代码与选出的固定代码要分清](0019-lowering-generation-is-observable.md) | [Lowering generation is observable](../../adr/0019-lowering-generation-is-observable.md) |
| 0020 | [用 KDA 历史找缺口，分清局部机制和完整程序](0020-kda-deltas-are-an-external-expressibility-corpus.md) | [KDA deltas are an external expressibility corpus](../../adr/0020-kda-deltas-are-an-external-expressibility-corpus.md) |
| 0021 | [top-k 同时返回数值和原位置](0021-top-k-is-a-deterministic-indexed-selection-primitive.md) | [top-k is a deterministic indexed-selection primitive](../../adr/0021-top-k-is-a-deterministic-indexed-selection-primitive.md) |
| 0022 | [tanh 必须说清使用哪种实现](0022-tanh-names-its-target-implementation-contract.md) | [tanh names its target implementation contract](../../adr/0022-tanh-names-its-target-implementation-contract.md) |
| 0023 | [量化缩放系数与数据轴建立明确关系](0023-block-scales-are-axis-relations.md) | [block scales are axis relations](../../adr/0023-block-scales-are-axis-relations.md) |
| 0024 | [固定容量里，有效长度可以运行时给出](0024-runtime-valid-extents-are-buffer-relations.md) | [runtime-valid extents are buffer relations](../../adr/0024-runtime-valid-extents-are-buffer-relations.md) |
| 0025 | [不等长分组矩阵乘法可以用已有操作组合](0025-ragged-grouped-gemm-is-primitive-composition.md) | [ragged grouped GEMM is primitive composition](../../adr/0025-ragged-grouped-gemm-is-primitive-composition.md) |
| 0026 | [读哪个位置，可以由前面算出的索引决定](0026-runtime-indexed-loads-are-access-map-composition.md) | [runtime-indexed loads are AccessMap composition](../../adr/0026-runtime-indexed-loads-are-access-map-composition.md) |
| 0027 | [后端做不到的事，在检查阶段就说明](0027-backend-preconditions-are-assessment-findings.md) | [Backend preconditions are Assessment Findings](../../adr/0027-backend-preconditions-are-assessment-findings.md) |
| 0028 | [实验模板写设计，实际版本在运行前固定](0028-study-templates-defer-revision-binding-to-campaign-lock.md) | [Study templates defer revision binding to CampaignLock](../../adr/0028-study-templates-defer-revision-binding-to-campaign-lock.md) |
| 0029 | [生成路线只说明后端和入口，不兼职定义题目](0029-lowering-route-is-not-a-workload-profile.md) | [Lowering route is not a workload profile](../../adr/0029-lowering-route-is-not-a-workload-profile.md) |
| 0030 | [准备发布的人不能顺便批准自己](0030-compiler-release-approval-is-external.md) | [Compiler release approval is external to the release cycle](../../adr/0030-compiler-release-approval-is-external.md) |
| 0031 | [文件完整，不等于保管过程可信](0031-archive-integrity-is-not-filesystem-custody.md) | [Archive integrity is not filesystem custody](../../adr/0031-archive-integrity-is-not-filesystem-custody.md) |
| 0032 | [访问边界由真正被访问的数组决定](0032-access-boundaries-use-the-accessed-buffer.md) | [access boundaries use the accessed Buffer](../../adr/0032-access-boundaries-use-the-accessed-buffer.md) |
| 0033 | [TinyGEMM2 的真实输入字节由 Workload 负责](0033-workload-materialization-owns-tinygemm2-input-bytes.md) | [Workload materialization owns TinyGEMM2 input bytes](../../adr/0033-workload-materialization-owns-tinygemm2-input-bytes.md) |
| 0034 | [KDA 加权合并使用已有操作组合](0034-kda-weighted-combine-is-primitive-composition.md) | [KDA weighted combine is primitive composition](../../adr/0034-kda-weighted-combine-is-primitive-composition.md) |
| 0035 | [用调用者状态和原子加法分配唯一位置](0035-atomic-slot-reservation-is-state-plus-rmw.md) | [atomic slot reservation is state plus RMW](../../adr/0035-atomic-slot-reservation-is-state-plus-rmw.md) |
| 0036 | [只有从原子分配推得唯一位置，才允许普通索引写入](0036-atomic-reservation-proves-indexed-store-ownership.md) | [atomic reservation proves indexed-store ownership](../../adr/0036-atomic-reservation-proves-indexed-store-ownership.md) |
| 0037 | [前缀和要保留每一步，不能冒充总和](0037-a-prefix-scan-is-not-a-fold-with-a-flag.md) | [A prefix scan is not a fold with a flag](../../adr/0037-a-prefix-scan-is-not-a-fold-with-a-flag.md) |
| 0038 | [AKA 是发现问题的外部题库，不是正式验收语料](0038-aka-is-a-challenge-corpus-not-a-compiler-corpus.md) | [AKA is a challenge corpus, not a Compiler Corpus](../../adr/0038-aka-is-a-challenge-corpus-not-a-compiler-corpus.md) |
| 0038 | [QSA 需要跨块选择状态和多次启动组合](0038-qsa-needs-stateful-selection-and-launch-composition.md) | [QSA needs stateful selection and launch composition](../../adr/0038-qsa-needs-stateful-selection-and-launch-composition.md) |
| 0039 | [逻辑寄存器压力不能当成真实寄存器下界](0039-logical-register-pressure-is-not-a-physical-bound.md) | [Logical register pressure is not a physical bound](../../adr/0039-logical-register-pressure-is-not-a-physical-bound.md) |
| 0039 | [单写入者的原地更新使用已有 store](0039-single-writer-state-store-is-a-store-effect.md) | [a single-writer state update is a proven store effect](../../adr/0039-single-writer-state-store-is-a-store-effect.md) |
| 0040 | [QSA 利用率看完整程序，并说明分母来自哪里](0040-qsa-utilization-and-component-attribution.md) | [QSA utilization is a whole-Program roofline claim](../../adr/0040-qsa-utilization-and-component-attribution.md) |
| 0041 | [驻留 top-k 可以按有符号 INT32 排序](0041-resident-top-k-supports-signed-int32.md) | [Resident top-k supports signed INT32 values](../../adr/0041-resident-top-k-supports-signed-int32.md) |
| 0042 | [跨循环 top-k 可以每两个来源块合并一次](0042-loop-carried-top-k-may-batch-two-source-tiles.md) | [Loop-carried top-k may batch two source tiles](../../adr/0042-loop-carried-top-k-may-batch-two-source-tiles.md) |
| 0043 | [FP32 存储可以明确选择 TF32 乘法精度](0043-fp32-mma-may-explicitly-request-tf32.md) | [FP32 MMA may explicitly request TF32 input precision](../../adr/0043-fp32-mma-may-explicitly-request-tf32.md) |
| 0044 | [已有 top-k 语义允许更省比较工作的精确生成方式](0044-loop-carried-top-k-canonical-lowering-may-use-exact-half-selection.md) | [Loop-carried top-k canonical lowering may use exact half-selection](../../adr/0044-loop-carried-top-k-canonical-lowering-may-use-exact-half-selection.md) |
| 0045 | [一个计划可以明确写多个 MMA 节点](0045-triton-lowering-admits-multiple-explicit-mma-dag-nodes.md) | [Triton lowering admits multiple explicit MMA DAG nodes](../../adr/0045-triton-lowering-admits-multiple-explicit-mma-dag-nodes.md) |
| 0046 | [动态停止条件决定整个网格实际做多少工作](0046-dynamic-loop-stop-owns-whole-grid-work.md) | [Dynamic loop stop owns whole-grid work](../../adr/0046-dynamic-loop-stop-owns-whole-grid-work.md) |
| 0047 | [稳定规则、旧记录和当前状态分开维护](0047-documentation-separates-stable-history-and-current-views.md) | [Documentation separates stable rules, history, and current views](../../adr/0047-documentation-separates-stable-history-and-current-views.md) |
| 0048 | [给 AI 两份固定文件，用外部 Ralph 控制迭代](0048-agent-runs-use-task-agents-and-ralph-control.md) | [Agent Runs use TASK.md, AGENTS.md, and external Ralph control](../../adr/0048-agent-runs-use-task-agents-and-ralph-control.md) |
| 0049 | [发布过的执行器身份，即使本地没人引用也不能复用](0049-released-executor-descriptors-reserve-their-identities.md) | [released Executor descriptors reserve their identities](../../adr/0049-released-executor-descriptors-reserve-their-identities.md) |
| 0050 | [发布过的编译器 lock 也永久保留身份](0050-released-compiler-locks-reserve-their-identities.md) | [a released Compiler lock reserves its identity](../../adr/0050-released-compiler-locks-reserve-their-identities.md) |
| 0051 | [load 的结果形状必须符合地址实际读出的范围](0051-load-values-follow-the-access-domain.md) | [load values follow the access domain](../../adr/0051-load-values-follow-the-access-domain.md) |
| 0052 | [独立代理会话也可以审查编译器发布](0052-independent-agent-release-review.md) | [Independent agent sessions may review Compiler releases](../../adr/0052-independent-agent-release-review.md) |

- [0054：Lab 只使用 Ralph](0054-lab-uses-only-ralph.md)
- [0055：任务实现从通用 Lab 中分离](0055-task-implementations-live-outside-the-common-lab.md)
- [0056：每个候选都与固定基线成对测量](0056-fixed-baseline-paired-execution.md)

## 原始决策补充索引 / Additional original decisions

- [ADR 0053: B300 is an exact target on the existing Triton path](../../adr/0053-b300-is-an-exact-target.md)
- [ADR 0053: Study-bound external advisory candidate order](../../adr/0053-study-bound-advisory-cost-selection.md)
- [ADR 0057: Metal and CLI harnesses use the existing Lab](../../adr/0057-metal-and-harnesses-use-the-existing-lab.md)

- [0060: Release review provenance and historical exceptions](../../adr/0060-release-review-provenance-and-historical-exceptions.md)

- [0058: External writer custody anchors](../../adr/0058-external-writer-custody-anchors.md)
