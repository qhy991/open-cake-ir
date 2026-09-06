# Architecture decision records

[中文目录](../../zh-CN/adr/README.md) · [English home](../README.md)

ADRs retain the rationale and status of a decision. A proposal, accepted decision, released revision, and GPU-qualified artifact are different facts. Consult the original status and any successor, then the [generated release view](../../../reports/current/STATUS.md).

There are two historical records numbered 0038 and two numbered 0039. Their full filenames distinguish their subjects; this index does not renumber history.

Start with 0001, 0029, 0047, 0048, and 0051. Proposed means the stated draft work is permitted within its scope, accepted records an accepted decision, superseded points to a successor, and rejected retains the reason a proposal was refused. None replaces current task authorization.

| Number | English original | 中文阅读 |
| --- | --- | --- |
| 0001 | [Make the compiler the product core and the research lab a dependent application](../../adr/0001-compiler-first-with-dependent-lab.md) | [编译器是核心，Lab 使用它](../../zh-CN/adr/0001-compiler-first-with-dependent-lab.md) |
| 0002 | [Portfolio is the second closed Study variant](../../adr/0002-portfolio-as-second-study-variant.md) | [portfolio 是同一实验系统里的第二种 Study](../../zh-CN/adr/0002-portfolio-as-second-study-variant.md) |
| 0003 | [Consider Rust only as a Corpus-equivalent v4 shadow engine](../../adr/0003-rust-shadow-engine-after-v3.md) | [如果改用 Rust，先做结果一致的影子实现](../../zh-CN/adr/0003-rust-shadow-engine-after-v3.md) |
| 0004 | [Restore provider-default features only for artifact optimization](../../adr/0004-tool-rich-artifact-optimization.md) | [丰富的 AI 工具用于单产物优化](../../zh-CN/adr/0004-tool-rich-artifact-optimization.md) |
| 0005 | [Adopt lifecycle-first repository governance without moving history](../../adr/0005-forward-compatible-lifecycle-layout.md) | [按材料生命周期整理，保留旧路径](../../zh-CN/adr/0005-forward-compatible-lifecycle-layout.md) |
| 0006 | [candidate sets, and a Schedule stays statically shaped](../../adr/0006-candidate-sets-and-static-shapes.md) | [一轮可以交多个方案，计划的形状仍然固定](../../zh-CN/adr/0006-candidate-sets-and-static-shapes.md) |
| 0007 | [candidate evidence has one identity, and every frozen reference is a witness](../../adr/0007-candidate-evidence-and-revision-witnesses.md) | [每个候选和发布版本都要认得准](../../zh-CN/adr/0007-candidate-evidence-and-revision-witnesses.md) |
| 0008 | [calibration coverage gates public ranking](../../adr/0008-calibration-coverage-gates-ranking.md) | [没有合格校准，就不能假装会排序](../../zh-CN/adr/0008-calibration-coverage-gates-ranking.md) |
| 0009 | [live candidate sets use one sealed envelope](../../adr/0009-live-candidate-set-envelope.md) | [多个候选装进一个固定提交文件](../../zh-CN/adr/0009-live-candidate-set-envelope.md) |
| 0010 | [Establish feedback before adding agent orchestration](../../adr/0010-feedback-before-agent-orchestration.md) | [先证明反馈有用，再考虑增加管理代理](../../zh-CN/adr/0010-feedback-before-agent-orchestration.md) |
| 0011 | [calibrate the decision the Lab actually makes](../../adr/0011-finite-domain-ranking-calibration.md) | [校准真实的“三选二”决定](../../zh-CN/adr/0011-finite-domain-ranking-calibration.md) |
| 0012 | [Profile each correctness-qualified search survivor](../../adr/0012-profile-each-correct-search-survivor.md) | [每个搜索后算对的候选都要留下瓶颈分析](../../zh-CN/adr/0012-profile-each-correct-search-survivor.md) |
| 0013 | [Keep candidate failure separate from missing scientific data](../../adr/0013-two-part-scientific-estimand.md) | [没找到好方案，与没有拿到有效数据要分开](../../zh-CN/adr/0013-two-part-scientific-estimand.md) |
| 0014 | [Close the matched Run semantic event vocabulary](../../adr/0014-close-matched-event-vocabulary.md) | [实验事件只能使用约定的词汇](../../zh-CN/adr/0014-close-matched-event-vocabulary.md) |
| 0015 | [Reasoning effort is an explicit treatment factor](../../adr/0015-reasoning-effort-is-treatment.md) | [AI 推理强度也是实验条件](../../zh-CN/adr/0015-reasoning-effort-is-treatment.md) |
| 0016 | [Clean-start references carry contract, not implementation](../../adr/0016-clean-start-references-carry-contract-not-implementation.md) | [从头探索只给题目接口，不给现成答案](../../zh-CN/adr/0016-clean-start-references-carry-contract-not-implementation.md) |
| 0017 | [Retain the exact provider reference bundle](../../adr/0017-retain-the-exact-provider-reference-bundle.md) | [保存每轮真正给 AI 的参考材料](../../zh-CN/adr/0017-retain-the-exact-provider-reference-bundle.md) |
| 0018 | [ranking is a preorder and calibration controls drift](../../adr/0018-ranking-is-a-preorder-and-calibration-controls-drift.md) | [并列就不强行淘汰，测量要控制时间漂移](../../zh-CN/adr/0018-ranking-is-a-preorder-and-calibration-controls-drift.md) |
| 0019 | [Lowering generation is observable](../../adr/0019-lowering-generation-is-observable.md) | [生成的代码与选出的固定代码要分清](../../zh-CN/adr/0019-lowering-generation-is-observable.md) |
| 0020 | [KDA deltas are an external expressibility corpus](../../adr/0020-kda-deltas-are-an-external-expressibility-corpus.md) | [用 KDA 历史找缺口，分清局部机制和完整程序](../../zh-CN/adr/0020-kda-deltas-are-an-external-expressibility-corpus.md) |
| 0021 | [top-k is a deterministic indexed-selection primitive](../../adr/0021-top-k-is-a-deterministic-indexed-selection-primitive.md) | [top-k 同时返回数值和原位置](../../zh-CN/adr/0021-top-k-is-a-deterministic-indexed-selection-primitive.md) |
| 0022 | [tanh names its target implementation contract](../../adr/0022-tanh-names-its-target-implementation-contract.md) | [tanh 必须说清使用哪种实现](../../zh-CN/adr/0022-tanh-names-its-target-implementation-contract.md) |
| 0023 | [block scales are axis relations](../../adr/0023-block-scales-are-axis-relations.md) | [量化缩放系数与数据轴建立明确关系](../../zh-CN/adr/0023-block-scales-are-axis-relations.md) |
| 0024 | [runtime-valid extents are buffer relations](../../adr/0024-runtime-valid-extents-are-buffer-relations.md) | [固定容量里，有效长度可以运行时给出](../../zh-CN/adr/0024-runtime-valid-extents-are-buffer-relations.md) |
| 0025 | [ragged grouped GEMM is primitive composition](../../adr/0025-ragged-grouped-gemm-is-primitive-composition.md) | [不等长分组矩阵乘法可以用已有操作组合](../../zh-CN/adr/0025-ragged-grouped-gemm-is-primitive-composition.md) |
| 0026 | [runtime-indexed loads are AccessMap composition](../../adr/0026-runtime-indexed-loads-are-access-map-composition.md) | [读哪个位置，可以由前面算出的索引决定](../../zh-CN/adr/0026-runtime-indexed-loads-are-access-map-composition.md) |
| 0027 | [Backend preconditions are Assessment Findings](../../adr/0027-backend-preconditions-are-assessment-findings.md) | [后端做不到的事，在检查阶段就说明](../../zh-CN/adr/0027-backend-preconditions-are-assessment-findings.md) |
| 0028 | [Study templates defer revision binding to CampaignLock](../../adr/0028-study-templates-defer-revision-binding-to-campaign-lock.md) | [实验模板写设计，实际版本在运行前固定](../../zh-CN/adr/0028-study-templates-defer-revision-binding-to-campaign-lock.md) |
| 0029 | [Lowering route is not a workload profile](../../adr/0029-lowering-route-is-not-a-workload-profile.md) | [生成路线只说明后端和入口，不兼职定义题目](../../zh-CN/adr/0029-lowering-route-is-not-a-workload-profile.md) |
| 0030 | [Compiler release approval is external to the release cycle](../../adr/0030-compiler-release-approval-is-external.md) | [准备发布的人不能顺便批准自己](../../zh-CN/adr/0030-compiler-release-approval-is-external.md) |
| 0031 | [Archive integrity is not filesystem custody](../../adr/0031-archive-integrity-is-not-filesystem-custody.md) | [文件完整，不等于保管过程可信](../../zh-CN/adr/0031-archive-integrity-is-not-filesystem-custody.md) |
| 0032 | [access boundaries use the accessed Buffer](../../adr/0032-access-boundaries-use-the-accessed-buffer.md) | [访问边界由真正被访问的数组决定](../../zh-CN/adr/0032-access-boundaries-use-the-accessed-buffer.md) |
| 0033 | [Workload materialization owns TinyGEMM2 input bytes](../../adr/0033-workload-materialization-owns-tinygemm2-input-bytes.md) | [TinyGEMM2 的真实输入字节由 Workload 负责](../../zh-CN/adr/0033-workload-materialization-owns-tinygemm2-input-bytes.md) |
| 0034 | [KDA weighted combine is primitive composition](../../adr/0034-kda-weighted-combine-is-primitive-composition.md) | [KDA 加权合并使用已有操作组合](../../zh-CN/adr/0034-kda-weighted-combine-is-primitive-composition.md) |
| 0035 | [atomic slot reservation is state plus RMW](../../adr/0035-atomic-slot-reservation-is-state-plus-rmw.md) | [用调用者状态和原子加法分配唯一位置](../../zh-CN/adr/0035-atomic-slot-reservation-is-state-plus-rmw.md) |
| 0036 | [atomic reservation proves indexed-store ownership](../../adr/0036-atomic-reservation-proves-indexed-store-ownership.md) | [只有从原子分配推得唯一位置，才允许普通索引写入](../../zh-CN/adr/0036-atomic-reservation-proves-indexed-store-ownership.md) |
| 0037 | [A prefix scan is not a fold with a flag](../../adr/0037-a-prefix-scan-is-not-a-fold-with-a-flag.md) | [前缀和要保留每一步，不能冒充总和](../../zh-CN/adr/0037-a-prefix-scan-is-not-a-fold-with-a-flag.md) |
| 0038 | [AKA is a challenge corpus, not a Compiler Corpus](../../adr/0038-aka-is-a-challenge-corpus-not-a-compiler-corpus.md) | [AKA 是发现问题的外部题库，不是正式验收语料](../../zh-CN/adr/0038-aka-is-a-challenge-corpus-not-a-compiler-corpus.md) |
| 0038 | [QSA needs stateful selection and launch composition](../../adr/0038-qsa-needs-stateful-selection-and-launch-composition.md) | [QSA 需要跨块选择状态和多次启动组合](../../zh-CN/adr/0038-qsa-needs-stateful-selection-and-launch-composition.md) |
| 0039 | [Logical register pressure is not a physical bound](../../adr/0039-logical-register-pressure-is-not-a-physical-bound.md) | [逻辑寄存器压力不能当成真实寄存器下界](../../zh-CN/adr/0039-logical-register-pressure-is-not-a-physical-bound.md) |
| 0039 | [a single-writer state update is a proven store effect](../../adr/0039-single-writer-state-store-is-a-store-effect.md) | [单写入者的原地更新使用已有 store](../../zh-CN/adr/0039-single-writer-state-store-is-a-store-effect.md) |
| 0040 | [QSA utilization is a whole-Program roofline claim](../../adr/0040-qsa-utilization-and-component-attribution.md) | [QSA 利用率看完整程序，并说明分母来自哪里](../../zh-CN/adr/0040-qsa-utilization-and-component-attribution.md) |
| 0041 | [Resident top-k supports signed INT32 values](../../adr/0041-resident-top-k-supports-signed-int32.md) | [驻留 top-k 可以按有符号 INT32 排序](../../zh-CN/adr/0041-resident-top-k-supports-signed-int32.md) |
| 0042 | [Loop-carried top-k may batch two source tiles](../../adr/0042-loop-carried-top-k-may-batch-two-source-tiles.md) | [跨循环 top-k 可以每两个来源块合并一次](../../zh-CN/adr/0042-loop-carried-top-k-may-batch-two-source-tiles.md) |
| 0043 | [FP32 MMA may explicitly request TF32 input precision](../../adr/0043-fp32-mma-may-explicitly-request-tf32.md) | [FP32 存储可以明确选择 TF32 乘法精度](../../zh-CN/adr/0043-fp32-mma-may-explicitly-request-tf32.md) |
| 0044 | [Loop-carried top-k canonical lowering may use exact half-selection](../../adr/0044-loop-carried-top-k-canonical-lowering-may-use-exact-half-selection.md) | [已有 top-k 语义允许更省比较工作的精确生成方式](../../zh-CN/adr/0044-loop-carried-top-k-canonical-lowering-may-use-exact-half-selection.md) |
| 0045 | [Triton lowering admits multiple explicit MMA DAG nodes](../../adr/0045-triton-lowering-admits-multiple-explicit-mma-dag-nodes.md) | [一个计划可以明确写多个 MMA 节点](../../zh-CN/adr/0045-triton-lowering-admits-multiple-explicit-mma-dag-nodes.md) |
| 0046 | [Dynamic loop stop owns whole-grid work](../../adr/0046-dynamic-loop-stop-owns-whole-grid-work.md) | [动态停止条件决定整个网格实际做多少工作](../../zh-CN/adr/0046-dynamic-loop-stop-owns-whole-grid-work.md) |
| 0047 | [Documentation separates stable rules, history, and current views](../../adr/0047-documentation-separates-stable-history-and-current-views.md) | [稳定规则、旧记录和当前状态分开维护](../../zh-CN/adr/0047-documentation-separates-stable-history-and-current-views.md) |
| 0048 | [Agent Runs use TASK.md, AGENTS.md, and external Ralph control](../../adr/0048-agent-runs-use-task-agents-and-ralph-control.md) | [给 AI 两份固定文件，用外部 Ralph 控制迭代](../../zh-CN/adr/0048-agent-runs-use-task-agents-and-ralph-control.md) |
| 0049 | [released Executor descriptors reserve their identities](../../adr/0049-released-executor-descriptors-reserve-their-identities.md) | [发布过的执行器身份，即使本地没人引用也不能复用](../../zh-CN/adr/0049-released-executor-descriptors-reserve-their-identities.md) |
| 0050 | [a released Compiler lock reserves its identity](../../adr/0050-released-compiler-locks-reserve-their-identities.md) | [发布过的编译器 lock 也永久保留身份](../../zh-CN/adr/0050-released-compiler-locks-reserve-their-identities.md) |
| 0051 | [load values follow the access domain](../../adr/0051-load-values-follow-the-access-domain.md) | [load 的结果形状必须符合地址实际读出的范围](../../zh-CN/adr/0051-load-values-follow-the-access-domain.md) |
| 0052 | [Independent agent sessions may review Compiler releases](../../adr/0052-independent-agent-release-review.md) | [独立代理会话也可以审查编译器发布](../../zh-CN/adr/0052-independent-agent-release-review.md) |
