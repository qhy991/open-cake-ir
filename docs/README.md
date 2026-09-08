# 文档总目录 / Documentation catalog

[中文入门](zh-CN/README.md) · [English start](en/README.md) · [职责与文档归属](../CONTEXT-MAP.md)

B300： [中文](B300.md) · [English](en/B300.md)

这里把原始 Markdown 文档对应到中文和英文阅读入口。初学者先按语言首页的路线读，不需要从第一份历史报告读到最后一份。

This catalog pairs original Markdown documents with Chinese and English reading routes. Start with a language home rather than reading the archive sequentially.

中文阅读版用例子解释背景、规则和结果；长篇技术表、精确来源和逐项证据保留在链接的原始材料中，并非逐句译本。英文原文不搬迁，已有中文正文不重复复制；翻译和导读不成为第二份合同或实验结论。术语仍由 [GLOSSARY](GLOSSARY.md) 负责，当前发布只看[生成状态](../reports/current/STATUS.md)。

Reading companions explain the material in simpler language. Detailed tables and evidence locators remain in the linked original. Original paths and dated conclusions stay intact; translations do not create new contracts or claims.

## 入门与使用 / Guides

| 主题 / Topic | 中文 | English |
| --- | --- | --- |
| 验收规则：做到什么，才可以进入下一步 | [阅读](zh-CN/ACCEPTANCE_GATES.md) | [Read](ACCEPTANCE_GATES.md) |
| 系统全貌：怎样把一个想法变成可验证的 GPU 程序 | [阅读](ARCHITECTURE.md) | [Read](en/ARCHITECTURE.md) |
| 第一次使用：先看懂一个乘加计划 | [阅读](GETTING_STARTED.md) | [Read](en/GETTING_STARTED.md) |
| 术语表：英文名与普通话解释 | [阅读](GLOSSARY.md) | [Read](en/GLOSSARY.md) |
| 论文对照约定：论文说了什么，本项目证明了什么 | [阅读](zh-CN/PAPER_CONTRACT.md) | [Read](PAPER_CONTRACT.md) |
| 执行手册：从准备到复查，一步步做什么 | [阅读](zh-CN/RUNBOOK.md) | [Read](RUNBOOK.md) |
| 用 Python 编写执行计划 | [阅读](zh-CN/PYTHON_FRONTEND.md) | [Read](en/PYTHON_FRONTEND.md) |
| Apple M1 Pro 的 TaskLab 归一化任务与测量 | [阅读](metal.zh-CN.md) | [Read](metal.md) |
| 当前 IR 详解：结构、语义与代码组织 | [阅读](IR_GUIDE.md) | [Read](en/IR_GUIDE.md) |
| 旧顶层设计入口：现在应该读哪里 | [阅读](zh-CN/TOP_LEVEL_DESIGN.md) | [Read](TOP_LEVEL_DESIGN.md) |
| Open Cake 中文 Wiki | [阅读](wiki/README.md) | [Read](en/wiki/README.md) |
| 一次实验怎样进行 | [阅读](wiki/experiments.md) | [Read](en/wiki/experiments.md) |
| 文档怎样保持准确 | [阅读](wiki/maintaining.md) | [Read](en/wiki/maintaining.md) |
| 常见算子：它们到底在算什么 | [阅读](wiki/operators.md) | [Read](en/wiki/operators.md) |
| IR 基本操作：拼出计算的积木 | [阅读](wiki/primitives.md) | [Read](en/wiki/primitives.md) |
| 怎样回放旧任务 | [阅读](wiki/replay.md) | [Read](en/wiki/replay.md) |
| 怎样读结果，也怎样排错 | [阅读](wiki/results.md) | [Read](en/wiki/results.md) |
| 读懂一份执行计划 | [阅读](wiki/schedule.md) | [Read](en/wiki/schedule.md) |
| Workload 目录：项目里有哪些完整任务定义 | [阅读](wiki/workloads.md) | [Read](en/wiki/workloads.md) |
| 独立 Tile Workload：共用 ABI、CPU oracle 与匹配基线 | [阅读](TILE_WORKLOADS.md) | [Read](en/TILE_WORKLOADS.md) |
| Triton TileLoop：两层循环的作用域、累积与写回 | [阅读](TRITON_LOOP_SCOPES.md) | [Read](en/TRITON_LOOP_SCOPES.md) |
| 同后端 Triton 配对：共同基线、隔离构建与 Evaluation | [阅读](PAIRED_TRITON.md) | [Read](en/PAIRED_TRITON.md) |

## 模块职责 / Contexts

| 主题 / Topic | 中文 | English |
| --- | --- | --- |
| Compiler：检查和翻译执行计划 | [阅读](zh-CN/contexts/compiler/CONTEXT.md) | [Read](contexts/compiler/CONTEXT.md) |
| Evaluation：核对答案，再公平地测量 | [阅读](zh-CN/contexts/evaluation/CONTEXT.md) | [Read](contexts/evaluation/CONTEXT.md) |
| Evidence：保存事实，让别人能复查 | [阅读](zh-CN/contexts/evidence/CONTEXT.md) | [Read](contexts/evidence/CONTEXT.md) |
| Research Lab：组织一次有规则的实验 | [阅读](zh-CN/contexts/lab/CONTEXT.md) | [Read](contexts/lab/CONTEXT.md) |

## 设计记录 / Decisions

| 主题 / Topic | 中文 | English |
| --- | --- | --- |
| ADR 0001：编译器是核心，Lab 使用它 | [阅读](zh-CN/adr/0001-compiler-first-with-dependent-lab.md) | [Read](adr/0001-compiler-first-with-dependent-lab.md) |
| ADR 0002：portfolio 是同一实验系统里的第二种 Study | [阅读](zh-CN/adr/0002-portfolio-as-second-study-variant.md) | [Read](adr/0002-portfolio-as-second-study-variant.md) |
| ADR 0003：如果改用 Rust，先做结果一致的影子实现 | [阅读](zh-CN/adr/0003-rust-shadow-engine-after-v3.md) | [Read](adr/0003-rust-shadow-engine-after-v3.md) |
| ADR 0004：丰富的 AI 工具用于单产物优化 | [阅读](zh-CN/adr/0004-tool-rich-artifact-optimization.md) | [Read](adr/0004-tool-rich-artifact-optimization.md) |
| ADR 0005：按材料生命周期整理，保留旧路径 | [阅读](zh-CN/adr/0005-forward-compatible-lifecycle-layout.md) | [Read](adr/0005-forward-compatible-lifecycle-layout.md) |
| ADR 0006：一轮可以交多个方案，计划的形状仍然固定 | [阅读](zh-CN/adr/0006-candidate-sets-and-static-shapes.md) | [Read](adr/0006-candidate-sets-and-static-shapes.md) |
| ADR 0007：每个候选和发布版本都要认得准 | [阅读](zh-CN/adr/0007-candidate-evidence-and-revision-witnesses.md) | [Read](adr/0007-candidate-evidence-and-revision-witnesses.md) |
| ADR 0008：没有合格校准，就不能假装会排序 | [阅读](zh-CN/adr/0008-calibration-coverage-gates-ranking.md) | [Read](adr/0008-calibration-coverage-gates-ranking.md) |
| ADR 0009：多个候选装进一个固定提交文件 | [阅读](zh-CN/adr/0009-live-candidate-set-envelope.md) | [Read](adr/0009-live-candidate-set-envelope.md) |
| ADR 0010：先证明反馈有用，再考虑增加管理代理 | [阅读](zh-CN/adr/0010-feedback-before-agent-orchestration.md) | [Read](adr/0010-feedback-before-agent-orchestration.md) |
| ADR 0011：校准真实的“三选二”决定 | [阅读](zh-CN/adr/0011-finite-domain-ranking-calibration.md) | [Read](adr/0011-finite-domain-ranking-calibration.md) |
| ADR 0012：每个搜索后算对的候选都要留下瓶颈分析 | [阅读](zh-CN/adr/0012-profile-each-correct-search-survivor.md) | [Read](adr/0012-profile-each-correct-search-survivor.md) |
| ADR 0013：没找到好方案，与没有拿到有效数据要分开 | [阅读](zh-CN/adr/0013-two-part-scientific-estimand.md) | [Read](adr/0013-two-part-scientific-estimand.md) |
| ADR 0014：实验事件只能使用约定的词汇 | [阅读](zh-CN/adr/0014-close-matched-event-vocabulary.md) | [Read](adr/0014-close-matched-event-vocabulary.md) |
| ADR 0015：AI 推理强度也是实验条件 | [阅读](zh-CN/adr/0015-reasoning-effort-is-treatment.md) | [Read](adr/0015-reasoning-effort-is-treatment.md) |
| ADR 0016：从头探索只给题目接口，不给现成答案 | [阅读](zh-CN/adr/0016-clean-start-references-carry-contract-not-implementation.md) | [Read](adr/0016-clean-start-references-carry-contract-not-implementation.md) |
| ADR 0017：保存每轮真正给 AI 的参考材料 | [阅读](zh-CN/adr/0017-retain-the-exact-provider-reference-bundle.md) | [Read](adr/0017-retain-the-exact-provider-reference-bundle.md) |
| ADR 0018：并列就不强行淘汰，测量要控制时间漂移 | [阅读](zh-CN/adr/0018-ranking-is-a-preorder-and-calibration-controls-drift.md) | [Read](adr/0018-ranking-is-a-preorder-and-calibration-controls-drift.md) |
| ADR 0019：生成的代码与选出的固定代码要分清 | [阅读](zh-CN/adr/0019-lowering-generation-is-observable.md) | [Read](adr/0019-lowering-generation-is-observable.md) |
| ADR 0020：用 KDA 历史找缺口，分清局部机制和完整程序 | [阅读](zh-CN/adr/0020-kda-deltas-are-an-external-expressibility-corpus.md) | [Read](adr/0020-kda-deltas-are-an-external-expressibility-corpus.md) |
| ADR 0021：top-k 同时返回数值和原位置 | [阅读](zh-CN/adr/0021-top-k-is-a-deterministic-indexed-selection-primitive.md) | [Read](adr/0021-top-k-is-a-deterministic-indexed-selection-primitive.md) |
| ADR 0022：tanh 必须说清使用哪种实现 | [阅读](zh-CN/adr/0022-tanh-names-its-target-implementation-contract.md) | [Read](adr/0022-tanh-names-its-target-implementation-contract.md) |
| ADR 0023：量化缩放系数与数据轴建立明确关系 | [阅读](zh-CN/adr/0023-block-scales-are-axis-relations.md) | [Read](adr/0023-block-scales-are-axis-relations.md) |
| ADR 0024：固定容量里，有效长度可以运行时给出 | [阅读](zh-CN/adr/0024-runtime-valid-extents-are-buffer-relations.md) | [Read](adr/0024-runtime-valid-extents-are-buffer-relations.md) |
| ADR 0025：不等长分组矩阵乘法可以用已有操作组合 | [阅读](zh-CN/adr/0025-ragged-grouped-gemm-is-primitive-composition.md) | [Read](adr/0025-ragged-grouped-gemm-is-primitive-composition.md) |
| ADR 0026：读哪个位置，可以由前面算出的索引决定 | [阅读](zh-CN/adr/0026-runtime-indexed-loads-are-access-map-composition.md) | [Read](adr/0026-runtime-indexed-loads-are-access-map-composition.md) |
| ADR 0027：后端做不到的事，在检查阶段就说明 | [阅读](zh-CN/adr/0027-backend-preconditions-are-assessment-findings.md) | [Read](adr/0027-backend-preconditions-are-assessment-findings.md) |
| ADR 0028：实验模板写设计，实际版本在运行前固定 | [阅读](zh-CN/adr/0028-study-templates-defer-revision-binding-to-campaign-lock.md) | [Read](adr/0028-study-templates-defer-revision-binding-to-campaign-lock.md) |
| ADR 0029：生成路线只说明后端和入口，不兼职定义题目 | [阅读](zh-CN/adr/0029-lowering-route-is-not-a-workload-profile.md) | [Read](adr/0029-lowering-route-is-not-a-workload-profile.md) |
| ADR 0030：准备发布的人不能顺便批准自己 | [阅读](zh-CN/adr/0030-compiler-release-approval-is-external.md) | [Read](adr/0030-compiler-release-approval-is-external.md) |
| ADR 0031：文件完整，不等于保管过程可信 | [阅读](zh-CN/adr/0031-archive-integrity-is-not-filesystem-custody.md) | [Read](adr/0031-archive-integrity-is-not-filesystem-custody.md) |
| ADR 0032：访问边界由真正被访问的数组决定 | [阅读](zh-CN/adr/0032-access-boundaries-use-the-accessed-buffer.md) | [Read](adr/0032-access-boundaries-use-the-accessed-buffer.md) |
| ADR 0033：TinyGEMM2 的真实输入字节由 Workload 负责 | [阅读](zh-CN/adr/0033-workload-materialization-owns-tinygemm2-input-bytes.md) | [Read](adr/0033-workload-materialization-owns-tinygemm2-input-bytes.md) |
| ADR 0034：KDA 加权合并使用已有操作组合 | [阅读](zh-CN/adr/0034-kda-weighted-combine-is-primitive-composition.md) | [Read](adr/0034-kda-weighted-combine-is-primitive-composition.md) |
| ADR 0035：用调用者状态和原子加法分配唯一位置 | [阅读](zh-CN/adr/0035-atomic-slot-reservation-is-state-plus-rmw.md) | [Read](adr/0035-atomic-slot-reservation-is-state-plus-rmw.md) |
| ADR 0036：只有从原子分配推得唯一位置，才允许普通索引写入 | [阅读](zh-CN/adr/0036-atomic-reservation-proves-indexed-store-ownership.md) | [Read](adr/0036-atomic-reservation-proves-indexed-store-ownership.md) |
| ADR 0037：前缀和要保留每一步，不能冒充总和 | [阅读](zh-CN/adr/0037-a-prefix-scan-is-not-a-fold-with-a-flag.md) | [Read](adr/0037-a-prefix-scan-is-not-a-fold-with-a-flag.md) |
| ADR 0038：AKA 是发现问题的外部题库，不是正式验收语料 | [阅读](zh-CN/adr/0038-aka-is-a-challenge-corpus-not-a-compiler-corpus.md) | [Read](adr/0038-aka-is-a-challenge-corpus-not-a-compiler-corpus.md) |
| ADR 0038：QSA 需要跨块选择状态和多次启动组合 | [阅读](zh-CN/adr/0038-qsa-needs-stateful-selection-and-launch-composition.md) | [Read](adr/0038-qsa-needs-stateful-selection-and-launch-composition.md) |
| ADR 0039：逻辑寄存器压力不能当成真实寄存器下界 | [阅读](zh-CN/adr/0039-logical-register-pressure-is-not-a-physical-bound.md) | [Read](adr/0039-logical-register-pressure-is-not-a-physical-bound.md) |
| ADR 0039：单写入者的原地更新使用已有 store | [阅读](zh-CN/adr/0039-single-writer-state-store-is-a-store-effect.md) | [Read](adr/0039-single-writer-state-store-is-a-store-effect.md) |
| ADR 0040：QSA 利用率看完整程序，并说明分母来自哪里 | [阅读](zh-CN/adr/0040-qsa-utilization-and-component-attribution.md) | [Read](adr/0040-qsa-utilization-and-component-attribution.md) |
| ADR 0041：驻留 top-k 可以按有符号 INT32 排序 | [阅读](zh-CN/adr/0041-resident-top-k-supports-signed-int32.md) | [Read](adr/0041-resident-top-k-supports-signed-int32.md) |
| ADR 0042：跨循环 top-k 可以每两个来源块合并一次 | [阅读](zh-CN/adr/0042-loop-carried-top-k-may-batch-two-source-tiles.md) | [Read](adr/0042-loop-carried-top-k-may-batch-two-source-tiles.md) |
| ADR 0043：FP32 存储可以明确选择 TF32 乘法精度 | [阅读](zh-CN/adr/0043-fp32-mma-may-explicitly-request-tf32.md) | [Read](adr/0043-fp32-mma-may-explicitly-request-tf32.md) |
| ADR 0044：已有 top-k 语义允许更省比较工作的精确生成方式 | [阅读](zh-CN/adr/0044-loop-carried-top-k-canonical-lowering-may-use-exact-half-selection.md) | [Read](adr/0044-loop-carried-top-k-canonical-lowering-may-use-exact-half-selection.md) |
| ADR 0045：一个计划可以明确写多个 MMA 节点 | [阅读](zh-CN/adr/0045-triton-lowering-admits-multiple-explicit-mma-dag-nodes.md) | [Read](adr/0045-triton-lowering-admits-multiple-explicit-mma-dag-nodes.md) |
| ADR 0046：动态停止条件决定整个网格实际做多少工作 | [阅读](zh-CN/adr/0046-dynamic-loop-stop-owns-whole-grid-work.md) | [Read](adr/0046-dynamic-loop-stop-owns-whole-grid-work.md) |
| ADR 0047：稳定规则、旧记录和当前状态分开维护 | [阅读](zh-CN/adr/0047-documentation-separates-stable-history-and-current-views.md) | [Read](adr/0047-documentation-separates-stable-history-and-current-views.md) |
| ADR 0048：给 AI 两份固定文件，用外部 Ralph 控制迭代 | [阅读](zh-CN/adr/0048-agent-runs-use-task-agents-and-ralph-control.md) | [Read](adr/0048-agent-runs-use-task-agents-and-ralph-control.md) |
| ADR 0049：发布过的执行器身份，即使本地没人引用也不能复用 | [阅读](zh-CN/adr/0049-released-executor-descriptors-reserve-their-identities.md) | [Read](adr/0049-released-executor-descriptors-reserve-their-identities.md) |
| ADR 0050：发布过的编译器 lock 也永久保留身份 | [阅读](zh-CN/adr/0050-released-compiler-locks-reserve-their-identities.md) | [Read](adr/0050-released-compiler-locks-reserve-their-identities.md) |
| ADR 0051：load 的结果形状必须符合地址实际读出的范围 | [阅读](zh-CN/adr/0051-load-values-follow-the-access-domain.md) | [Read](adr/0051-load-values-follow-the-access-domain.md) |
| ADR 0052：独立代理会话也可以审查编译器发布 | [阅读](zh-CN/adr/0052-independent-agent-release-review.md) | [Read](adr/0052-independent-agent-release-review.md) |
| 设计决策：按问题阅读记录 | [阅读](zh-CN/adr/README.md) | [Read](en/adr/README.md) |

## 历史调查与报告 / Historical surveys and reports

| 主题 / Topic | 中文 | English |
| --- | --- | --- |
| 从 FMA 再审修复读取与向量检查 | [阅读](ACCESS_DOMAIN_REPAIR_20260906.md) | [Read](en/ACCESS_DOMAIN_REPAIR_20260906.md) |
| 仿射计算：一次完整的 B200 正确性检查 | [阅读](AFFINE_PARENT_B200_CANARY_20260906.md) | [Read](en/AFFINE_PARENT_B200_CANARY_20260906.md) |
| FMA 在 B200 上的正确性：两个 kernel、十二组输入 | [阅读](zh-CN/AKA_FMA_B200_CORRECTNESS_20260904.md) | [Read](AKA_FMA_B200_CORRECTNESS_20260904.md) |
| FMA 实现检查点：代码已有，发布仍待审 | [阅读](zh-CN/AKA_FMA_IMPLEMENTATION_20260904.md) | [Read](AKA_FMA_IMPLEMENTATION_20260904.md) |
| 补了 FMA 后，十二个完整父项还剩什么问题 | [阅读](zh-CN/AKA_FMA_PARENT_REAUDIT_V41_20260906.md) | [Read](AKA_FMA_PARENT_REAUDIT_V41_20260906.md) |
| FMA v41 发布审查：静态版本正式完成 | [阅读](zh-CN/AKA_FMA_RELEASE_REVIEW_20260904.md) | [Read](AKA_FMA_RELEASE_REVIEW_20260904.md) |
| AKA v6 设计审查：先接受三个范围明确的数学操作 | [阅读](zh-CN/AKA_IR_OWNER_REVIEW_20260904.md) | [Read](AKA_IR_OWNER_REVIEW_20260904.md) |
| 100 个便携父算子的历史 IR 审查 | [阅读](zh-CN/AKA_PORTABLE_PARENT_V2_IR_REVIEW_20260829.md) | [Read](AKA_PORTABLE_PARENT_V2_IR_REVIEW_20260829.md) |
| AKA 大批量审查：677 道题怎样筛到 56 个固定样本 | [阅读](zh-CN/AKA_QUALIFIED_IR_REVIEW_AND_LAB_PLAN_20260903.md) | [Read](en/AKA_QUALIFIED_IR_REVIEW_AND_LAB_PLAN_20260903.md) |
| FMA 与余弦的定向探测：先问是否真的缺一种运算 | [阅读](zh-CN/AKA_TARGETED_IR_PROBES_20260901.md) | [Read](AKA_TARGETED_IR_PROBES_20260901.md) |
| 用实测检验静态分析：能被推翻的尺子才有用 | [阅读](zh-CN/ANALYSIS_CALIBRATION.md) | [Read](ANALYSIS_CALIBRATION.md) |
| 历史审计问题册：哪些修了，哪些不是缺陷 | [阅读](zh-CN/AUDIT_FINDINGS_20260825.md) | [Read](AUDIT_FINDINGS_20260825.md) |
| IR 能表达什么：从真实优化工作找需求 | [阅读](zh-CN/IR_COVERAGE.md) | [Read](IR_COVERAGE.md) |
| 注意力调查：有些工作安排到运行时才知道 | [阅读](zh-CN/IR_COVERAGE_ATTENTION.md) | [Read](IR_COVERAGE_ATTENTION.md) |
| CUTLASS 调查：除了数据大小，还要描述协作规则 | [阅读](zh-CN/IR_COVERAGE_CUTLASS.md) | [Read](IR_COVERAGE_CUTLASS.md) |
| Megakernel 与 KDA 调查：一个固定计划装不下动态程序 | [阅读](zh-CN/IR_COVERAGE_MEGAKERNEL.md) | [Read](IR_COVERAGE_MEGAKERNEL.md) |
| DeepGEMM 调查：量化不只是换一种数字类型 | [阅读](zh-CN/IR_COVERAGE_QUANTIZED.md) | [Read](IR_COVERAGE_QUANTIZED.md) |
| 多份调查的共同结论：缺口要分三种处理 | [阅读](zh-CN/IR_SURVEY_SYNTHESIS.md) | [Read](IR_SURVEY_SYNTHESIS.md) |
| 历史迁移计划：把实验脚本集合变成编译器产品 | [阅读](zh-CN/MIGRATION_PLAN.md) | [Read](MIGRATION_PLAN.md) |
| 原地状态更新：前四次交接结果与后继准备 | [阅读](zh-CN/STATE_STORE_B200_CORRECTNESS_20260902.md) | [Read](STATE_STORE_B200_CORRECTNESS_20260902.md) |
| 2026 年第 35 周：算子优化与系统证据 | [阅读](zh-CN/WEEKLY_PROGRESS_2026-W35.md) | [Read](WEEKLY_PROGRESS_2026-W35.md) |

## 数据说明 / Dataset notes

| 主题 / Topic | 中文 | English |
| --- | --- | --- |
| AKA 合格父项审查数据：每份文件能说明什么 | [阅读](zh-CN/data/aka-qualified-ir-v6-review-20260904/DATASET.md) | [Read](data/aka-qualified-ir-v6-review-20260904/DATASET.md) |
| IR 缺口聚类：把说法相近的需求按真正含义分组 | [阅读](zh-CN/data/aka-qualified-ir-v6-review-20260904/IR_GAP_CLUSTERS.md) | [Read](data/aka-qualified-ir-v6-review-20260904/IR_GAP_CLUSTERS.md) |

## 文档范围外的正式依据 / Authorities outside docs

- [Compiler Authoring Contract](../compiler/AUTHORING_CONTRACT.md)：精确 IR 字段与生成约定；入门解释见[执行计划](wiki/schedule.md)。
- [Workload contracts](../contracts/workloads/README.md)：精确题目；阅读版见[中文](wiki/workloads.md) / [English](en/wiki/workloads.md)。
- [Context map](../CONTEXT-MAP.md)：导航职责依据；[中文阅读](zh-CN/CONTEXT-MAP.md)。
- [Generated current status](../reports/current/STATUS.md)：发布依据的派生视图；不在翻译中手写另一份版本清单。

## 原始决策补充索引 / Additional original decisions

- [ADR 0053: B300 is an exact target on the existing Triton path](adr/0053-b300-is-an-exact-target.md)
- [ADR 0053: Study-bound external advisory candidate order](adr/0053-study-bound-advisory-cost-selection.md)
- [ADR 0054: Lab uses only Ralph](adr/0054-lab-uses-only-ralph.md)
- [ADR 0055: Task implementations live outside the common Lab](adr/0055-task-implementations-live-outside-the-common-lab.md)
- [ADR 0056 — One fixed-baseline paired assay per candidate](adr/0056-fixed-baseline-paired-execution.md)
- [ADR 0057: Metal and CLI harnesses use the existing Lab](adr/0057-metal-and-harnesses-use-the-existing-lab.md)

- [Development branches](DEVELOPMENT_BRANCHES.md): historical integration boundaries and merged Metal work.

- [0060: Release review provenance and historical exceptions](adr/0060-release-review-provenance-and-historical-exceptions.md)

- [0058: External writer custody anchors](adr/0058-external-writer-custody-anchors.md)

- [0059: Final release identities bind complete authority](adr/0059-final-release-identities-bind-complete-authority.md)
