# 文档总目录 / Documentation catalog

[Triton CTA 宽度特化与 tick-tock 经验总结](TRITON_CTA_WIDTH.md)：显式候选变换、经验归属、适用条件与配对测量。

[任务效率评分](PERFORMANCE_SCORING.md)：固定任务字节数、带宽参考、审计报告与校准覆盖。

[Claude artifact-only v4](CLAUDE_PROVIDER_V4.md)：上下文压缩记录、精确终态请求与旧契约兼容。

原生 CUDA/PTX：[中文](zh-CN/NATIVE_CUDA.md) · [English](NATIVE_CUDA.md)

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
| Apple M1 Pro / M2 / M4 的 TaskLab 归一化任务与测量 | [阅读](metal.zh-CN.md) | [Read](metal.md) |
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
| 每个任务的已验证最佳实现怎样成为下一轮 baseline | [阅读](zh-CN/TASK_INCUMBENTS.md) | [Read](en/TASK_INCUMBENTS.md) |

## 模块职责 / Contexts

| 主题 / Topic | 中文 | English |
| --- | --- | --- |
| Compiler：检查和翻译执行计划 | [阅读](zh-CN/contexts/compiler/CONTEXT.md) | [Read](contexts/compiler/CONTEXT.md) |
| Evaluation：核对答案，再公平地测量 | [阅读](zh-CN/contexts/evaluation/CONTEXT.md) | [Read](contexts/evaluation/CONTEXT.md) |
| Evidence：保存事实，让别人能复查 | [阅读](zh-CN/contexts/evidence/CONTEXT.md) | [Read](contexts/evidence/CONTEXT.md) |
| Research Lab：组织一次有规则的实验 | [阅读](zh-CN/contexts/lab/CONTEXT.md) | [Read](contexts/lab/CONTEXT.md) |

## 设计记录 / Decisions

完整决策索引只维护在 [ADR 目录](adr/README.md)。

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

