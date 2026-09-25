# 文档总目录 / Documentation catalog

[项目首页](../README.md) · [技术报告 PDF](open-cake-ir-technical-report.pdf) · [报告索引与引用](README.md) · [中文入门](zh-CN/README.md) · [English start](en/README.md)

本页按主题收纳详细文档与历史材料。首次访问先读项目 README 或技术报告首页；查找具体接口、实验方法和原始报告时使用本目录。
This catalog groups detailed references and dated material. Use the repository README or report entry for a short reading route.

**目录：** [入门与使用](#入门与使用--guides) · [实验与硬件](#实验与硬件--experiments-and-hardware) · [模块职责](#模块职责--contexts) · [设计记录](#设计记录--decisions) · [历史调查](#历史调查与报告--historical-surveys-and-reports) · [数据说明](#数据说明--dataset-notes)

章节沿用原路径。阅读版可以解释或摘要，但不会创建第二份合同或实验结论；术语由 [GLOSSARY](GLOSSARY.md)维护，当前能力见[生成状态页](../reports/current/STATUS.md)。
Original paths and dated conclusions stay intact. Reading companions do not create independent contracts or evidence.

## 入门与使用 / Guides

| 主题 / Topic | 中文 | English |
| --- | --- | --- |
| 验收规则：做到什么，才可以进入下一步 | [阅读](zh-CN/ACCEPTANCE_GATES.md) | [Read](ACCEPTANCE_GATES.md) |
| 相关工程：Croqtile 与 Open-Cake | [阅读](CROQTILE_COMPARISON.md) | [Read](en/CROQTILE_COMPARISON.md) |
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

## 实验与硬件 / Experiments and hardware

| 查找内容 / Topic | 文档与数据入口 / References |
|---|---|
| 各平台已发布观察、失败与历史结果 | [完整硬件汇总](RESULTS.md) · [交互目录（下载后打开）](results/index.html) · [发布维护流程](RESULTS_MAINTENANCE.md) |
| FlashInfer 改写、starter 与外部实现的差距 | [逐任务实验综述及 English summary](results/nvidia/FLASHINFER_STATUS.md) · [改写任务包](../experiments/flashinfer_rewrites/README.md) |
| CAKE 参考机制与本地实现边界 | [NVIDIA CAKE 对照](NVIDIA_CAKE_REPRODUCTION.md) · [已有 Kernel 改写](KERNEL_REPRODUCTION.md) |
| NVIDIA B200 / B300 | [B300 中文](B300.md) · [English](en/B300.md) · [发布记录](results/nvidia/README.md) |
| Apple Metal | [中文指南](metal.zh-CN.md) · [English](metal.md) · [发布记录](results/metal/README.md) |
| AMD | [任务入口](amd-task-entrypoints.md) · [发布记录](results/amd/README.md) |
| Hygon DCU | [设计与运行](dcu-gfx938-design.md) · [设备结果](dcu-gfx938-results.md) · [发布记录](results/dcu/README.md) |
| MetaX C550 | [当前路径与验收范围](metax-c550.md) · [bring-up 记录](metax-c550-bringup.md) |
| 原生 CUDA / PTX 与 CuTe DSL | [CUDA 中文](zh-CN/NATIVE_CUDA.md) · [CUDA English](NATIVE_CUDA.md) · [CuTe 中文](zh-CN/PAIRED_CUTE.md) · [CuTe English](en/PAIRED_CUTE.md) |
| 优化知识迁移与受控消融 | [机制设计](OPTIMIZATION_TRANSFER.md) · [English](en/OPTIMIZATION_TRANSFER.md) · [消融方法](OPTIMIZATION_TRANSFER_ABLATION.md) |
| 显式变换与性能解释 | [CTA 宽度](TRITON_CTA_WIDTH.md) · [输出列特化](OUTPUT_COLUMN_SPECIALIZATION_PASS.md) · [Epilogue fusion](EPILOGUE_FUSION_PASS.md) · [效率评分](PERFORMANCE_SCORING.md) |
| Agent、预算与实验操作 | [Lab 用途与流程](wiki/experiments.md) · [English](en/wiki/experiments.md) · [执行手册](RUNBOOK.md) · [Claude artifact-only v4](CLAUDE_PROVIDER_V4.md) |
| 实验输入与可复查依据 | [Workload 定义](../contracts/workloads/README.md) · [问题与改进记录](../findings/README.md) · [当前状态](../reports/current/STATUS.md) |

具体结果只在各自绑定的源码、硬件、输入与协议内解释；指南入口不表示全部路径已获设备或性能验收。

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
