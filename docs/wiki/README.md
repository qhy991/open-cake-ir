# Open Cake 中文 Wiki

[中文首页](../zh-CN/README.md) · [English](../en/wiki/README.md) · [中英文对照](../README.md)

**这里讲清楚一件事：怎样把计算想法变成可检查、可测量的 GPU 程序。**
不要求先学编译器。先读熟悉的问题，再按需查代码里的英文名。

## 第一条阅读路线

1. [系统有什么用](../ARCHITECTURE.md)：四部分各负责什么。
2. [第一次试用](../GETTING_STARTED.md)：不需要 GPU，检查并生成一个乘加计划。
3. [读懂执行计划](schedule.md)：数据放哪里，谁算哪一行，先后怎样安排。
4. [算子图解](operators.md)：矩阵乘法、归一化、注意力和状态更新怎样工作。
5. [怎样读结果](results.md)：哪些结论已经有证据，哪些还不能下。

## 按问题查阅

| 我想知道 | 去哪里 |
| --- | --- |
| `load`、`mma`、`top_k` 等名字是什么意思 | [IR 基本操作](primitives.md) |
| 用 Python 写计划而不是手填 JSON | [Python 入门](../zh-CN/PYTHON_FRONTEND.md) |
| 当前有哪些完整任务定义 | [Workload 目录](workloads.md) |
| 怎样准备三个独立算子的共用 ABI 和匹配基线 | [Tile Workload 指南](../TILE_WORKLOADS.md) |
| 两层静态 TileLoop 怎样组织作用域 | [循环作用域](../TRITON_LOOP_SCOPES.md) · [带尾部的 GEMM 计划](../../corpus/schedules/gemm-bias-two-deep-tail-b1-smoke.json) |
| IR 与原生 Triton 怎样共用基线、构建和评测 | [Triton 配对流程](../PAIRED_TRITON.md) |
| AI 怎样改代码，怎样控制预算 | [实验流程](experiments.md) |
| 怎么开始一次真实 GPU 实验 | [实验流程](experiments.md#运行前看什么)和 [执行手册](../zh-CN/RUNBOOK.md) |
| 某个检查没通过，该看什么 | [结果与排错](results.md) |
| 主线升级后怎样重建旧任务 | [历史任务回放](replay.md) |
| 一个新操作为什么不等于整个算子都支持 | [FMA 再审与读取检查案例](../ACCESS_DOMAIN_REPAIR_20260906.md) |
| 怎样证明一个固定样本在 GPU 上算对 | [8 个输出的仿射检查实例](../AFFINE_PARENT_B200_CANARY_20260906.md) |
| 现在发布的是哪个版本 | [自动生成的发布状态](../../reports/current/STATUS.md) |
| 一个英文术语的正式含义 | [术语表](../GLOSSARY.md) |
| 代码应该放在哪一部分 | [职责地图](../zh-CN/CONTEXT-MAP.md) |
| 为什么采用某种设计 | [设计决策目录](../zh-CN/adr/README.md) |
| 文档怎样保持准确 | [文档维护](maintaining.md) |

## 怎样使用这个 Wiki

这是仓库内的 Wiki：点链接即可阅读，文档和代码一起检查、提交、更新。
没有另一份需要手动同步的 GitHub Wiki 副本。

这里的数字小例子用于理解计算。真实任务的形状、精度和判分规则，以链接中的合同为准；
当前发布能力以本次编译器检查为准；实验成绩必须回到它绑定的原始记录。
