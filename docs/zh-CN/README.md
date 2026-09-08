# Open Cake 中文文档：从零看懂这个系统

[English](../en/README.md) · [全部中英文对照](../README.md) · [项目首页](../../README.md)

在 B300 上运行三个 Python 算子起点，见 [B300 指南](../B300.md)。

**这个系统帮助人和 AI 写出清楚的 GPU 计算计划，先找明显错误，再生成代码，最后用真实答案和测量检查。** 你不用先学会编程；先看小例子，想动手时再打开终端。

## 先用四句话理解

GPU 是擅长让很多小组同时算题的处理器。算子是一道明确的计算题，例如把每行求和。Schedule 是分工表，说明谁读哪些数、怎么算、写哪里。Compiler 是检查并翻译分工表的程序；真正算得对不对、快不快，还要独立评测。

例如 `2×3+4=10`：Workload 规定要做这道题，Schedule 规定怎样分工，Evaluation 核对 10，Evidence 保存输入、输出与过程。研究不同写程序方式时，Study 还要事先规定公平比较的规则。

## 第一条阅读路线

1. [系统全貌](../ARCHITECTURE.md)：每部分的用途，先看图和表。
2. [算子的小数字例子](../wiki/operators.md)：先认识要计算的题。
3. [第一次动手](../GETTING_STARTED.md)：无需 GPU，检查一个乘加计划。
4. [逐项读计划](../wiki/schedule.md)：数据、线程分工、顺序和地址。
5. [读结果与排错](../wiki/results.md)：区分检查、生成、正确性与速度。
6. [实验怎样进行](../wiki/experiments.md)：看 AI、预算、评测怎样合作。

英文代码名是查找键，不用先背。遇到术语查[统一词表](../GLOSSARY.md)；例如“冻结”就是这次实验使用的内容固定下来，“oracle”就是独立的标准答案计算方法。

## 按需要继续

| 我想知道 | 中文页面 |
| --- | --- |
| 在 B300 上用 Cake 和原生 CuTeDSL 优化同一个 GEMM | [CuTeDSL 同后端路径](PAIRED_CUTE.md) |
| load、FMA、矩阵乘和前缀和分别做什么 | [基本操作](../wiki/primitives.md) |
| 项目有哪些完整任务 | [任务目录](../wiki/workloads.md) |
| 三个独立 Tile Workload 的 ABI、CPU 参考和匹配基线 | [Tile Workload 指南](../TILE_WORKLOADS.md) |
| 用 Python 变量和表达式写执行计划 | [Python 入门](PYTHON_FRONTEND.md) |
| 在 Apple M2 上运行与测量逐元素计算、行归约和加权 RMSNorm | [Metal 入门](../metal.zh-CN.md) |
| 理解 IR 对象、代码组织与扩展位置 | [当前 IR 详解](../IR_GUIDE.md) |
| 两层 TileLoop 怎样初始化、累积和写回 | [Triton 循环作用域](../TRITON_LOOP_SCOPES.md) |
| IR 与原生 Triton 怎样从同一基线进行配对优化 | [Triton 配对流程](../PAIRED_TRITON.md) |
| 从准备到真实实验应该按什么顺序 | [执行手册](RUNBOOK.md) |
| 什么证据足够进入下一步 | [验收规则](ACCEPTANCE_GATES.md) |
| 论文结果与本地结果有什么区别 | [论文对照约定](PAPER_CONTRACT.md) |
| 四个模块各负责什么 | [职责地图](CONTEXT-MAP.md) |
| 为什么这样设计 | [完整设计记录目录](adr/README.md) |
| 旧实验怎样重建 | [历史回放](../wiki/replay.md) |
| 677 条审查和 56 个 GPU 样本是什么意思 | [大批量审查导读](AKA_QUALIFIED_IR_REVIEW_AND_LAB_PLAN_20260903.md) |
| 一个最小真实 GPU 样本怎样判对 | [八个输出的仿射计算](../AFFINE_PARENT_B200_CANARY_20260906.md) |
| 今日发布版由哪里决定 | [自动生成的状态页](../../reports/current/STATUS.md) |
| 其余调查、报告和数据说明 | [全部中英文对照](../README.md) |

## 阅读时记住三条

“计划通过”“源码编译成功”“GPU 算对”“速度提高”各需要自己的证据。旧报告中的日期、版本和数字属于当时条件。小数字教学例子用于理解，不等于当前机器上的测试成绩。

已有中文正文保留原路径；缺失的中文解释在本目录。英文原文和完整历史长表从每页顶部直达。阅读版按问题重写，不是逐句翻译；原始合同、决定状态和证据仍各有原来的负责位置。
