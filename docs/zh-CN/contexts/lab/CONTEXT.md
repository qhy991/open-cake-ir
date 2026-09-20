# Research Lab：组织一次有规则的实验

[中文目录](../../README.md) · [英文原文](../../../contexts/lab/CONTEXT.md) · [统一术语](../../../GLOSSARY.md#workload-and-research-lab-terms)

Lab 像考试组织者：分配题目和工具、控制时间、收作业，再按事先规定的方法分析成绩。它把 Workload、Compiler 和 Executor 的固定版本当作输入。

## 负责什么

- 将执行输入、权限和预算固定到 RunSpecification；旧 Study/CampaignLock 只作同一 Run 引擎的输入适配。
- 为后继 Run 生成不可改动的 `TASK.md`、`AGENTS.md`；外部 Ralph 只提供从证据生成的本轮状态。
- 为每个 Run 指定完整写程序环境，保留一个主要候选提交者。
- 管理轮数、作者预算、GPU 搜索预算、检查点和停止规则。
- 把封存候选交给共同 Evaluation；只采用执行前固定的分析计划和声明范围。

它不重新定义题目，不替评测器判对或写收据，也不能把系统资格检查写成科学比较。

## 实验怎样分层

工程优化直接使用 Run，无需 Study。研究由 StudyPlan 预分配条件和重复次数，使用同一个 Run 引擎。
旧 Campaign 输入保留原 Study 的比较约束；跨次汇总仍需分析计划事先允许。

Run 中包含有顺序的 Turn，可以提交多个不可变 Candidate。同一个 Candidate 可以分别有搜索、确认、profiler 评测记录。科学 Study 预先声明 Estimand；系统资格和工程优化没有组间效果估计。

KernelSeed 和独立 portfolio artifact 工具仍保留，但没有 Portfolio Study 生命周期。单个种子不能证明任意形状或完整服务能力。

Run 在 preflight 中解析固定输入；任务层在运行副作用之前检查准入，旧 Study 策略在输入边界检查。模块归属见[实现导航](../../../../src/open_cake_ir/lab/README.md)。

## 两种没得到好结果的情况

按约定跑完，始终没找到合格候选，是观察到的负结果。AI 服务、GPU 调度、文件保管或评测框架坏了，是缺失数据。它们不能互换。

Artifact Promotion 只是在一个 Run 中选择通过确认性评测的候选，不是比较实验的胜方、因果结论或生产部署决定。

具体任务实现与 TaskLab 组合入口位于通用引擎之外，见[任务说明](../../../TASKS.md)。
