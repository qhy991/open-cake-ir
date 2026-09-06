# Research Lab：组织一次有规则的实验

[中文目录](../../README.md) · [英文原文](../../../contexts/lab/CONTEXT.md) · [统一术语](../../../GLOSSARY.md#workload-and-research-lab-terms)

Lab 像考试组织者：分配题目和工具、控制时间、收作业，再按事先规定的方法分析成绩。它把 Workload、Compiler 和 Executor 的固定版本当作输入。

## 负责什么

- 把 Study 模板和已发布输入解析一次，写成固定的 CampaignLock。
- 为后继 Run 生成不可改动的 `TASK.md`、`AGENTS.md`；外部 Ralph 只提供从证据生成的本轮状态。
- 为每个 Run 指定完整写程序环境，保留一个主要候选提交者。
- 管理轮数、作者预算、GPU 搜索预算、检查点和停止规则。
- 把封存候选交给共同 Evaluation；只采用执行前固定的分析计划和声明范围。

它不重新定义题目，不替评测器判对或写收据，也不能把系统资格检查写成科学比较。

## 实验怎样分层

Study 引用一个 Workload，并规定每个 Run 使用哪个环境。Campaign 是这个 Study 的一次实际执行；只有分析计划事先允许，才可合并不同 Campaign。

Run 中包含有顺序的 Turn，可以提交多个不可变 Candidate。同一个 Candidate 可以分别有搜索、确认、profiler 评测记录。科学 Study 预先声明 Estimand；系统资格和工程优化没有组间效果估计。

合格的 KernelSeed 可以进入另一个 portfolio Study，但不能据此宣布支持任意形状或完整服务。

## 两种没得到好结果的情况

按约定跑完，始终没找到合格候选，是观察到的负结果。AI 服务、GPU 调度、文件保管或评测框架坏了，是缺失数据。它们不能互换。

Artifact Promotion 只是在一个 Run 中选择通过确认性评测的候选，不是比较实验的胜方、因果结论或生产部署决定。
