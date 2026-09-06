# 文档与职责地图

[中文目录](README.md) · [英文原文](../../CONTEXT-MAP.md)

英文原文是仓库导航职责的正式说明。本页是中文阅读版；词义统一查[术语表](../GLOSSARY.md)，精确字段仍由代码和合同负责。

## 五种材料分开放

| 类别 | 用途 | 更新办法 |
| --- | --- | --- |
| 稳定说明 | 架构、术语、职责、使用规则 | 支持的约定变化时更新 |
| 决策历史 | 记录为何这样设计 | 新写或取代旧 ADR，保留旧理由 |
| 可执行依据 | 精确数学、版本、目标、规则 | 通过负责它的流程创建后继 |
| 历史快照 | 某日、某版观察到了什么 | 保留；纠正另写勘误或后继 |
| 当前视图 | 把正式依据整理给人读 | 可删除再生成，不能单独授权或产生结论 |

`inventory/CURRENT_STATE.md` 名字虽有“当前”，实际是 2026-08-25 的冻结迁移快照。今天发布了什么，查[生成的状态页](../../reports/current/STATUS.md)。

## 四个模块的边界

对用户提供独立编译器和依赖它的研究 Lab 两种能力。评测与证据支持 Lab，不是另外两个产品。

| 模块 | 它回答的问题 |
| --- | --- |
| [Compiler](contexts/compiler/CONTEXT.md) | 计划是否合法，能生成什么目标源码 |
| [Research Lab](contexts/lab/CONTEXT.md) | AI 如何按已固定的实验约定改进候选 |
| [Evaluation](contexts/evaluation/CONTEXT.md) | 一个封存候选算对了吗，测量有效吗 |
| [Evidence](contexts/evidence/CONTEXT.md) | 别人能否从记录重建观察和结论 |

Lab 调用 Compiler，也通过 Evaluation 保存 Evidence；Evidence 的只读审计供 Lab 分析。Compiler 不依赖这些实验模块。Workload 的数学定义来自外部固定合同，Lab 不重新解释题目。

## 新信息放哪里

候选和测量进实验 Evidence，结束的 Run 进终态归档及审计，接受的研究结论进 Study Report，再生成可重建视图。它们不会自动修改稳定说明。

新的数学题目或实验设计建立后继合同；反复出现的编译器问题形成 ADR、编译器后继和 Corpus Gate。纠正旧观察另写勘误；新论文版本另立来源。运行证据只能促成提案，不能直接改正在用的编译器。

初学者按[系统全貌](../ARCHITECTURE.md) → [入门教程](../GETTING_STARTED.md) → [术语表](../GLOSSARY.md)读。实验操作人员继续读[论文对照](PAPER_CONTRACT.md)、[Lab 职责](contexts/lab/CONTEXT.md)和[执行手册](RUNBOOK.md)。维护者继续读[验收规则](ACCEPTANCE_GATES.md)、状态页和本次审查绑定的原始证据。
