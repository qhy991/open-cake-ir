# 设计决策目录

[中文首页](../zh-CN/README.md) · [English](../en/adr/README.md) · [中英文对照](../README.md)

ADR 记录“为什么这样设计”。当前版本看 [发布状态](../../reports/current/STATUS.md)，一次实验的成绩看它自己的报告。
新决策写新记录，并说明取代哪条旧决定；不改写历史证据。

| 想了解什么 | 主要记录 |
| --- | --- |
| 为什么以编译器为核心 | [0001](../zh-CN/adr/0001-compiler-first-with-dependent-lab.md) |
| 为什么单独区分 portfolio | [0002](../zh-CN/adr/0002-portfolio-as-second-study-variant.md) |
| 代码与运行产物放在哪里 | [0005](../zh-CN/adr/0005-forward-compatible-lifecycle-layout.md) |
| 为什么估算需要校准 | [0008](../zh-CN/adr/0008-calibration-coverage-gates-ranking.md) |
| 生成源码与固定源码有什么区别 | [0019](../zh-CN/adr/0019-lowering-generation-is-observable.md) |
| 为什么 Corpus 与 Workload 不同 | [0029](../zh-CN/adr/0029-lowering-route-is-not-a-workload-profile.md)、[0038](../zh-CN/adr/0038-aka-is-a-challenge-corpus-not-a-compiler-corpus.md) |
| 发布如何使用独立人类或模型会话的批准 | [0052](../zh-CN/adr/0052-independent-agent-release-review.md)，背景：[0030](../zh-CN/adr/0030-compiler-release-approval-is-external.md) |
| 为什么完整文件不等于可信的权限历史 | [0031](../zh-CN/adr/0031-archive-integrity-is-not-filesystem-custody.md) |
| 文档怎样避免重复维护事实 | [0047](../zh-CN/adr/0047-documentation-separates-stable-history-and-current-views.md) |
| AI 怎样读取任务并受预算约束 | [0048](../zh-CN/adr/0048-agent-runs-use-task-agents-and-ralph-control.md) |
| 已发布身份怎样保留 | [Executor：0049](../zh-CN/adr/0049-released-executor-descriptors-reserve-their-identities.md)、[Compiler：0050](../zh-CN/adr/0050-released-compiler-locks-reserve-their-identities.md) |
| 为什么读取不能隐式复制成一组数 | [0051](../zh-CN/adr/0051-load-values-follow-the-access-domain.md) |
| Study 怎样显式使用外部模型给候选排序 | [0053](0053-study-bound-advisory-cost-selection.md) |

[完整的 54 份中文设计记录](../zh-CN/adr/README.md)逐条对应英文原文。历史中重复的编号按完整文件名区分。状态含义：

- **proposed：** 提案，可按授权范围实现和审查，不代表已发布或 GPU 通过。
- **accepted：** 决策已被接受；具体权限范围仍看该记录和当前任务授权。
- **superseded：** 后继记录负责当前决定，旧记录保留背景。
- **rejected：** 保留被拒绝的方案及理由，不据此实施。

- [0054: Lab uses only Ralph](0054-lab-uses-only-ralph.md)
- [0055: Task implementations live outside the common Lab](0055-task-implementations-live-outside-the-common-lab.md)
- [0056: One fixed-baseline paired assay per candidate](0056-fixed-baseline-paired-execution.md)
