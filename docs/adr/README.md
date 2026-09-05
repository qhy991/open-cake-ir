# 设计决策目录

ADR 记录“为什么这样设计”。当前版本看 [发布状态](../../reports/current/STATUS.md)，一次实验的成绩看它自己的报告。
新决策写新记录，并说明取代哪条旧决定；不改写历史证据。

| 想了解什么 | 主要记录 |
| --- | --- |
| 为什么以编译器为核心 | [0001](0001-compiler-first-with-dependent-lab.md) |
| 为什么单独区分 portfolio | [0002](0002-portfolio-as-second-study-variant.md) |
| 代码与运行产物放在哪里 | [0005](0005-forward-compatible-lifecycle-layout.md) |
| 为什么估算需要校准 | [0008](0008-calibration-coverage-gates-ranking.md) |
| 生成源码与固定源码有什么区别 | [0019](0019-lowering-generation-is-observable.md) |
| 为什么 Corpus 与 Workload 不同 | [0029](0029-lowering-route-is-not-a-workload-profile.md)、[0038](0038-aka-is-a-challenge-corpus-not-a-compiler-corpus.md) |
| 发布如何使用明确的批准 | [0030](0030-compiler-release-approval-is-external.md) |
| 为什么完整文件不等于可信的权限历史 | [0031](0031-archive-integrity-is-not-filesystem-custody.md) |
| 文档怎样避免重复维护事实 | [0047](0047-documentation-separates-stable-history-and-current-views.md) |
| AI 怎样读取任务并受预算约束 | [0048](0048-agent-runs-use-task-agents-and-ralph-control.md) |
| 已发布身份怎样保留 | [Executor：0049](0049-released-executor-descriptors-reserve-their-identities.md)、[Compiler：0050](0050-released-compiler-locks-reserve-their-identities.md) |
| 为什么读取不能隐式复制成一组数 | [0051](0051-load-values-follow-the-access-domain.md) |

其余记录按本目录文件名查阅。状态含义：

- **proposed：** 提案，可按授权范围实现和审查，不代表已发布或 GPU 通过。
- **accepted：** 决策已被接受；具体权限范围仍看该记录和当前任务授权。
- **superseded：** 后继记录负责当前决定，旧记录保留背景。
- **rejected：** 保留被拒绝的方案及理由，不据此实施。
