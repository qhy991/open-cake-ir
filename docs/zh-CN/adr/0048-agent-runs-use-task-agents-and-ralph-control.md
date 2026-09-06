# ADR 0048：给 AI 两份固定文件，用外部 Ralph 控制迭代

[设计目录](README.md) · [英文原文与完整证据](../../adr/0048-agent-runs-use-task-agents-and-ralph-control.md)

**原文状态：仓库拥有者已接受，2026-08-30；原文记录由 schema-v2 和 Executor v42 实现。**

把题目、规则、参考和动态分数全塞在提示模板里，会重复已有事实，换题又容易复制模板。后继接口 `task_agents_ralph_v1` 用两份明确文件：TASK 从 Workload、Study、Target、脚手架和参考生成题目；AGENTS 说明稳定工具、工作区与单写入者规则，二者都不保存可变结果。

起始工作区恰好两份只读文件，AI 只增加或更新 `candidate-set.json`。调用只要求读文件，并给证据派生的 StateCard。同一 provider 会话持续，外部 Ralph 决定停止，每轮实际任务和状态字节全部保留。

预算分别限制 token、总时间、活跃写作时间、轮数、搜索、确认和分析评测。剩余评测不足以容纳下一轮最坏需求就不启动；禁止 Run 重试或替补，只保留同一逻辑评测中已有的受限零工作准入恢复。

修改任务文件是污染，状态卡缺失或错误是协议故障；预算耗尽是正常停止，不是候选算错；外部故障仍是缺失数据。可信编译器、评测、调度和证据都在 AI 控制之外，Markdown 不变成评分权威。

验收两边文件、首次／继续轮、全部预算、两轮回放和篡改拒绝。旧 schema-v1 按原 Prompt 回放，新 Ralph Study 不再含 prompt_template。
