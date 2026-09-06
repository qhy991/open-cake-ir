# ADR 0028：实验模板写设计，实际版本在运行前固定

[设计目录](README.md) · [英文原文与完整证据](../../adr/0028-study-templates-defer-revision-binding-to-campaign-lock.md)

**原文状态：已接受，2026-08-25。**

编译器每升一版就复制五份几乎相同的实验合同，会让一个设计变成多个要维护的副本。因此 Study 的 `state` 只分 `template` 与 `frozen`。

模板的 Compiler 和 Executor 都写 `{"binding":"current_release"}`，保留稳定实验设计；冻结 Study 的两者都用精确内容引用。`Lab.preflight` 是唯一解析者，把这次选择一次固定到 CampaignLock。模板不能只固定其中一个，冻结 Study 不能偷偷跟随主线。

像报名表写“使用考试时指定教材”，开考后考卷就明确记录具体版本。之后教材更新，不改变这场考试。后继生成工具必须一起替换两个引用再验证。

当前编译器未发布、执行器清单无法核验、绑定拼写错误或冻结内容仍含移动引用，都应失败。CampaignLock 保留模板本身与两个解析版本，便于复查设计和执行各来自哪里。旧实验字节不改；尚未运行的旧 Study 要回原源码环境准备，而不是因此得到一份新的执行器归档。
