# Evaluation：核对答案，再公平地测量

[中文目录](../../README.md) · [英文原文](../../../contexts/evaluation/CONTEXT.md) · [统一术语](../../../GLOSSARY.md#evaluation-terms)

评测像独立阅卷和计时：作者先交一份已经固定、可以启动的程序，再由相同的规则检查。它记录发生了什么，不替研究者事后挑选想回答的问题。

## 负责什么

- 两种写程序环境都先达到相同的 LaunchableCandidate 边界。
- 按 Workload 生成真实输入，用外部标准答案和容差判对。
- 先正确性，再计时或收集 profiler；搜索、确认、profiler 分别保留收据。
- 保存原始计时组、启动与备用路径次数，以及可重建分析的原始 profiler 输出。
- portfolio 遇到不支持的输入键，在启动前拒绝。

Workload 提供题目和判分方法；CampaignLock 提供本次精确运行准入；Evidence 保存候选和收据。Study 分析读取 Run Audit，而不是只读评测器的一行最快时间。

## 看懂几个常见误会

计时 CV 不合格表示样本波动太大，不等于候选算错。CV 可以理解为波动相对于平均值的大小，实际门槛由测量约定固定。

profiler 会观察内部行为，它的启动记录用于诊断，不能当作无 profiler 的延迟样本。“源码”也要标明是作者写的、生成的、展开的，还是 PTX、CUBIN、SASS 等不同阶段产物。

研究目标、允许的声明、哪些 Run 纳入分析和最终结论视图，都由相应 Study 与报告规则负责。
