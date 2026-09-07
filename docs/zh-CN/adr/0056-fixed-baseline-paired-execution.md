# ADR 0056：每个候选都与固定基线成对测量

状态：Executor 后继提案，需要独立审查和发布。

每个候选与同一份已封存基线交错测量。Study 固定顺序、样本数和稳定性规则；
worker 在同一独占 GPU 任务中加载双方，逐次检查输出、输入不变和调用次数。
Receipt 和 audit 从原始成对记录重算。正确、稳定但更慢或没有明显差异的候选，
仍是有效结果。确认测量重新运行，profile 单独记录。

三个 B300 Study 保持稳定。机器、provider、Executor 和已封存基线通过 checkout
外的执行绑定进入 CampaignLock。TASK/AGENTS 正文来自同一个不可变投影，
每轮只改变 StateCard。新的 Code Mode 禁用配置需要新的真实双轮资格检查。

TaskLab 提供严格的任务 loader 和 manifest 解析，任务执行位于 `tasks`。
通用 Lab、Evaluation 和 Compiler 不反向导入具体任务。
历史发布和原始证据保留在原 Git 树；新组合必须生成新的 Executor。
CPU 检查不等于 GPU 正确性、计时、provider 资格或框架验收。

完整协议和数值规则见[英文原文](../../adr/0056-fixed-baseline-paired-execution.md)。
