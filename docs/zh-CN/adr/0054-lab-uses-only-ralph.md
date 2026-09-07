# ADR 0054：Lab 只使用 Ralph

[英文决定](../../adr/0054-lab-uses-only-ralph.md)

仓库拥有者于 2026-09-07 明确决定不再兼容旧 prompt 流程。本决定替代 ADR 0048 的兼容条款。

所有新的 matched-search 实验，包括 Cake / Native Triton 对照，都用同一条 Ralph 路径：
读取 TASK.md 和 AGENTS.md，根据每轮 StateCard 更新 candidate-set.json，由外部控制器决定何时停止。
旧模板、旧渲染分支和旧 Study 入口移除。资格验证也默认使用 Ralph。

历史发布、审核和实验原始证据不改写。旧实验需要切回对应的历史 Git 版本才能回放。
本次属于 Executor 后继源代码变更；CPU 测试通过不代表已获得 GPU 或正式实验资格。
