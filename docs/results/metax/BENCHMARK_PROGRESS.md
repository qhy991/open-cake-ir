# MetaX C550：独立 Bench 进步记录

[共同标准](../../BENCHMARK_PROTOCOL.md) · [独立 Bench](https://github.com/qhy991/c550-bench)

## 2026-10-06：建立可比较的起点

读取的 Bench 版本：[`ababa4c0`](https://github.com/qhy991/c550-bench/commit/ababa4c0656b89bdf0a9c60ef0d2e90e0aebb425)。

10 题、160 个原始 workload 的独立正确性工作集。L1/069 的独立 Torch 候选已有 16 workload × 10 轮单题资格；其他任务不能据此获得整套资格。

当前 Bench 未提供延迟或 speedup 结果。C550 E/P 试点、求差平方改写和 FP8 机制记录是开发/局部设备证据，不能移作这个 Bench 的成绩。

| 对比 | Compiler 前版 → 后版 | 完整正确性覆盖 | 同口径性能变化 | 状态 |
|---|---|---|---|---|
| 固定 Bench 上的新旧版本搜索 | 尚未绑定 | 未测 | 未测 | 待资格与匹配实验 |

[Bench 范围与验证来源](https://github.com/qhy991/c550-bench/blob/ababa4c0656b89bdf0a9c60ef0d2e90e0aebb425/README.md) · [已有开发证据](../../metax-c550.md)。
本条只建立发布起点，没有新启动 provider 或 GPU 实验。后续每轮追加固定版本、逐任务
正确性/性能、失败和封存报告来源；不把这个起点表回填为旧实验成绩。

## 2026-10-10：验证状态与后续工作

[完整验证交接](VERIFICATION_HANDOFF_20261010.md)区分了原生接口、原始正确性、计时组件和作者资格。正式 Bench 优化仍为零。旧 RMS A/A 的拒绝保留；完整 Program 排队计时组件已完成有界设备控制，生产接入待公共 worker 安全边界审查和独立后继验收。
