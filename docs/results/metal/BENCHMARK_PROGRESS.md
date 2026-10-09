# Apple M4：独立 Bench 进步记录

[共同标准](../../BENCHMARK_PROTOCOL.md) · [独立 Bench](https://github.com/qhy991/metal-bench)

## 2026-10-06：建立可比较的起点

读取的 Bench 版本：[`afac2386`](https://github.com/qhy991/metal-bench/commit/afac23867528cd3f2b9ac9eddce353c2e9089552)。

公开 RMSNorm FP32 开发工作集：3 个形状、5 种分布、2 个 seed。15 项软件测试、30 个 CPU 正确性 case 和 Swift runner 编译通过；尚未取得 GPU 资格。

当前 Bench 没有 GPU 时间或 Cake 性能进步结论。此前 GLM mean30 pilots 属于固定开发任务，不转换为独立 Bench 分数。

| 对比 | Compiler 前版 → 后版 | 完整正确性覆盖 | 同口径性能变化 | 状态 |
|---|---|---|---|---|
| 固定 Bench 上的新旧版本搜索 | 尚未绑定 | 未测 | 未测 | 待资格与匹配实验 |

[Bench 范围与验证来源](https://github.com/qhy991/metal-bench/blob/afac23867528cd3f2b9ac9eddce353c2e9089552/docs/VALIDATION.md) · [已有开发证据](../../METAL_MEAN30_GLM.md)。
本条只建立发布起点，没有新启动 provider 或 GPU 实验。后续每轮追加固定版本、逐任务
正确性/性能、失败和封存报告来源；不把这个起点表回填为旧实验成绩。
