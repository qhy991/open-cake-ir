# Hygon BW1100 / gfx938：独立 Bench 进步记录

[共同标准](../../BENCHMARK_PROTOCOL.md) · [独立 Bench](https://github.com/qhy991/bw1100-bench)

## 2026-10-06：建立可比较的起点

读取的 Bench 版本：[`77a2848f`](https://github.com/qhy991/bw1100-bench/commit/77a2848f95bd2a13fea0ce69dcc1147ccac39a93)。

10 题、160 个原始 workload、每 workload 10 轮新输入的独立正确性入口，保留原始 dtype、shape、容差和 UUID；社区基线按单题资格说明。

本次未取得新旧 Compiler 在固定 Bench 上的配对性能报告。已有 decoder 全调用 replay 约 1.254× 是保留候选相对社区基线的复核，不是本轮 Compiler 造成的搜索进步。

| 对比 | Compiler 前版 → 后版 | 完整正确性覆盖 | 同口径性能变化 | 状态 |
|---|---|---|---|---|
| 固定 Bench 上的新旧版本搜索 | 尚未绑定 | 未测 | 未测 | 待资格与匹配实验 |

[Bench 范围与验证来源](https://github.com/qhy991/bw1100-bench/blob/77a2848f95bd2a13fea0ce69dcc1147ccac39a93/README.md) · [已有开发证据](compiler-trig-scan-20261006.md)。
本条只建立发布起点，没有新启动 provider 或 GPU 实验。后续每轮追加固定版本、逐任务
正确性/性能、失败和封存报告来源；不把这个起点表回填为旧实验成绩。

## Compiler 开发验证参考（不计入独立 Bench）

- 2026-10-08：[多区域 MMA 有界设备验证](multi-region-mma-qualification-20261008.md)。
  合法方案已能生成并通过原任务正确性；新候选比旧 incumbent 慢 6.30–6.67%，保留旧实现。
  该记录不更新上面的独立 Bench 分数，也不证明固定三小时搜索收益。
