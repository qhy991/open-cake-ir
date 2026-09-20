# AMD 实验结果

由 `tools/render_hardware_results.py` 从各平台发布数据生成。原始报告与 Finding 保留权威。

[交互目录](index.html) · [发布数据](records.json) · [main 汇总](https://github.com/qhy991/open-cake-ir/blob/main/docs/RESULTS.md) · [维护流程](../../RESULTS_MAINTENANCE.md)

**加速比 = 基线耗时 ÷ 候选耗时，超过 1× 表示更快。** 各行仅在自己的硬件、Workload、版本与计时协议内比较，不求跨平台平均值。

平台分支分别更新自己的发布数据；main 展示已经合入当前检出的版本。这里不会在线抓取平台分支最新值，也不是实时冠军榜。失败、无显著差异与未实测记录均保留。

## 数据归属

| 平台 | 维护分支 | 数据日期 | 观察条目 | 发布数据 |
|---|---|---|---:|---|
| AMD | [amd](https://github.com/qhy991/open-cake-ir/tree/amd/docs/results/amd) | 2026-09-20 | 1 | [amd/records.json](records.json) |

观察条目数不等于任务数：同一任务可以有不同形状、实验集合和历史尝试。

## AMD

gfx1151 的两种设备计时器尚未对齐。保留支持状态与调查入口，不给出合格性能或加速比。

- 此记录来自设备调查，未形成 Campaign；两种计时器的绝对耗时不可互换。

| 设备 / 集合 | Task | 输入 / Workload | 基线 µs | 候选 µs | 加速比 | 状态 | 详情 |
|---|---|---|---:|---:|---:|---|---|
| Radeon / Strix Halo | `rmsnorm smoke` | gfx1151-rmsnorm-b8-smoke | — | — | — | 计时边界未解决 | [amd-result-080](#amd-result-080) |

## 演进与更新

交互页保留同一 Campaign 内已通过确认的候选时间序列，包括变慢点。它不把不同形状、版本和计时协议拼成长期晋升曲线。
历史晋升记录不等于当前 registry 冠军。跨编译器版本时分别看固定基线的变化与候选相对基线的改善；完整失败过程见来源。
新增实验先在对应平台的 records.json 追加有稳定 id 和固定来源提交的观察，再生成该平台页面。合入 main 的集成分支重建总览；不要手工改生成表格。具体命令见维护流程。

## 逐项来源

### amd-result-080

**Radeon / Strix Halo · rmsnorm smoke** — 2026-09-17 / 计时边界未解决

- Workload：`gfx1151-rmsnorm-b8-smoke`；目标：`gfx1151`；版本：`见来源`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[findings/2026-09-17-002-two-device-timers-disagree-on-gfx1151.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/2026-09-17-002-two-device-timers-disagree-on-gfx1151.json)。
- 原记录 / 实现定位：`infplane；原调查未产生 Campaign`。
- 31.858 与 24.224 µs 来自不同计时器，不作为合格延迟或加速比。
