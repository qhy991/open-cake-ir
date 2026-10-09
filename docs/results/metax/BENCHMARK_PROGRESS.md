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

## 2026-10-09：冻结 C1 通过 L1/069 完整设备正确性评估

CAKE 候选在 C550-2 上通过 `L1/069_rms_norm` 的全部 **16 个原始 workload × 10 轮新输入**，
共 160 次比较，失败 0 次。Bench 回执为 `status=passed`、`full_device_correctness=true`、
`performance=not_measured`，评测进程退出码为 0。

| 固定对象 | 版本或范围 |
| --- | --- |
| Compiler C1 | [`5bb474c6`](https://github.com/qhy991/open-cake-ir/commit/5bb474c6df2948d1cc50b2d46f9ab828525be3c2) |
| 独立评估适配器 | [`3af313eb`](https://github.com/qhy991/open-cake-ir/commit/3af313eb266fadd6fc32fd5f5a0d82b388843ced) |
| c550-bench | [`ababa4c0`](https://github.com/qhy991/c550-bench/commit/ababa4c0656b89bdf0a9c60ef0d2e90e0aebb425) |
| 本轮 CAKE 覆盖 | 1/10 题、16/160 个原始 workload；其余 9 题尚未适配 |
| 正确性 | L1/069：160/160 次比较通过，保留原始维度和输入生成规则 |
| 性能 | 未测，未形成新旧 Compiler 对照或加速比 |

候选以 Schedule 表达，由冻结 C1 检查并生成 Triton，再通过隔离编译器预编译为 9 个 native 变体。
适配器只用 Torch 分配输出、检查 Tensor 元信息和取得当前 stream，算术由 CAKE 生成的原生二进制执行。
候选保留残差相加后的 BF16 舍入，以及归一化结果乘 weight 前的 BF16 舍入。
本次允许读取 Bench 的高层参考和返回值 ABI，未启动 Bench Agent 搜索，记录的是冻结 C1 后的独立评估。

设备执行前，Bench 的 11 项 CPU 合同、适配器的 2 项 CPU 合同和 9 个变体的原生编译全部通过。
两位独立审查者完成只读检查，未发现阻塞项。适配器提交与 Compiler 提交分别记录，适配器没有修改 C1。

本题沿用上游有效容差：`max_atol=1e-5`、`max_rtol=0.05`、`required_matched_ratio=0.99`、
`max_error_cap=null`。通过表示满足这组容差，不表示逐位相等。
160 次比较中观察到的最大相对误差为 0.013888889，最大绝对误差为 0.0625。
本次使用 C550-2 物理设备 7、PCI `0000:d7:00`，原生运行库来自 `/opt/maca-3.8.0.16/`，
PyTorch 报告 `torch.version.maca=3.8.0.4.c600u`，
本机独占锁 job 为 `maca-ec6464461d80`。评测结束后已核对该进程退出。

回执与构建材料保留在 C550-2 外置工作根，原始数据和生成报告未纳入仓库：

```text
/root/open-cake-runs-reviewed/c550-evolution-parallel-author-20261006/bench-c1-eval-20261009/
  compile-attempt-001/compiled/index.json
  l1-069-evaluation-001/result.json
```
