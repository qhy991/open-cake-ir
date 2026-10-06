# MetaX 设备事件均值计时

本后继实现将新 MetaX Run 的 `--metax-timing mean10-events` 接到已有配对 Evaluation。
默认仍为旧 MCPTI 路线；冻结 Run 不改写。用户已于 2026-10-06 明确同意新事件区间并另建基线。

- 使用当前已捕获的 MACA PyTorch 设备事件，记录同一默认 stream 上包围封存原生 launch 的区间。
- 区间包含暴露的 host 提交空隙，不是 MCPTI kernel-only 时间；两种结果不可直接拼接。
- 两块按 candidate/baseline、baseline/candidate 执行，每臂每块五个正式样本，共十个。
- 每块沿用十一轮预热，另计；均值为主指标，CV 和块胜数只作诊断。
- 每次正式采样前，在相同默认 stream 填充四倍声明 L2 的 FP32 缓冲区。
  这是可信执行器的 host 顺序保证，没有声称 profiler 验证了缓存实际清空或排除了外部工作。
- 计时路径不启用 MCPTI。MCPTI 仅在独立 attribution 阶段采集，失败仍保留。
- 正式计时数、正值、有限值、事件同步、reset 顺序、manifest 和设备身份均为硬门。
- 普通正确性仍覆盖完整 Workload。单 dispatch 事件计时不自动授予完整 Program 资格。

已有一次独立能力探针在 C550-2、MACA PyTorch 2.10.0+metax3.8.0.4.c600u 上观察到事件 API
可返回有效正值并通过固定加法输出检查；这不替代本实现的原生任务 A/A、独立 profile 和作者准入。

每个新 Run 显式绑定本协议。确认使用固定候选与新鲜的十样本均值；三小时包含确认预算。
统计口径与 interval、cache reset、原始事件样本一起保留。旧零宽 MCPTI 记录仍然无效，
新事件计时不重新解释它们。后继设备资格与批次结果保存在 checkout 外。
