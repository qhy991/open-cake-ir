# 按硬件查看实验结果

2026-09-20 整理。由 `tools/render_hardware_results.py` 生成；原报告与 Finding 为依据。

[交互目录](results/index.html) · [返回首页](../README.md) · [CAKE 复现](NVIDIA_CAKE_REPRODUCTION.md) · [最佳实现维护](zh-CN/TASK_INCUMBENTS.md)

**加速比 = 基线耗时 ÷ 候选耗时，超过 1× 表示更快。** 各行仅在自己的硬件、Workload、版本与计时协议内比较，不求跨平台平均值。

这是有明确来源的结果目录，不是全项目实时冠军榜。保留了失败、未实测与无显著差异的条目。服务实验展示每个 Campaign 的最佳合格候选；晋升记录和历史确认另作标记。

## 数据范围与更新

- B300 服务实验：本轮只读收集 29 份 report，涉及 27 个不同任务；27 个合格终点及 2 次未合格尝试。报告中的审计结论照录，没有重新审计或晋升。
- DCU：保留 29 个任务中的 27 个首个合格结果、2 个未合格任务，以及 3 次后续重复。13 个双方均为 5.439 µs 的结果不写成性能持平。原主机本轮 SSH 超时，尺寸与协议未从远端重新提取；表内明确保留缺项。
- Apple 晋升与跨版本结果来自注明日期的 Finding；尚未刷新 M4 registry 当前代，不能称作今天的全量最佳实现。
- 每项来源下列出原 Workload、版本、事件与产物定位。远端产物需要相应主机权限；文档与图不授予新验收资格。

更新流程：用 `tools/read_task_results.py --runs /原实验目录` 只读导出到仓库外的新文件，核对范围后保留带日期的公开字段投影；更新来源选择，再运行 `python tools/render_hardware_results.py --write`。
图使用 Matplotlib 3.10.6，运行 `--write --figures` 更新，`--check --figures` 检查图与目录是否过期。不要手改生成表格；原始实验报告继续保留在原证据根。

## NVIDIA

B300 的服务实验、CTA 宽度验证与 CAKE 改写各自保留基线和版本，不混成一个榜。B200 单列正确性证据。

| 设备 / 集合 | Task | 输入 / Workload | 基线 µs | 候选 µs | 加速比 | 状态 | 详情 |
|---|---|---|---:|---:|---:|---|---|
| B300 · 服务实验 | `absmax_rescale` | absmax-rescale-fp32-triton-b300-r128-c1024-v1 | 2.320 | 2.272 | 1.021× | 报告已确认 | [result-001](#result-001) |
| B300 · 服务实验 | `adadelta` | adadelta-fp32-triton-b300-r128-c1024-v1 | 4.160 | 4.096 | 1.016× | 报告已确认 | [result-002](#result-002) |
| B300 · 服务实验 | `adamw` | adamw-fp32-triton-b300-r128-c1024-v1 | 4.064 | 2.592 | 1.568× | 报告已确认 | [result-003](#result-003) |
| B300 · 服务实验 | `attention_decode` | contraction-attention-decode-fp32-triton-b300-r1024-k1024-n64-v1 | 339.874 | 230.769 | 1.473× | 报告已确认 | [result-004](#result-004) |
| B300 · 服务实验 | `bias_gradient_reduction` | bias-gradient-reduction-fp32-triton-b300-r128-c1024-v1 | — | — | — | 未合格终点 | [result-005](#result-005) |
| B300 · 服务实验 | `bias_gradient_reduction` | bias-gradient-reduction-fp32-triton-b300-r128-c1024-v1 | 2.784 | 2.752 | 1.012× | 报告已确认 | [result-006](#result-006) |
| B300 · 服务实验 | `channel_absmax_scale` | channel-absmax-scale-fp32-triton-b300-r128-c1024-v1 | 2.784 | 2.783 | 1.000× | 报告已确认 | [result-007](#result-007) |
| B300 · 服务实验 | `cosine_similarity` | cosine-similarity-fp32-triton-b300-r128-c1024-v1 | 2.976 | 2.977 | 1.000× | 报告已确认 | [result-008](#result-008) |
| B300 · 服务实验 | `gelu_tanh` | gelu-tanh-fp32-triton-b300-r128-c1024-v1 | 4.415 | 4.352 | 1.014× | 报告已确认 | [result-009](#result-009) |
| B300 · 服务实验 | `gelu_tanh_backward` | gelu-tanh-backward-fp32-triton-b300-r128-c1024-v1 | 4.960 | 2.720 | 1.824× | 报告已确认 | [result-010](#result-010) |
| B300 · 服务实验 | `gemm` | contraction-gemm-fp32-triton-b300-r1024-k1024-n64-v1 | 169.249 | 154.945 | 1.092× | 报告已确认 | [result-011](#result-011) |
| B300 · 服务实验 | `gemm_silu` | contraction-gemm-silu-fp32-triton-b300-r1024-k1024-n64-v1 | 166.081 | 178.305 | 0.931× | 报告已确认 | [result-012](#result-012) |
| B300 · 服务实验 | `layernorm` | layernorm-fp32-triton-b300-r128-c1024-v1 | 3.616 | 2.464 | 1.468× | 报告已确认 | [result-013](#result-013) |
| B300 · 服务实验 | `layernorm_backward_input` | layernorm-backward-input-fp32-triton-b300-r128-c1024-v2 | 3.168 | 3.040 | 1.042× | 报告已确认 | [result-014](#result-014) |
| B300 · 服务实验 | `momentum_sgd` | momentum-sgd-fp32-triton-b300-r128-c1024-v1 | 3.520 | 3.520 | 1.000× | 报告已确认 | [result-015](#result-015) |
| B300 · 服务实验 | `pairwise_sqdist` | contraction-pairwise-sqdist-fp32-triton-b300-r1024-k1024-n64-v1 | 411.795 | 53.856 | 7.646× | 报告已确认 | [result-016](#result-016) |
| B300 · 服务实验 | `per_channel_moments` | per-channel-moments-fp32-triton-b300-r128-c1024-v1 | 2.784 | 2.784 | 1.000× | 报告已确认 | [result-017](#result-017) |
| B300 · 服务实验 | `prelu` | prelu-fp32-triton-b300-r128-c1024-v1 | 2.896 | 2.832 | 1.023× | 报告已确认 | [result-018](#result-018) |
| B300 · 服务实验 | `residual_rmsnorm` | residual-rmsnorm-fp32-triton-b300-r128-c1024-v1 | 3.264 | 2.400 | 1.360× | 报告已确认 | [result-019](#result-019) |
| B300 · 服务实验 | `rmsnorm` | rmsnorm-fp32-triton-b300-r128-c1024-v3 | — | — | — | 未合格终点 | [result-020](#result-020) |
| B300 · 服务实验 | `rmsnorm` | rmsnorm-fp32-triton-b300-r128-c1024-v3 | 2.575 | 2.272 | 1.134× | 报告已确认 | [result-021](#result-021) |
| B300 · 服务实验 | `rmsnorm_input_gradient` | rmsnorm-input-gradient-fp32-triton-b300-r128-c1024-v2 | 3.040 | 2.432 | 1.250× | 报告已确认 | [result-022](#result-022) |
| B300 · 服务实验 | `selu` | selu-fp32-triton-b300-r128-c1024-v1 | 2.368 | 2.096 | 1.130× | 报告已确认 | [result-023](#result-023) |
| B300 · 服务实验 | `silu` | silu-fp32-triton-b300-r128-c1024-v1 | 2.432 | 2.112 | 1.152× | 报告已确认 | [result-024](#result-024) |
| B300 · 服务实验 | `softmax` | softmax-fp32-triton-b300-r128-c1024-v1 | 2.495 | 2.464 | 1.013× | 报告已确认 | [result-025](#result-025) |
| B300 · 服务实验 | `softmax_backward` | softmax-backward-fp32-triton-b300-r128-c1024-v1 | 2.944 | 2.592 | 1.136× | 报告已确认 | [result-026](#result-026) |
| B300 · 服务实验 | `softplus_gradient` | softplus-gradient-fp32-triton-b300-r128-c1024-v2 | 2.977 | 2.208 | 1.348× | 报告已确认 | [result-027](#result-027) |
| B300 · 服务实验 | `softsign` | softsign-fp32-triton-b300-r128-c1024-v1 | 2.432 | 2.400 | 1.013× | 报告已确认 | [result-028](#result-028) |
| B300 · 服务实验 | `swiglu` | swiglu-fp32-triton-b300-r128-c1024-v1 | 3.520 | 3.232 | 1.089× | 报告已确认 | [result-029](#result-029) |
| B300 · CTA 宽度验证 | `rmsnorm_input_gradient` | FP32 · R=128, C=1024 | 3.104 | 2.497 | 1.243× | 已确认 | [result-030](#result-030) |
| B300 · CTA 宽度验证 | `swiglu` | FP32 · R=128, C=1024 | 3.648 | 2.208 | 1.652× | 已确认 | [result-031](#result-031) |
| B300 · CTA 宽度验证 | `softmax_backward` | FP32 · R=128, C=1024 | 2.927 | 2.560 | 1.144× | 已确认 | [result-032](#result-032) |
| B300 · CTA 宽度验证 | `cosine_similarity` | FP32 · R=128, C=1024 | 2.976 | 2.528 | 1.177× | 已确认 | [result-033](#result-033) |
| B300 · CAKE 对照 | `TinyGEMM2 s2` | BF16+bias · B/N/K=1 / 128 / 720 | 2.720 | 99.969 | 0.027× | 正确；性能落后 | [result-034](#result-034) |
| B300 · CAKE 对照 | `TinyGEMM2 s4` | BF16+bias · B/N/K=1 / 128 / 720 | 2.720 | 100.192 | 0.027× | 正确；性能落后 | [result-035](#result-035) |
| B300 · CAKE 对照 | `TinyGEMM2 s2` | BF16+bias · B/N/K=16 / 1024 / 1024 | 3.040 | 106.016 | 0.029× | 正确；性能落后 | [result-036](#result-036) |
| B300 · CAKE 对照 | `TinyGEMM2 s4` | BF16+bias · B/N/K=16 / 1024 / 1024 | 3.040 | 105.825 | 0.029× | 正确；性能落后 | [result-037](#result-037) |
| B300 · CAKE 对照 | `TinyGEMM2 s2` | BF16+bias · B/N/K=64 / 4096 / 3072 | 21.216 | 239.969 | 0.088× | 正确；性能落后 | [result-038](#result-038) |
| B300 · CAKE 对照 | `TinyGEMM2 s4` | BF16+bias · B/N/K=64 / 4096 / 3072 | 21.216 | 289.186 | 0.073× | 正确；性能落后 | [result-039](#result-039) |
| B300 · CAKE 对照 | `KDA prefill` | 完整任务尚未完成 | — | — | — | 未实测 | [result-040](#result-040) |
| B300 · CAKE 对照 | `KDA decode` | 完整任务尚未完成 | — | — | — | 未实测 | [result-041](#result-041) |
| B300 · CAKE 对照 | `Alpha-MoE` | 完整任务尚未完成 | — | — | — | 未实测 | [result-042](#result-042) |
| B200 | `FMA / affine / state-store` | 固定正确性实例，详见各报告 | — | — | — | 仅正确性 | [result-043](#result-043) |

## Apple

M1 Pro、M4、M2 分开。Metal 测量整个 command buffer，并按重复 dispatch 均摊；不能与 CUPTI 绝对耗时横比。晋升标签仅表示保留记录记载过晋升，未重新读取远端 registry 的当前代。

| 设备 / 集合 | Task | 输入 / Workload | 基线 µs | 候选 µs | 加速比 | 状态 | 详情 |
|---|---|---|---:|---:|---:|---|---|
| M4 | `channel_absmax_scale` | FP32 · R=2；完整 C/协议由原 Workload 固定 | 0.788 | 0.712 | 1.107× | 有晋升记录 | [result-044](#result-044) |
| M4 | `bias_gradient_reduction` | FP32 · R=2；完整 C/协议由原 Workload 固定 | 0.772 | 0.678 | 1.138× | 有晋升记录 | [result-045](#result-045) |
| M1 Pro | `gemm_silu` | FP32 · M=128, K=256, N=32 | 1878.596 | 79.588 | 23.604× | 历史确认 | [result-046](#result-046) |
| M2 | `gemm` | 原记录的三个 GEMM 候选 | — | — | — | 计时未通过 | [result-047](#result-047) |

## Hygon DCU

BW1101 / gfx938：保留首个合格结果，同时展示未合格任务和后续重复运行。首个合格结果不等于历史最快；使用原记录的分类，不从显示的小数重新判定晋升。

| 设备 / 集合 | Task | 输入 / Workload | 基线 µs | 候选 µs | 加速比 | 状态 | 详情 |
|---|---|---|---:|---:|---:|---|---|
| BW1101 | `absmax_rescale` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 5.439 | — | 测量分辨能力待查 | [result-048](#result-048) |
| BW1101 | `adadelta` | 原 Campaign 的固定形状（汇总未转录尺寸） | 6.719 | 6.719 | 1.000× | 未检出显著差异 | [result-049](#result-049) |
| BW1101 | `adamw` | 原 Campaign 的固定形状（汇总未转录尺寸） | 7.839 | 7.839 | 1.000× | 未检出显著差异 | [result-050](#result-050) |
| BW1101 | `attention_decode` | 原 Campaign 的固定形状（汇总未转录尺寸） | 138.559 | 219.519 | 0.631× | 确认变慢 | [result-051](#result-051) |
| BW1101 | `bias_gradient_reduction` | 原 Campaign 的固定形状（汇总未转录尺寸） | 10.079 | 10.399 | 0.969× | 未检出显著差异 | [result-052](#result-052) |
| BW1101 | `channel_absmax_scale` | 原 Campaign 的固定形状（汇总未转录尺寸） | 9.599 | 9.599 | 1.000× | 未检出显著差异 | [result-053](#result-053) |
| BW1101 | `cosine_similarity` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 5.439 | — | 测量分辨能力待查 | [result-054](#result-054) |
| BW1101 | `gelu_tanh` | 原 Campaign 的固定形状（汇总未转录尺寸） | 6.399 | 6.239 | 1.026× | 未检出显著差异 | [result-055](#result-055) |
| BW1101 | `gelu_tanh_backward` | 原 Campaign 的固定形状（汇总未转录尺寸） | 6.719 | 12.639 | 0.532× | 确认变慢 | [result-056](#result-056) |
| BW1101 | `gemm` | 原 Campaign 的固定形状（汇总未转录尺寸） | 305.599 | 409.279 | 0.747× | 确认变慢 | [result-057](#result-057) |
| BW1101 | `layernorm` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 5.439 | — | 测量分辨能力待查 | [result-058](#result-058) |
| BW1101 | `layernorm_backward_input` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 5.439 | — | 测量分辨能力待查 | [result-059](#result-059) |
| BW1101 | `layernorm_gamma_beta_backward` | 原 Campaign 的固定形状（汇总未转录尺寸） | 14.879 | 15.199 | 0.979× | 未检出显著差异 | [result-060](#result-060) |
| BW1101 | `momentum_sgd` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.919 | 5.599 | 1.057× | 确认加速 | [result-061](#result-061) |
| BW1101 | `pairwise_sqdist` | 原 Campaign 的固定形状（汇总未转录尺寸） | 24.959 | 23.679 | 1.054× | 确认加速 | [result-062](#result-062) |
| BW1101 | `per_channel_moments` | 原 Campaign 的固定形状（汇总未转录尺寸） | 9.439 | 6.719 | 1.405× | 确认加速 | [result-063](#result-063) |
| BW1101 | `prelu` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 9.759 | 0.557× | 确认变慢 | [result-064](#result-064) |
| BW1101 | `residual_rmsnorm` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 5.439 | — | 测量分辨能力待查 | [result-065](#result-065) |
| BW1101 | `rmsnorm` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 5.439 | — | 测量分辨能力待查 | [result-066](#result-066) |
| BW1101 | `rmsnorm_input_gradient` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 5.439 | — | 测量分辨能力待查 | [result-067](#result-067) |
| BW1101 | `selu` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 5.439 | — | 测量分辨能力待查 | [result-068](#result-068) |
| BW1101 | `silu` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 5.439 | — | 测量分辨能力待查 | [result-069](#result-069) |
| BW1101 | `softmax` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 5.439 | — | 测量分辨能力待查 | [result-070](#result-070) |
| BW1101 | `softmax_backward` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 5.439 | — | 测量分辨能力待查 | [result-071](#result-071) |
| BW1101 | `softplus_gradient` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 5.439 | — | 测量分辨能力待查 | [result-072](#result-072) |
| BW1101 | `softsign` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 5.439 | — | 测量分辨能力待查 | [result-073](#result-073) |
| BW1101 | `swiglu` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.599 | 5.439 | 1.029× | 未检出显著差异 | [result-074](#result-074) |
| BW1101 | `gemm_bias` | 保留 sweep 的验证形状 | — | — | — | 基线未通过 | [result-075](#result-075) |
| BW1101 | `gemm_silu` | 保留 sweep 的验证形状 | — | — | — | 候选未通过 | [result-076](#result-076) |
| BW1101 · 后续重复 | `gelu_tanh` | 原 Campaign 的固定形状（汇总未转录尺寸） | 6.399 | 5.439 | — | 保留的后续运行 | [result-077](#result-077) |
| BW1101 · 后续重复 | `gemm` | 原 Campaign 的固定形状（汇总未转录尺寸） | 305.759 | 409.279 | — | 保留的后续运行 | [result-078](#result-078) |
| BW1101 · 后续重复 | `rmsnorm` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 5.439 | — | 保留的后续运行 | [result-079](#result-079) |

## AMD

gfx1151 的两种设备计时器尚未对齐。保留支持状态与调查入口，不给出合格性能或加速比。

| 设备 / 集合 | Task | 输入 / Workload | 基线 µs | 候选 µs | 加速比 | 状态 | 详情 |
|---|---|---|---:|---:|---:|---|---|
| Radeon / Strix Halo | `rmsnorm smoke` | gfx1151-rmsnorm-b8-smoke | — | — | — | 计时边界未解决 | [result-080](#result-080) |

## 演进记录怎么读

交互页展开 B300 服务实验时，可以看到同一 Campaign 内已通过确认的候选时间序列；它保留变慢点，不跨 Campaign 拼接成长期晋升曲线。
M1 Pro 的 GEMM-SiLU 曾在旧版本确认 23.60×，后继版本保留了计时未通过的 33.9× 搜索值以及另一个机制的 14.19× 合格终点；这些不是单调提升曲线。
M4 两项记录的 generation 0 证明晋升已发生，不足以画多代曲线。TinyGEMM2 的 22/30 到 30/30 是数值与资源处理进展，早期无有效计时。
跨编译器版本时分别看固定基线的变化与候选相对基线的改善，完整失败过程见各条来源。

## 逐项来源

### result-001

**B300 · 服务实验 · absmax_rescale** — 2026-09-16 / 报告已确认

- Workload：`absmax-rescale-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](../docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/absmax_rescale-20260916-100813/report.json; open_cake-1; event=17; candidate=a363e12177e7c493ca3748b23b5fad0515f84efc761d0bd62d6c37e02fa2adb5`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 2 | 17 | 2.272 | 1.021× |

### result-002

**B300 · 服务实验 · adadelta** — 2026-09-16 / 报告已确认

- Workload：`adadelta-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](../docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/adadelta-20260916-060824/report.json; open_cake-1; event=17; candidate=1eb5ed817fcf0ddc63179af3c7eed9d6cda9354e7d6e3ebed6506e8f08960904`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 2 | 17 | 4.096 | 1.016× |

### result-003

**B300 · 服务实验 · adamw** — 2026-09-16 / 报告已确认

- Workload：`adamw-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](../docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/adamw-20260916-053018/report.json; open_cake-1; event=26; candidate=398044880e6094b2c0e585775b7d0753650db52ad51d2aecc6aa0200ea09cc4f`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 4.000 | 1.016× |
| 2 | 26 | 2.592 | 1.568× |

### result-004

**B300 · 服务实验 · attention_decode** — 2026-09-16 / 报告已确认

- Workload：`contraction-attention-decode-fp32-triton-b300-r1024-k1024-n64-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](../docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/attention_decode-20260916-184301/report.json; open_cake-1; event=13; candidate=62bc95ec21df257e1f1f7f7b28d6845dc47a6af5d32c5d970b1f840af27ad5bf`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 230.769 | 1.473× |

### result-005

**B300 · 服务实验 · bias_gradient_reduction** — 2026-09-16 / 未合格终点

- Workload：`bias-gradient-reduction-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](../docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/bias_gradient_reduction-20260916-065950/report.json; open_cake-1; event=None; candidate=None`。
- provider_fault; missing; None

### result-006

**B300 · 服务实验 · bias_gradient_reduction** — 2026-09-16 / 报告已确认

- Workload：`bias-gradient-reduction-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](../docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/bias_gradient_reduction-20260916-130258/report.json; open_cake-1; event=13; candidate=1f9b4ad199fc61c3ae923d998bd1ac59d1fbcde65649725e66f64e989b791df8`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 2.752 | 1.012× |

### result-007

**B300 · 服务实验 · channel_absmax_scale** — 2026-09-16 / 报告已确认

- Workload：`channel-absmax-scale-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](../docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/channel_absmax_scale-20260916-124804/report.json; open_cake-1; event=26; candidate=bb02cc80ae5a1785f89440d86203f1d058c61b80074629edf8a0d7f4c915e743`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 2.784 | 1.000× |
| 2 | 26 | 2.783 | 1.000× |

### result-008

**B300 · 服务实验 · cosine_similarity** — 2026-09-16 / 报告已确认

- Workload：`cosine-similarity-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](../docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/cosine_similarity-20260916-102543/report.json; open_cake-1; event=13; candidate=f739fb48dec1050ae77b2abbe731a106d055be484eb657fada04e1b108a4fea4`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 2.977 | 1.000× |

### result-009

**B300 · 服务实验 · gelu_tanh** — 2026-09-16 / 报告已确认

- Workload：`gelu-tanh-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](../docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/gelu_tanh-20260916-042956/report.json; open_cake-1; event=17; candidate=f42e4152b65487a489d44963da52469166425ebf2e7862416ff128272dbc65b8`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 2 | 17 | 4.352 | 1.014× |

### result-010

**B300 · 服务实验 · gelu_tanh_backward** — 2026-09-16 / 报告已确认

- Workload：`gelu-tanh-backward-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](../docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/gelu_tanh_backward-20260916-120441/report.json; open_cake-1; event=17; candidate=62aca837508c4a2dbd5cc1c14235d6833b26cbc858dad707288bec74f48c92aa`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 2 | 17 | 2.720 | 1.824× |

### result-011

**B300 · 服务实验 · gemm** — 2026-09-16 / 报告已确认

- Workload：`contraction-gemm-fp32-triton-b300-r1024-k1024-n64-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](../docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/gemm-20260916-170620/report.json; open_cake-1; event=17; candidate=445d2727b4e39d3aae77f38415ea7ebc40190db07ffd325674ebb05c380fd454`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 2 | 17 | 154.945 | 1.092× |

### result-012

**B300 · 服务实验 · gemm_silu** — 2026-09-16 / 报告已确认

- Workload：`contraction-gemm-silu-fp32-triton-b300-r1024-k1024-n64-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](../docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/gemm_silu-20260916-172338/report.json; open_cake-1; event=26; candidate=0cfa7400e15901e6e39968f14f956117ca8ca6c9d622cceae04e339f6ea34860`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 324.274 | 0.509× |
| 2 | 26 | 178.305 | 0.931× |

### result-013

**B300 · 服务实验 · layernorm** — 2026-09-16 / 报告已确认

- Workload：`layernorm-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](../docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/layernorm-20260916-034037/report.json; open_cake-1; event=13; candidate=d110d259af65053f62a441195bd51dd4d3827ad3210324e3be89ffa87c6ebf9c`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 2.464 | 1.468× |
| 2 | 26 | 2.465 | 1.467× |

### result-014

**B300 · 服务实验 · layernorm_backward_input** — 2026-09-16 / 报告已确认

- Workload：`layernorm-backward-input-fp32-triton-b300-r128-c1024-v2`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](../docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/layernorm_backward_input-20260916-094015/report.json; open_cake-1; event=26; candidate=b123db1855356f93ed21bb3063101963857e893d120c33dd837b05b0bcb9819b`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 3.200 | 0.990× |
| 2 | 26 | 3.040 | 1.042× |

### result-015

**B300 · 服务实验 · momentum_sgd** — 2026-09-16 / 报告已确认

- Workload：`momentum-sgd-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](../docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/momentum_sgd-20260916-051315/report.json; open_cake-1; event=26; candidate=a17d866faea721d4a7ea9ee3579af848a649d0625d7470b64430d3256f80d62f`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 4.864 | 0.727× |
| 2 | 26 | 3.520 | 1.000× |

### result-016

**B300 · 服务实验 · pairwise_sqdist** — 2026-09-16 / 报告已确认

- Workload：`contraction-pairwise-sqdist-fp32-triton-b300-r1024-k1024-n64-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](../docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/pairwise_sqdist-20260916-180016/report.json; open_cake-1; event=17; candidate=1c24a27ae8410b36407e837607d0349d0d152634f6759da19283697afccc0962`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 2 | 17 | 53.856 | 7.646× |

### result-017

**B300 · 服务实验 · per_channel_moments** — 2026-09-16 / 报告已确认

- Workload：`per-channel-moments-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](../docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/per_channel_moments-20260916-104523/report.json; open_cake-1; event=13; candidate=b087e9645f5ba0d469c650fde3fc65fda1f79c50b8d150c0f5c633feee3ad3df`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 2.784 | 1.000× |

### result-018

**B300 · 服务实验 · prelu** — 2026-09-16 / 报告已确认

- Workload：`prelu-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](../docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/prelu-20260916-122027/report.json; open_cake-1; event=13; candidate=56e850a984d225bdc58d359bc91fd12a28074af6c56aba9df64ee1a77591ebcf`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 2.832 | 1.023× |

### result-019

**B300 · 服务实验 · residual_rmsnorm** — 2026-09-16 / 报告已确认

- Workload：`residual-rmsnorm-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](../docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/residual_rmsnorm-20260916-044120/report.json; open_cake-1; event=26; candidate=533df099cbd8ced4e5be0305fa946ea1691c7249797ff9917f194c92aa35c999`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 3.232 | 1.000× |
| 2 | 26 | 2.400 | 1.360× |

### result-020

**B300 · 服务实验 · rmsnorm** — 2026-09-16 / 未合格终点

- Workload：`rmsnorm-fp32-triton-b300-r128-c1024-v3`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](../docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/rmsnorm-20260916-031407/report.json; open_cake-1; event=None; candidate=None`。
- broker_fault; missing; None

### result-021

**B300 · 服务实验 · rmsnorm** — 2026-09-16 / 报告已确认

- Workload：`rmsnorm-fp32-triton-b300-r128-c1024-v3`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](../docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/rmsnorm-20260916-032057/report.json; open_cake-1; event=26; candidate=3d9385205d3413906415268c8c6e2b85cc82a8de9dfa213708f0870989f77802`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 2.400 | 1.080× |
| 2 | 26 | 2.272 | 1.134× |

### result-022

**B300 · 服务实验 · rmsnorm_input_gradient** — 2026-09-16 / 报告已确认

- Workload：`rmsnorm-input-gradient-fp32-triton-b300-r128-c1024-v2`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](../docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/rmsnorm_input_gradient-20260916-064534/report.json; open_cake-1; event=26; candidate=77fdf24695c20187bd7e22ccb6bb332c329d425498f1a098c1c38950952fc940`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 2.688 | 1.095× |
| 2 | 26 | 2.432 | 1.250× |

### result-023

**B300 · 服务实验 · selu** — 2026-09-16 / 报告已确认

- Workload：`selu-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](../docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/selu-20260916-112941/report.json; open_cake-1; event=13; candidate=8bc67babc39b24758c111e18d6f669c04940077f4fef70c965945aee5bec83fa`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 2.096 | 1.130× |
| 2 | 26 | 2.112 | 1.121× |

### result-024

**B300 · 服务实验 · silu** — 2026-09-16 / 报告已确认

- Workload：`silu-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](../docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/silu-20260916-040643/report.json; open_cake-1; event=26; candidate=282f390f27dc28fcb86e9bafad4c5d1aeb1ee77bf86c135d6bee41e4fb3d5ddc`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 2.432 | 1.000× |
| 2 | 26 | 2.112 | 1.152× |

### result-025

**B300 · 服务实验 · softmax** — 2026-09-16 / 报告已确认

- Workload：`softmax-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](../docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/softmax-20260916-035525/report.json; open_cake-1; event=17; candidate=4484a1e9ff3a2b46bf87bf1971d1c824321c225f3c0ab3c3f3ab6bd67edea845`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 2 | 17 | 2.464 | 1.013× |

### result-026

**B300 · 服务实验 · softmax_backward** — 2026-09-16 / 报告已确认

- Workload：`softmax-backward-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](../docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/softmax_backward-20260916-062444/report.json; open_cake-1; event=13; candidate=9a920f573083bc5a81c073abcd497aeafe1a2ac7371477e51b600b2673a5ac79`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 2.592 | 1.136× |
| 2 | 26 | 2.592 | 1.136× |

### result-027

**B300 · 服务实验 · softplus_gradient** — 2026-09-16 / 报告已确认

- Workload：`softplus-gradient-fp32-triton-b300-r128-c1024-v2`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](../docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/softplus_gradient-20260916-114305/report.json; open_cake-1; event=26; candidate=7c347746d32c355786837f6905864de79bcd3cafa34fbf92642a91ac10bb6d8a`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 2.944 | 1.000× |
| 2 | 26 | 2.208 | 1.348× |

### result-028

**B300 · 服务实验 · softsign** — 2026-09-16 / 报告已确认

- Workload：`softsign-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](../docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/softsign-20260916-111344/report.json; open_cake-1; event=13; candidate=c980c8bfc21b13b7e500597320d127c21771cd0bc8131b77e671c1c66493d142`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 2.400 | 1.013× |

### result-029

**B300 · 服务实验 · swiglu** — 2026-09-16 / 报告已确认

- Workload：`swiglu-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](../docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/swiglu-20260916-041831/report.json; open_cake-1; event=13; candidate=312b3a5636d5fef356c884dfc408b0e2d062467d3e7ea04e748ae6e002e14fd2`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 3.232 | 1.089× |
| 2 | 26 | 4.544 | 0.796× |

### result-030

**B300 · CTA 宽度验证 · rmsnorm_input_gradient** — 2026-09-15 / 已确认

- Workload：`FP32 · R=128, C=1024`；目标：`sm_103a`；版本：`6198976b`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[findings/2026-09-15-005-explicit-triton-cta-width.json](../findings/2026-09-15-005-explicit-triton-cta-width.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oc-width-ticktock-20260915-DBihzZ/evaluation/rmsnorm_input_gradient/w8/confirmatory/receipt.json`。
- 显式选择 8 warps；10/10 配对胜出，五种输入与稳定性通过，保留 NCU。无通用最优宽度结论。

### result-031

**B300 · CTA 宽度验证 · swiglu** — 2026-09-15 / 已确认

- Workload：`FP32 · R=128, C=1024`；目标：`sm_103a`；版本：`6198976b`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[findings/2026-09-15-005-explicit-triton-cta-width.json](../findings/2026-09-15-005-explicit-triton-cta-width.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oc-width-ticktock-20260915-DBihzZ/evaluation/swiglu/w16/confirmatory/receipt.json`。
- 显式选择 16 warps；10/10 配对胜出，五种输入与稳定性通过，保留 NCU。无通用最优宽度结论。

### result-032

**B300 · CTA 宽度验证 · softmax_backward** — 2026-09-15 / 已确认

- Workload：`FP32 · R=128, C=1024`；目标：`sm_103a`；版本：`a5734239`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[findings/2026-09-15-005-explicit-triton-cta-width.json](../findings/2026-09-15-005-explicit-triton-cta-width.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oc-width-transfer-20260915-KoekDV/evaluation/softmax_backward/w16/confirmatory/receipt.json`。
- 显式选择 16 warps；10/10 配对胜出，五种输入与稳定性通过，保留 NCU。无通用最优宽度结论。

### result-033

**B300 · CTA 宽度验证 · cosine_similarity** — 2026-09-15 / 已确认

- Workload：`FP32 · R=128, C=1024`；目标：`sm_103a`；版本：`a5734239`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[findings/2026-09-15-005-explicit-triton-cta-width.json](../findings/2026-09-15-005-explicit-triton-cta-width.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oc-width-transfer-20260915-KoekDV/evaluation/cosine_similarity/w8/confirmatory/receipt.json`。
- 显式选择 8 warps；10/10 配对胜出，五种输入与稳定性通过，保留 NCU。无通用最优宽度结论。

### result-034

**B300 · CAKE 对照 · TinyGEMM2 s2** — 2026-09-20 / 正确；性能落后

- Workload：`BF16+bias · B/N/K=1 / 128 / 720`；目标：`sm_103a`；版本：`539c6c81`。
- 基线：官方 CAKE stage4 固定导出；比值口径：`inverse_rounded_slowdown`。
- 来源：[docs/NVIDIA_CAKE_REPRODUCTION.md](../docs/NVIDIA_CAKE_REPRODUCTION.md)。
- 原记录 / 实现定位：`B300-M2:cake-tinygemm-compare-b300-539c6c81-a98903082f5c/stages/comparison/checks/report.json`。
- 30/30 Open-Cake 与 15/15 CAKE 检查通过。此加速比由报告中四舍五入的 36.72x 耗时倍数取倒数，约数；非新配对计算。s2/s4 在短 K 无循环时不是两种流水线。

### result-035

**B300 · CAKE 对照 · TinyGEMM2 s4** — 2026-09-20 / 正确；性能落后

- Workload：`BF16+bias · B/N/K=1 / 128 / 720`；目标：`sm_103a`；版本：`539c6c81`。
- 基线：官方 CAKE stage4 固定导出；比值口径：`inverse_rounded_slowdown`。
- 来源：[docs/NVIDIA_CAKE_REPRODUCTION.md](../docs/NVIDIA_CAKE_REPRODUCTION.md)。
- 原记录 / 实现定位：`B300-M2:cake-tinygemm-compare-b300-539c6c81-a98903082f5c/stages/comparison/checks/report.json`。
- 30/30 Open-Cake 与 15/15 CAKE 检查通过。此加速比由报告中四舍五入的 36.77x 耗时倍数取倒数，约数；非新配对计算。s2/s4 在短 K 无循环时不是两种流水线。

### result-036

**B300 · CAKE 对照 · TinyGEMM2 s2** — 2026-09-20 / 正确；性能落后

- Workload：`BF16+bias · B/N/K=16 / 1024 / 1024`；目标：`sm_103a`；版本：`539c6c81`。
- 基线：官方 CAKE stage4 固定导出；比值口径：`inverse_rounded_slowdown`。
- 来源：[docs/NVIDIA_CAKE_REPRODUCTION.md](../docs/NVIDIA_CAKE_REPRODUCTION.md)。
- 原记录 / 实现定位：`B300-M2:cake-tinygemm-compare-b300-539c6c81-a98903082f5c/stages/comparison/checks/report.json`。
- 30/30 Open-Cake 与 15/15 CAKE 检查通过。此加速比由报告中四舍五入的 34.87x 耗时倍数取倒数，约数；非新配对计算。s2/s4 在短 K 无循环时不是两种流水线。

### result-037

**B300 · CAKE 对照 · TinyGEMM2 s4** — 2026-09-20 / 正确；性能落后

- Workload：`BF16+bias · B/N/K=16 / 1024 / 1024`；目标：`sm_103a`；版本：`539c6c81`。
- 基线：官方 CAKE stage4 固定导出；比值口径：`inverse_rounded_slowdown`。
- 来源：[docs/NVIDIA_CAKE_REPRODUCTION.md](../docs/NVIDIA_CAKE_REPRODUCTION.md)。
- 原记录 / 实现定位：`B300-M2:cake-tinygemm-compare-b300-539c6c81-a98903082f5c/stages/comparison/checks/report.json`。
- 30/30 Open-Cake 与 15/15 CAKE 检查通过。此加速比由报告中四舍五入的 34.81x 耗时倍数取倒数，约数；非新配对计算。s2/s4 在短 K 无循环时不是两种流水线。

### result-038

**B300 · CAKE 对照 · TinyGEMM2 s2** — 2026-09-20 / 正确；性能落后

- Workload：`BF16+bias · B/N/K=64 / 4096 / 3072`；目标：`sm_103a`；版本：`539c6c81`。
- 基线：官方 CAKE stage4 固定导出；比值口径：`inverse_rounded_slowdown`。
- 来源：[docs/NVIDIA_CAKE_REPRODUCTION.md](../docs/NVIDIA_CAKE_REPRODUCTION.md)。
- 原记录 / 实现定位：`B300-M2:cake-tinygemm-compare-b300-539c6c81-a98903082f5c/stages/comparison/checks/report.json`。
- 30/30 Open-Cake 与 15/15 CAKE 检查通过。此加速比由报告中四舍五入的 11.31x 耗时倍数取倒数，约数；非新配对计算。s2/s4 在短 K 无循环时不是两种流水线。

### result-039

**B300 · CAKE 对照 · TinyGEMM2 s4** — 2026-09-20 / 正确；性能落后

- Workload：`BF16+bias · B/N/K=64 / 4096 / 3072`；目标：`sm_103a`；版本：`539c6c81`。
- 基线：官方 CAKE stage4 固定导出；比值口径：`inverse_rounded_slowdown`。
- 来源：[docs/NVIDIA_CAKE_REPRODUCTION.md](../docs/NVIDIA_CAKE_REPRODUCTION.md)。
- 原记录 / 实现定位：`B300-M2:cake-tinygemm-compare-b300-539c6c81-a98903082f5c/stages/comparison/checks/report.json`。
- 30/30 Open-Cake 与 15/15 CAKE 检查通过。此加速比由报告中四舍五入的 13.63x 耗时倍数取倒数，约数；非新配对计算。s2/s4 在短 K 无循环时不是两种流水线。

### result-040

**B300 · CAKE 对照 · KDA prefill** — 2026-09-20 / 未实测

- Workload：`完整任务尚未完成`；目标：`sm_103a`；版本：`见来源`。
- 基线：官方 CAKE 导出；比值口径：`paired`。
- 来源：[docs/NVIDIA_CAKE_REPRODUCTION.md](../docs/NVIDIA_CAKE_REPRODUCTION.md)。
- 原记录 / 实现定位：`见来源所引用的原始实验`。
- 已有源参考与准备规格；尚无完整合格候选。

### result-041

**B300 · CAKE 对照 · KDA decode** — 2026-09-20 / 未实测

- Workload：`完整任务尚未完成`；目标：`sm_103a`；版本：`见来源`。
- 基线：官方 CAKE 导出；比值口径：`paired`。
- 来源：[docs/NVIDIA_CAKE_REPRODUCTION.md](../docs/NVIDIA_CAKE_REPRODUCTION.md)。
- 原记录 / 实现定位：`见来源所引用的原始实验`。
- 已有源参考与准备规格；尚无完整合格候选。

### result-042

**B300 · CAKE 对照 · Alpha-MoE** — 2026-09-20 / 未实测

- Workload：`完整任务尚未完成`；目标：`sm_103a`；版本：`见来源`。
- 基线：官方 CAKE 导出；比值口径：`paired`。
- 来源：[docs/NVIDIA_CAKE_REPRODUCTION.md](../docs/NVIDIA_CAKE_REPRODUCTION.md)。
- 原记录 / 实现定位：`见来源所引用的原始实验`。
- 已有源参考与准备规格；尚无完整合格候选。

### result-043

**B200 · FMA / affine / state-store** — 2026-09-06 / 仅正确性

- Workload：`固定正确性实例，详见各报告`；目标：`sm_100a`；版本：`见来源`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/AKA_FMA_B200_CORRECTNESS_20260904.md](../docs/AKA_FMA_B200_CORRECTNESS_20260904.md)。
- 原记录 / 实现定位：`见来源所引用的原始实验`。
- 没有此组实例的合格计时；另见 AFFINE_PARENT_B200_CANARY_20260906 与 STATE_STORE_B200_CORRECTNESS_20260903。

### result-044

**M4 · channel_absmax_scale** — 2026-09-12 / 有晋升记录

- Workload：`FP32 · R=2；完整 C/协议由原 Workload 固定`；目标：`apple_gpu_family9`；版本：`v74/v104`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[findings/2026-09-12-001-r2-row-reduction-scalarization.json](../findings/2026-09-12-001-r2-row-reduction-scalarization.json)。
- 原记录 / 实现定位：`open-cake-ir-experiments:m4-family9/matrix-kimi-k3-high-v104-20260911/channel_absmax_scale; open_cake-1; event 13`。
- 记录中的 generation 0；并非本轮重新查询的当前冠军。channel_absmax 的晋升另见 F-2026-09-12-003。

### result-045

**M4 · bias_gradient_reduction** — 2026-09-13 / 有晋升记录

- Workload：`FP32 · R=2；完整 C/协议由原 Workload 固定`；目标：`apple_gpu_family9`；版本：`v78/v113`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[findings/2026-09-12-001-r2-row-reduction-scalarization.json](../findings/2026-09-12-001-r2-row-reduction-scalarization.json)。
- 原记录 / 实现定位：`open-cake-ir-experiments:m4-family9/matrix-kimi-k3-high-v78-v113-incumbent-20260913/bias_gradient_reduction; open_cake-1; event 13`。
- 记录中的 generation 0；并非本轮重新查询的当前冠军。channel_absmax 的晋升另见 F-2026-09-12-003。

### result-046

**M1 Pro · gemm_silu** — 2026-09-11 / 历史确认

- Workload：`FP32 · M=128, K=256, N=32`；目标：`apple_gpu_family7`；版本：`v73/v98`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[findings/2026-09-11-004-metal-output-column-specialization.json](../findings/2026-09-11-004-metal-output-column-specialization.json)。
- 原记录 / 实现定位：`open-cake-ir-experiments:m1pro-family7/gemm-silu-glm53-high-20260910n32; confirmatory event 37; candidate 450d8350`。
- 固定小形状相对朴素基线；后继合格候选为另一种 K-split 机制，未证明跨版本持续加速。

### result-047

**M2 · gemm** — 2026-09-10 / 计时未通过

- Workload：`原记录的三个 GEMM 候选`；目标：`apple_gpu_family8`；版本：`见来源`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[findings/2026-09-10-010-contraction-regime-shows-material-headroom.json](../findings/2026-09-10-010-contraction-regime-shows-material-headroom.json)。
- 原记录 / 实现定位：`见来源所引用的原始实验`。
- 搜索曾观察 6–7×，均未通过测量稳定性门；不列入性能图。

### result-048

**BW1101 · absmax_rescale** — 2026-09-18 / 测量分辨能力待查

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@0970a36e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/absmax_rescale-20260918-002242/campaign-evidence/objects/sha256/93/936ba860b59b2716f843de288526fa8659f6915fdb121d537108108d9d0cca66`。
- 每任务首个合格运行，不是 fastest-of-all。两者均读到 5.439 µs；不写成 1× 性能持平。

### result-049

**BW1101 · adadelta** — 2026-09-18 / 未检出显著差异

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@0970a36e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/adadelta-20260918-015416/campaign-evidence/objects/sha256/5a/5a8ca7e823848cd2623dde3178f95d9da3e7198fdb68d6433107a40debeab742`。
- 每任务首个合格运行，不是 fastest-of-all。比值来自确认中位数；是否超过提升门槛沿用原 classification。

### result-050

**BW1101 · adamw** — 2026-09-18 / 未检出显著差异

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@5151954e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/adamw-20260917-195709/campaign-evidence/objects/sha256/de/de177a4c4a0ae0866d2a63ab2117d09142ebf2f57f43287a9bd9e1bae2eef9e4`。
- 每任务首个合格运行，不是 fastest-of-all。比值来自确认中位数；是否超过提升门槛沿用原 classification。

### result-051

**BW1101 · attention_decode** — 2026-09-18 / 确认变慢

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@4278caf2`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/attention_decode-20260918-070030/campaign-evidence/objects/sha256/40/4098e094b242916b4edcadb7f8cf096e3bbbed5d765673eb35a02667f0b4bb0a`。
- 每任务首个合格运行，不是 fastest-of-all。比值来自确认中位数；是否超过提升门槛沿用原 classification。

### result-052

**BW1101 · bias_gradient_reduction** — 2026-09-18 / 未检出显著差异

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@0970a36e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/bias_gradient_reduction-20260918-011329/campaign-evidence/objects/sha256/60/60268d83cf58bc39f68d44f68c6cd850331fd9ee7f11418c1fa212b45d9e1121`。
- 每任务首个合格运行，不是 fastest-of-all。比值来自确认中位数；是否超过提升门槛沿用原 classification。

### result-053

**BW1101 · channel_absmax_scale** — 2026-09-18 / 未检出显著差异

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@4278caf2`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/channel_absmax_scale-20260918-074053/campaign-evidence/objects/sha256/39/39e552743464224b12d914af1d2a716bfb5ac468d4d763248be0e961b26f7942`。
- 每任务首个合格运行，不是 fastest-of-all。比值来自确认中位数；是否超过提升门槛沿用原 classification。

### result-054

**BW1101 · cosine_similarity** — 2026-09-18 / 测量分辨能力待查

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@4278caf2`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/cosine_similarity-20260918-052136/campaign-evidence/objects/sha256/73/73a5efe324e6974474d66691f18458010d973221f007a12c076f3a40b0afda51`。
- 每任务首个合格运行，不是 fastest-of-all。两者均读到 5.439 µs；不写成 1× 性能持平。

### result-055

**BW1101 · gelu_tanh** — 2026-09-18 / 未检出显著差异

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@0970a36e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/gelu_tanh-20260917-224115/campaign-evidence/objects/sha256/87/87033d2251a648103290097aeea51f78b07166b7185804835eb3be2a557ef252`。
- 每任务首个合格运行，不是 fastest-of-all。比值来自确认中位数；是否超过提升门槛沿用原 classification。

### result-056

**BW1101 · gelu_tanh_backward** — 2026-09-18 / 确认变慢

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@0970a36e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/gelu_tanh_backward-20260917-213519/campaign-evidence/objects/sha256/fd/fddc68b73a893a3135d7cbab7d78752217011d5fdd9f4f039169fc975526b901`。
- 每任务首个合格运行，不是 fastest-of-all。比值来自确认中位数；是否超过提升门槛沿用原 classification。

### result-057

**BW1101 · gemm** — 2026-09-18 / 确认变慢

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@0970a36e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/gemm-20260917-225918/campaign-evidence/objects/sha256/29/293418506b77308dd35947557b6bc7899846a2714bba71af1c2a2f83aca12aaa`。
- 每任务首个合格运行，不是 fastest-of-all。比值来自确认中位数；是否超过提升门槛沿用原 classification。

### result-058

**BW1101 · layernorm** — 2026-09-18 / 测量分辨能力待查

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@0970a36e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/layernorm-20260917-222113/campaign-evidence/objects/sha256/52/529ff05598a641f3a4719abbdc9146f0848e5d3d7e1830eea33275653a6373cb`。
- 每任务首个合格运行，不是 fastest-of-all。两者均读到 5.439 µs；不写成 1× 性能持平。

### result-059

**BW1101 · layernorm_backward_input** — 2026-09-18 / 测量分辨能力待查

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@0970a36e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/layernorm_backward_input-20260918-005411/campaign-evidence/objects/sha256/a9/a965a96d2f1fed63a12cf28e045845bb53e16c884103cb9b0ec0281966314aad`。
- 每任务首个合格运行，不是 fastest-of-all。两者均读到 5.439 µs；不写成 1× 性能持平。

### result-060

**BW1101 · layernorm_gamma_beta_backward** — 2026-09-18 / 未检出显著差异

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@4278caf2`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/layernorm_gamma_beta_backward-20260918-062624/campaign-evidence/objects/sha256/9c/9ce5553d9c030a548d4b0d7759be7d83bbe67275edfc94b7138f20df567b2226`。
- 每任务首个合格运行，不是 fastest-of-all。比值来自确认中位数；是否超过提升门槛沿用原 classification。

### result-061

**BW1101 · momentum_sgd** — 2026-09-18 / 确认加速

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@0970a36e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/momentum_sgd-20260918-022953/campaign-evidence/objects/sha256/cb/cb1ad30f1dc8eaf56071d45203d0c57dbea7c6b8c695e75846c3e8e2d496526c`。
- 每任务首个合格运行，不是 fastest-of-all。比值来自确认中位数；是否超过提升门槛沿用原 classification。

### result-062

**BW1101 · pairwise_sqdist** — 2026-09-18 / 确认加速

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@4278caf2`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/pairwise_sqdist-20260918-045643/campaign-evidence/objects/sha256/26/264e9c4a30b4183bf6494aff778434daee991b233cd4918a748ccbaabf18d1c9`。
- 每任务首个合格运行，不是 fastest-of-all。比值来自确认中位数；是否超过提升门槛沿用原 classification。

### result-063

**BW1101 · per_channel_moments** — 2026-09-18 / 确认加速

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@4278caf2`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/per_channel_moments-20260918-060857/campaign-evidence/objects/sha256/12/122489d13e79061e068d709b428a80e7ac039f7efeaa5a2c9242dfac13c133ef`。
- 每任务首个合格运行，不是 fastest-of-all。比值来自确认中位数；是否超过提升门槛沿用原 classification。

### result-064

**BW1101 · prelu** — 2026-09-18 / 确认变慢

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@0970a36e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/prelu-20260917-232247/campaign-evidence/objects/sha256/10/10d7db433322f8dfa54c0d66aa1d81a207832ae8092358be9319ed2320a58404`。
- 每任务首个合格运行，不是 fastest-of-all。比值来自确认中位数；是否超过提升门槛沿用原 classification。

### result-065

**BW1101 · residual_rmsnorm** — 2026-09-18 / 测量分辨能力待查

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@4278caf2`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/residual_rmsnorm-20260918-053055/campaign-evidence/objects/sha256/5e/5e0889655e4306927b8f750728613fb933806170899e8220a0cf0715f975d506`。
- 每任务首个合格运行，不是 fastest-of-all。两者均读到 5.439 µs；不写成 1× 性能持平。

### result-066

**BW1101 · rmsnorm** — 2026-09-18 / 测量分辨能力待查

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@5151954e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/rmsnorm-20260917-174616/campaign-evidence/objects/sha256/2b/2b07b39b143778658c71feb364c710b0b460647a5d2970d5dbd5745e34f66818`。
- 每任务首个合格运行，不是 fastest-of-all。两者均读到 5.439 µs；不写成 1× 性能持平。

### result-067

**BW1101 · rmsnorm_input_gradient** — 2026-09-18 / 测量分辨能力待查

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@5151954e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/rmsnorm_input_gradient-20260917-192819/campaign-evidence/objects/sha256/78/78b89806977d0387aadcff006ae95a538254240f20750322f8616bbea774d8ed`。
- 每任务首个合格运行，不是 fastest-of-all。两者均读到 5.439 µs；不写成 1× 性能持平。

### result-068

**BW1101 · selu** — 2026-09-18 / 测量分辨能力待查

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@0970a36e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/selu-20260917-234230/campaign-evidence/objects/sha256/74/74e244f32261efc0b57817644eccc5f3b2f89d8674de7be6f03f86993e2be98f`。
- 每任务首个合格运行，不是 fastest-of-all。两者均读到 5.439 µs；不写成 1× 性能持平。

### result-069

**BW1101 · silu** — 2026-09-18 / 测量分辨能力待查

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@5151954e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/silu-20260917-184123/campaign-evidence/objects/sha256/25/25a10f330efbf8bd169e8ad1415198b9108b4ba5b33586fff3bdf790ca5fbd91`。
- 每任务首个合格运行，不是 fastest-of-all。两者均读到 5.439 µs；不写成 1× 性能持平。

### result-070

**BW1101 · softmax** — 2026-09-18 / 测量分辨能力待查

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@4278caf2`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/softmax-20260918-073121/campaign-evidence/objects/sha256/09/09a45892d4a2d420ebd999610cc41cc9df9c973e4ffc0423646bee19fd72109a`。
- 每任务首个合格运行，不是 fastest-of-all。两者均读到 5.439 µs；不写成 1× 性能持平。

### result-071

**BW1101 · softmax_backward** — 2026-09-18 / 测量分辨能力待查

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@5151954e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/softmax_backward-20260917-191303/campaign-evidence/objects/sha256/68/68dbdcaba2946f42d631f862943f467f2e1eb9b1732ed04447eeaec9103fd425`。
- 每任务首个合格运行，不是 fastest-of-all。两者均读到 5.439 µs；不写成 1× 性能持平。

### result-072

**BW1101 · softplus_gradient** — 2026-09-18 / 测量分辨能力待查

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@0970a36e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/softplus_gradient-20260917-235820/campaign-evidence/objects/sha256/84/846fbbfc6d668811b686158d772ef05a2cbfc5b2855fbad8cd8a8955dd831e57`。
- 每任务首个合格运行，不是 fastest-of-all。两者均读到 5.439 µs；不写成 1× 性能持平。

### result-073

**BW1101 · softsign** — 2026-09-18 / 测量分辨能力待查

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@4278caf2`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/softsign-20260918-054910/campaign-evidence/objects/sha256/4a/4a332cf3b614046addbcea3e8258cde774b690d195112db039a687e5d8d5f913`。
- 每任务首个合格运行，不是 fastest-of-all。两者均读到 5.439 µs；不写成 1× 性能持平。

### result-074

**BW1101 · swiglu** — 2026-09-18 / 未检出显著差异

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@5151954e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/swiglu-20260917-185702/campaign-evidence/objects/sha256/54/5491c5924ad5437ee20ec11f63cf3d88452252db9e0c8990b469c1e437889045`。
- 每任务首个合格运行，不是 fastest-of-all。比值来自确认中位数；是否超过提升门槛沿用原 classification。

### result-075

**BW1101 · gemm_bias** — 2026-09-18 / 基线未通过

- Workload：`保留 sweep 的验证形状`；目标：`gfx938`；版本：`见来源`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/dcu-gfx938-results.md](../docs/dcu-gfx938-results.md)。
- 原记录 / 实现定位：`见来源所引用的原始实验`。
- 本集合无合格终点；保留在任务覆盖分母中。

### result-076

**BW1101 · gemm_silu** — 2026-09-18 / 候选未通过

- Workload：`保留 sweep 的验证形状`；目标：`gfx938`；版本：`见来源`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/dcu-gfx938-results.md](../docs/dcu-gfx938-results.md)。
- 原记录 / 实现定位：`见来源所引用的原始实验`。
- 本集合无合格终点；保留在任务覆盖分母中。

### result-077

**BW1101 · 后续重复 · gelu_tanh** — 2026-09-18 / 保留的后续运行

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@a8a4885d`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`gelu_tanh-20260918-104531/campaign-evidence/objects/sha256/94/94f0cc691770709577e02094c39a51b1dd6aea7abf62d53bf818f9d6aa546d02`。
- 按原汇总的首个合格选择规则未替换主行；不作为新的最佳实现。gelu_tanh 后继落在待查的 5.439 µs 读数。

### result-078

**BW1101 · 后续重复 · gemm** — 2026-09-18 / 保留的后续运行

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@4278caf2`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`gemm-20260918-042832/campaign-evidence/objects/sha256/59/59cf80cd04c6e9a6d4f4d3f11f2a5a037566b4174a5627c6b21305a207c1483a`。
- 按原汇总的首个合格选择规则未替换主行；不作为新的最佳实现。gelu_tanh 后继落在待查的 5.439 µs 读数。

### result-079

**BW1101 · 后续重复 · rmsnorm** — 2026-09-18 / 保留的后续运行

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@5151954e`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](../findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`rmsnorm-20260917-181121/campaign-evidence/objects/sha256/6b/6b31786a9cc813648dc1e386c47bd25656cd536e93dab084b1554a22dde3f9d7`。
- 按原汇总的首个合格选择规则未替换主行；不作为新的最佳实现。gelu_tanh 后继落在待查的 5.439 µs 读数。

### result-080

**Radeon / Strix Halo · rmsnorm smoke** — 2026-09-17 / 计时边界未解决

- Workload：`gfx1151-rmsnorm-b8-smoke`；目标：`gfx1151`；版本：`见来源`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[findings/2026-09-17-002-two-device-timers-disagree-on-gfx1151.json](../findings/2026-09-17-002-two-device-timers-disagree-on-gfx1151.json)。
- 原记录 / 实现定位：`infplane；原调查未产生 Campaign`。
- 31.858 与 24.224 µs 来自不同计时器，不作为合格延迟或加速比。
