# 按硬件查看实验结果

由 `tools/render_hardware_results.py` 从各平台发布数据生成。原始报告与 Finding 保留权威。

[交互目录](results/index.html) · [返回首页](../README.md) · [分支维护流程](RESULTS_MAINTENANCE.md)

**加速比 = 基线耗时 ÷ 候选耗时，超过 1× 表示更快。** 各行仅在自己的硬件、Workload、版本与计时协议内比较，不求跨平台平均值。

平台分支分别更新自己的发布数据；main 展示已经合入当前检出的版本。这里不会在线抓取平台分支最新值，也不是实时冠军榜。失败、无显著差异与未实测记录均保留。

## 数据归属

| 平台 | 维护分支 | 数据日期 | 观察条目 | 发布数据 |
|---|---|---|---:|---|
| NVIDIA | [nvidia](https://github.com/qhy991/open-cake-ir/tree/nvidia/docs/results/nvidia) | 2026-09-20 | 84 | [nvidia/records.json](results/nvidia/records.json) |
| Apple | [metal](https://github.com/qhy991/open-cake-ir/tree/metal/docs/results/metal) | 2026-09-20 | 4 | [metal/records.json](results/metal/records.json) |
| Hygon DCU | [dcu](https://github.com/qhy991/open-cake-ir/tree/dcu/docs/results/dcu) | 2026-09-20 | 32 | [dcu/records.json](results/dcu/records.json) |
| AMD | [amd](https://github.com/qhy991/open-cake-ir/tree/amd/docs/results/amd) | 2026-09-20 | 1 | [amd/records.json](results/amd/records.json) |

观察条目数不等于任务数：同一任务可以有不同形状、实验集合和历史尝试。

## NVIDIA

B300 的服务实验、CTA 宽度验证与 CAKE 改写各自保留基线和版本，不混成一个榜。B200 单列正确性证据。

- B300 服务实验投影保留每个 Campaign 的最佳合格候选及确认历史；不是远端 registry 的当前冠军。
- CTA 宽度验证与 CAKE 对照分别保留自身基线、版本及协议。
- FlashInfer新增7项固定shape三方对比：210/210数值检查通过。003仍比派生外部参考慢4.73倍；其余外部候选边保留CV失败，不发布合格加速比。022的原starter另以质量通过的1.151倍胜过外部参考。
- 六项三方NCU均为单kernel且未观察到local-memory sectors。003/022优先检查工作分配与并行度；001/002/025对齐launch几何后继续核对访存。详见F-2026-09-20-010。
- F-2026-09-20-011隔离了AOT对齐信息机制：仅CPU编译、假设16字节pointer alignment，三项kernel由标量b16变为128位向量访存。该历史probe时ABI未保证该假设；未做GPU运行、速度或晋升声明。002原有.cg提示并未解决该向量化信息缺口。
- PR #108已将运行时检查的通用/对齐AOT变体合入main，保留合法非对齐输入与原v1基线；本页历史测量仍指向原提交，新变体尚无GPU性能结论。
- With 005, 023 (h1536) and 024, ten tasks have three-way comparisons and all 300 pre/postflight checks pass. The three new external timing edges fail the unchanged CV gate. The qualified starter edges show 2.475x for 024 and 0.944x for 005; keep the faster 005 starter.
- Guarded AOT validation now passes 180/180 checks on B300: 001=65, 002=65, 025=50. CPU preparation and numerical verification run without a GPU lease; the device stage only launches and retains snapshots. Performance comparison of these new binaries remains pending.
- 001 aligned AOT timing is qualified: 2.304 us versus the unchanged old optimized binary at 2.496 us (1.083x; latency -7.69%). The supplied external is 2.336 us, classified close_null under the original 5% materiality threshold. All 2550 snapshots passed; 002 and 025 remain pending CPU verification.
- All three alignment runs now pass 7650/7650 snapshots. Qualified external edges: 001 close_null (2.304/2.336 us), 002 faster (3.008/3.296 us, 1.096x), 025 close_null (2.720/2.720 us). The direct new/old edges for 002 and 025 fail CV; their nominal gains are not published as qualified speedups. No blanket promotion.
- Work-assignment follow-up: 6/7 runs have passed all 15300 complete snapshots; 1 remains pending CPU verification. 003 w1 is a qualified compiler-floor close_null; w4/w16 improve over the frozen old optimized control by 1.463x/2.125x. Their external edges fail CV. 022 w4 beats the external by a qualified 1.164x, with no qualified improvement over the original starter. Failed edges remain descriptive; no promotion.
- Final work-assignment verification: all 7 runs and 17,850 complete snapshots pass numerically. The previously pending 003 w8 edge beats the frozen old optimized control by a qualified 1.398x (10.368/14.496 us), while its external edge fails CV. Only 003 w1 has all three timing edges quality-passing; the other six complete reports fail measurement quality. No blanket promotion.
- 003 whole-row at 16 groups passes all 2550 snapshots and all three quality edges: 1.896x versus sliced w16 (3.713/7.040 us), still 23.158% slower than the derived external (3.744/3.040 us). Subsequent guarded alignment passes 65 pointer checks and another 2550 snapshots, but new/control and new/external edges fail CV; its nominal 1.114x is not a qualified gain. No new IR primitive, profiler attribution or promotion.

| 设备 / 集合 | Task | 输入 / Workload | 基线 µs | 候选 µs | 加速比 | 状态 | 详情 |
|---|---|---|---:|---:|---:|---|---|
| B300 · 服务实验 | `absmax_rescale` | absmax-rescale-fp32-triton-b300-r128-c1024-v1 | 2.320 | 2.272 | 1.021× | 报告已确认 | [nvidia-result-001](#nvidia-result-001) |
| B300 · 服务实验 | `adadelta` | adadelta-fp32-triton-b300-r128-c1024-v1 | 4.160 | 4.096 | 1.016× | 报告已确认 | [nvidia-result-002](#nvidia-result-002) |
| B300 · 服务实验 | `adamw` | adamw-fp32-triton-b300-r128-c1024-v1 | 4.064 | 2.592 | 1.568× | 报告已确认 | [nvidia-result-003](#nvidia-result-003) |
| B300 · 服务实验 | `attention_decode` | contraction-attention-decode-fp32-triton-b300-r1024-k1024-n64-v1 | 339.874 | 230.769 | 1.473× | 报告已确认 | [nvidia-result-004](#nvidia-result-004) |
| B300 · 服务实验 | `bias_gradient_reduction` | bias-gradient-reduction-fp32-triton-b300-r128-c1024-v1 | — | — | — | 未合格终点 | [nvidia-result-005](#nvidia-result-005) |
| B300 · 服务实验 | `bias_gradient_reduction` | bias-gradient-reduction-fp32-triton-b300-r128-c1024-v1 | 2.784 | 2.752 | 1.012× | 报告已确认 | [nvidia-result-006](#nvidia-result-006) |
| B300 · 服务实验 | `channel_absmax_scale` | channel-absmax-scale-fp32-triton-b300-r128-c1024-v1 | 2.784 | 2.783 | 1.000× | 报告已确认 | [nvidia-result-007](#nvidia-result-007) |
| B300 · 服务实验 | `cosine_similarity` | cosine-similarity-fp32-triton-b300-r128-c1024-v1 | 2.976 | 2.977 | 1.000× | 报告已确认 | [nvidia-result-008](#nvidia-result-008) |
| B300 · 服务实验 | `gelu_tanh` | gelu-tanh-fp32-triton-b300-r128-c1024-v1 | 4.415 | 4.352 | 1.014× | 报告已确认 | [nvidia-result-009](#nvidia-result-009) |
| B300 · 服务实验 | `gelu_tanh_backward` | gelu-tanh-backward-fp32-triton-b300-r128-c1024-v1 | 4.960 | 2.720 | 1.824× | 报告已确认 | [nvidia-result-010](#nvidia-result-010) |
| B300 · 服务实验 | `gemm` | contraction-gemm-fp32-triton-b300-r1024-k1024-n64-v1 | 169.249 | 154.945 | 1.092× | 报告已确认 | [nvidia-result-011](#nvidia-result-011) |
| B300 · 服务实验 | `gemm_silu` | contraction-gemm-silu-fp32-triton-b300-r1024-k1024-n64-v1 | 166.081 | 178.305 | 0.931× | 报告已确认 | [nvidia-result-012](#nvidia-result-012) |
| B300 · 服务实验 | `layernorm` | layernorm-fp32-triton-b300-r128-c1024-v1 | 3.616 | 2.464 | 1.468× | 报告已确认 | [nvidia-result-013](#nvidia-result-013) |
| B300 · 服务实验 | `layernorm_backward_input` | layernorm-backward-input-fp32-triton-b300-r128-c1024-v2 | 3.168 | 3.040 | 1.042× | 报告已确认 | [nvidia-result-014](#nvidia-result-014) |
| B300 · 服务实验 | `momentum_sgd` | momentum-sgd-fp32-triton-b300-r128-c1024-v1 | 3.520 | 3.520 | 1.000× | 报告已确认 | [nvidia-result-015](#nvidia-result-015) |
| B300 · 服务实验 | `pairwise_sqdist` | contraction-pairwise-sqdist-fp32-triton-b300-r1024-k1024-n64-v1 | 411.795 | 53.856 | 7.646× | 报告已确认 | [nvidia-result-016](#nvidia-result-016) |
| B300 · 服务实验 | `per_channel_moments` | per-channel-moments-fp32-triton-b300-r128-c1024-v1 | 2.784 | 2.784 | 1.000× | 报告已确认 | [nvidia-result-017](#nvidia-result-017) |
| B300 · 服务实验 | `prelu` | prelu-fp32-triton-b300-r128-c1024-v1 | 2.896 | 2.832 | 1.023× | 报告已确认 | [nvidia-result-018](#nvidia-result-018) |
| B300 · 服务实验 | `residual_rmsnorm` | residual-rmsnorm-fp32-triton-b300-r128-c1024-v1 | 3.264 | 2.400 | 1.360× | 报告已确认 | [nvidia-result-019](#nvidia-result-019) |
| B300 · 服务实验 | `rmsnorm` | rmsnorm-fp32-triton-b300-r128-c1024-v3 | — | — | — | 未合格终点 | [nvidia-result-020](#nvidia-result-020) |
| B300 · 服务实验 | `rmsnorm` | rmsnorm-fp32-triton-b300-r128-c1024-v3 | 2.575 | 2.272 | 1.134× | 报告已确认 | [nvidia-result-021](#nvidia-result-021) |
| B300 · 服务实验 | `rmsnorm_input_gradient` | rmsnorm-input-gradient-fp32-triton-b300-r128-c1024-v2 | 3.040 | 2.432 | 1.250× | 报告已确认 | [nvidia-result-022](#nvidia-result-022) |
| B300 · 服务实验 | `selu` | selu-fp32-triton-b300-r128-c1024-v1 | 2.368 | 2.096 | 1.130× | 报告已确认 | [nvidia-result-023](#nvidia-result-023) |
| B300 · 服务实验 | `silu` | silu-fp32-triton-b300-r128-c1024-v1 | 2.432 | 2.112 | 1.152× | 报告已确认 | [nvidia-result-024](#nvidia-result-024) |
| B300 · 服务实验 | `softmax` | softmax-fp32-triton-b300-r128-c1024-v1 | 2.495 | 2.464 | 1.013× | 报告已确认 | [nvidia-result-025](#nvidia-result-025) |
| B300 · 服务实验 | `softmax_backward` | softmax-backward-fp32-triton-b300-r128-c1024-v1 | 2.944 | 2.592 | 1.136× | 报告已确认 | [nvidia-result-026](#nvidia-result-026) |
| B300 · 服务实验 | `softplus_gradient` | softplus-gradient-fp32-triton-b300-r128-c1024-v2 | 2.977 | 2.208 | 1.348× | 报告已确认 | [nvidia-result-027](#nvidia-result-027) |
| B300 · 服务实验 | `softsign` | softsign-fp32-triton-b300-r128-c1024-v1 | 2.432 | 2.400 | 1.013× | 报告已确认 | [nvidia-result-028](#nvidia-result-028) |
| B300 · 服务实验 | `swiglu` | swiglu-fp32-triton-b300-r128-c1024-v1 | 3.520 | 3.232 | 1.089× | 报告已确认 | [nvidia-result-029](#nvidia-result-029) |
| B300 · CTA 宽度验证 | `rmsnorm_input_gradient` | FP32 · R=128, C=1024 | 3.104 | 2.497 | 1.243× | 已确认 | [nvidia-result-030](#nvidia-result-030) |
| B300 · CTA 宽度验证 | `swiglu` | FP32 · R=128, C=1024 | 3.648 | 2.208 | 1.652× | 已确认 | [nvidia-result-031](#nvidia-result-031) |
| B300 · CTA 宽度验证 | `softmax_backward` | FP32 · R=128, C=1024 | 2.927 | 2.560 | 1.144× | 已确认 | [nvidia-result-032](#nvidia-result-032) |
| B300 · CTA 宽度验证 | `cosine_similarity` | FP32 · R=128, C=1024 | 2.976 | 2.528 | 1.177× | 已确认 | [nvidia-result-033](#nvidia-result-033) |
| B300 · CAKE 对照 | `TinyGEMM2 s2` | BF16+bias · B/N/K=1 / 128 / 720 | 2.720 | 99.969 | 0.027× | 正确；性能落后 | [nvidia-result-034](#nvidia-result-034) |
| B300 · CAKE 对照 | `TinyGEMM2 s4` | BF16+bias · B/N/K=1 / 128 / 720 | 2.720 | 100.192 | 0.027× | 正确；性能落后 | [nvidia-result-035](#nvidia-result-035) |
| B300 · CAKE 对照 | `TinyGEMM2 s2` | BF16+bias · B/N/K=16 / 1024 / 1024 | 3.040 | 106.016 | 0.029× | 正确；性能落后 | [nvidia-result-036](#nvidia-result-036) |
| B300 · CAKE 对照 | `TinyGEMM2 s4` | BF16+bias · B/N/K=16 / 1024 / 1024 | 3.040 | 105.825 | 0.029× | 正确；性能落后 | [nvidia-result-037](#nvidia-result-037) |
| B300 · CAKE 对照 | `TinyGEMM2 s2` | BF16+bias · B/N/K=64 / 4096 / 3072 | 21.216 | 239.969 | 0.088× | 正确；性能落后 | [nvidia-result-038](#nvidia-result-038) |
| B300 · CAKE 对照 | `TinyGEMM2 s4` | BF16+bias · B/N/K=64 / 4096 / 3072 | 21.216 | 289.186 | 0.073× | 正确；性能落后 | [nvidia-result-039](#nvidia-result-039) |
| B300 · CAKE 对照 | `KDA prefill` | 完整任务尚未完成 | — | — | — | 未实测 | [nvidia-result-040](#nvidia-result-040) |
| B300 · CAKE 对照 | `KDA decode` | 完整任务尚未完成 | — | — | — | 未实测 | [nvidia-result-041](#nvidia-result-041) |
| B300 · CAKE 对照 | `Alpha-MoE` | 完整任务尚未完成 | — | — | — | 未实测 | [nvidia-result-042](#nvidia-result-042) |
| B200 | `FMA / affine / state-store` | 固定正确性实例，详见各报告 | — | — | — | 仅正确性 | [nvidia-result-043](#nvidia-result-043) |
| B300 · FlashInfer外部对照 | `001_fused_add_rmsnorm_h2048` | fib_fused_add_rmsnorm_h2048 / R=79, C=2048 / BF16 | 2.336 | 2.496 | — | 正确；计时质量未通过 | [nvidia-fib-external-001-20260920](#nvidia-fib-external-001-20260920) |
| B300 · FlashInfer外部对照 | `002_fused_add_rmsnorm_h4096` | fib_fused_add_rmsnorm_h4096 / R=170, C=4096 / BF16 | 3.296 | 3.712 | — | 正确；计时质量未通过 | [nvidia-fib-external-002-20260920](#nvidia-fib-external-002-20260920) |
| B300 · FlashInfer外部对照 | `003_fused_add_rmsnorm_h7168` | fib_fused_add_rmsnorm_h7168 / R=64, C=7168 / BF16 | 3.040 | 14.368 | 0.212× | 正确；外部参考更快（质量通过） | [nvidia-fib-external-003-20260920](#nvidia-fib-external-003-20260920) |
| B300 · FlashInfer外部对照 | `004_gemm_n128_k2048` | fib_gemm_n128_k2048 / R=1, C=128 / FP16 | 2.368 | 2.688 | — | 正确；计时质量未通过 | [nvidia-fib-external-004-20260920](#nvidia-fib-external-004-20260920) |
| B300 · FlashInfer外部对照 | `021_rmsnorm_h128` | fib_rmsnorm_h128 / R=2528, C=128 / BF16 | 2.560 | 2.912 | — | 正确；计时质量未通过 | [nvidia-fib-external-021-20260920](#nvidia-fib-external-021-20260920) |
| B300 · FlashInfer外部对照 | `022_rmsnorm_h512` | fib_rmsnorm_h512 / R=539, C=512 / BF16 | 2.720 | 5.152 | — | 正确；计时质量未通过 | [nvidia-fib-external-022-20260920](#nvidia-fib-external-022-20260920) |
| B300 · FlashInfer外部对照 | `025_rmsnorm_h4096` | fib_rmsnorm_h4096 / R=170, C=4096 / BF16 | 2.752 | 3.168 | — | 正确；计时质量未通过 | [nvidia-fib-external-025-20260920](#nvidia-fib-external-025-20260920) |
| B300 · FlashInfer外部对照 | `022_rmsnorm_h512_starter` | fib_rmsnorm_h512 / R=539, C=512 / BF16 | 2.688 | 2.336 | 1.151× | 正确；starter更快（质量通过） | [nvidia-fib-external-022-starter-20260920](#nvidia-fib-external-022-starter-20260920) |
| B300 / FlashInfer comparison | `005_gemm_n256_k7168` | fib_gemm_n256_k7168 / R=1, C=256 / FP16 | 6.880 | 9.152 | — | Correct; timing quality failed | [nvidia-fib-external-005-20260920](#nvidia-fib-external-005-20260920) |
| B300 / FlashInfer comparison | `005_gemm_n256_k7168` | fib_gemm_n256_k7168 / R=1, C=256 / FP16 | 8.641 | 9.152 | 0.944× | Correct; timing qualified | [nvidia-fib-starter-005-20260920](#nvidia-fib-starter-005-20260920) |
| B300 / FlashInfer comparison | `023_rmsnorm_h1536` | fib_rmsnorm_h1536 / R=539, C=1536 / BF16 | 4.095 | 2.880 | — | Correct; timing quality failed | [nvidia-fib-external-023-20260920](#nvidia-fib-external-023-20260920) |
| B300 / FlashInfer comparison | `023_rmsnorm_h1536` | fib_rmsnorm_h1536 / R=539, C=1536 / BF16 | 4.064 | 2.880 | — | Correct; timing quality failed | [nvidia-fib-starter-023-20260920](#nvidia-fib-starter-023-20260920) |
| B300 / FlashInfer comparison | `024_rmsnorm_h2048` | fib_rmsnorm_h2048 / R=79, C=2048 / BF16 | 2.272 | 2.464 | — | Correct; timing quality failed | [nvidia-fib-external-024-20260920](#nvidia-fib-external-024-20260920) |
| B300 / FlashInfer comparison | `024_rmsnorm_h2048` | fib_rmsnorm_h2048 / R=79, C=2048 / BF16 | 6.177 | 2.496 | 2.475× | Correct; timing qualified | [nvidia-fib-starter-024-20260920](#nvidia-fib-starter-024-20260920) |
| B300 / guarded AOT validation | `001_fused_add_rmsnorm_h2048` | Retained exact shape / BF16 / five distributions / aligned and each pointer offset 2, 4, 8 bytes | — | — | — | Correctness only: 65 checks passed | [nvidia-alignment-guard-001-20260920](#nvidia-alignment-guard-001-20260920) |
| B300 / guarded AOT validation | `002_fused_add_rmsnorm_h4096` | Retained exact shape / BF16 / five distributions / aligned and each pointer offset 2, 4, 8 bytes | — | — | — | Correctness only: 65 checks passed | [nvidia-alignment-guard-002-20260920](#nvidia-alignment-guard-002-20260920) |
| B300 / guarded AOT validation | `025_rmsnorm_h4096` | Retained exact shape / BF16 / five distributions / aligned and each pointer offset 2, 4, 8 bytes | — | — | — | Correctness only: 50 checks passed | [nvidia-alignment-guard-025-20260920](#nvidia-alignment-guard-025-20260920) |
| B300 / AOT alignment ablation | `001_fused_add_rmsnorm_h2048` | R=79, H=2048, BF16; same source/grid/options, alignment treatment only | 2.496 | 2.304 | 1.083× | Correct; timing qualified; first_arm_faster | [nvidia-alignment-001-optimized_vs_starter-20260920](#nvidia-alignment-001-optimized_vs_starter-20260920) |
| B300 / AOT alignment ablation | `001_fused_add_rmsnorm_h2048` | R=79, H=2048, BF16; same source/grid/options, alignment treatment only | 2.336 | 2.304 | 1.014× | Correct; timing qualified; close_null | [nvidia-alignment-001-optimized_vs_external-20260920](#nvidia-alignment-001-optimized_vs_external-20260920) |
| B300 / AOT alignment ablation | `002_fused_add_rmsnorm_h4096` | R=170, H=4096, BF16; same source/grid/options, alignment treatment only | 3.296 | 3.008 | 1.096× | Correct; first_arm_faster | [nvidia-alignment-002-optimized_vs_external-20260920](#nvidia-alignment-002-optimized_vs_external-20260920) |
| B300 / AOT alignment ablation | `002_fused_add_rmsnorm_h4096` | R=170, H=4096, BF16; same source/grid/options, alignment treatment only | 3.648 | 3.024 | — | Correct; measurement_quality_failed | [nvidia-alignment-002-optimized_vs_starter-20260920](#nvidia-alignment-002-optimized_vs_starter-20260920) |
| B300 / AOT alignment ablation | `025_rmsnorm_h4096` | R=170, H=4096, BF16; same source/grid/options, alignment treatment only | 2.720 | 2.720 | 1.000× | Correct; close_null | [nvidia-alignment-025-optimized_vs_external-20260920](#nvidia-alignment-025-optimized_vs_external-20260920) |
| B300 / AOT alignment ablation | `025_rmsnorm_h4096` | R=170, H=4096, BF16; same source/grid/options, alignment treatment only | 3.136 | 2.720 | — | Correct; measurement_quality_failed | [nvidia-alignment-025-optimized_vs_starter-20260920](#nvidia-alignment-025-optimized_vs_starter-20260920) |
| B300 / work-assignment ablation | `003_fused_add_rmsnorm_h7168` | R=64, H=7168, BF16; 1 execution groups; generic AOT | 14.656 | 14.320 | 1.023× | Correct; close_null | [nvidia-assignment-003-w1-optimized_vs_starter-20260920](#nvidia-assignment-003-w1-optimized_vs_starter-20260920) |
| B300 / work-assignment ablation | `003_fused_add_rmsnorm_h7168` | R=64, H=7168, BF16; 1 execution groups; generic AOT | 3.040 | 14.272 | 0.213× | Correct; second_arm_faster | [nvidia-assignment-003-w1-optimized_vs_external-20260920](#nvidia-assignment-003-w1-optimized_vs_external-20260920) |
| B300 / work-assignment ablation | `003_fused_add_rmsnorm_h7168` | R=64, H=7168, BF16; 4 execution groups; generic AOT | 14.656 | 10.016 | 1.463× | Correct; first_arm_faster | [nvidia-assignment-003-w4-optimized_vs_starter-20260920](#nvidia-assignment-003-w4-optimized_vs_starter-20260920) |
| B300 / work-assignment ablation | `003_fused_add_rmsnorm_h7168` | R=64, H=7168, BF16; 4 execution groups; generic AOT | 3.104 | 10.016 | — | Correct; measurement_quality_failed | [nvidia-assignment-003-w4-optimized_vs_external-20260920](#nvidia-assignment-003-w4-optimized_vs_external-20260920) |
| B300 / work-assignment ablation | `003_fused_add_rmsnorm_h7168` | R=64, H=7168, BF16; 8 execution groups; generic AOT | 14.496 | 10.368 | 1.398× | Correct; first_arm_faster | [nvidia-assignment-003-w8-optimized_vs_starter-20260920](#nvidia-assignment-003-w8-optimized_vs_starter-20260920) |
| B300 / work-assignment ablation | `003_fused_add_rmsnorm_h7168` | R=64, H=7168, BF16; 8 execution groups; generic AOT | 3.072 | 10.368 | — | Correct; measurement_quality_failed | [nvidia-assignment-003-w8-optimized_vs_external-20260920](#nvidia-assignment-003-w8-optimized_vs_external-20260920) |
| B300 / work-assignment ablation | `003_fused_add_rmsnorm_h7168` | R=64, H=7168, BF16; 16 execution groups; generic AOT | 14.688 | 6.912 | 2.125× | Correct; first_arm_faster | [nvidia-assignment-003-w16-optimized_vs_starter-20260920](#nvidia-assignment-003-w16-optimized_vs_starter-20260920) |
| B300 / work-assignment ablation | `003_fused_add_rmsnorm_h7168` | R=64, H=7168, BF16; 16 execution groups; generic AOT | 3.103 | 6.944 | — | Correct; measurement_quality_failed | [nvidia-assignment-003-w16-optimized_vs_external-20260920](#nvidia-assignment-003-w16-optimized_vs_external-20260920) |
| B300 / work-assignment ablation | `022_rmsnorm_h512` | R=539, H=512, BF16; 1 execution groups; generic AOT | 2.336 | 2.336 | — | Correct; measurement_quality_failed | [nvidia-assignment-022-w1-optimized_vs_starter-20260920](#nvidia-assignment-022-w1-optimized_vs_starter-20260920) |
| B300 / work-assignment ablation | `022_rmsnorm_h512` | R=539, H=512, BF16; 1 execution groups; generic AOT | 2.688 | 2.305 | — | Correct; measurement_quality_failed | [nvidia-assignment-022-w1-optimized_vs_external-20260920](#nvidia-assignment-022-w1-optimized_vs_external-20260920) |
| B300 / work-assignment ablation | `022_rmsnorm_h512` | R=539, H=512, BF16; 4 execution groups; generic AOT | 2.368 | 2.336 | — | Correct; measurement_quality_failed | [nvidia-assignment-022-w4-optimized_vs_starter-20260920](#nvidia-assignment-022-w4-optimized_vs_starter-20260920) |
| B300 / work-assignment ablation | `022_rmsnorm_h512` | R=539, H=512, BF16; 4 execution groups; generic AOT | 2.720 | 2.336 | 1.164× | Correct; first_arm_faster | [nvidia-assignment-022-w4-optimized_vs_external-20260920](#nvidia-assignment-022-w4-optimized_vs_external-20260920) |
| B300 / work-assignment ablation | `022_rmsnorm_h512` | R=539, H=512, BF16; 8 execution groups; generic AOT | 2.336 | 2.304 | — | Correct; measurement_quality_failed | [nvidia-assignment-022-w8-optimized_vs_starter-20260920](#nvidia-assignment-022-w8-optimized_vs_starter-20260920) |
| B300 / work-assignment ablation | `022_rmsnorm_h512` | R=539, H=512, BF16; 8 execution groups; generic AOT | 2.688 | 2.305 | — | Correct; measurement_quality_failed | [nvidia-assignment-022-w8-optimized_vs_external-20260920](#nvidia-assignment-022-w8-optimized_vs_external-20260920) |
| B300 / whole-row structure | `003_fused_add_rmsnorm_h7168` | R=64, H=7168, BF16; masked whole row; 16 execution groups | 7.040 | 3.713 | 1.896× | Correct; first_arm_faster | [nvidia-003-whole-row-structure-optimized_vs_starter-20260921](#nvidia-003-whole-row-structure-optimized_vs_starter-20260921) |
| B300 / whole-row structure | `003_fused_add_rmsnorm_h7168` | R=64, H=7168, BF16; masked whole row; 16 execution groups | 3.040 | 3.744 | 0.812× | Correct; second_arm_faster | [nvidia-003-whole-row-structure-optimized_vs_external-20260921](#nvidia-003-whole-row-structure-optimized_vs_external-20260921) |
| B300 / whole-row alignment | `003_fused_add_rmsnorm_h7168` | R=64, H=7168, BF16; masked whole row; 16 execution groups | 3.744 | 3.360 | — | Correct; measurement_quality_failed | [nvidia-003-whole-row-alignment-optimized_vs_starter-20260921](#nvidia-003-whole-row-alignment-optimized_vs_starter-20260921) |
| B300 / whole-row alignment | `003_fused_add_rmsnorm_h7168` | R=64, H=7168, BF16; masked whole row; 16 execution groups | 3.040 | 3.360 | — | Correct; measurement_quality_failed | [nvidia-003-whole-row-alignment-optimized_vs_external-20260921](#nvidia-003-whole-row-alignment-optimized_vs_external-20260921) |

## Apple

M1 Pro、M4、M2 分开。Metal 测量整个 command buffer，并按重复 dispatch 均摊；不能与 CUPTI 绝对耗时横比。晋升标签仅表示保留记录记载过晋升，未重新读取远端 registry 的当前代。

- M4 晋升记录来自历史 Finding，本次未刷新 registry 当前代。
- M1 Pro 后继实验的搜索高分和合格 K-split 终点不构成单调加速曲线。

| 设备 / 集合 | Task | 输入 / Workload | 基线 µs | 候选 µs | 加速比 | 状态 | 详情 |
|---|---|---|---:|---:|---:|---|---|
| M4 | `channel_absmax_scale` | FP32 · R=2；完整 C/协议由原 Workload 固定 | 0.788 | 0.712 | 1.107× | 有晋升记录 | [metal-result-044](#metal-result-044) |
| M4 | `bias_gradient_reduction` | FP32 · R=2；完整 C/协议由原 Workload 固定 | 0.772 | 0.678 | 1.138× | 有晋升记录 | [metal-result-045](#metal-result-045) |
| M1 Pro | `gemm_silu` | FP32 · M=128, K=256, N=32 | 1878.596 | 79.588 | 23.604× | 历史确认 | [metal-result-046](#metal-result-046) |
| M2 | `gemm` | 原记录的三个 GEMM 候选 | — | — | — | 计时未通过 | [metal-result-047](#metal-result-047) |

## Hygon DCU

BW1101 / gfx938：保留首个合格结果，同时展示未合格任务和后续重复运行。首个合格结果不等于历史最快；使用原记录的分类，不从显示的小数重新判定晋升。

- 每个任务取保留 sweep 的首个合格结果，并展示后续重复；不作最快结果选择。
- 原主机在 2026-09-20 整理时 SSH 超时，未重新提取形状；缺失字段明确保留。

| 设备 / 集合 | Task | 输入 / Workload | 基线 µs | 候选 µs | 加速比 | 状态 | 详情 |
|---|---|---|---:|---:|---:|---|---|
| BW1101 | `absmax_rescale` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 5.439 | — | 测量分辨能力待查 | [dcu-result-048](#dcu-result-048) |
| BW1101 | `adadelta` | 原 Campaign 的固定形状（汇总未转录尺寸） | 6.719 | 6.719 | 1.000× | 未检出显著差异 | [dcu-result-049](#dcu-result-049) |
| BW1101 | `adamw` | 原 Campaign 的固定形状（汇总未转录尺寸） | 7.839 | 7.839 | 1.000× | 未检出显著差异 | [dcu-result-050](#dcu-result-050) |
| BW1101 | `attention_decode` | 原 Campaign 的固定形状（汇总未转录尺寸） | 138.559 | 219.519 | 0.631× | 确认变慢 | [dcu-result-051](#dcu-result-051) |
| BW1101 | `bias_gradient_reduction` | 原 Campaign 的固定形状（汇总未转录尺寸） | 10.079 | 10.399 | 0.969× | 未检出显著差异 | [dcu-result-052](#dcu-result-052) |
| BW1101 | `channel_absmax_scale` | 原 Campaign 的固定形状（汇总未转录尺寸） | 9.599 | 9.599 | 1.000× | 未检出显著差异 | [dcu-result-053](#dcu-result-053) |
| BW1101 | `cosine_similarity` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 5.439 | — | 测量分辨能力待查 | [dcu-result-054](#dcu-result-054) |
| BW1101 | `gelu_tanh` | 原 Campaign 的固定形状（汇总未转录尺寸） | 6.399 | 6.239 | 1.026× | 未检出显著差异 | [dcu-result-055](#dcu-result-055) |
| BW1101 | `gelu_tanh_backward` | 原 Campaign 的固定形状（汇总未转录尺寸） | 6.719 | 12.639 | 0.532× | 确认变慢 | [dcu-result-056](#dcu-result-056) |
| BW1101 | `gemm` | 原 Campaign 的固定形状（汇总未转录尺寸） | 305.599 | 409.279 | 0.747× | 确认变慢 | [dcu-result-057](#dcu-result-057) |
| BW1101 | `layernorm` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 5.439 | — | 测量分辨能力待查 | [dcu-result-058](#dcu-result-058) |
| BW1101 | `layernorm_backward_input` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 5.439 | — | 测量分辨能力待查 | [dcu-result-059](#dcu-result-059) |
| BW1101 | `layernorm_gamma_beta_backward` | 原 Campaign 的固定形状（汇总未转录尺寸） | 14.879 | 15.199 | 0.979× | 未检出显著差异 | [dcu-result-060](#dcu-result-060) |
| BW1101 | `momentum_sgd` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.919 | 5.599 | 1.057× | 确认加速 | [dcu-result-061](#dcu-result-061) |
| BW1101 | `pairwise_sqdist` | 原 Campaign 的固定形状（汇总未转录尺寸） | 24.959 | 23.679 | 1.054× | 确认加速 | [dcu-result-062](#dcu-result-062) |
| BW1101 | `per_channel_moments` | 原 Campaign 的固定形状（汇总未转录尺寸） | 9.439 | 6.719 | 1.405× | 确认加速 | [dcu-result-063](#dcu-result-063) |
| BW1101 | `prelu` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 9.759 | 0.557× | 确认变慢 | [dcu-result-064](#dcu-result-064) |
| BW1101 | `residual_rmsnorm` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 5.439 | — | 测量分辨能力待查 | [dcu-result-065](#dcu-result-065) |
| BW1101 | `rmsnorm` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 5.439 | — | 测量分辨能力待查 | [dcu-result-066](#dcu-result-066) |
| BW1101 | `rmsnorm_input_gradient` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 5.439 | — | 测量分辨能力待查 | [dcu-result-067](#dcu-result-067) |
| BW1101 | `selu` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 5.439 | — | 测量分辨能力待查 | [dcu-result-068](#dcu-result-068) |
| BW1101 | `silu` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 5.439 | — | 测量分辨能力待查 | [dcu-result-069](#dcu-result-069) |
| BW1101 | `softmax` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 5.439 | — | 测量分辨能力待查 | [dcu-result-070](#dcu-result-070) |
| BW1101 | `softmax_backward` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 5.439 | — | 测量分辨能力待查 | [dcu-result-071](#dcu-result-071) |
| BW1101 | `softplus_gradient` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 5.439 | — | 测量分辨能力待查 | [dcu-result-072](#dcu-result-072) |
| BW1101 | `softsign` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 5.439 | — | 测量分辨能力待查 | [dcu-result-073](#dcu-result-073) |
| BW1101 | `swiglu` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.599 | 5.439 | 1.029× | 未检出显著差异 | [dcu-result-074](#dcu-result-074) |
| BW1101 | `gemm_bias` | 保留 sweep 的验证形状 | — | — | — | 基线未通过 | [dcu-result-075](#dcu-result-075) |
| BW1101 | `gemm_silu` | 保留 sweep 的验证形状 | — | — | — | 候选未通过 | [dcu-result-076](#dcu-result-076) |
| BW1101 · 后续重复 | `gelu_tanh` | 原 Campaign 的固定形状（汇总未转录尺寸） | 6.399 | 5.439 | — | 保留的后续运行 | [dcu-result-077](#dcu-result-077) |
| BW1101 · 后续重复 | `gemm` | 原 Campaign 的固定形状（汇总未转录尺寸） | 305.759 | 409.279 | — | 保留的后续运行 | [dcu-result-078](#dcu-result-078) |
| BW1101 · 后续重复 | `rmsnorm` | 原 Campaign 的固定形状（汇总未转录尺寸） | 5.439 | 5.439 | — | 保留的后续运行 | [dcu-result-079](#dcu-result-079) |

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

### nvidia-result-001

**B300 · 服务实验 · absmax_rescale** — 2026-09-16 / 报告已确认

- Workload：`absmax-rescale-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/absmax_rescale-20260916-100813/report.json; open_cake-1; event=17; candidate=a363e12177e7c493ca3748b23b5fad0515f84efc761d0bd62d6c37e02fa2adb5`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 2 | 17 | 2.272 | 1.021× |

### nvidia-result-002

**B300 · 服务实验 · adadelta** — 2026-09-16 / 报告已确认

- Workload：`adadelta-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/adadelta-20260916-060824/report.json; open_cake-1; event=17; candidate=1eb5ed817fcf0ddc63179af3c7eed9d6cda9354e7d6e3ebed6506e8f08960904`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 2 | 17 | 4.096 | 1.016× |

### nvidia-result-003

**B300 · 服务实验 · adamw** — 2026-09-16 / 报告已确认

- Workload：`adamw-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/adamw-20260916-053018/report.json; open_cake-1; event=26; candidate=398044880e6094b2c0e585775b7d0753650db52ad51d2aecc6aa0200ea09cc4f`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 4.000 | 1.016× |
| 2 | 26 | 2.592 | 1.568× |

### nvidia-result-004

**B300 · 服务实验 · attention_decode** — 2026-09-16 / 报告已确认

- Workload：`contraction-attention-decode-fp32-triton-b300-r1024-k1024-n64-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/attention_decode-20260916-184301/report.json; open_cake-1; event=13; candidate=62bc95ec21df257e1f1f7f7b28d6845dc47a6af5d32c5d970b1f840af27ad5bf`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 230.769 | 1.473× |

### nvidia-result-005

**B300 · 服务实验 · bias_gradient_reduction** — 2026-09-16 / 未合格终点

- Workload：`bias-gradient-reduction-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/bias_gradient_reduction-20260916-065950/report.json; open_cake-1; event=None; candidate=None`。
- provider_fault; missing; None

### nvidia-result-006

**B300 · 服务实验 · bias_gradient_reduction** — 2026-09-16 / 报告已确认

- Workload：`bias-gradient-reduction-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/bias_gradient_reduction-20260916-130258/report.json; open_cake-1; event=13; candidate=1f9b4ad199fc61c3ae923d998bd1ac59d1fbcde65649725e66f64e989b791df8`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 2.752 | 1.012× |

### nvidia-result-007

**B300 · 服务实验 · channel_absmax_scale** — 2026-09-16 / 报告已确认

- Workload：`channel-absmax-scale-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/channel_absmax_scale-20260916-124804/report.json; open_cake-1; event=26; candidate=bb02cc80ae5a1785f89440d86203f1d058c61b80074629edf8a0d7f4c915e743`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 2.784 | 1.000× |
| 2 | 26 | 2.783 | 1.000× |

### nvidia-result-008

**B300 · 服务实验 · cosine_similarity** — 2026-09-16 / 报告已确认

- Workload：`cosine-similarity-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/cosine_similarity-20260916-102543/report.json; open_cake-1; event=13; candidate=f739fb48dec1050ae77b2abbe731a106d055be484eb657fada04e1b108a4fea4`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 2.977 | 1.000× |

### nvidia-result-009

**B300 · 服务实验 · gelu_tanh** — 2026-09-16 / 报告已确认

- Workload：`gelu-tanh-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/gelu_tanh-20260916-042956/report.json; open_cake-1; event=17; candidate=f42e4152b65487a489d44963da52469166425ebf2e7862416ff128272dbc65b8`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 2 | 17 | 4.352 | 1.014× |

### nvidia-result-010

**B300 · 服务实验 · gelu_tanh_backward** — 2026-09-16 / 报告已确认

- Workload：`gelu-tanh-backward-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/gelu_tanh_backward-20260916-120441/report.json; open_cake-1; event=17; candidate=62aca837508c4a2dbd5cc1c14235d6833b26cbc858dad707288bec74f48c92aa`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 2 | 17 | 2.720 | 1.824× |

### nvidia-result-011

**B300 · 服务实验 · gemm** — 2026-09-16 / 报告已确认

- Workload：`contraction-gemm-fp32-triton-b300-r1024-k1024-n64-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/gemm-20260916-170620/report.json; open_cake-1; event=17; candidate=445d2727b4e39d3aae77f38415ea7ebc40190db07ffd325674ebb05c380fd454`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 2 | 17 | 154.945 | 1.092× |

### nvidia-result-012

**B300 · 服务实验 · gemm_silu** — 2026-09-16 / 报告已确认

- Workload：`contraction-gemm-silu-fp32-triton-b300-r1024-k1024-n64-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/gemm_silu-20260916-172338/report.json; open_cake-1; event=26; candidate=0cfa7400e15901e6e39968f14f956117ca8ca6c9d622cceae04e339f6ea34860`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 324.274 | 0.509× |
| 2 | 26 | 178.305 | 0.931× |

### nvidia-result-013

**B300 · 服务实验 · layernorm** — 2026-09-16 / 报告已确认

- Workload：`layernorm-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/layernorm-20260916-034037/report.json; open_cake-1; event=13; candidate=d110d259af65053f62a441195bd51dd4d3827ad3210324e3be89ffa87c6ebf9c`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 2.464 | 1.468× |
| 2 | 26 | 2.465 | 1.467× |

### nvidia-result-014

**B300 · 服务实验 · layernorm_backward_input** — 2026-09-16 / 报告已确认

- Workload：`layernorm-backward-input-fp32-triton-b300-r128-c1024-v2`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/layernorm_backward_input-20260916-094015/report.json; open_cake-1; event=26; candidate=b123db1855356f93ed21bb3063101963857e893d120c33dd837b05b0bcb9819b`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 3.200 | 0.990× |
| 2 | 26 | 3.040 | 1.042× |

### nvidia-result-015

**B300 · 服务实验 · momentum_sgd** — 2026-09-16 / 报告已确认

- Workload：`momentum-sgd-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/momentum_sgd-20260916-051315/report.json; open_cake-1; event=26; candidate=a17d866faea721d4a7ea9ee3579af848a649d0625d7470b64430d3256f80d62f`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 4.864 | 0.727× |
| 2 | 26 | 3.520 | 1.000× |

### nvidia-result-016

**B300 · 服务实验 · pairwise_sqdist** — 2026-09-16 / 报告已确认

- Workload：`contraction-pairwise-sqdist-fp32-triton-b300-r1024-k1024-n64-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/pairwise_sqdist-20260916-180016/report.json; open_cake-1; event=17; candidate=1c24a27ae8410b36407e837607d0349d0d152634f6759da19283697afccc0962`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 2 | 17 | 53.856 | 7.646× |

### nvidia-result-017

**B300 · 服务实验 · per_channel_moments** — 2026-09-16 / 报告已确认

- Workload：`per-channel-moments-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/per_channel_moments-20260916-104523/report.json; open_cake-1; event=13; candidate=b087e9645f5ba0d469c650fde3fc65fda1f79c50b8d150c0f5c633feee3ad3df`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 2.784 | 1.000× |

### nvidia-result-018

**B300 · 服务实验 · prelu** — 2026-09-16 / 报告已确认

- Workload：`prelu-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/prelu-20260916-122027/report.json; open_cake-1; event=13; candidate=56e850a984d225bdc58d359bc91fd12a28074af6c56aba9df64ee1a77591ebcf`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 2.832 | 1.023× |

### nvidia-result-019

**B300 · 服务实验 · residual_rmsnorm** — 2026-09-16 / 报告已确认

- Workload：`residual-rmsnorm-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/residual_rmsnorm-20260916-044120/report.json; open_cake-1; event=26; candidate=533df099cbd8ced4e5be0305fa946ea1691c7249797ff9917f194c92aa35c999`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 3.232 | 1.000× |
| 2 | 26 | 2.400 | 1.360× |

### nvidia-result-020

**B300 · 服务实验 · rmsnorm** — 2026-09-16 / 未合格终点

- Workload：`rmsnorm-fp32-triton-b300-r128-c1024-v3`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/rmsnorm-20260916-031407/report.json; open_cake-1; event=None; candidate=None`。
- broker_fault; missing; None

### nvidia-result-021

**B300 · 服务实验 · rmsnorm** — 2026-09-16 / 报告已确认

- Workload：`rmsnorm-fp32-triton-b300-r128-c1024-v3`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/rmsnorm-20260916-032057/report.json; open_cake-1; event=26; candidate=3d9385205d3413906415268c8c6e2b85cc82a8de9dfa213708f0870989f77802`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 2.400 | 1.080× |
| 2 | 26 | 2.272 | 1.134× |

### nvidia-result-022

**B300 · 服务实验 · rmsnorm_input_gradient** — 2026-09-16 / 报告已确认

- Workload：`rmsnorm-input-gradient-fp32-triton-b300-r128-c1024-v2`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/rmsnorm_input_gradient-20260916-064534/report.json; open_cake-1; event=26; candidate=77fdf24695c20187bd7e22ccb6bb332c329d425498f1a098c1c38950952fc940`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 2.688 | 1.095× |
| 2 | 26 | 2.432 | 1.250× |

### nvidia-result-023

**B300 · 服务实验 · selu** — 2026-09-16 / 报告已确认

- Workload：`selu-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/selu-20260916-112941/report.json; open_cake-1; event=13; candidate=8bc67babc39b24758c111e18d6f669c04940077f4fef70c965945aee5bec83fa`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 2.096 | 1.130× |
| 2 | 26 | 2.112 | 1.121× |

### nvidia-result-024

**B300 · 服务实验 · silu** — 2026-09-16 / 报告已确认

- Workload：`silu-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/silu-20260916-040643/report.json; open_cake-1; event=26; candidate=282f390f27dc28fcb86e9bafad4c5d1aeb1ee77bf86c135d6bee41e4fb3d5ddc`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 2.432 | 1.000× |
| 2 | 26 | 2.112 | 1.152× |

### nvidia-result-025

**B300 · 服务实验 · softmax** — 2026-09-16 / 报告已确认

- Workload：`softmax-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/softmax-20260916-035525/report.json; open_cake-1; event=17; candidate=4484a1e9ff3a2b46bf87bf1971d1c824321c225f3c0ab3c3f3ab6bd67edea845`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 2 | 17 | 2.464 | 1.013× |

### nvidia-result-026

**B300 · 服务实验 · softmax_backward** — 2026-09-16 / 报告已确认

- Workload：`softmax-backward-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/softmax_backward-20260916-062444/report.json; open_cake-1; event=13; candidate=9a920f573083bc5a81c073abcd497aeafe1a2ac7371477e51b600b2673a5ac79`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 2.592 | 1.136× |
| 2 | 26 | 2.592 | 1.136× |

### nvidia-result-027

**B300 · 服务实验 · softplus_gradient** — 2026-09-16 / 报告已确认

- Workload：`softplus-gradient-fp32-triton-b300-r128-c1024-v2`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/softplus_gradient-20260916-114305/report.json; open_cake-1; event=26; candidate=7c347746d32c355786837f6905864de79bcd3cafa34fbf92642a91ac10bb6d8a`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 2.944 | 1.000× |
| 2 | 26 | 2.208 | 1.348× |

### nvidia-result-028

**B300 · 服务实验 · softsign** — 2026-09-16 / 报告已确认

- Workload：`softsign-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/softsign-20260916-111344/report.json; open_cake-1; event=13; candidate=c980c8bfc21b13b7e500597320d127c21771cd0bc8131b77e671c1c66493d142`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 2.400 | 1.013× |

### nvidia-result-029

**B300 · 服务实验 · swiglu** — 2026-09-16 / 报告已确认

- Workload：`swiglu-fp32-triton-b300-r128-c1024-v1`；目标：`sm_103a`；版本：`open-cake-ir-sm100a-v84`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/results/b300-service-20260920.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/results/b300-service-20260920.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oci-service-runs/swiglu-20260916-041831/report.json; open_cake-1; event=13; candidate=312b3a5636d5fef356c884dfc408b0e2d062467d3e7ea04e748ae6e002e14fd2`。
- 该 Campaign 记录的最佳合格候选；基线选择=starter_reference。仅投影原审计结果，不是当前 registry 冠军。

| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |
|---|---|---:|---:|
| 1 | 13 | 3.232 | 1.089× |
| 2 | 26 | 4.544 | 0.796× |

### nvidia-result-030

**B300 · CTA 宽度验证 · rmsnorm_input_gradient** — 2026-09-15 / 已确认

- Workload：`FP32 · R=128, C=1024`；目标：`sm_103a`；版本：`6198976b`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[findings/2026-09-15-005-explicit-triton-cta-width.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/2026-09-15-005-explicit-triton-cta-width.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oc-width-ticktock-20260915-DBihzZ/evaluation/rmsnorm_input_gradient/w8/confirmatory/receipt.json`。
- 显式选择 8 warps；10/10 配对胜出，五种输入与稳定性通过，保留 NCU。无通用最优宽度结论。

### nvidia-result-031

**B300 · CTA 宽度验证 · swiglu** — 2026-09-15 / 已确认

- Workload：`FP32 · R=128, C=1024`；目标：`sm_103a`；版本：`6198976b`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[findings/2026-09-15-005-explicit-triton-cta-width.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/2026-09-15-005-explicit-triton-cta-width.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oc-width-ticktock-20260915-DBihzZ/evaluation/swiglu/w16/confirmatory/receipt.json`。
- 显式选择 16 warps；10/10 配对胜出，五种输入与稳定性通过，保留 NCU。无通用最优宽度结论。

### nvidia-result-032

**B300 · CTA 宽度验证 · softmax_backward** — 2026-09-15 / 已确认

- Workload：`FP32 · R=128, C=1024`；目标：`sm_103a`；版本：`a5734239`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[findings/2026-09-15-005-explicit-triton-cta-width.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/2026-09-15-005-explicit-triton-cta-width.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oc-width-transfer-20260915-KoekDV/evaluation/softmax_backward/w16/confirmatory/receipt.json`。
- 显式选择 16 warps；10/10 配对胜出，五种输入与稳定性通过，保留 NCU。无通用最优宽度结论。

### nvidia-result-033

**B300 · CTA 宽度验证 · cosine_similarity** — 2026-09-15 / 已确认

- Workload：`FP32 · R=128, C=1024`；目标：`sm_103a`；版本：`a5734239`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[findings/2026-09-15-005-explicit-triton-cta-width.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/2026-09-15-005-explicit-triton-cta-width.json)。
- 原记录 / 实现定位：`B300-M2:/mnt/b300-shared/home/qinhaiyan/oc-width-transfer-20260915-KoekDV/evaluation/cosine_similarity/w8/confirmatory/receipt.json`。
- 显式选择 8 warps；10/10 配对胜出，五种输入与稳定性通过，保留 NCU。无通用最优宽度结论。

### nvidia-result-034

**B300 · CAKE 对照 · TinyGEMM2 s2** — 2026-09-20 / 正确；性能落后

- Workload：`BF16+bias · B/N/K=1 / 128 / 720`；目标：`sm_103a`；版本：`539c6c81`。
- 基线：官方 CAKE stage4 固定导出；比值口径：`inverse_rounded_slowdown`。
- 来源：[docs/NVIDIA_CAKE_REPRODUCTION.md](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/NVIDIA_CAKE_REPRODUCTION.md)。
- 原记录 / 实现定位：`B300-M2:cake-tinygemm-compare-b300-539c6c81-a98903082f5c/stages/comparison/checks/report.json`。
- 30/30 Open-Cake 与 15/15 CAKE 检查通过。此加速比由报告中四舍五入的 36.72x 耗时倍数取倒数，约数；非新配对计算。s2/s4 在短 K 无循环时不是两种流水线。

### nvidia-result-035

**B300 · CAKE 对照 · TinyGEMM2 s4** — 2026-09-20 / 正确；性能落后

- Workload：`BF16+bias · B/N/K=1 / 128 / 720`；目标：`sm_103a`；版本：`539c6c81`。
- 基线：官方 CAKE stage4 固定导出；比值口径：`inverse_rounded_slowdown`。
- 来源：[docs/NVIDIA_CAKE_REPRODUCTION.md](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/NVIDIA_CAKE_REPRODUCTION.md)。
- 原记录 / 实现定位：`B300-M2:cake-tinygemm-compare-b300-539c6c81-a98903082f5c/stages/comparison/checks/report.json`。
- 30/30 Open-Cake 与 15/15 CAKE 检查通过。此加速比由报告中四舍五入的 36.77x 耗时倍数取倒数，约数；非新配对计算。s2/s4 在短 K 无循环时不是两种流水线。

### nvidia-result-036

**B300 · CAKE 对照 · TinyGEMM2 s2** — 2026-09-20 / 正确；性能落后

- Workload：`BF16+bias · B/N/K=16 / 1024 / 1024`；目标：`sm_103a`；版本：`539c6c81`。
- 基线：官方 CAKE stage4 固定导出；比值口径：`inverse_rounded_slowdown`。
- 来源：[docs/NVIDIA_CAKE_REPRODUCTION.md](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/NVIDIA_CAKE_REPRODUCTION.md)。
- 原记录 / 实现定位：`B300-M2:cake-tinygemm-compare-b300-539c6c81-a98903082f5c/stages/comparison/checks/report.json`。
- 30/30 Open-Cake 与 15/15 CAKE 检查通过。此加速比由报告中四舍五入的 34.87x 耗时倍数取倒数，约数；非新配对计算。s2/s4 在短 K 无循环时不是两种流水线。

### nvidia-result-037

**B300 · CAKE 对照 · TinyGEMM2 s4** — 2026-09-20 / 正确；性能落后

- Workload：`BF16+bias · B/N/K=16 / 1024 / 1024`；目标：`sm_103a`；版本：`539c6c81`。
- 基线：官方 CAKE stage4 固定导出；比值口径：`inverse_rounded_slowdown`。
- 来源：[docs/NVIDIA_CAKE_REPRODUCTION.md](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/NVIDIA_CAKE_REPRODUCTION.md)。
- 原记录 / 实现定位：`B300-M2:cake-tinygemm-compare-b300-539c6c81-a98903082f5c/stages/comparison/checks/report.json`。
- 30/30 Open-Cake 与 15/15 CAKE 检查通过。此加速比由报告中四舍五入的 34.81x 耗时倍数取倒数，约数；非新配对计算。s2/s4 在短 K 无循环时不是两种流水线。

### nvidia-result-038

**B300 · CAKE 对照 · TinyGEMM2 s2** — 2026-09-20 / 正确；性能落后

- Workload：`BF16+bias · B/N/K=64 / 4096 / 3072`；目标：`sm_103a`；版本：`539c6c81`。
- 基线：官方 CAKE stage4 固定导出；比值口径：`inverse_rounded_slowdown`。
- 来源：[docs/NVIDIA_CAKE_REPRODUCTION.md](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/NVIDIA_CAKE_REPRODUCTION.md)。
- 原记录 / 实现定位：`B300-M2:cake-tinygemm-compare-b300-539c6c81-a98903082f5c/stages/comparison/checks/report.json`。
- 30/30 Open-Cake 与 15/15 CAKE 检查通过。此加速比由报告中四舍五入的 11.31x 耗时倍数取倒数，约数；非新配对计算。s2/s4 在短 K 无循环时不是两种流水线。

### nvidia-result-039

**B300 · CAKE 对照 · TinyGEMM2 s4** — 2026-09-20 / 正确；性能落后

- Workload：`BF16+bias · B/N/K=64 / 4096 / 3072`；目标：`sm_103a`；版本：`539c6c81`。
- 基线：官方 CAKE stage4 固定导出；比值口径：`inverse_rounded_slowdown`。
- 来源：[docs/NVIDIA_CAKE_REPRODUCTION.md](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/NVIDIA_CAKE_REPRODUCTION.md)。
- 原记录 / 实现定位：`B300-M2:cake-tinygemm-compare-b300-539c6c81-a98903082f5c/stages/comparison/checks/report.json`。
- 30/30 Open-Cake 与 15/15 CAKE 检查通过。此加速比由报告中四舍五入的 13.63x 耗时倍数取倒数，约数；非新配对计算。s2/s4 在短 K 无循环时不是两种流水线。

### nvidia-result-040

**B300 · CAKE 对照 · KDA prefill** — 2026-09-20 / 未实测

- Workload：`完整任务尚未完成`；目标：`sm_103a`；版本：`见来源`。
- 基线：官方 CAKE 导出；比值口径：`paired`。
- 来源：[docs/NVIDIA_CAKE_REPRODUCTION.md](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/NVIDIA_CAKE_REPRODUCTION.md)。
- 原记录 / 实现定位：`见来源所引用的原始实验`。
- 已有源参考与准备规格；尚无完整合格候选。

### nvidia-result-041

**B300 · CAKE 对照 · KDA decode** — 2026-09-20 / 未实测

- Workload：`完整任务尚未完成`；目标：`sm_103a`；版本：`见来源`。
- 基线：官方 CAKE 导出；比值口径：`paired`。
- 来源：[docs/NVIDIA_CAKE_REPRODUCTION.md](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/NVIDIA_CAKE_REPRODUCTION.md)。
- 原记录 / 实现定位：`见来源所引用的原始实验`。
- 已有源参考与准备规格；尚无完整合格候选。

### nvidia-result-042

**B300 · CAKE 对照 · Alpha-MoE** — 2026-09-20 / 未实测

- Workload：`完整任务尚未完成`；目标：`sm_103a`；版本：`见来源`。
- 基线：官方 CAKE 导出；比值口径：`paired`。
- 来源：[docs/NVIDIA_CAKE_REPRODUCTION.md](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/NVIDIA_CAKE_REPRODUCTION.md)。
- 原记录 / 实现定位：`见来源所引用的原始实验`。
- 已有源参考与准备规格；尚无完整合格候选。

### nvidia-result-043

**B200 · FMA / affine / state-store** — 2026-09-06 / 仅正确性

- Workload：`固定正确性实例，详见各报告`；目标：`sm_100a`；版本：`见来源`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/AKA_FMA_B200_CORRECTNESS_20260904.md](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/AKA_FMA_B200_CORRECTNESS_20260904.md)。
- 原记录 / 实现定位：`见来源所引用的原始实验`。
- 没有此组实例的合格计时；另见 AFFINE_PARENT_B200_CANARY_20260906 与 STATE_STORE_B200_CORRECTNESS_20260903。

### nvidia-fib-external-001-20260920

**B300 · FlashInfer外部对照 · 001_fused_add_rmsnorm_h2048** — 2026-09-20 / 正确；计时质量未通过

- Workload：`fib_fused_add_rmsnorm_h2048 / R=79, C=2048 / BF16`；目标：`sm_103a`；版本：`candidate 2812dd95; judge cb673fba`。
- 基线：用户提供的外部优秀实现（对应固定shape）；比值口径：`descriptive_quality_failed`。
- 来源：[findings/2026-09-20-007-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/be5e3e879b21ea782cdd0db0767a708d8d3ddda5/findings/2026-09-20-007-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/cake-reference-cxx20-20260920-e34e7a1aeb6a/stages/comparison/comparison-report.json`。
- 五种输入pre/postflight共30项全部通过。CUPTI device-activity、cold L2、10对×25样本、每cohort42次调用；最大CV=5.186%，1/20个cohort超出5%上限。不计host dispatch/编译/准备成本，不是模型端到端、官方多shape总榜或晋升。延迟仅为描述性中位数，发布加速比留空，不放宽CV门禁。外部源码不变，C++20为显式构建兼容条件。

### nvidia-fib-external-002-20260920

**B300 · FlashInfer外部对照 · 002_fused_add_rmsnorm_h4096** — 2026-09-20 / 正确；计时质量未通过

- Workload：`fib_fused_add_rmsnorm_h4096 / R=170, C=4096 / BF16`；目标：`sm_103a`；版本：`candidate 2812dd95; judge 1848f80d`。
- 基线：用户提供的外部优秀实现（对应固定shape）；比值口径：`descriptive_quality_failed`。
- 来源：[findings/2026-09-20-007-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/be5e3e879b21ea782cdd0db0767a708d8d3ddda5/findings/2026-09-20-007-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/cake-native-isolation-comparison-20260920-0fc73116ae8d/stages/comparison/comparison-report.json`。
- 五种输入pre/postflight共30项全部通过。CUPTI device-activity、cold L2、10对×25样本、每cohort42次调用；最大CV=5.134%，1/20个cohort超出5%上限。不计host dispatch/编译/准备成本，不是模型端到端、官方多shape总榜或晋升。延迟仅为描述性中位数，发布加速比留空，不放宽CV门禁。外部源码不变，C++20为显式构建兼容条件。

### nvidia-fib-external-003-20260920

**B300 · FlashInfer外部对照 · 003_fused_add_rmsnorm_h7168** — 2026-09-20 / 正确；外部参考更快（质量通过）

- Workload：`fib_fused_add_rmsnorm_h7168 / R=64, C=7168 / BF16`；目标：`sm_103a`；版本：`candidate 2812dd95; judge 1848f80d`。
- 基线：003派生修正外部参考；比值口径：`paired_external`。
- 来源：[findings/2026-09-20-007-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/be5e3e879b21ea782cdd0db0767a708d8d3ddda5/findings/2026-09-20-007-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/cake-native-isolation-comparison-20260920-167e275db2a8/stages/comparison/comparison-report.json`。
- 五种输入pre/postflight共30项全部通过。CUPTI device-activity、cold L2、10对×25样本、每cohort42次调用；最大CV=4.899%，0/20个cohort超出5%上限。不计host dispatch/编译/准备成本，不是模型端到端、官方多shape总榜或晋升。外部为修正host Graph参数ABI的派生参考，device kernel不变；原始参考失败记录保留。

### nvidia-fib-external-004-20260920

**B300 · FlashInfer外部对照 · 004_gemm_n128_k2048** — 2026-09-20 / 正确；计时质量未通过

- Workload：`fib_gemm_n128_k2048 / R=1, C=128 / FP16`；目标：`sm_103a`；版本：`candidate 1848f80d; judge beb875df`。
- 基线：用户提供的外部优秀实现（对应固定shape）；比值口径：`descriptive_quality_failed`。
- 来源：[findings/2026-09-20-007-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/be5e3e879b21ea782cdd0db0767a708d8d3ddda5/findings/2026-09-20-007-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-gap-analysis-20260920-comparison-89bd3818cbba/stages/comparison/comparison-report.json`。
- 五种输入pre/postflight共30项全部通过。CUPTI device-activity、cold L2、10对×25样本、每cohort42次调用；最大CV=6.235%，4/20个cohort超出5%上限。不计host dispatch/编译/准备成本，不是模型端到端、官方多shape总榜或晋升。延迟仅为描述性中位数，发布加速比留空，不放宽CV门禁。外部源码不变，C++20为显式构建兼容条件。

### nvidia-fib-external-021-20260920

**B300 · FlashInfer外部对照 · 021_rmsnorm_h128** — 2026-09-20 / 正确；计时质量未通过

- Workload：`fib_rmsnorm_h128 / R=2528, C=128 / BF16`；目标：`sm_103a`；版本：`candidate 2812dd95; judge 40c4876b`。
- 基线：用户提供的外部优秀实现（对应固定shape）；比值口径：`descriptive_quality_failed`。
- 来源：[findings/2026-09-20-007-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/be5e3e879b21ea782cdd0db0767a708d8d3ddda5/findings/2026-09-20-007-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/cake-full-reference-comparison-20260920-01473a3ec07b/stages/comparison/comparison-report.json`。
- 五种输入pre/postflight共30项全部通过。CUPTI device-activity、cold L2、10对×25样本、每cohort42次调用；最大CV=5.433%，3/20个cohort超出5%上限。不计host dispatch/编译/准备成本，不是模型端到端、官方多shape总榜或晋升。延迟仅为描述性中位数，发布加速比留空，不放宽CV门禁。

### nvidia-fib-external-022-20260920

**B300 · FlashInfer外部对照 · 022_rmsnorm_h512** — 2026-09-20 / 正确；计时质量未通过

- Workload：`fib_rmsnorm_h512 / R=539, C=512 / BF16`；目标：`sm_103a`；版本：`candidate 2812dd95; judge 40c4876b`。
- 基线：用户提供的外部优秀实现（对应固定shape）；比值口径：`descriptive_quality_failed`。
- 来源：[findings/2026-09-20-007-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/be5e3e879b21ea782cdd0db0767a708d8d3ddda5/findings/2026-09-20-007-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/cake-full-reference-comparison-20260920-213dc1d85874/stages/comparison/comparison-report.json`。
- 五种输入pre/postflight共30项全部通过。CUPTI device-activity、cold L2、10对×25样本、每cohort42次调用；最大CV=5.723%，1/20个cohort超出5%上限。不计host dispatch/编译/准备成本，不是模型端到端、官方多shape总榜或晋升。延迟仅为描述性中位数，发布加速比留空，不放宽CV门禁。

### nvidia-fib-external-025-20260920

**B300 · FlashInfer外部对照 · 025_rmsnorm_h4096** — 2026-09-20 / 正确；计时质量未通过

- Workload：`fib_rmsnorm_h4096 / R=170, C=4096 / BF16`；目标：`sm_103a`；版本：`candidate 2812dd95; judge 40c4876b`。
- 基线：用户提供的外部优秀实现（对应固定shape）；比值口径：`descriptive_quality_failed`。
- 来源：[findings/2026-09-20-007-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/be5e3e879b21ea782cdd0db0767a708d8d3ddda5/findings/2026-09-20-007-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/cake-full-reference-comparison-20260920-26982f378ab5/stages/comparison/comparison-report.json`。
- 五种输入pre/postflight共30项全部通过。CUPTI device-activity、cold L2、10对×25样本、每cohort42次调用；最大CV=5.122%，1/20个cohort超出5%上限。不计host dispatch/编译/准备成本，不是模型端到端、官方多shape总榜或晋升。延迟仅为描述性中位数，发布加速比留空，不放宽CV门禁。

### nvidia-fib-external-022-starter-20260920

**B300 · FlashInfer外部对照 · 022_rmsnorm_h512_starter** — 2026-09-20 / 正确；starter更快（质量通过）

- Workload：`fib_rmsnorm_h512 / R=539, C=512 / BF16`；目标：`sm_103a`；版本：`starter 2812dd95; judge 40c4876b`。
- 基线：用户提供的外部优秀实现（对应固定shape）；比值口径：`paired_external`。
- 来源：[findings/2026-09-20-007-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/be5e3e879b21ea782cdd0db0767a708d8d3ddda5/findings/2026-09-20-007-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/cake-full-reference-comparison-20260920-213dc1d85874/stages/comparison/comparison-report.json; starter_vs_external`。
- 一行/CTA的固定starter胜过该外部参考；这不等于四行打包的改写候选也更快。相同shape和精度、五种输入检查通过；10对×25样本、cold L2/CUPTI、CV门禁通过。外部优秀实现不保证每个固定shape最优。

### nvidia-fib-external-005-20260920

**B300 / FlashInfer comparison · 005_gemm_n256_k7168** — 2026-09-20 / Correct; timing quality failed

- Workload：`fib_gemm_n256_k7168 / R=1, C=256 / FP16`；目标：`sm_103a`；版本：`open-cake-ir@1848f80dfaf60d89581f98c01659243a44df7b50; judge 76cf8762`。
- 基线：Supplied external implementation；比值口径：`descriptive_quality_failed`。
- 来源：[findings/2026-09-20-010-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/839e8b8a4629e8b664724e22c98818337c6bc82c/findings/2026-09-20-010-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-gap-successors-20260920-comparison-2126e322eb9c/stages/comparison/comparison-report.json`。
- All 30 pre/postflight checks over five distributions passed. CUPTI device activity, cold L2 per sample, 10 pairs x 25 samples, 42 calls per cohort. Maximum CV=5.456%; 1/20 cohorts exceed the unchanged 5% limit. Historical frozen candidate, not the new AOT alignment implementation; no E2E or official leaderboard claim.

### nvidia-fib-starter-005-20260920

**B300 / FlashInfer comparison · 005_gemm_n256_k7168** — 2026-09-20 / Correct; timing qualified

- Workload：`fib_gemm_n256_k7168 / R=1, C=256 / FP16`；目标：`sm_103a`；版本：`open-cake-ir@1848f80dfaf60d89581f98c01659243a44df7b50; judge 76cf8762`。
- 基线：Original Cake starter；比值口径：`paired`。
- 来源：[findings/2026-09-20-010-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/839e8b8a4629e8b664724e22c98818337c6bc82c/findings/2026-09-20-010-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-gap-successors-20260920-comparison-2126e322eb9c/stages/comparison/comparison-report.json`。
- All 30 pre/postflight checks over five distributions passed. CUPTI device activity, cold L2 per sample, 10 pairs x 25 samples, 42 calls per cohort. Maximum CV=2.755%; 0/20 cohorts exceed the unchanged 5% limit. Historical frozen candidate, not the new AOT alignment implementation; no E2E or official leaderboard claim. Keep the starter: this candidate is slower in a qualified comparison.

### nvidia-fib-external-023-20260920

**B300 / FlashInfer comparison · 023_rmsnorm_h1536** — 2026-09-20 / Correct; timing quality failed

- Workload：`fib_rmsnorm_h1536 / R=539, C=1536 / BF16`；目标：`sm_103a`；版本：`open-cake-ir@1848f80dfaf60d89581f98c01659243a44df7b50; judge 76cf8762`。
- 基线：Supplied external implementation；比值口径：`descriptive_quality_failed`。
- 来源：[findings/2026-09-20-010-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/839e8b8a4629e8b664724e22c98818337c6bc82c/findings/2026-09-20-010-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-gap-successors-20260920-comparison-c2e26eb06cd5/stages/comparison/comparison-report.json`。
- All 30 pre/postflight checks over five distributions passed. CUPTI device activity, cold L2 per sample, 10 pairs x 25 samples, 42 calls per cohort. Maximum CV=5.202%; 1/20 cohorts exceed the unchanged 5% limit. Historical frozen candidate, not the new AOT alignment implementation; no E2E or official leaderboard claim.

### nvidia-fib-starter-023-20260920

**B300 / FlashInfer comparison · 023_rmsnorm_h1536** — 2026-09-20 / Correct; timing quality failed

- Workload：`fib_rmsnorm_h1536 / R=539, C=1536 / BF16`；目标：`sm_103a`；版本：`open-cake-ir@1848f80dfaf60d89581f98c01659243a44df7b50; judge 76cf8762`。
- 基线：Original Cake starter；比值口径：`descriptive_quality_failed`。
- 来源：[findings/2026-09-20-010-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/839e8b8a4629e8b664724e22c98818337c6bc82c/findings/2026-09-20-010-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-gap-successors-20260920-comparison-c2e26eb06cd5/stages/comparison/comparison-report.json`。
- All 30 pre/postflight checks over five distributions passed. CUPTI device activity, cold L2 per sample, 10 pairs x 25 samples, 42 calls per cohort. Maximum CV=5.228%; 1/20 cohorts exceed the unchanged 5% limit. Historical frozen candidate, not the new AOT alignment implementation; no E2E or official leaderboard claim.

### nvidia-fib-external-024-20260920

**B300 / FlashInfer comparison · 024_rmsnorm_h2048** — 2026-09-20 / Correct; timing quality failed

- Workload：`fib_rmsnorm_h2048 / R=79, C=2048 / BF16`；目标：`sm_103a`；版本：`open-cake-ir@1848f80dfaf60d89581f98c01659243a44df7b50; judge cb5e73b3`。
- 基线：Supplied external implementation；比值口径：`descriptive_quality_failed`。
- 来源：[findings/2026-09-20-010-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/839e8b8a4629e8b664724e22c98818337c6bc82c/findings/2026-09-20-010-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-gap-024-20260920-comparison-6d911313e6b2/stages/comparison/comparison-report.json`。
- All 30 pre/postflight checks over five distributions passed. CUPTI device activity, cold L2 per sample, 10 pairs x 25 samples, 42 calls per cohort. Maximum CV=7.124%; 4/20 cohorts exceed the unchanged 5% limit. Historical frozen candidate, not the new AOT alignment implementation; no E2E or official leaderboard claim.

### nvidia-fib-starter-024-20260920

**B300 / FlashInfer comparison · 024_rmsnorm_h2048** — 2026-09-20 / Correct; timing qualified

- Workload：`fib_rmsnorm_h2048 / R=79, C=2048 / BF16`；目标：`sm_103a`；版本：`open-cake-ir@1848f80dfaf60d89581f98c01659243a44df7b50; judge cb5e73b3`。
- 基线：Original Cake starter；比值口径：`paired`。
- 来源：[findings/2026-09-20-010-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/839e8b8a4629e8b664724e22c98818337c6bc82c/findings/2026-09-20-010-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-gap-024-20260920-comparison-6d911313e6b2/stages/comparison/comparison-report.json`。
- All 30 pre/postflight checks over five distributions passed. CUPTI device activity, cold L2 per sample, 10 pairs x 25 samples, 42 calls per cohort. Maximum CV=3.979%; 0/20 cohorts exceed the unchanged 5% limit. Historical frozen candidate, not the new AOT alignment implementation; no E2E or official leaderboard claim.

### nvidia-alignment-guard-001-20260920

**B300 / guarded AOT validation · 001_fused_add_rmsnorm_h2048** — 2026-09-20 / Correctness only: 65 checks passed

- Workload：`Retained exact shape / BF16 / five distributions / aligned and each pointer offset 2, 4, 8 bytes`；目标：`sm_103a`；版本：`compiler 88aab6b2; judge f1499b70`。
- 基线：Original Workload oracle; no timing comparison；比值口径：`correctness_only`。
- 来源：[findings/2026-09-20-011-triton-aot-pointer-alignment.json](https://github.com/qhy991/open-cake-ir/blob/e1e550b44bf0f06866468c16b0bb253305ce6655/findings/2026-09-20-011-triton-aot-pointer-alignment.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-alignment-guards-20260920-0e762344f212/stages/verify/guard-report.json`。
- CPU preparation precedes the broker GPU stage. Every input/output snapshot is checked by the following CPU stage after GPU worker exit. All input mutation checks passed. Generic and aligned binaries are both exercised; the extracted restricted leaf rejects unaligned calls. No speedup or broad-shape claim.

### nvidia-alignment-guard-002-20260920

**B300 / guarded AOT validation · 002_fused_add_rmsnorm_h4096** — 2026-09-20 / Correctness only: 65 checks passed

- Workload：`Retained exact shape / BF16 / five distributions / aligned and each pointer offset 2, 4, 8 bytes`；目标：`sm_103a`；版本：`compiler 88aab6b2; judge f1499b70`。
- 基线：Original Workload oracle; no timing comparison；比值口径：`correctness_only`。
- 来源：[findings/2026-09-20-011-triton-aot-pointer-alignment.json](https://github.com/qhy991/open-cake-ir/blob/e1e550b44bf0f06866468c16b0bb253305ce6655/findings/2026-09-20-011-triton-aot-pointer-alignment.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-alignment-guards-20260920-13447a2cfa61/stages/verify/guard-report.json`。
- CPU preparation precedes the broker GPU stage. Every input/output snapshot is checked by the following CPU stage after GPU worker exit. All input mutation checks passed. Generic and aligned binaries are both exercised; the extracted restricted leaf rejects unaligned calls. No speedup or broad-shape claim.

### nvidia-alignment-guard-025-20260920

**B300 / guarded AOT validation · 025_rmsnorm_h4096** — 2026-09-20 / Correctness only: 50 checks passed

- Workload：`Retained exact shape / BF16 / five distributions / aligned and each pointer offset 2, 4, 8 bytes`；目标：`sm_103a`；版本：`compiler 88aab6b2; judge f1499b70`。
- 基线：Original Workload oracle; no timing comparison；比值口径：`correctness_only`。
- 来源：[findings/2026-09-20-011-triton-aot-pointer-alignment.json](https://github.com/qhy991/open-cake-ir/blob/e1e550b44bf0f06866468c16b0bb253305ce6655/findings/2026-09-20-011-triton-aot-pointer-alignment.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-alignment-guards-20260920-33445a79498a/stages/verify/guard-report.json`。
- CPU preparation precedes the broker GPU stage. Every input/output snapshot is checked by the following CPU stage after GPU worker exit. All input mutation checks passed. Generic and aligned binaries are both exercised; the extracted restricted leaf rejects unaligned calls. No speedup or broad-shape claim.

### nvidia-alignment-001-optimized_vs_starter-20260920

**B300 / AOT alignment ablation · 001_fused_add_rmsnorm_h2048** — 2026-09-20 / Correct; timing qualified; first_arm_faster

- Workload：`R=79, H=2048, BF16; same source/grid/options, alignment treatment only`；目标：`sm_103a`；版本：`compiler 88aab6b2; judge d31c551a`。
- 基线：Unchanged pre-specialization optimized binary；比值口径：`paired`。
- 来源：[findings/2026-09-20-011-triton-aot-pointer-alignment.json](https://github.com/qhy991/open-cake-ir/blob/8c97440e90a1d55e14160269d93f412b709e8472/findings/2026-09-20-011-triton-aot-pointer-alignment.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-staged-alignment-comparison-20260920-df5de212bcdd/stages/verify/comparison-report.json`。
- All 2550 complete snapshots passed, including 30 pre/post checks and 2520 fresh cohort calls. All timing-quality gates passed. Cold L2 CUPTI, 10 ordered pairs and 25 samples/cohort; all three edges in one exclusive allocation. The external edge is close_null under the unchanged 5 percent materiality threshold, not a material external win. No official leaderboard or model E2E claim.

### nvidia-alignment-001-optimized_vs_external-20260920

**B300 / AOT alignment ablation · 001_fused_add_rmsnorm_h2048** — 2026-09-20 / Correct; timing qualified; close_null

- Workload：`R=79, H=2048, BF16; same source/grid/options, alignment treatment only`；目标：`sm_103a`；版本：`compiler 88aab6b2; judge d31c551a`。
- 基线：Supplied external CUDA implementation；比值口径：`paired`。
- 来源：[findings/2026-09-20-011-triton-aot-pointer-alignment.json](https://github.com/qhy991/open-cake-ir/blob/8c97440e90a1d55e14160269d93f412b709e8472/findings/2026-09-20-011-triton-aot-pointer-alignment.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-staged-alignment-comparison-20260920-df5de212bcdd/stages/verify/comparison-report.json`。
- All 2550 complete snapshots passed, including 30 pre/post checks and 2520 fresh cohort calls. All timing-quality gates passed. Cold L2 CUPTI, 10 ordered pairs and 25 samples/cohort; all three edges in one exclusive allocation. The external edge is close_null under the unchanged 5 percent materiality threshold, not a material external win. No official leaderboard or model E2E claim.

### nvidia-alignment-002-optimized_vs_external-20260920

**B300 / AOT alignment ablation · 002_fused_add_rmsnorm_h4096** — 2026-09-20 / Correct; first_arm_faster

- Workload：`R=170, H=4096, BF16; same source/grid/options, alignment treatment only`；目标：`sm_103a`；版本：`compiler 88aab6b2; judge d31c551a`。
- 基线：Supplied external implementation；比值口径：`paired`。
- 来源：[findings/2026-09-20-011-triton-aot-pointer-alignment.json](https://github.com/qhy991/open-cake-ir/blob/b7abaf3e9e386f56da27f26e5fc940f515541078/findings/2026-09-20-011-triton-aot-pointer-alignment.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-staged-alignment-comparison-20260920-69481876670f/stages/verify/comparison-report.json`。
- All 2550 complete snapshots passed. Each comparison edge retains its own quality decision. The overall three-edge report fails quality because the new/old-control edge fails the unchanged CV gate; it is not a blanket promotion. Cold L2 CUPTI, 10 ordered pairs, 25 samples per cohort, one continuous paired allocation. No model E2E or official leaderboard claim.

### nvidia-alignment-002-optimized_vs_starter-20260920

**B300 / AOT alignment ablation · 002_fused_add_rmsnorm_h4096** — 2026-09-20 / Correct; measurement_quality_failed

- Workload：`R=170, H=4096, BF16; same source/grid/options, alignment treatment only`；目标：`sm_103a`；版本：`compiler 88aab6b2; judge d31c551a`。
- 基线：Unchanged pre-specialization optimized binary；比值口径：`descriptive_quality_failed`。
- 来源：[findings/2026-09-20-011-triton-aot-pointer-alignment.json](https://github.com/qhy991/open-cake-ir/blob/b7abaf3e9e386f56da27f26e5fc940f515541078/findings/2026-09-20-011-triton-aot-pointer-alignment.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-staged-alignment-comparison-20260920-69481876670f/stages/verify/comparison-report.json`。
- All 2550 complete snapshots passed. Each comparison edge retains its own quality decision. The overall three-edge report fails quality because the new/old-control edge fails the unchanged CV gate; it is not a blanket promotion. Cold L2 CUPTI, 10 ordered pairs, 25 samples per cohort, one continuous paired allocation. No model E2E or official leaderboard claim.

### nvidia-alignment-025-optimized_vs_external-20260920

**B300 / AOT alignment ablation · 025_rmsnorm_h4096** — 2026-09-20 / Correct; close_null

- Workload：`R=170, H=4096, BF16; same source/grid/options, alignment treatment only`；目标：`sm_103a`；版本：`compiler 88aab6b2; judge d31c551a`。
- 基线：Supplied external implementation；比值口径：`paired`。
- 来源：[findings/2026-09-20-011-triton-aot-pointer-alignment.json](https://github.com/qhy991/open-cake-ir/blob/b7abaf3e9e386f56da27f26e5fc940f515541078/findings/2026-09-20-011-triton-aot-pointer-alignment.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-staged-alignment-comparison-20260920-68716e4d66e1/stages/verify/comparison-report.json`。
- All 2550 complete snapshots passed. Each comparison edge retains its own quality decision. The overall three-edge report fails quality because the new/old-control edge fails the unchanged CV gate; it is not a blanket promotion. Cold L2 CUPTI, 10 ordered pairs, 25 samples per cohort, one continuous paired allocation. No model E2E or official leaderboard claim.

### nvidia-alignment-025-optimized_vs_starter-20260920

**B300 / AOT alignment ablation · 025_rmsnorm_h4096** — 2026-09-20 / Correct; measurement_quality_failed

- Workload：`R=170, H=4096, BF16; same source/grid/options, alignment treatment only`；目标：`sm_103a`；版本：`compiler 88aab6b2; judge d31c551a`。
- 基线：Unchanged pre-specialization optimized binary；比值口径：`descriptive_quality_failed`。
- 来源：[findings/2026-09-20-011-triton-aot-pointer-alignment.json](https://github.com/qhy991/open-cake-ir/blob/b7abaf3e9e386f56da27f26e5fc940f515541078/findings/2026-09-20-011-triton-aot-pointer-alignment.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-staged-alignment-comparison-20260920-68716e4d66e1/stages/verify/comparison-report.json`。
- All 2550 complete snapshots passed. Each comparison edge retains its own quality decision. The overall three-edge report fails quality because the new/old-control edge fails the unchanged CV gate; it is not a blanket promotion. Cold L2 CUPTI, 10 ordered pairs, 25 samples per cohort, one continuous paired allocation. No model E2E or official leaderboard claim.

### nvidia-assignment-003-w1-optimized_vs_starter-20260920

**B300 / work-assignment ablation · 003_fused_add_rmsnorm_h7168** — 2026-09-20 / Correct; close_null

- Workload：`R=64, H=7168, BF16; 1 execution groups; generic AOT`；目标：`sm_103a`；版本：`compiler/judge ba4537fd`。
- 基线：Frozen old optimized binary；比值口径：`paired`。
- 来源：[findings/2026-09-20-010-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/f1bf6497ac0659e2eff8a5ee952d591cd8cd22fe/findings/2026-09-20-010-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-work-assignment-comparison-20260920-5e937a53efea/stages/verify/comparison-report.json`。
- All 2550 complete snapshots passed. 003 control is the old optimized binary; 022 control is the original one-row starter. Width one separates compiler-floor movement. Other widths retain the authored operation graph, accesses, loops and grid. Each edge keeps its own CV decision; the full report may fail quality. No new profiler, model E2E, official leaderboard or promotion claim.

### nvidia-assignment-003-w1-optimized_vs_external-20260920

**B300 / work-assignment ablation · 003_fused_add_rmsnorm_h7168** — 2026-09-20 / Correct; second_arm_faster

- Workload：`R=64, H=7168, BF16; 1 execution groups; generic AOT`；目标：`sm_103a`；版本：`compiler/judge ba4537fd`。
- 基线：Derived external reference (Graph host-argument correction)；比值口径：`paired`。
- 来源：[findings/2026-09-20-010-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/f1bf6497ac0659e2eff8a5ee952d591cd8cd22fe/findings/2026-09-20-010-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-work-assignment-comparison-20260920-5e937a53efea/stages/verify/comparison-report.json`。
- All 2550 complete snapshots passed. 003 control is the old optimized binary; 022 control is the original one-row starter. Width one separates compiler-floor movement. Other widths retain the authored operation graph, accesses, loops and grid. Each edge keeps its own CV decision; the full report may fail quality. No new profiler, model E2E, official leaderboard or promotion claim.

### nvidia-assignment-003-w4-optimized_vs_starter-20260920

**B300 / work-assignment ablation · 003_fused_add_rmsnorm_h7168** — 2026-09-20 / Correct; first_arm_faster

- Workload：`R=64, H=7168, BF16; 4 execution groups; generic AOT`；目标：`sm_103a`；版本：`compiler/judge ba4537fd`。
- 基线：Frozen old optimized binary；比值口径：`paired`。
- 来源：[findings/2026-09-20-010-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/f1bf6497ac0659e2eff8a5ee952d591cd8cd22fe/findings/2026-09-20-010-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-work-assignment-comparison-20260920-7abcd87339da/stages/verify/comparison-report.json`。
- All 2550 complete snapshots passed. 003 control is the old optimized binary; 022 control is the original one-row starter. Width one separates compiler-floor movement. Other widths retain the authored operation graph, accesses, loops and grid. Each edge keeps its own CV decision; the full report may fail quality. No new profiler, model E2E, official leaderboard or promotion claim.

### nvidia-assignment-003-w4-optimized_vs_external-20260920

**B300 / work-assignment ablation · 003_fused_add_rmsnorm_h7168** — 2026-09-20 / Correct; measurement_quality_failed

- Workload：`R=64, H=7168, BF16; 4 execution groups; generic AOT`；目标：`sm_103a`；版本：`compiler/judge ba4537fd`。
- 基线：Derived external reference (Graph host-argument correction)；比值口径：`descriptive_quality_failed`。
- 来源：[findings/2026-09-20-010-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/f1bf6497ac0659e2eff8a5ee952d591cd8cd22fe/findings/2026-09-20-010-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-work-assignment-comparison-20260920-7abcd87339da/stages/verify/comparison-report.json`。
- All 2550 complete snapshots passed. 003 control is the old optimized binary; 022 control is the original one-row starter. Width one separates compiler-floor movement. Other widths retain the authored operation graph, accesses, loops and grid. Each edge keeps its own CV decision; the full report may fail quality. No new profiler, model E2E, official leaderboard or promotion claim.

### nvidia-assignment-003-w8-optimized_vs_starter-20260920

**B300 / work-assignment ablation · 003_fused_add_rmsnorm_h7168** — 2026-09-20 / Correct; first_arm_faster

- Workload：`R=64, H=7168, BF16; 8 execution groups; generic AOT`；目标：`sm_103a`；版本：`compiler/judge ba4537fd`。
- 基线：Frozen old optimized binary；比值口径：`paired`。
- 来源：[findings/2026-09-20-010-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/230ae65a5d6f62b14639b72f3d87c94ddc78a28f/findings/2026-09-20-010-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-work-assignment-comparison-20260920-c3bed83a0d53/stages/verify/comparison-report.json`。
- All 2550 complete snapshots passed. 003 control is the old optimized binary; 022 control is the original one-row starter. Width one separates compiler-floor movement. Other widths retain the authored operation graph, accesses, loops and grid. Each edge keeps its own CV decision; the full report may fail quality. No new profiler, model E2E, official leaderboard or promotion claim.

### nvidia-assignment-003-w8-optimized_vs_external-20260920

**B300 / work-assignment ablation · 003_fused_add_rmsnorm_h7168** — 2026-09-20 / Correct; measurement_quality_failed

- Workload：`R=64, H=7168, BF16; 8 execution groups; generic AOT`；目标：`sm_103a`；版本：`compiler/judge ba4537fd`。
- 基线：Derived external reference (Graph host-argument correction)；比值口径：`descriptive_quality_failed`。
- 来源：[findings/2026-09-20-010-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/230ae65a5d6f62b14639b72f3d87c94ddc78a28f/findings/2026-09-20-010-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-work-assignment-comparison-20260920-c3bed83a0d53/stages/verify/comparison-report.json`。
- All 2550 complete snapshots passed. 003 control is the old optimized binary; 022 control is the original one-row starter. Width one separates compiler-floor movement. Other widths retain the authored operation graph, accesses, loops and grid. Each edge keeps its own CV decision; the full report may fail quality. No new profiler, model E2E, official leaderboard or promotion claim.

### nvidia-assignment-003-w16-optimized_vs_starter-20260920

**B300 / work-assignment ablation · 003_fused_add_rmsnorm_h7168** — 2026-09-20 / Correct; first_arm_faster

- Workload：`R=64, H=7168, BF16; 16 execution groups; generic AOT`；目标：`sm_103a`；版本：`compiler/judge ba4537fd`。
- 基线：Frozen old optimized binary；比值口径：`paired`。
- 来源：[findings/2026-09-20-010-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/f1bf6497ac0659e2eff8a5ee952d591cd8cd22fe/findings/2026-09-20-010-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-work-assignment-comparison-20260920-a081f2ee0d4f/stages/verify/comparison-report.json`。
- All 2550 complete snapshots passed. 003 control is the old optimized binary; 022 control is the original one-row starter. Width one separates compiler-floor movement. Other widths retain the authored operation graph, accesses, loops and grid. Each edge keeps its own CV decision; the full report may fail quality. No new profiler, model E2E, official leaderboard or promotion claim.

### nvidia-assignment-003-w16-optimized_vs_external-20260920

**B300 / work-assignment ablation · 003_fused_add_rmsnorm_h7168** — 2026-09-20 / Correct; measurement_quality_failed

- Workload：`R=64, H=7168, BF16; 16 execution groups; generic AOT`；目标：`sm_103a`；版本：`compiler/judge ba4537fd`。
- 基线：Derived external reference (Graph host-argument correction)；比值口径：`descriptive_quality_failed`。
- 来源：[findings/2026-09-20-010-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/f1bf6497ac0659e2eff8a5ee952d591cd8cd22fe/findings/2026-09-20-010-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-work-assignment-comparison-20260920-a081f2ee0d4f/stages/verify/comparison-report.json`。
- All 2550 complete snapshots passed. 003 control is the old optimized binary; 022 control is the original one-row starter. Width one separates compiler-floor movement. Other widths retain the authored operation graph, accesses, loops and grid. Each edge keeps its own CV decision; the full report may fail quality. No new profiler, model E2E, official leaderboard or promotion claim.

### nvidia-assignment-022-w1-optimized_vs_starter-20260920

**B300 / work-assignment ablation · 022_rmsnorm_h512** — 2026-09-20 / Correct; measurement_quality_failed

- Workload：`R=539, H=512, BF16; 1 execution groups; generic AOT`；目标：`sm_103a`；版本：`compiler/judge ba4537fd`。
- 基线：Original single-row starter；比值口径：`descriptive_quality_failed`。
- 来源：[findings/2026-09-20-010-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/f1bf6497ac0659e2eff8a5ee952d591cd8cd22fe/findings/2026-09-20-010-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-work-assignment-comparison-20260920-9f123f05addb/stages/verify/comparison-report.json`。
- All 2550 complete snapshots passed. 003 control is the old optimized binary; 022 control is the original one-row starter. Width one separates compiler-floor movement. Other widths retain the authored operation graph, accesses, loops and grid. Each edge keeps its own CV decision; the full report may fail quality. No new profiler, model E2E, official leaderboard or promotion claim.

### nvidia-assignment-022-w1-optimized_vs_external-20260920

**B300 / work-assignment ablation · 022_rmsnorm_h512** — 2026-09-20 / Correct; measurement_quality_failed

- Workload：`R=539, H=512, BF16; 1 execution groups; generic AOT`；目标：`sm_103a`；版本：`compiler/judge ba4537fd`。
- 基线：Supplied external implementation；比值口径：`descriptive_quality_failed`。
- 来源：[findings/2026-09-20-010-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/f1bf6497ac0659e2eff8a5ee952d591cd8cd22fe/findings/2026-09-20-010-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-work-assignment-comparison-20260920-9f123f05addb/stages/verify/comparison-report.json`。
- All 2550 complete snapshots passed. 003 control is the old optimized binary; 022 control is the original one-row starter. Width one separates compiler-floor movement. Other widths retain the authored operation graph, accesses, loops and grid. Each edge keeps its own CV decision; the full report may fail quality. No new profiler, model E2E, official leaderboard or promotion claim.

### nvidia-assignment-022-w4-optimized_vs_starter-20260920

**B300 / work-assignment ablation · 022_rmsnorm_h512** — 2026-09-20 / Correct; measurement_quality_failed

- Workload：`R=539, H=512, BF16; 4 execution groups; generic AOT`；目标：`sm_103a`；版本：`compiler/judge ba4537fd`。
- 基线：Original single-row starter；比值口径：`descriptive_quality_failed`。
- 来源：[findings/2026-09-20-010-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/f1bf6497ac0659e2eff8a5ee952d591cd8cd22fe/findings/2026-09-20-010-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-work-assignment-comparison-20260920-0ca4fb237cc1/stages/verify/comparison-report.json`。
- All 2550 complete snapshots passed. 003 control is the old optimized binary; 022 control is the original one-row starter. Width one separates compiler-floor movement. Other widths retain the authored operation graph, accesses, loops and grid. Each edge keeps its own CV decision; the full report may fail quality. No new profiler, model E2E, official leaderboard or promotion claim.

### nvidia-assignment-022-w4-optimized_vs_external-20260920

**B300 / work-assignment ablation · 022_rmsnorm_h512** — 2026-09-20 / Correct; first_arm_faster

- Workload：`R=539, H=512, BF16; 4 execution groups; generic AOT`；目标：`sm_103a`；版本：`compiler/judge ba4537fd`。
- 基线：Supplied external implementation；比值口径：`paired`。
- 来源：[findings/2026-09-20-010-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/f1bf6497ac0659e2eff8a5ee952d591cd8cd22fe/findings/2026-09-20-010-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-work-assignment-comparison-20260920-0ca4fb237cc1/stages/verify/comparison-report.json`。
- All 2550 complete snapshots passed. 003 control is the old optimized binary; 022 control is the original one-row starter. Width one separates compiler-floor movement. Other widths retain the authored operation graph, accesses, loops and grid. Each edge keeps its own CV decision; the full report may fail quality. No new profiler, model E2E, official leaderboard or promotion claim.

### nvidia-assignment-022-w8-optimized_vs_starter-20260920

**B300 / work-assignment ablation · 022_rmsnorm_h512** — 2026-09-20 / Correct; measurement_quality_failed

- Workload：`R=539, H=512, BF16; 8 execution groups; generic AOT`；目标：`sm_103a`；版本：`compiler/judge ba4537fd`。
- 基线：Original single-row starter；比值口径：`descriptive_quality_failed`。
- 来源：[findings/2026-09-20-010-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/f1bf6497ac0659e2eff8a5ee952d591cd8cd22fe/findings/2026-09-20-010-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-work-assignment-comparison-20260920-125fa0d071ac/stages/verify/comparison-report.json`。
- All 2550 complete snapshots passed. 003 control is the old optimized binary; 022 control is the original one-row starter. Width one separates compiler-floor movement. Other widths retain the authored operation graph, accesses, loops and grid. Each edge keeps its own CV decision; the full report may fail quality. No new profiler, model E2E, official leaderboard or promotion claim.

### nvidia-assignment-022-w8-optimized_vs_external-20260920

**B300 / work-assignment ablation · 022_rmsnorm_h512** — 2026-09-20 / Correct; measurement_quality_failed

- Workload：`R=539, H=512, BF16; 8 execution groups; generic AOT`；目标：`sm_103a`；版本：`compiler/judge ba4537fd`。
- 基线：Supplied external implementation；比值口径：`descriptive_quality_failed`。
- 来源：[findings/2026-09-20-010-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/f1bf6497ac0659e2eff8a5ee952d591cd8cd22fe/findings/2026-09-20-010-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-work-assignment-comparison-20260920-125fa0d071ac/stages/verify/comparison-report.json`。
- All 2550 complete snapshots passed. 003 control is the old optimized binary; 022 control is the original one-row starter. Width one separates compiler-floor movement. Other widths retain the authored operation graph, accesses, loops and grid. Each edge keeps its own CV decision; the full report may fail quality. No new profiler, model E2E, official leaderboard or promotion claim.

### nvidia-003-whole-row-structure-optimized_vs_starter-20260921

**B300 / whole-row structure · 003_fused_add_rmsnorm_h7168** — 2026-09-21 / Correct; first_arm_faster

- Workload：`R=64, H=7168, BF16; masked whole row; 16 execution groups`；目标：`sm_103a`；版本：`compiler/judge ba4537fd`。
- 基线：Frozen sliced 16-group candidate；比值口径：`paired`。
- 来源：[findings/2026-09-20-010-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/5190e4f6f6f9d661da8b50e44d0f5cf306638982/findings/2026-09-20-010-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-whole-row-comparison-20260921-66e2f67c5ab3/stages/verify/comparison-report.json`。
- All 2550 snapshots passed. Each timing edge keeps its own quality decision; failed ratios are not qualified. Controls are the specified frozen intermediate binaries; the original starter is preserved separately. Same oracle, cold L2 and continuous paired allocation. No cross-run ratio multiplication, new profiler, model E2E or promotion claim.

### nvidia-003-whole-row-structure-optimized_vs_external-20260921

**B300 / whole-row structure · 003_fused_add_rmsnorm_h7168** — 2026-09-21 / Correct; second_arm_faster

- Workload：`R=64, H=7168, BF16; masked whole row; 16 execution groups`；目标：`sm_103a`；版本：`compiler/judge ba4537fd`。
- 基线：Derived external (Graph host-argument correction)；比值口径：`paired`。
- 来源：[findings/2026-09-20-010-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/5190e4f6f6f9d661da8b50e44d0f5cf306638982/findings/2026-09-20-010-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-whole-row-comparison-20260921-66e2f67c5ab3/stages/verify/comparison-report.json`。
- All 2550 snapshots passed. Each timing edge keeps its own quality decision; failed ratios are not qualified. Controls are the specified frozen intermediate binaries; the original starter is preserved separately. Same oracle, cold L2 and continuous paired allocation. No cross-run ratio multiplication, new profiler, model E2E or promotion claim.

### nvidia-003-whole-row-alignment-optimized_vs_starter-20260921

**B300 / whole-row alignment · 003_fused_add_rmsnorm_h7168** — 2026-09-21 / Correct; measurement_quality_failed

- Workload：`R=64, H=7168, BF16; masked whole row; 16 execution groups`；目标：`sm_103a`；版本：`compiler/judge ba4537fd`。
- 基线：Frozen generic whole-row candidate；比值口径：`descriptive_quality_failed`。
- 来源：[findings/2026-09-20-010-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/5190e4f6f6f9d661da8b50e44d0f5cf306638982/findings/2026-09-20-010-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-whole-row-alignment-comparison-20260921-4a7ffebc7903/stages/verify/comparison-report.json`。
- All 2550 snapshots passed. Each timing edge keeps its own quality decision; failed ratios are not qualified. Controls are the specified frozen intermediate binaries; the original starter is preserved separately. Same oracle, cold L2 and continuous paired allocation. No cross-run ratio multiplication, new profiler, model E2E or promotion claim.

### nvidia-003-whole-row-alignment-optimized_vs_external-20260921

**B300 / whole-row alignment · 003_fused_add_rmsnorm_h7168** — 2026-09-21 / Correct; measurement_quality_failed

- Workload：`R=64, H=7168, BF16; masked whole row; 16 execution groups`；目标：`sm_103a`；版本：`compiler/judge ba4537fd`。
- 基线：Derived external (Graph host-argument correction)；比值口径：`descriptive_quality_failed`。
- 来源：[findings/2026-09-20-010-rewrite-external-performance-gap.json](https://github.com/qhy991/open-cake-ir/blob/5190e4f6f6f9d661da8b50e44d0f5cf306638982/findings/2026-09-20-010-rewrite-external-performance-gap.json)。
- 原记录 / 实现定位：`B300-M3:/mnt/b300-shared/home/qinhaiyan/workspace/aka-gpu-infra-b300-m3-20260908/state/runs/nvidia-whole-row-alignment-comparison-20260921-4a7ffebc7903/stages/verify/comparison-report.json`。
- All 2550 snapshots passed. Each timing edge keeps its own quality decision; failed ratios are not qualified. Controls are the specified frozen intermediate binaries; the original starter is preserved separately. Same oracle, cold L2 and continuous paired allocation. No cross-run ratio multiplication, new profiler, model E2E or promotion claim.

### metal-result-044

**M4 · channel_absmax_scale** — 2026-09-12 / 有晋升记录

- Workload：`FP32 · R=2；完整 C/协议由原 Workload 固定`；目标：`apple_gpu_family9`；版本：`v74/v104`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[findings/2026-09-12-001-r2-row-reduction-scalarization.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/2026-09-12-001-r2-row-reduction-scalarization.json)。
- 原记录 / 实现定位：`open-cake-ir-experiments:m4-family9/matrix-kimi-k3-high-v104-20260911/channel_absmax_scale; open_cake-1; event 13`。
- 记录中的 generation 0；并非本轮重新查询的当前冠军。channel_absmax 的晋升另见 F-2026-09-12-003。

### metal-result-045

**M4 · bias_gradient_reduction** — 2026-09-13 / 有晋升记录

- Workload：`FP32 · R=2；完整 C/协议由原 Workload 固定`；目标：`apple_gpu_family9`；版本：`v78/v113`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[findings/2026-09-12-001-r2-row-reduction-scalarization.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/2026-09-12-001-r2-row-reduction-scalarization.json)。
- 原记录 / 实现定位：`open-cake-ir-experiments:m4-family9/matrix-kimi-k3-high-v78-v113-incumbent-20260913/bias_gradient_reduction; open_cake-1; event 13`。
- 记录中的 generation 0；并非本轮重新查询的当前冠军。channel_absmax 的晋升另见 F-2026-09-12-003。

### metal-result-046

**M1 Pro · gemm_silu** — 2026-09-11 / 历史确认

- Workload：`FP32 · M=128, K=256, N=32`；目标：`apple_gpu_family7`；版本：`v73/v98`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[findings/2026-09-11-004-metal-output-column-specialization.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/2026-09-11-004-metal-output-column-specialization.json)。
- 原记录 / 实现定位：`open-cake-ir-experiments:m1pro-family7/gemm-silu-glm53-high-20260910n32; confirmatory event 37; candidate 450d8350`。
- 固定小形状相对朴素基线；后继合格候选为另一种 K-split 机制，未证明跨版本持续加速。

### metal-result-047

**M2 · gemm** — 2026-09-10 / 计时未通过

- Workload：`原记录的三个 GEMM 候选`；目标：`apple_gpu_family8`；版本：`见来源`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[findings/2026-09-10-010-contraction-regime-shows-material-headroom.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/2026-09-10-010-contraction-regime-shows-material-headroom.json)。
- 原记录 / 实现定位：`见来源所引用的原始实验`。
- 搜索曾观察 6–7×，均未通过测量稳定性门；不列入性能图。

### dcu-result-048

**BW1101 · absmax_rescale** — 2026-09-18 / 测量分辨能力待查

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@0970a36e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/absmax_rescale-20260918-002242/campaign-evidence/objects/sha256/93/936ba860b59b2716f843de288526fa8659f6915fdb121d537108108d9d0cca66`。
- 每任务首个合格运行，不是 fastest-of-all。两者均读到 5.439 µs；不写成 1× 性能持平。

### dcu-result-049

**BW1101 · adadelta** — 2026-09-18 / 未检出显著差异

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@0970a36e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/adadelta-20260918-015416/campaign-evidence/objects/sha256/5a/5a8ca7e823848cd2623dde3178f95d9da3e7198fdb68d6433107a40debeab742`。
- 每任务首个合格运行，不是 fastest-of-all。比值来自确认中位数；是否超过提升门槛沿用原 classification。

### dcu-result-050

**BW1101 · adamw** — 2026-09-18 / 未检出显著差异

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@5151954e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/adamw-20260917-195709/campaign-evidence/objects/sha256/de/de177a4c4a0ae0866d2a63ab2117d09142ebf2f57f43287a9bd9e1bae2eef9e4`。
- 每任务首个合格运行，不是 fastest-of-all。比值来自确认中位数；是否超过提升门槛沿用原 classification。

### dcu-result-051

**BW1101 · attention_decode** — 2026-09-18 / 确认变慢

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@4278caf2`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/attention_decode-20260918-070030/campaign-evidence/objects/sha256/40/4098e094b242916b4edcadb7f8cf096e3bbbed5d765673eb35a02667f0b4bb0a`。
- 每任务首个合格运行，不是 fastest-of-all。比值来自确认中位数；是否超过提升门槛沿用原 classification。

### dcu-result-052

**BW1101 · bias_gradient_reduction** — 2026-09-18 / 未检出显著差异

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@0970a36e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/bias_gradient_reduction-20260918-011329/campaign-evidence/objects/sha256/60/60268d83cf58bc39f68d44f68c6cd850331fd9ee7f11418c1fa212b45d9e1121`。
- 每任务首个合格运行，不是 fastest-of-all。比值来自确认中位数；是否超过提升门槛沿用原 classification。

### dcu-result-053

**BW1101 · channel_absmax_scale** — 2026-09-18 / 未检出显著差异

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@4278caf2`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/channel_absmax_scale-20260918-074053/campaign-evidence/objects/sha256/39/39e552743464224b12d914af1d2a716bfb5ac468d4d763248be0e961b26f7942`。
- 每任务首个合格运行，不是 fastest-of-all。比值来自确认中位数；是否超过提升门槛沿用原 classification。

### dcu-result-054

**BW1101 · cosine_similarity** — 2026-09-18 / 测量分辨能力待查

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@4278caf2`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/cosine_similarity-20260918-052136/campaign-evidence/objects/sha256/73/73a5efe324e6974474d66691f18458010d973221f007a12c076f3a40b0afda51`。
- 每任务首个合格运行，不是 fastest-of-all。两者均读到 5.439 µs；不写成 1× 性能持平。

### dcu-result-055

**BW1101 · gelu_tanh** — 2026-09-18 / 未检出显著差异

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@0970a36e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/gelu_tanh-20260917-224115/campaign-evidence/objects/sha256/87/87033d2251a648103290097aeea51f78b07166b7185804835eb3be2a557ef252`。
- 每任务首个合格运行，不是 fastest-of-all。比值来自确认中位数；是否超过提升门槛沿用原 classification。

### dcu-result-056

**BW1101 · gelu_tanh_backward** — 2026-09-18 / 确认变慢

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@0970a36e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/gelu_tanh_backward-20260917-213519/campaign-evidence/objects/sha256/fd/fddc68b73a893a3135d7cbab7d78752217011d5fdd9f4f039169fc975526b901`。
- 每任务首个合格运行，不是 fastest-of-all。比值来自确认中位数；是否超过提升门槛沿用原 classification。

### dcu-result-057

**BW1101 · gemm** — 2026-09-18 / 确认变慢

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@0970a36e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/gemm-20260917-225918/campaign-evidence/objects/sha256/29/293418506b77308dd35947557b6bc7899846a2714bba71af1c2a2f83aca12aaa`。
- 每任务首个合格运行，不是 fastest-of-all。比值来自确认中位数；是否超过提升门槛沿用原 classification。

### dcu-result-058

**BW1101 · layernorm** — 2026-09-18 / 测量分辨能力待查

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@0970a36e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/layernorm-20260917-222113/campaign-evidence/objects/sha256/52/529ff05598a641f3a4719abbdc9146f0848e5d3d7e1830eea33275653a6373cb`。
- 每任务首个合格运行，不是 fastest-of-all。两者均读到 5.439 µs；不写成 1× 性能持平。

### dcu-result-059

**BW1101 · layernorm_backward_input** — 2026-09-18 / 测量分辨能力待查

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@0970a36e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/layernorm_backward_input-20260918-005411/campaign-evidence/objects/sha256/a9/a965a96d2f1fed63a12cf28e045845bb53e16c884103cb9b0ec0281966314aad`。
- 每任务首个合格运行，不是 fastest-of-all。两者均读到 5.439 µs；不写成 1× 性能持平。

### dcu-result-060

**BW1101 · layernorm_gamma_beta_backward** — 2026-09-18 / 未检出显著差异

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@4278caf2`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/layernorm_gamma_beta_backward-20260918-062624/campaign-evidence/objects/sha256/9c/9ce5553d9c030a548d4b0d7759be7d83bbe67275edfc94b7138f20df567b2226`。
- 每任务首个合格运行，不是 fastest-of-all。比值来自确认中位数；是否超过提升门槛沿用原 classification。

### dcu-result-061

**BW1101 · momentum_sgd** — 2026-09-18 / 确认加速

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@0970a36e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/momentum_sgd-20260918-022953/campaign-evidence/objects/sha256/cb/cb1ad30f1dc8eaf56071d45203d0c57dbea7c6b8c695e75846c3e8e2d496526c`。
- 每任务首个合格运行，不是 fastest-of-all。比值来自确认中位数；是否超过提升门槛沿用原 classification。

### dcu-result-062

**BW1101 · pairwise_sqdist** — 2026-09-18 / 确认加速

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@4278caf2`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/pairwise_sqdist-20260918-045643/campaign-evidence/objects/sha256/26/264e9c4a30b4183bf6494aff778434daee991b233cd4918a748ccbaabf18d1c9`。
- 每任务首个合格运行，不是 fastest-of-all。比值来自确认中位数；是否超过提升门槛沿用原 classification。

### dcu-result-063

**BW1101 · per_channel_moments** — 2026-09-18 / 确认加速

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@4278caf2`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/per_channel_moments-20260918-060857/campaign-evidence/objects/sha256/12/122489d13e79061e068d709b428a80e7ac039f7efeaa5a2c9242dfac13c133ef`。
- 每任务首个合格运行，不是 fastest-of-all。比值来自确认中位数；是否超过提升门槛沿用原 classification。

### dcu-result-064

**BW1101 · prelu** — 2026-09-18 / 确认变慢

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@0970a36e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/prelu-20260917-232247/campaign-evidence/objects/sha256/10/10d7db433322f8dfa54c0d66aa1d81a207832ae8092358be9319ed2320a58404`。
- 每任务首个合格运行，不是 fastest-of-all。比值来自确认中位数；是否超过提升门槛沿用原 classification。

### dcu-result-065

**BW1101 · residual_rmsnorm** — 2026-09-18 / 测量分辨能力待查

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@4278caf2`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/residual_rmsnorm-20260918-053055/campaign-evidence/objects/sha256/5e/5e0889655e4306927b8f750728613fb933806170899e8220a0cf0715f975d506`。
- 每任务首个合格运行，不是 fastest-of-all。两者均读到 5.439 µs；不写成 1× 性能持平。

### dcu-result-066

**BW1101 · rmsnorm** — 2026-09-18 / 测量分辨能力待查

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@5151954e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/rmsnorm-20260917-174616/campaign-evidence/objects/sha256/2b/2b07b39b143778658c71feb364c710b0b460647a5d2970d5dbd5745e34f66818`。
- 每任务首个合格运行，不是 fastest-of-all。两者均读到 5.439 µs；不写成 1× 性能持平。

### dcu-result-067

**BW1101 · rmsnorm_input_gradient** — 2026-09-18 / 测量分辨能力待查

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@5151954e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/rmsnorm_input_gradient-20260917-192819/campaign-evidence/objects/sha256/78/78b89806977d0387aadcff006ae95a538254240f20750322f8616bbea774d8ed`。
- 每任务首个合格运行，不是 fastest-of-all。两者均读到 5.439 µs；不写成 1× 性能持平。

### dcu-result-068

**BW1101 · selu** — 2026-09-18 / 测量分辨能力待查

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@0970a36e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/selu-20260917-234230/campaign-evidence/objects/sha256/74/74e244f32261efc0b57817644eccc5f3b2f89d8674de7be6f03f86993e2be98f`。
- 每任务首个合格运行，不是 fastest-of-all。两者均读到 5.439 µs；不写成 1× 性能持平。

### dcu-result-069

**BW1101 · silu** — 2026-09-18 / 测量分辨能力待查

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@5151954e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/silu-20260917-184123/campaign-evidence/objects/sha256/25/25a10f330efbf8bd169e8ad1415198b9108b4ba5b33586fff3bdf790ca5fbd91`。
- 每任务首个合格运行，不是 fastest-of-all。两者均读到 5.439 µs；不写成 1× 性能持平。

### dcu-result-070

**BW1101 · softmax** — 2026-09-18 / 测量分辨能力待查

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@4278caf2`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/softmax-20260918-073121/campaign-evidence/objects/sha256/09/09a45892d4a2d420ebd999610cc41cc9df9c973e4ffc0423646bee19fd72109a`。
- 每任务首个合格运行，不是 fastest-of-all。两者均读到 5.439 µs；不写成 1× 性能持平。

### dcu-result-071

**BW1101 · softmax_backward** — 2026-09-18 / 测量分辨能力待查

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@5151954e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/softmax_backward-20260917-191303/campaign-evidence/objects/sha256/68/68dbdcaba2946f42d631f862943f467f2e1eb9b1732ed04447eeaec9103fd425`。
- 每任务首个合格运行，不是 fastest-of-all。两者均读到 5.439 µs；不写成 1× 性能持平。

### dcu-result-072

**BW1101 · softplus_gradient** — 2026-09-18 / 测量分辨能力待查

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@0970a36e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/softplus_gradient-20260917-235820/campaign-evidence/objects/sha256/84/846fbbfc6d668811b686158d772ef05a2cbfc5b2855fbad8cd8a8955dd831e57`。
- 每任务首个合格运行，不是 fastest-of-all。两者均读到 5.439 µs；不写成 1× 性能持平。

### dcu-result-073

**BW1101 · softsign** — 2026-09-18 / 测量分辨能力待查

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@4278caf2`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/softsign-20260918-054910/campaign-evidence/objects/sha256/4a/4a332cf3b614046addbcea3e8258cde774b690d195112db039a687e5d8d5f913`。
- 每任务首个合格运行，不是 fastest-of-all。两者均读到 5.439 µs；不写成 1× 性能持平。

### dcu-result-074

**BW1101 · swiglu** — 2026-09-18 / 未检出显著差异

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@5151954e`。
- 基线：任务固定初始基线；比值口径：`ratio_of_medians`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`bw1100:/home/testuser01/oci-dcu-runs/swiglu-20260917-185702/campaign-evidence/objects/sha256/54/5491c5924ad5437ee20ec11f63cf3d88452252db9e0c8990b469c1e437889045`。
- 每任务首个合格运行，不是 fastest-of-all。比值来自确认中位数；是否超过提升门槛沿用原 classification。

### dcu-result-075

**BW1101 · gemm_bias** — 2026-09-18 / 基线未通过

- Workload：`保留 sweep 的验证形状`；目标：`gfx938`；版本：`见来源`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/dcu-gfx938-results.md](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/dcu-gfx938-results.md)。
- 原记录 / 实现定位：`见来源所引用的原始实验`。
- 本集合无合格终点；保留在任务覆盖分母中。

### dcu-result-076

**BW1101 · gemm_silu** — 2026-09-18 / 候选未通过

- Workload：`保留 sweep 的验证形状`；目标：`gfx938`；版本：`见来源`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[docs/dcu-gfx938-results.md](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/docs/dcu-gfx938-results.md)。
- 原记录 / 实现定位：`见来源所引用的原始实验`。
- 本集合无合格终点；保留在任务覆盖分母中。

### dcu-result-077

**BW1101 · 后续重复 · gelu_tanh** — 2026-09-18 / 保留的后续运行

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@a8a4885d`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`gelu_tanh-20260918-104531/campaign-evidence/objects/sha256/94/94f0cc691770709577e02094c39a51b1dd6aea7abf62d53bf818f9d6aa546d02`。
- 按原汇总的首个合格选择规则未替换主行；不作为新的最佳实现。gelu_tanh 后继落在待查的 5.439 µs 读数。

### dcu-result-078

**BW1101 · 后续重复 · gemm** — 2026-09-18 / 保留的后续运行

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@4278caf2`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`gemm-20260918-042832/campaign-evidence/objects/sha256/59/59cf80cd04c6e9a6d4f4d3f11f2a5a037566b4174a5627c6b21305a207c1483a`。
- 按原汇总的首个合格选择规则未替换主行；不作为新的最佳实现。gelu_tanh 后继落在待查的 5.439 µs 读数。

### dcu-result-079

**BW1101 · 后续重复 · rmsnorm** — 2026-09-18 / 保留的后续运行

- Workload：`原 Campaign 的固定形状（汇总未转录尺寸）`；目标：`gfx938`；版本：`open-cake-ir@5151954e`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[findings/data/2026-09-18-dcu-confirmatory-medians.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/data/2026-09-18-dcu-confirmatory-medians.json)。
- 原记录 / 实现定位：`rmsnorm-20260917-181121/campaign-evidence/objects/sha256/6b/6b31786a9cc813648dc1e386c47bd25656cd536e93dab084b1554a22dde3f9d7`。
- 按原汇总的首个合格选择规则未替换主行；不作为新的最佳实现。gelu_tanh 后继落在待查的 5.439 µs 读数。

### amd-result-080

**Radeon / Strix Halo · rmsnorm smoke** — 2026-09-17 / 计时边界未解决

- Workload：`gfx1151-rmsnorm-b8-smoke`；目标：`gfx1151`；版本：`见来源`。
- 基线：任务固定初始基线；比值口径：`paired`。
- 来源：[findings/2026-09-17-002-two-device-timers-disagree-on-gfx1151.json](https://github.com/qhy991/open-cake-ir/blob/f95af8005619a9356d5891873b38a9322d32f102/findings/2026-09-17-002-two-device-timers-disagree-on-gfx1151.json)。
- 原记录 / 实现定位：`infplane；原调查未产生 Campaign`。
- 31.858 与 24.224 µs 来自不同计时器，不作为合格延迟或加速比。
