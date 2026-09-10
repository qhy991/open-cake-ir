# OMOE 经验的首批迁移

本入口把 OMOE 的三组经验准备成 Lab 作者材料，并提供一个保留 BF16 舍入和两个
输出的 add-RMSNorm 任务。它只做合同生成、静态验证和 lowering，不启动 provider
或 GPU，也不把原项目的 WIN 继承为新任务的结果。

## 生成材料

在本 checkout 使用 Python >= 3.10；若默认 python3 不符合项目要求，指定已有的
受支持解释器。输出目录必须不存在，且位于所有 Git checkout 之外。

```bash
python3 tools/prepare_omoe_transfer.py \
  --omoe-root /Users/haiyan-infiniai/omoe \
  --omoe-ref 19356783c9c58f49a603472c3ecfd9db92bb608f \
  --task add_rmsnorm_bf16 --backend triton-b200 --rows 128 --columns 2560 \
  --representation steps \
  --output /Users/haiyan-infiniai/.local/share/open-cake-ir/omoe-addnorm-steps
```

`case` 保留六条完整原始 recipe。`steps` 把原始前提、失败条件和迁移修正放在
机制/步骤之前，保留正文及证据。两者都是从同一个 Git commit 读取，忽略 OMOE
工作区的未提交改动。它们是工程材料的两种组织方式，尚不是语义/长度配平的
Case-vs-Skill 因果实验。

| 组 | 正例 / 条件例 | 反例 / 限制 |
|---|---|---|
| fusion | gemm-swiglu-epilogue-fusion | geglu-frontend-fusion-sm87-compute-floor |
| residency | hybrid-shared-register-state-residency | register-budget-requires-occupancy-limiter-match |
| rounding | bitexact-triton-needs-scalar-lowering-sm100 | autoregressive-rollout-compounds-fusion-drift |

这些组不是适用于每一个任务的推荐集合。作者应根据任务检查每条经验的条件，
明确不适用的原因；本入口不提供新的自动检索、路由或隐藏编译 Pass。

输出包含 `workload.json`、`candidate.py`、`lowered.py`、`scaffold.md`、
`transfer.json` 和 `authoring-references.json`。最后一份文件提供现有 Lab 的
`workload` 与 `arm` 引用字段，包含所需的路径和身份绑定；不是第二个 CampaignLock。
将它的 `workload` 和 `arm` 字段合入相应 Study 的已声明位置，再由现有
`lab preflight` 冻结实际执行。源文件、工具输出和身份在该边界必须保持一致。

材料含低层实现细节，只允许 `known_kernel_reproduction`。不能给 clean-start
作者使用，也不能在比较 Cake 与原生实现时只给一臂经验、再把差异归因于 IR。
正式实验仍需完整的 target、Executor、provider、baseline、预算和共同评测准入。

## add-RMSNorm 的语义

输入 `delta[R,C]`、`residual[R,C]`、`weight[C]` 都是 BF16。

1. 在 FP32 中相加，再按 ties-to-even 舍入到 BF16，得到 `z`。
2. 输出 `residual_out = z`，检查 BF16 字节相等，包括带符号零。
3. 用同一个已舍入的 `z` 计算 FP32 RMSNorm，epsilon=1e-6；输出再舍入到 BF16。
4. `out` 使用预声明的数值容差，不能把该容差用于 `residual_out`。

五组输入覆盖一般分布、带符号零、BF16 舍入中点、抵消及混合幅值。
输入保持不变，两个输出都是新分配且不别名。这里提取的是 OMOE 算子的值语义；
OMOE 的原地 buffer 更新需要在将来的模型接入层另行验证。

目标只允许 B200/B300 的 Triton 路线，分别生成独立 Workload。C 可以不是 2 的幂：
例如 C=2560 对应 4096 宽的 masked tile，padding 为零，均值仍除以 2560。
C=1 使用至少两个 lane 的 masked tile，避免把向量 reduction 变成标量操作。
当前限定 1<=C<=16384；这是任务的明确范围，不是新硬件规律。

Compiler 仍使用现有 load/cast/add/reduce/rsqrt/store。新增的 `per_output` 比较
由共同 Evaluation 比较函数消费，任务严格校验每一个输出规则。
残差不正确、缺少一个输出、输入被改动、容差被放宽都应失败。

## GEMM epilogue 衔接与后续任务

准备工具使用 `tasks.workloads.create_task` 作为唯一任务选择入口，没有复制在建的
contraction 实现。`gemm_silu` 在该任务正式集成后可直接使用同一工具：
`--task gemm_silu --rows 128 --depth 128 --columns 64`。它没有注册时会拒绝，
不会悄悄换成其他 GEMM。当前已有 `gemm_bias` 可用于相同接口的非融合控制。

SiLU 只消费一个累加器；SwiGLU 需要成对 gate/up 累加器和相应写回安排。因此
GEMM+SiLU 材料不证明完整 Tensor Core SwiGLU fusion 已实现。

下一步依次验证：融合收益与 GEMM 代价、tile 缩小与延迟隐藏的反例、有序递推的
状态驻留。经验若暴露了具体 Compiler 能力缺口，使用 `findings/` 记录实际
Revision/target 下的探测，再走 successor cycle。性能反例不能变成正确性硬拒绝。

## 发布与验证边界

本变化涉及任务与 Evaluation，是 Executor successor 的输入；Compiler 源集合
无需变化。旧 Executor 描述和旧 Workload 不得重写。新任务的静态通过、CPU
语义检查和生成源码不等于 GPU 编译、正确性、性能或 OMOE 端到端资格。

GPU 上仍需验证真实路径、所有输入分布、两种输出规则、冷 L2 的配对 CUPTI 计时
及 profiler。新算子最终回接 OMOE 时，以该模型的既有 gate 和端到端测量决定收益。

## English

The preparer exports source-bound case/step material for known-kernel reproduction and a
BF16 add-RMSNorm tensor task. The residual is rounded before normalization and checked
exactly, while the normalized output has its own tolerance. Inputs remain unchanged;
model-level in-place integration is outside this contract. Non-power-of-two widths use
zero-masked tiles without changing the true shape. No provider/GPU run or performance
qualification is implied. Existing task registration and Lab reference bindings remain
the only integration paths; the in-progress GEMM epilogue task is not duplicated here.
