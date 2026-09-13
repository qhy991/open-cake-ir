# 显式 output-column 特化 pass

Compiler 现在提供 `specialize_output_columns`。调用者给出一个 Schedule 并为结果
选择新的 schedule id 和 entry point；pass 检查适用条件后返回列特化的候选
Schedule。`assess()` 和 `lower()` 不会自动调用它。

它对应 F-2026-09-11-004 保留的机制证据：M1 Pro 的 gemm_silu 基线给每个输出行
分配一个 program，把整个 K×N 第二操作数载入私有值并对所有 N 个输出归约 K；
候选增加独立的输出列 program 轴，只载入一个 K 向量 `B[:, col]`，归约到标量后
施加同样的 bias/SiLU 逐点 epilogue，写 `out[row, col]`。这以更多 program 和
跨列访问换取每 program 私有 contraction 状态减少 N 倍。是否更快不由 Compiler
判断：Lab 在原 Schedule 与候选之间做选择和共同评测。

## 匹配的不变量

结构匹配不是算子名识别：行编程的秩 2 contraction（`a[M,K] · b[K,N] → out[M,N]`，
一个 program 一行），输出 N 轴在标量/逐点 epilogue 下独立。变换追加一个列
program 轴，`b` 的访问变为 `[dim0, program:col]`，秩 1 全轴向量载入变为
`[program:col]`，store 变为 `[row, col]`，broadcast 乘法失去 `broadcast_axis`，
`[K,N]` 中间值变为 `[K]`，fold 之后的所有逐列 `[N]` 值变为 `[1]` 标量约定。

全局 ABI 不变：输入、输出与 `metadata` 全部保留，因此原 Workload 契约直接
评测候选（与 epilogue 融合 pass 不同，那里 ABI 改变需要重新绑定）。结果使用
新的 schedule id 和 entry point，输入 Schedule 不被修改。

## 首版支持范围与拒绝

- 仅 Metal 路线：能力证据只存在于 Apple family7/family9 目标，不在其他后端
  声称此变换。
- 单 role、无 state、无别名、无循环/持久调度/同步/流水线/分配。
- 操作只允许 load、elementwise、cast、恰好一个 fold（sum over axis 0）和一个
  最终 store。
- 第二操作数整载入、第一操作数整行载入、输出整行 store、逐列向量整轴载入；
  切片或偏移访问拒绝。
- 跨 N 的第二次 fold（attention/softmax 式耦合）拒绝：列 program 看不到其他
  列，在那里 fold 会改变数值。epilogue 只做逐点算术。

拒绝按序报告具体原因：`input_refused`、`result_identity`、`target_route`、
`unsupported_effects`、`program_shape`、`coupled_output_axis`、
`operation_domain`、`contraction_shape`、`access_domain`、`result_refused`。
返回 `applied=False` 时输入保持不变，不产生半成品候选；接受后对完整结果重新
`assess`。`SpecializationResult.schedule` 每次返回 assessment 字节的独立 JSON
投影，不是第二份可变权威。

## 使用

```python
from open_cake_ir.compiler import Compiler, frontend

compiler = Compiler.load(".", "compiler/revision.lock.json")
schedule = frontend.read_schedule("row_gemm_silu.py").document
result = compiler.specialize_output_columns(
    schedule, schedule_id="gemm_silu_columns", entry_point="gemm_silu_columns")
if result.applied:
    schedule = result.schedule
    source = compiler.lower(result.assessment).source
else:
    print(result.reason, result.message)
```

命令行在 checkout 外生成候选 JSON、lowered source 和结果说明：

```bash
python3 tools/apply_column_specialization.py \
  --schedule row_gemm_silu.py \
  --schedule-id gemm_silu_columns --entry-point gemm_silu_columns \
  --output /Users/haiyan-infiniai/.local/share/open-cake-ir/columns-demo
```

输出目录必须是新的、规范的 checkout 外路径。成功应用退出 0；不适用退出 2。

## 验证与经验边界

合同检查镜像机制证据（M1 Pro 合格候选 450d8350 的 program/access 结构），
包含 GEMM 与 GEMM-SiLU 正例、逐列 CPU 数值等价（随机、零和对抗输入，借用
Metal 合同的 SIMD 适配器）、耦合/切片/非 Metal/已列编程/多 store 的合法但不
适用反例（每个先证明原 Schedule 本身通过检查，再确认是本 pass 的适用条件
拒绝），以及 python 级 epilogue 重结合反例说明耦合拒绝的原因。

注意借用阻塞的事实：state、多 stage、多 role、scan 等构造在 Metal 路线上先被
verifier 拒绝，无法作为“合法但不适用”独立复现；pass 的效果检查仍防御性拒绝
它们，但该拒绝借用的是其他规则。CPU 检查不建立 GPU 正确性、timing 或性能
资格。能力验证仍需 M1 Pro 与 M4 的匹配 campaign（分别等待 family7 Executor
后继与 M4 SDK 修复），报告基线移动与候选减基线两个事实。

## English

`Compiler.specialize_output_columns` rewrites one row-programmed rank-2 Metal
contraction with an independent output-column axis into per-column programs,
preserving the global ABI and Workload metadata while reducing per-program private
contraction state by the column extent (F-2026-09-11-004). It refuses coupled
cross-column folds, sliced or offset access, non-Metal routes and effectful
schedules with named reasons, never mutates the input, and re-assesses the complete
candidate. The pass claims no speedup; Lab selection chooses between the schedules.
CPU contract checks establish no GPU, timing or performance qualification.
