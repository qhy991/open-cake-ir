# B300 KDA prefill：Cake IR 与 native CUDA lowering 的设计及证据

本页记录从 KDA prefill 性能差距导出的 **IR 语义、Verifier 合同、native CUDA/PTX 映射**。它是设计和证据索引，不替代 [Workload Contract](../contracts/workloads/cake-kda-prefill-b300-v1.json)、[Finding F-2026-09-24-003](../findings/2026-09-24-003-kda-prefill-b300-chunk-state-lowering.json) 或 [技术报告](open-cake-ir-technical-report.pdf)。下列 K32/K128 数值试验都是两 chunk 的合成 Target 原型；完整 KDA 的输出、全部六形状、框架验收及超过 CAKE 的性能仍未实现。

## 1. 问题和比较边界

Workload 的状态是 BF16、V-first 的 `[sequence, head, V128, K128]`。每个 token 先衰减旧状态、用 K 做预测，再以 beta 缩放残差更新状态并 **逐 token 舍入到 BF16**，最后用 Q 从更新后的状态产生输出。完整验收比较所有输出和最终原位状态元素，固定与 packed 序列均须覆盖；B300 计时指定 CUPTI、每样本冷 L2、计时前恢复初始状态。只比较某个投影或最后状态，不能宣称完成该 Workload。

原始 CAKE 的 B200 导出实现仅改目标检查及编译架构后，成为这里的 B300 适配参考；它不是原论文的 B300 结果。固定 H64/T8192 的完整参考在同机配对作业中约 **456.260 µs**。可见的参考发射使用 32-token 分块、TMA、`tcgen05`、TMEM 状态以及 producer/compute/epilogue 流水。当前最快的另一路完整 H64 固定输入候选为 **1,444.107 µs**，同作业比值 **3.165× 慢**；其两-token 入口状态近似在 T257 常规输入只有 19/20 组通过，并在高保留输入失败，不能推广为正确的六形状实现。

| 路线 | 已测事实 | 从中得到的设计要求 |
| --- | --- | --- |
| 现有 Triton chunk32/V64 | 96,080.520 µs；每线程静态 176 字节 stack | 大 live tile 映射的寄存器/本地存储是风险，需要让 TMEM 和转移相位成为可检查的承诺。 |
| 缩小后的 Triton chunk16/V32 | 83,264.993 µs；静态 stack 与 local SASS 消失 | 消除静态 stack 信号并未解决串行块遍历与重复工作；两次实验还同时改变了 tile，不能把差值全归因于 spill。 |
| 直接逐 token native CUDA | 16,979.920 µs；正确的固定 H64 结果 | CUDA 语言能够实现语义，但每 token 重读写大状态过慢，必须减少状态搬运。 |
| 两次 MMA 的独立预处理 | 389.507 µs 组件中位数；七个中间输出 | 它单独已接近完整参考的 456 µs；继续物化全部中间值并另启状态 kernel，缺少可信的超过 CAKE 的余量。 |

这里的“现有 CUDA/Triton 路径无法实现”特指 **仓库当时的后端准入和映射**，并非 CUDA 或 Triton 语言不具备这些指令。原始 CAKE 的 CUDA/PTX 已证明硬件路线存在；新增 lowering 的任务是用 Cake IR 表达同样关键的状态、指令、地址和同步合同，并寻找可测的进一步收益。Triton 可写自定义 asm，但当前 `triton` 后端没有这套 TMEM 状态生命周期及相位证明；直接把 backend 名字改为 Triton 不会创建它们。

## 2. 从算子语义到设备指令

![KDA B300 有界状态 lowering 数据流](figures/kda-b300-native-lowering.svg)

可编辑源图是 [kda-b300-native-lowering.mmd](figures/kda-b300-native-lowering.mmd)。图中准备阶段到状态 kernel 的连线表示 **当前物化的输入接口**，不表示已经融合成一个生产 kernel；图中也没有完整 KDA 输出路径。

### 2.1 严格下三角前代入，而非展开矩阵逆

`OperationKind.FORWARD_SUBSTITUTE` 的一个规范语义是 BF16 `P[C,C]`、FP32 `RHS[M,C]` 到 FP32 `U[M,C]`；对每行、每个 token `t`，从 `RHS[t]` 开始，按 `i=0..t-1` 次序做 `float(P[t,i]) * U[i]` 的 FP32 FMA，跳过对角线及上三角。C32 每行有 496 项 FMA，后续 BF16 表示由独立 `cast` 承担。这一语义归 [IR](../src/open_cake_ir/compiler/ir/operations.py)、[类型验证](../src/open_cake_ir/compiler/verifier/data_consistency.py)及 [native emission](../src/open_cake_ir/compiler/backends/native_cuda.py) 各自负责。

选择前代入的原因是 state-dependent RHS 无法提前逆掉；原先五步逆矩阵预处理需要 21 个 BF16 MMA，B300 上该准备组件为 637.606 µs，而独立两 MMA + P 准备组件为 389.507 µs。这是 **不同作业、不同计算范围** 的诊断，不能直接当作端到端加速比。native root 发射用一个 copy warp 将 P 放入 2 KiB、64B swizzle 的 shared memory，以一个 `mbarrier` 交给四个各自负责 128 行中一部分的 compute warp，后者发射具名 `__fmaf_rn` 更新。具名值替换临时数组后，root AOT 从 256 字节 stack 降为 0，使用 119 寄存器；这只是 root 子问题的资源证据。

### 2.2 BF16 TMEM 状态、两次收缩与相位

每个 head/CTA 的有界原型将 `[128,128]` BF16 状态放在 TMEM。四个 compute warp 用 `tcgen05.st.sync.aligned.32x32b.x8.b32` 把两个 BF16 值显式打包为一个 32-bit word；各 warp 等待 store 完成后，由每组一个 leader 向 count=4 的 `state_ready` mbarrier 到达。基础 MMA 的 issuing warp 等待当前 phase，再对 TMEM-A 和 shared-B 发出 `tcgen05.mma`。基础投影读出 FP32 RHS 后，前代入产生 U；U 显式 BF16 舍入并写到另一块 TMEM，再通过独立的 `u_ready` 交给校正 MMA。校正结果有自己的 completion barrier，状态发布必须排在它之后。

`TileLoop.carried_buffers` 只承认一个静态外层循环、一个 BF16 TMEM tile、循环前初始化和循环内一次更新。`analyze_carried_tmem` 证明所有读取发生在更新之前、读者等待相同 barrier、写者角色及相位一致。原始 state 生产者发布 phase 0，第 `trip` 次读取等待 `trip & 1`，该次更新发布下一 phase；无声明的第二写者仍报 `BUFFER_MULTIPLE_WRITERS`。native 的 `NATIVE_TWO_PHASE_ORDER` 另要求基础 MMA → solve → U 发布 → 校正 MMA。这样，一个偶然被其它规则拦住的错误不会冒充真正的同步证明。

### 2.3 K128 的 B 方向和标量块坐标

`tcgen05` 的 K-major shared B 行最多由本路由的 128B swizzle 表达；直接将 BF16 K128 放进 `[N32,K128]` 行需要 256B，原路径以 `NATIVE_SMEM_LAYOUT` 拒绝。B300 PTX 探针和独立设备乘积检查支持精确的 **MN-major B** 子集：物理 B `[K128,N32]`、64B swizzle、M128/N32/K128、每个 K16 atom 的 shared descriptor 前进 1024B。native 后端只在 `sm_103a` 为此几何/方向开放 `_TMEM_A_MN_B_EVIDENCE`，其它组合以 `NATIVE_MN_MAJOR_B_UNQUALIFIED` 拒绝；校正 MMA 则用 K-major B `[N128,K32]`。这些是 Target 及后端自己的硬件词汇，不是共享布局代数。

每个 chunk 的真实 B 输入还有 chunk 轴：`[chunks,K128,N32]` 或 `[chunks,N128,K32]`。原有 `loop_tile` 即使 tile=1 仍产生一个长度为 1 的 **向量** 轴，不能悄悄挤掉后假装得到二维 shared tile。新增 `AccessIndexKind.LOOP` 明确取单元循环当前块号作 **标量** 坐标；Verifier 检查作用域、unit tile、源维度上界和 load 的局部结果形状。native 发射三级 TMA 坐标时，chunk 是第三个标量坐标，其余两个轴匹配 declared shared box。无对应循环/形状的访问被 `ACCESS_LOOP_*`、`LOAD_ACCESS_SHAPE_MISMATCH`、`NATIVE_CARRIED_MMA_DOMAIN` 或 `NATIVE_TMA_COORDINATES` 定位拒绝。

### 2.4 BF16 TMEM 回读与衰减状态合并

完整 chunk 状态转移包含 `old_state * prefix_end + correction`，不能仅用校正 MMA 覆盖旧状态。已有 TMEM load 只准 FP32 累加器；共享 IR 现在也准许 BF16 TMEM tile 到同形同 dtype 的 BF16 register tile。每个 32-bit TMEM word 对应两个相邻 BF16 列，`source_atom` 的重复数按 word 计，列宽须容纳完整重复。native 发射 `tcgen05.ld.sync.aligned.32x32b.x16.b32`，先等待 carried state's 当前 mbarrier phase，再用 `__ushort_as_bfloat16` 原样解包位模式；不把 BF16 数据假设为 FP32 累加器。这个 bounded admission 名为 `NATIVE_TMEM_BF16_STATE`。

合成 K128/C32 两 chunk Schedule 随后将旧状态显式转 FP32，乘 FP32 `prefix_end`，加 FP32 校正结果，最后显式舍入 BF16 并发布下一 phase。它在 `sm_103a` 编译为 255 寄存器、0 stack、0 spill；B300 三组完整状态对照各覆盖 16,384 元素、0 超差，最大绝对误差不超过 `1.1920928955078125e-07`。255 寄存器是紧迫的资源信号，但没有 CUPTI/occupancy 测量可证明它对完整 KDA 延迟的贡献。

## 3. 对照：谁拥有哪个拒绝

| 合同/硬件选择 | 共享 IR/Verifier 的职责 | native CUDA 的职责 | 最小反例与证据 |
| --- | --- | --- | --- |
| `forward_substitute` | 形状、dtype、strict-lower 顺序、舍入边界 | P shared stage、barrier、128 行 FP32 FMA 发射 | `test_forward_substitute_ir.py`、`test_native_forward_substitute.py`；错误 stage/wait 被 `NATIVE_SOLVE_P_STAGE` 等拒绝。 |
| carried BF16 TMEM | 一个 initializer、一个 updater、读先于写、同 barrier phase | `tcgen05.st` 打包、四 warp 到达及循环 phase | `test_tmem_state_contract.py`、`test_native_two_phase_carried.py`；缺 wait/多写者不准入。 |
| 标量 `loop` 地址 | unit tile、作用域、地址范围和局部结果维度 | rank-3 TMA 坐标与两个 staged B 域 | `test_scalar_loop_access.py`、`test_native_two_phase_k128.py`；错误首轴由 `NATIVE_CARRIED_MMA_DOMAIN` 拒绝。 |
| BF16 TMEM 回读 | 同 dtype、32-bit word/双 BF16 packing、carried read-before-update | 当前 phase 等待、word load、位模式解包 | `test_tmem_state_contract.py`、`test_native_bf16_tmem_read.py`；错误 atom/wait/非 carried 源被拒绝。 |
| 旧状态衰减 | FP32 cast/mul/add 与 BF16 cast 保持独立语义 | 行拥有者逐列计算并发布下一 TMEM phase | `test_native_kda_decayed_state.py`；去掉 prefix producer 触发 `BUFFER_UNPRODUCED`。 |

### 一次没有推广的 lowering 尝试

`task/nvidia-kda-fused-state-epilogue` 试图在后端把“BF16→FP32、乘前缀、加校正、转 BF16”四个 **相邻且独占中间值** 的操作映射为一轮逐列计算，仍使用 `__fmul_rn`、`__fadd_rn` 和 `__float2bfloat16_rn`。额外读者存在时，它回到原始逐操作发射，测试固定这个反例。AOT 与未融合版均使用 **255 寄存器、0 stack、0 spill**；`cuobjdump` 均列出 3,376 条 SASS 指令，但发射内容并非逐条相同。这些静态观察没有证明降低资源或提升完整 Workload 性能，**当前 disposition：No promotion**。不能因为源代码看起来更短，就将此特化自动加入 Compiler pass 或声称加速。

## 4. 证据与尚未完成的证明

| 固定源码 / 作业 | 已证明 | 没有证明 |
| --- | --- | --- |
| `2e365222` / `gpuq-f0aeefc31e6b` | K32/C32 两次 MMA + solve，三组每组 4,096 BF16 最终状态全部通过；119 寄存器，0 spill | K128、完整 KDA 输出或延迟 |
| `a81a6948` / `gpuq-2eac9e60295d` | K128/C32 两阶段，三组每组 16,384 最终状态全部通过；230 寄存器，0 spill | 旧状态衰减、beta、查询输出或延迟 |
| `d579e917` / `gpuq-85485a66ccbf` | K128/C32 加 BF16 TMEM 回读和 FP32 prefix 衰减，三组每组 16,384 最终状态全部通过；255 寄存器，0 spill | 逐 token BF16 舍入、高保留输入、完整输出或延迟 |
| `5bd8e2c1` / CPU-only AOT | 独占四操作的后端融合可编译，仍为 255 寄存器、0 spill | 设备数值、动态成本或任何可推广收益 |

上述作业均使用 exact `sm_103a`、broker 分配；设备输出在作业终结、租约释放后由独立 host oracle 比对，输入保持性也经检查。`d579e917` 的 full applicable CPU contracts 为 2,607 passed、5 skipped，另有 1 项本机 Apple MLX 实机测试因缺 `device_info` 接口而未作为 NVIDIA 门禁；Corpus Gate 为 179/179。GPU 程序正确仅覆盖本表对应的合成 Schedule，**不是** Workload Contract 的全形状验收。

## 5. 下一步性能闭环

1. 把准备阶段的 base/query、P、beta、prefix、final-key 接到同一状态/输出路径，明确哪些值留片上、哪些必须物化；核算额外 CTA、TMA 和 global traffic。原先七输出准备的 389.507 µs 是组件成本，不可与完整 CAKE 456 µs 非配对相减后宣称剩余预算。
2. 增加更新后状态的 query 输出和 in-place 最终状态效果；同一外部 oracle 检查 H64/T8192 全部 67,108,864 输出与 1,048,576 状态元素，并覆盖 T65/T257、packed 边界和六个 benchmark 形状。
3. 块级代数省略了逐 token BF16 状态舍入，高保留反例已失败。需要可证明的精确路径或有完整 fallback 的 guard；一组固定输入的通过不能授权 dispatcher。
4. 在完整可启动 Schedule 和 Target admission、固定 Executor 源码、适用门禁之后，同机同作业使用 CUPTI、冷 L2、状态重置及 profiler 做配对测量；静态寄存器数、WorkBound 和单个组件延迟只筛候选，不决定接受。
5. 将复现性语义归共享 core，硬件方向/相位/发射归 NVIDIA 后端，何时采用某种映射归 Lab recipe。只有跨候选反复受益且有反例保护的改写才提升为 pass；当前四操作融合没有这类证据。

这里的原始 CUDA、现有 Triton、native CUDA 是三个不同的证据层级。要声称“追上或超过原始 CAKE”，还需要在相同 B300 Workload、完整输出和状态、同一配对计时合同下测得稳定的胜出，而不是从某一个已通过的 PTX 指令或合成状态原型外推。

## 6. 后续 lowering 的记录单位

每次后续 tick 在相应小节补齐五项：**触发它的失败或性能证据**、**与原始 CAKE CUDA / 既有 native CUDA / 既有 Triton 的具体差别**、**IR 与 Target/后端各自承担的合同及反例**、**固定源码下的 CPU/AOT/设备/计时证据**、**promotion disposition**。尚未测量的硬件假设直接写作假设；一次合成形状的成功不填补完整 Workload 的格子。重复缺口才考虑公共 pass，目标专属的地址和指令位继续归 NVIDIA lowering，实验中的参数选择归 Lab recipe。这样后继优化能沿同一论证和证据链继续，而不是重新发明一个看似相近的 CUDA kernel。

## 7. 硬件依据与源码入口

- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)：`tcgen05.mma` 的 A/B 来源与 major mode、shared descriptor、`tcgen05.st`/`tcgen05.ld` 的形状和同步语义。这里的指令参数还必须经本机 `sm_103a` AOT 与设备试验限定；读到 ISA 说明不自动扩大 Target 资格。
- [CUDA Bfloat16 数据移动 API](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__MISC.html)：`__bfloat16_as_ushort` / `__ushort_as_bfloat16` 是位解释，`__float2bfloat16_rn` 是显式舍入。状态 bit 模式搬运与数值转换不能混用。
- [B300 Target 文档](../compiler/targets/sm_103a.json)、[native CUDA backend](../src/open_cake_ir/compiler/backends/native_cuda.py)、[core carried-state analysis](../src/open_cake_ir/compiler/verifier/carried_tmem.py)、[共享 IR 指南](IR_GUIDE.md)：分别持有硬件事实、发射、合法性和作者可见语义。
