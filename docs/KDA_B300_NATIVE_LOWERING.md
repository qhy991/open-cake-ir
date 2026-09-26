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

可编辑源图是 [kda-b300-native-lowering.mmd](figures/kda-b300-native-lowering.mmd)。图中准备阶段到状态 kernel 的连线表示 **当前物化的输入接口**，不表示已经融合成一个生产 kernel；输出与状态路径只对应 H64、两或 256 chunk 的合成准备值原型。

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

### 2.5 两组投影、两组校正与延迟 TMEM 回读

有界后继 `3ce392dc` 将基础 pipeline 增为两次 K128 投影：`state @ base_key` 供 solve，`state @ base_query` 供输出；U 发布后的 pipeline 也增为两次 K32 收缩：`U @ final_key` 更新状态，`U @ output_coupling` 修正输出。每条 TMA load 仍有自己的 shared B view，两条 load 共用 count=2 的 ready barrier；每个 MMA 有独立 completion barrier，输出路径显式 FP32 相加、scale 与 BF16 舍入。所需 TMEM view 在 512-column 分配内互不重叠。此前 `NATIVE_TWO_PHASE_ORDER` 只准每条 pipeline 一次 MMA，这个后继将上限有界扩成两次，并要求首组都读 carried state、第二组都读 BF16 U；错误的 tensor-A owner 有专门反例。

首个四 MMA 排列过早将 base-query FP32 `[V128,C32]` 从 TMEM 读到寄存器，穿过全部 496 项/行的顺序 solve 才用于输出。`sm_103a` AOT 使用 255 寄存器、112 字节 stack，报告 108 字节 spill 读写；SASS 的 `STL` 出现在 `subtract_base` 源码附近。后继 `fbafa719` 只把这个 query 回读移到输出合并前，保留同一 TMEM accumulator、输出公式与 BF16 边界，AOT 仍用 255 寄存器但 **0 stack、0 spill**。这给出一个明确的 lowering 原因：跨顺序求解保留早期 FP32 结果会制造寄存器活跃区间；延迟具有独立 completion barrier 的 TMEM readout 可缩短该区间。在完整适用 CPU 合同与 Corpus Gate 通过后，`fbafa719` 的 broker-shared B300 抓取完成了 3 组检查：每组 8,192 个 BF16 chunk 输出和 16,384 个 BF16 最终状态均为 0 超差，输入未改写，作业终结后才运行 host oracle。这证明有界合成 RHS 的输出/状态发射正确；尚无配对延迟或公共 pass 收益证据。合成 RHS 仍未接入公开 V/beta，因此不能当作完整 KDA 输出。

公开输出的物理方向仍是独立缺口：当前 CTA 按 V 行拥有寄存器，合成输出是 `[chunk,V128,token32]`，而 Workload 以 token 为前轴。`59855c8b` 的合同探针在共享 IR 中显式加入 `transpose` 与 token-major store，通过通用 Verifier 和 Target 检查；当前 native CUDA 后端以 `BACKEND_OPERATION_UNEMITTABLE` 拒绝。不能在 CUDA store 地址里私下转置、却让 Schedule 继续声称旧形状。下一步可以为严格限定的 transpose→store 链实现正确的行线程写回，也可以由另一阶段显式承担转换，但必须同一 Workload 测量额外流量与启动成本。原始 V 输入的 `[token,V]` 到行拥有者 `[V,token]` 也需同等显式映射，尚未完成后端准入。

### 2.6 显式 transpose 后的 token-major 直接写回

`59855c8b` 首先证明共享 IR 的 BF16 `[V128,C32] → [C32,V128]` `transpose` 与对应 global `store` 可通过 Verifier/Target，而原 native emitter 报 `BACKEND_OPERATION_UNEMITTABLE`。`42b78a5d` 仅准入一条紧邻且独占的 transpose→store 链：transpose 在 IR 中保留语义，但物理数据仍由 128 个 V 行线程持有；store 在每个 token 列迭代时，把相邻线程的 V 行值写到连续的 `[chunk,token,V]` 地址。没有额外全局中间张量，也没有让作者暗中接受另一种输出布局。多一个读取者、错误形状/类型/AccessMap 或不在同一 carried scope 都被 `NATIVE_TRANSPOSE_STORE_DOMAIN` 拒绝。

该固定源码通过 Corpus Gate 179/179、适用 CPU 合同 2610 passed，以及 exact-B300 AOT（255 寄存器、0 stack/spill）。broker-shared 三组数值试验各对照 8,192 个 token-major BF16 输出与 16,384 个 V-first BF16 最终状态，全部 0 超差并保持输入不变。这是**合成 RHS** 的物理输出映射资格；原始 V 输入 `[token,V]` 到行拥有者 `[V,token]` 的读取和公开 beta 仍需单独实现，性能更未比较。这个限定也解释了为什么不能把 Triton 的通用 `tl.trans` 或原始 CAKE CUDA 地址拼写直接当作本 backend 的合法性证明。

### 2.7 原始 V/beta 到行拥有 RHS 的有界映射

公开 V 输入按 `[chunk,token32,V128]` 存储，而四 MMA 状态 kernel 的 128 个 compute 线程各自拥有一个 V 行。`ac102f0e` 的 Schedule 使用已有的 BF16 `load` `[token32,V128]`、显式 `transpose` `[V128,token32]` 和 BF16→FP32 `cast`；后端在有且只有这个紧邻读者、unit chunk 轴及精确 AccessMap 时，直接让每个 V 行线程对每个 token 从 `[chunk,token,V]` 读取连续线程地址。没有隐式寄存器 reshape，也不新增全局转置张量。第二条 FP32 beta-gate `[chunk,token]` load 在每行形成 32 值列向量，`RHS = beta * (V - base_prediction)` 仍由两个有类型 FP32 elementwise 操作显式承担，再交给 strict-lower solve。额外读取原始 `[token,V]` register tile 时由 `NATIVE_TRANSPOSE_LOAD_DOMAIN` 拒绝。

这条输入映射与 §2.6 的 token-major 输出写回构成方向相反、但相互独立的两个有界规则。定向合同与 Corpus Gate 179/179 已通过；精确 B300 AOT 为 255 寄存器、0 stack/spill。本机完整 CPU 套件因复制历史/Lab 夹具时磁盘空间耗尽而失败；随后在远端固定 `ac102f0e` 的隔离检出、标准 `umask 0022` 下重新运行，得到 2,601 passed、16 skipped，另有 1 项 Apple MLX 实机测试不适用而取消选择，退出码为 0。远端默认 `umask 0002` 曾使 `test_author_home` 的私有目录权限检查失败，故两次运行及环境差异均保留在 Finding 中，没有改写夹具或期望。broker-shared B300 作业 `gpuq-c841e1995072` 在 3 组输入上分别检查 8,192 个 BF16 token-major 输出和 16,384 个 BF16 最终状态，均为 0 超差、无非有限值、输入未改写；最大绝对误差分别为 `1.1920928955078125e-07` 和 `3.0517578125e-05`。这证明合成准备值接入**公开 V/beta 的有界输入/输出映射**；真实上游 Q/K/G/prefix、逐 token BF16 状态舍入、64-head/256-chunk 及完整 Workload 尚待验收。它不能靠一句“CUDA 或 Triton 无法转置”来解释：缺口是当前 native 后端此前未兑现已存在的 IR `transpose`，而让 Triton 另起转置 kernel 则必须把额外启动和读写算进同一 Workload。

### 2.8 H64 的标量 head 所有权与 rank-4 TMA

原有合成设备证明只绑定一个 head、两个 chunk。真实准备输出以 `[chunk256,head64,...]` 组织，V/输出沿 `[token,head,V]`，初始/最终状态还多一个 head 轴。只读探针先证实共享 IR 的 `ProgramMap` 标量 head、`AccessIndexKind.PROGRAM` 与 `AccessIndexKind.LOOP` 可以无阻塞地表达这些轴；当时的 native preflight 分别以 `NATIVE_CARRIED_MMA_DOMAIN`、`NATIVE_TMA_DESCRIPTOR`、`NATIVE_TRANSPOSE_LOAD_DOMAIN`、`NATIVE_TRANSPOSE_STORE_DOMAIN` 拒绝，而随后的 register-layout/store-role 报告是转置未准入的连带诊断。这些是具体 backend 缺口，不是 CUDA 或 Triton 语言的不可表达性。

后继 `d5a9648a` 不把 `[chunk,head]` 私自 flatten：一个 `ProgramMap` 标量 head 坐标选定一个 CTA，循环标量 chunk 选定当前块；四个 `[chunk2,head64,K/N,C]` BF16 B 输入以 rank-4 `CUtensorMap` 描述，shared box 仍为原 `[K/N,C]`，高两维 box 均为 1。native helper 把二维块坐标、head、chunk 按 TMA 的内到外顺序送入 `cp.async.bulk.tensor.4d`；P 的普通 global→shared 地址、初始/最终状态、prefix 与 beta 的地址也都显式带 head。公开 V 的 `[chunk,token,head,V]` load 和相同方向的输出 store 沿用独占转置规则，让 128 个 V 行线程直接访问正确的 token-major 地址，不生成全局转置缓冲。NVIDIA 的 [PTX tensor copy 指令](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)与 [tensor-map 编码合同](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__TENSOR__MEMORY.html)允许这类 4D 坐标；具体组合仍由本 Target 的离线编译及设备数值证据限定。

准入只覆盖 head64、unit chunk、一个标量 head CTA 轴与上述精确映射。错误 head 选择、rank-5 TMA、H32 直接沿用 H64 规则、漏掉输出 head 所有权都有专门反例。最初将 `cake_tma4` helper 无条件加入所有 native 源码，改变了旧 Corpus 的发射源快照；后继仅在有 rank-4 TMA 的 Schedule 发射该 helper，固定源码的 Corpus Gate 恢复为 179/179。远端隔离完整适用 CPU 套件在标准 `umask 0022` 下为 2,672 passed、16 skipped、1 项 Apple MLX 实机测试取消选择，退出码 0。精确 `sm_103a` CPU-only AOT 为 255 寄存器、0 stack/spill。第一次 broker 作业 `gpuq-337fef5a99a0` 在 kernel 加载前因 CPU-only 链接漏掉 `libcuda` 而失败；只修正链接后，同一源码的 `gpuq-ad57c793c89c` 完成，租约释放后独立 host oracle 对三组各 **1,048,576 状态元素和 524,288 输出元素** 均检查为 0 超差，所有不可变输入未改写。最大绝对状态误差 `6.103515625e-05`，输出误差 `3.814697265625e-06`。这证明 H64 两 chunk 的地址及发射路径；长循环继承性由 §2.9 单独验证，准备值仍是合成的。

### 2.9 256 chunk 的相位复用与完整固定形状地址

`61892f81` 将同一 H64 Schedule 的 chunk 轴从 2 精确扩展为 256，保留一个 head/CTA、标量 chunk、rank-4 TMA 与 token-major V/输出地址。后端仍拒绝未资格化的 3-chunk 形状；这并非硬件不能运行 3 chunk，而是资格证据目前只覆盖 2 和 256。所有 carried mbarrier 在 256 次迭代中重复使用相位，AOT 源码保留 `#pragma unroll 1` 并按 `trip & 1` 等待/发布；不能从两次迭代推断其跨偶奇相位循环正确。固定提交通过 Corpus Gate 179/179 和远端完整适用 CPU 合同 2,674 passed、16 skipped、1 项 Apple MLX 实机测试取消选择；精确 `sm_103a` AOT 为 254 寄存器、0 stack/spill。

broker-shared 作业 `gpuq-1f1423e7d3da` 完成并释放 GPU1 后，独立 host oracle 用一组各 head 不同的合成准备值检查完整固定 H64/T8192 地址范围：**67,108,864 个 BF16 token-major 输出和 1,048,576 个 BF16 最终状态**均为 0 超差、无非有限值，所有不可变输入未改写。最大绝对输出/状态误差分别是 `3.814697265625e-06` / `6.103515625e-05`。这是 256 次 chunk 循环、相位及输出地址的设备证明，**不是完整 KDA 的外部 oracle 通过**：Q/K/G/prefix/P/query/coupling 等准备值仍为合成输入，状态仍在 chunk 边界而非逐 token 舍入，packed、六形状与框架 ABI 尚未验收。

首次独占 CUPTI 组件诊断因 Python 环境缺失已安装的 `cupti-python` 路径而被严格封装拒绝，不能使用其 CUDA event fallback。补齐依赖后，两次作业分别在 GPU0、GPU1 启动前发现外部计算进程并退出；broker 均已终结释放，外部进程未被干预。因此 **没有有效状态 kernel 延迟，更没有相对原始 CAKE 的加速比**。准备阶段单独 389.507 µs 的旧诊断已接近完整参考 456.260 µs；即使该状态路径未来测得很快，超越 CAKE 仍需要减少准备/状态间物化、额外启动和状态搬运，并以完整 Workload 的配对 CUPTI 实验检验。

### 2.10 为什么长循环数值通过仍不能证明逐 token BF16 语义

[Workload oracle](../src/open_cake_ir/tasks/kda_prefill/oracle.py) 在每个 token 计算衰减、预测、beta 残差及状态外积后，立刻把**整个状态**舍入到 BF16，再用该状态计算输出和下一个 token。[chunked algebra](../src/open_cake_ir/tasks/kda_prefill/chunked.py) 将同一 chunk 内的旧状态保持不变，用预先算好的前缀/P 耦合及 FP32 前代入推出 U，只在 chunk 末尾舍入新状态。它减少状态搬运并使 TMA/MMA 可分块，但量化算子 `Q` 不与线性组合交换。例如从已量化的 `s=0.10009765625` 出发，两步均取 `d=0.99`、`u=0.01`：`Q(Q(s·d+u)·d+u)=0.11767578125`，而只在末尾舍入得 `Q(s·d²+u·d+u)=0.1181640625`。这个两步差值本身低于 Workload 的 0.01 容差，却证明块级恒等式**并非精确的 BF16 递推**；真实 KDA 的下一步 U 还会读取已量化状态，误差可以随高保留轨迹累积。

Finding `F-2026-09-24-003` 的 event 40 给出算子级反例：同一高保留输入下 chunk1 的最终状态与独立 oracle 完全一致，chunk2/4/8/16/32 均出现超容差状态元素；常数 gate 试验从 `g=-4`（decay 约 0.914）通过，变为 `g=-6`（decay 约 0.988）时有 123 个失败元素。它未给出可证明的分流阈值。当前 `61892f81` 的 67,108,864 输出检查只与**同一块级代数的合成 host oracle**比较，不能覆盖这个缺陷。原始 CAKE CUDA 在冻结 Workload 的完整输入上通过外部 oracle；不能因此推断它对任意高保留输入也完全逐 token 精确。既有 Triton 路线可以写循环与 BF16 cast，但当前实测的 chunk16/32 映射仍有昂贵串行/重复工作；既有 native 路线尚无逐 token 状态舍入的有界同步与存储合同。这些是具体实现与资格差距，不是 CUDA 或 Triton 语言的表达上限。

可验证的后继有两条：在片上保留 BF16 状态并**每 token 更新和舍入**，用外部 oracle 与高保留反例先证明正确，再测其顺序成本；或为快速块级路径推导能在运行前检查的误差上界，并为不满足条件的输入提供已验证的精确 fallback。前者不能退回每 token 全局读写状态（该已正确的 direct CUDA 路线为 16,979.920 µs）；后者不能用一次通过的普通随机输入猜 guard。两者都需要把 IR 可见的舍入位置、状态所有权、phase 和读写分析同时补齐，然后在完整六形状及配对计时下决定是否保留。

## 3. 对照：谁拥有哪个拒绝

| 合同/硬件选择 | 共享 IR/Verifier 的职责 | native CUDA 的职责 | 最小反例与证据 |
| --- | --- | --- | --- |
| `forward_substitute` | 形状、dtype、strict-lower 顺序、舍入边界 | P shared stage、barrier、128 行 FP32 FMA 发射 | `test_forward_substitute_ir.py`、`test_native_forward_substitute.py`；错误 stage/wait 被 `NATIVE_SOLVE_P_STAGE` 等拒绝。 |
| carried BF16 TMEM | 一个 initializer、一个 updater、读先于写、同 barrier phase | `tcgen05.st` 打包、四 warp 到达及循环 phase | `test_tmem_state_contract.py`、`test_native_two_phase_carried.py`；缺 wait/多写者不准入。 |
| 标量 `loop` 地址 | unit tile、作用域、地址范围和局部结果维度 | rank-3 TMA 坐标与两个 staged B 域 | `test_scalar_loop_access.py`、`test_native_two_phase_k128.py`；错误首轴由 `NATIVE_CARRIED_MMA_DOMAIN` 拒绝。 |
| BF16 TMEM 回读 | 同 dtype、32-bit word/双 BF16 packing、carried read-before-update | 当前 phase 等待、word load、位模式解包 | `test_tmem_state_contract.py`、`test_native_bf16_tmem_read.py`；错误 atom/wait/非 carried 源被拒绝。 |
| 旧状态衰减 | FP32 cast/mul/add 与 BF16 cast 保持独立语义 | 行拥有者逐列计算并发布下一 TMEM phase | `test_native_kda_decayed_state.py`；去掉 prefix producer 触发 `BUFFER_UNPRODUCED`。 |
| H64 标量 head / rank-4 TMA | `ProgramMap` 和 AccessMap 明示 head、chunk 与本地二维 tile | 只为 H64 两 chunk 发射 4D tensor-map、TMA4 与直接 V/输出地址 | `test_native_kda_head_address.py`；错误 head、rank-5、H32、漏 store 所有权各有拒绝。 |

### 一次没有推广的 lowering 尝试

`task/nvidia-kda-fused-state-epilogue` 试图在后端把“BF16→FP32、乘前缀、加校正、转 BF16”四个 **相邻且独占中间值** 的操作映射为一轮逐列计算，仍使用 `__fmul_rn`、`__fadd_rn` 和 `__float2bfloat16_rn`。额外读者存在时，它回到原始逐操作发射，测试固定这个反例。AOT 与未融合版均使用 **255 寄存器、0 stack、0 spill**；`cuobjdump` 均列出 3,376 条 SASS 指令，但发射内容并非逐条相同。这些静态观察没有证明降低资源或提升完整 Workload 性能，**当前 disposition：No promotion**。不能因为源代码看起来更短，就将此特化自动加入 Compiler pass 或声称加速。

## 4. 证据与尚未完成的证明

| 固定源码 / 作业 | 已证明 | 没有证明 |
| --- | --- | --- |
| `2e365222` / `gpuq-f0aeefc31e6b` | K32/C32 两次 MMA + solve，三组每组 4,096 BF16 最终状态全部通过；119 寄存器，0 spill | K128、完整 KDA 输出或延迟 |
| `a81a6948` / `gpuq-2eac9e60295d` | K128/C32 两阶段，三组每组 16,384 最终状态全部通过；230 寄存器，0 spill | 旧状态衰减、beta、查询输出或延迟 |
| `d579e917` / `gpuq-85485a66ccbf` | K128/C32 加 BF16 TMEM 回读和 FP32 prefix 衰减，三组每组 16,384 最终状态全部通过；255 寄存器，0 spill | 逐 token BF16 舍入、高保留输入、完整输出或延迟 |
| `5bd8e2c1` / CPU-only AOT | 独占四操作的后端融合可编译，仍为 255 寄存器、0 spill | 设备数值、动态成本或任何可推广收益 |
| `3ce392dc` → `fbafa719` / `gpuq-1b12eb3b5932` | 同一四 MMA 合成输出/状态图，延迟 query TMEM 回读将 stack 112B、spill 108B 降到 0；两版均 255 寄存器。后版三组各 8,192 输出及 16,384 状态均通过 B300 检查 | 公开 V/beta ABI、逐 token BF16 状态舍入、六形状及配对延迟 |
| `42b78a5d` / `gpuq-7177fb25c220` | 显式 transpose→直接 token-major store；三组各 8,192 输出、16,384 状态在 B300 均通过，AOT 0 spill | 原始 V/beta RHS、完整 Workload 与性能胜出 |
| `ac102f0e` / `gpuq-c841e1995072` | 公开 V/beta 输入到行拥有 RHS；远端完整适用 CPU 合同 2,601 passed、Corpus Gate 179/179，B300 三组各 8,192 输出及 16,384 状态均通过，AOT 255 寄存器、0 stack/spill | 真实上游准备值、逐 token BF16 状态舍入、64-head/256-chunk、六形状和配对延迟 |
| `d5a9648a` / `gpuq-ad57c793c89c` | H64 两 chunk 的 rank-4 TMA、公开 V/beta 与 token-major 输出；完整适用 CPU 合同 2,672 passed、Corpus Gate 179/179，B300 三组各 524,288 输出及 1,048,576 状态均通过，AOT 255 寄存器、0 stack/spill | 256 chunk、真实准备值、逐 token BF16 舍入、完整六形状和配对延迟 |
| `61892f81` / `gpuq-1f1423e7d3da` | H64/256 chunk 的长循环相位和地址；完整适用 CPU 合同 2,674 passed、Corpus Gate 179/179，B300 一组 67,108,864 输出及 1,048,576 状态均通过，AOT 254 寄存器、0 stack/spill | 真实准备值、逐 token BF16 舍入、packed/六形状、组件及配对延迟 |

表中带 `gpuq-` 的数值作业使用 exact `sm_103a` 与 broker 分配；设备输出在作业终结、租约释放后由独立 host oracle 比对，输入保持性也经检查。只有 CPU-only AOT 的行不含设备结论。`d579e917` 的 full applicable CPU contracts 为 2,607 passed、5 skipped，另有 1 项本机 Apple MLX 实机测试因缺 `device_info` 接口而未作为 NVIDIA 门禁；Corpus Gate 为 179/179。GPU 程序正确仅覆盖本表对应的合成 Schedule，**不是** Workload Contract 的全形状验收。

## 5. 下一步性能闭环

| 完整 KDA 的数据边 | 当前最低证据 | 尚缺的工作 |
| --- | --- | --- |
| Q/K 归一化、decay、beta 与 32-token 耦合 | 独立准备组件在 B300 通过 H64 元素检查；两 MMA 版本单独计时 389.507 µs | 与状态/输出 CTA 融合或有证据地选择物化边界，避免七个中间张量往返。 |
| `state @ base_key` 与 P 前代入 | `61892f81` 的 H64/256 chunk 地址、公开 V/beta 到 RHS 映射在完整适用 CPU 门禁和 B300 全输出设备数值检查均通过；准备值仍为合成输入 | 接入真实上游 Q/K/G/prefix，并验证完整状态语义。 |
| `state * prefix_end + U @ final_key` | BF16 TMEM 回读/FP32 合并的两 chunk 合成状态在 B300 三种输入通过 | 接入准备组件的真实 FP32 prefix/final-key，并检验更多 chunk、尾块及原位状态别名。 |
| `state_after_update @ query` 与输出耦合 | `61892f81` 已在 B300 固定 H64/T8192 合成输入上通过公开 V/beta、token-major 输出及最终状态，AOT 0 spill | 真实准备值、逐 token 状态舍入、packed/tail 与完整 Workload 写回仍未证明。 |
| 逐 token BF16 状态舍入 | 独立 Workload oracle 和高保留失败反例 | 当前块代数只在块边界舍入；需精确路径或有证明且含 fallback 的输入 guard。 |
| 六形状、packed/tail、框架 ABI | Workload 与 guardrail 已冻结 | 完整候选、Target admission、正式 Evaluation、CUPTI 配对和 profiler 均未完成。 |

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
