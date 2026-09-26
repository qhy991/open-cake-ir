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

`61892f81` 将同一 H64 Schedule 的 chunk 轴从 2 精确扩展为 256，保留一个 head/CTA、标量 chunk、rank-4 TMA 与 token-major V/输出地址。后端仍拒绝未资格化的 3-chunk 形状；这并非硬件不能运行 3 chunk，而是资格证据目前只覆盖 2 和 256。carried state 与 P 的 mbarrier 按 `trip & 1` 重复使用相位，两个 MMA 流水的 ready/free/completion barrier **每 chunk 重新初始化与失效**；AOT 源码保留 `#pragma unroll 1`。不能从两次迭代推断其跨偶奇相位循环正确。固定提交通过 Corpus Gate 179/179 和远端完整适用 CPU 合同 2,674 passed、16 skipped、1 项 Apple MLX 实机测试取消选择；精确 `sm_103a` AOT 为 254 寄存器、0 stack/spill。

broker-shared 作业 `gpuq-1f1423e7d3da` 完成并释放 GPU1 后，独立 host oracle 用一组各 head 不同的合成准备值检查完整固定 H64/T8192 地址范围：**67,108,864 个 BF16 token-major 输出和 1,048,576 个 BF16 最终状态**均为 0 超差、无非有限值，所有不可变输入未改写。最大绝对输出/状态误差分别是 `3.814697265625e-06` / `6.103515625e-05`。这是 256 次 chunk 循环、相位及输出地址的设备证明，**不是完整 KDA 的外部 oracle 通过**：Q/K/G/prefix/P/query/coupling 等准备值仍为合成输入，状态仍在 chunk 边界而非逐 token 舍入，packed、六形状与框架 ABI 尚未验收。

首次独占 CUPTI 组件诊断因 Python 环境缺失已安装的 `cupti-python` 路径而被严格封装拒绝，不能使用其 CUDA event fallback。补齐依赖后，两次作业分别在 GPU0、GPU1 启动前发现外部计算进程并退出；broker 均已终结释放，外部进程未被干预。新作业 `gpuq-95ff801f95c4` 在干净的独占 GPU1 上取得 5 轮各 25 个 CUPTI 冷 L2 样本、无 graph，计时后输出/状态与已资格化抓取逐位一致：组件中位数 **7,678.607 µs**，轮中位数 7,677.390–7,681.713 µs，轮内变异系数均小于 0.13%。这比适配 CAKE 完整参考约 456 µs 还慢很多；两者不是同作业、同计算范围的 speedup 比较，但已足以否决“再加独立准备阶段即可追平”的想法。准备阶段单独 389.507 µs 的旧诊断也已接近完整参考；超越 CAKE 需要改变求解/流水与准备/状态间的物化边界，再做完整 Workload 的配对实验。

### 2.10 为什么长循环数值通过仍不能证明逐 token BF16 语义

[Workload oracle](../src/open_cake_ir/tasks/kda_prefill/oracle.py) 在每个 token 计算衰减、预测、beta 残差及状态外积后，立刻把**整个状态**舍入到 BF16，再用该状态计算输出和下一个 token。[chunked algebra](../src/open_cake_ir/tasks/kda_prefill/chunked.py) 将同一 chunk 内的旧状态保持不变，用预先算好的前缀/P 耦合及 FP32 前代入推出 U，只在 chunk 末尾舍入新状态。它减少状态搬运并使 TMA/MMA 可分块，但量化算子 `Q` 不与线性组合交换。例如从已量化的 `s=0.10009765625` 出发，两步均取 `d=0.99`、`u=0.01`：`Q(Q(s·d+u)·d+u)=0.11767578125`，而只在末尾舍入得 `Q(s·d²+u·d+u)=0.1181640625`。这个两步差值本身低于 Workload 的 0.01 容差，却证明块级恒等式**并非精确的 BF16 递推**；真实 KDA 的下一步 U 还会读取已量化状态，误差可以随高保留轨迹累积。

Finding `F-2026-09-24-003` 的 event 40 给出算子级反例：同一高保留输入下 chunk1 的最终状态与独立 oracle 完全一致，chunk2/4/8/16/32 均出现超容差状态元素；常数 gate 试验从 `g=-4`（decay 约 0.914）通过，变为 `g=-6`（decay 约 0.988）时有 123 个失败元素。它未给出可证明的分流阈值。当前 `61892f81` 的 67,108,864 输出检查只与**同一块级代数的合成 host oracle**比较，不能覆盖这个缺陷。原始 CAKE CUDA 在冻结 Workload 的完整输入上通过外部 oracle；不能因此推断它对任意高保留输入也完全逐 token 精确。既有 Triton 路线可以写循环与 BF16 cast，但当前实测的 chunk16/32 映射仍有昂贵串行/重复工作；既有 native 路线尚无逐 token 状态舍入的有界同步与存储合同。这些是具体实现与资格差距，不是 CUDA 或 Triton 语言的表达上限。

可验证的后继有两条：在片上保留 BF16 状态并**每 token 更新和舍入**，用外部 oracle 与高保留反例先证明正确，再测其顺序成本；或为快速块级路径推导能在运行前检查的误差上界，并为不满足条件的输入提供已验证的精确 fallback。前者不能退回每 token 全局读写状态（该已正确的 direct CUDA 路线为 16,979.920 µs）；后者不能用一次通过的普通随机输入猜 guard。两者都需要把 IR 可见的舍入位置、状态所有权、phase 和读写分析同时补齐，然后在完整六形状及配对计时下决定是否保留。

### 2.11 测量驱动的 barrier 生命周期修订及其限度

`61892f81` 每个 chunk 为两条 TMA/MMA 流水重新执行 ready/free 与四个 completion mbarrier 的 `init`、CTA 同步、`inval`。为了定位耗时，保留 P stage 和 wait、但故意把 `forward_substitute` 改成 `U=RHS` 的**不正确**独立消融，在另一独占作业测得 7,232.558 µs；与有效组件的 7,678.607 µs 相差约 446 µs，但这两个数字跨作业、跨设备，只能削弱“496 项顺序 FMA 单独解释 7 ms”的假设，不能当作正式消融加速比。生成源码还有每 chunk 多次 `__syncthreads`；原始 CAKE 导出的 B300 适配源码则使用五槽 shared 流水、分开的 producer/compute/MMA/epilogue 角色，以及每 head 两个 M64 value slice。它并不靠缩短一个 solve 函数得到完整的约 456 µs。

目标专属后继 `92994711` 在**同一 H64、256 chunk、两条单槽流水且每条恰好两次 MMA** 的结构域，将两条 ready/free 和四个 completion barrier 的初始化/失效移到 carried loop 两端；循环内 producer/consumer 使用 `trip & 1`，每次重新到达前仍等待上一 phase 完成，保留每 chunk 的 CTA drain。其它两 chunk 路径保持原发射，专门合同固定这组前后条件与反例。[NVIDIA PTX mbarrier 生命周期](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)说明 phase 完成后自动重新就绪、下一 phase 的 arrive 之前须有成功的 wait；这支持该映射的硬件意图，设备验证仍不可省略。固定后继通过 Corpus Gate 179/179、远端完整适用 CPU 合同 2,676 passed、16 skipped、1 项 Apple MLX 实机测试取消选择；AOT 为 255 寄存器、0 stack/spill。broker-shared `gpuq-072c72413d0e` 释放后，独立 host oracle 对同一固定 H64/T8192 合成输入的全部 67,108,864 输出及 1,048,576 最终状态元素均为 0 超差，输入未改写。

随后 `gpuq-259eacb16124` 在**同一独占 GPU0、同一输入、交替顺序**比较两个固定 cubin：五轮每臂各 25 个 CUPTI 冷 L2 样本、无 graph，两臂末次输出/状态均与先前抓取逐位相同、后检查无其它计算 PID。旧版 pooled 中位数 **7,611.606 µs**，复用版 **7,228.691 µs**；五轮比值 1.05218–1.05306，pooled 比值 **1.05297×**，轮内变异系数均低于 0.16%。这是同范围组件的约 5.3% 收益，但 **7.23 ms 仍无法作为追赶 CAKE 的主体路线**。当前 disposition：保留 NVIDIA 任务分支的有界机制，不提升为共享 pass 或完整 KDA 候选。后继需要使 B 输入预取与状态消费跨 chunk 重叠，并考虑原始 CAKE 的多槽/多角色分工与准备阶段片上融合；每一项仍须保留完整数值和配对计时证据。

### 2.12 末态写回：把每块的死覆盖缩成最后一次

测过 barrier 后再检查生成源码，发现 `store_final` 位于 256 次 carried loop **内部**，但它的全局目的地 `[head64,V128,K128]` 与 chunk 无关，且无内核内读者；每次都完整覆盖前一次结果。全局目的地每轮约 2 MiB，其中前 255 轮是约 510 MiB 的逻辑死写回。直接把 `store_final` 移到 Schedule 的循环外时，通用 Verifier 以 `BUFFER_ESCAPES_LOOP` 拒绝循环内 register `next_state` 的越界读取。这是正确的生命周期门禁，不应为优化而绕开；未来的共享 IR 若要表达循环出口值，应连同生存期和别名分析一并设计。

有界 NVIDIA 后继 `5ac55ad4` 保留可验证的 Schedule 结构，只在精确 H64/256 carried 路径中，确认最终 STORE 是 loop 末操作、紧随使用同一 register 状态的 TMEM 发布、目的地 BF16 `[64,128,128]` 完整写入、AccessMap 只含标量 head 与本地 row/column、唯一 writer 且没有任何 reader，才发射 `if (it0 == 255)`。其他形状与读者存在的反例不走此映射，两个 chunk 的控制版本保持原发射。这个规则使性能决定可在 emitter 源码里检查，但尚未将 loop-exit 值推广成共享 IR 的新语法。固定源码通过 Corpus Gate 179/179、远端完整适用 CPU 合同 2,680 passed、16 skipped、1 项 Apple MLX 实机测试取消选择；精确 `sm_103a` AOT 为 255 寄存器、0 stack/spill。broker-shared `gpuq-4548098ec2d4` 释放设备后，独立 host oracle 对 H64/T8192 的 67,108,864 输出和 1,048,576 最终状态元素均为 0 超差，所有不可变输入未改写。

同一独占 GPU0、同一输入、交替顺序的 `gpuq-5dee2913c2e2` 做五轮每臂各 25 个冷 L2 CUPTI 样本、无 graph。两臂计时后均与各自先前设备抓取**逐位相同**，且没有其它计算 PID。原版 pooled 中位数 **7,217.556 µs**，仅末块写回版 **2,999.349 µs**，五轮比值 2.40552–2.40744，pooled **2.40637×**，轮内变异系数均低于 0.1%。这是同范围的真实组件收益，也定位了重复的行线程全状态写回是主要成本之一；它仍不能与适配 CAKE 的完整约 456 µs 非配对地称作加速比。当前 disposition：保留在 NVIDIA 任务分支，尚不推广共享 pass 或完整 KDA 候选。下一轮先分辨 token-major 输出交通与 TMEM/MMA 相位成本，再决定 TMA store、更多并行 CTA 或多槽流水的设计。

### 2.13 输出 epilogue 的上界诊断与下一种流水

为了判断剩余约 3 ms 是否主要耗在 token-major 输出，保留 `5ac55ad4` 的状态递推、故意去掉最终输出 STORE 的独立源码消融；其输出**不正确**，且编译器可能连带删除仅供输出使用的寄存器计算，因此不能把时差归到 store 指令。`gpuq-98fa41502e4e` 在同一独占 GPU0、同一输入、交替顺序五轮每臂各 25 个 CUPTI 冷 L2 样本中，得到有效版 pooled **2,997.653 µs**、消融版 **2,846.931 µs**，比值 1.05294；有效版输出与状态逐位复核、消融版的状态逐位复核均通过，后检查无外部计算 PID。最大轮内变异系数为 0.10%。这给输出 epilogue 与写回合计约 151 µs 的**诊断上界**，不支持优先把直接 token-major store 换成更复杂的 TMA store 来追赶 CAKE。

剩余成本主要落在 256 次依赖串行的状态转移、TMEM/MMA 相位与未重叠的 B 输入准备上，具体份额仍需同作业阶段探针或有效硬件计数器才能归因。可见的原始 CAKE CUDA 用五槽 shared ring 和分开的 producer/compute/MMA/epilogue 角色，当前 native 仅有单槽、每 head 一个 CTA 且在两条流水及 chunk 边界做完整 CTA drain。下一种 lowering 应先明确多槽 B 预取的所有权、phase、free/ready 反例和 carried-state 依赖，保证同一 chunk 的状态语义；再考虑准备值片上融合与 M64 价值行切分。现有输出消融的 disposition 是 **No promotion**，它只选择下一步工作，不能成为候选或性能结论。

### 2.14 阶段时钟证据与分角色多槽预取的设计边界

针对 `5ac55ad4` 的独立阶段探针只给 CTA0 在每个 chunk 写五个 `clock64` 值，trace 放在**显式扩容**的输出末尾，逻辑输出和最终状态在设备后检查仍与未插桩版本逐位相同。第一次作业把 trace 错接到未扩容的 V 输入，发生 illegal access 并由 broker 终结释放；修正为输出指针后的 `gpuq-3dcb525965c7` 完成且释放。修正探针的 AOT 比原 kernel 多 8 字节 stack、4 字节 spill store/load，因此它只用于**粗粒度、单 CTA 的阶段占比**，不用于接受延迟。在稳态 chunk 1–254，单块四段 clock 中位数依次为 142、13,567、91、6,609 cycles，合计的中位数约 20,487 cycles；第二段占各段中位数之和约 66.5%，第四段约 32.4%。前两段的分界是基础流水发射结束而非 MMA 完成，异步完成的等待可能落在第二段；校正流水同理。更细的 11 点和前半段五点探针分别造成 1,016 与 488 字节 stack/spill，已在 CPU-only AOT 门禁停止，没有引用它们的设备计时。这些数据支持改变相位重叠，但不支持把某条 TMA、MMA、V load 或 FMA 单独定为瓶颈。

下一种 **尚未实现或资格化** 的映射应让 copy warp 在独立循环中预取未来 chunk 的四个 B box，计算/MMA warp 仍按 carried state 的顺序消费当前 chunk。Schedule 已能声明 Pipeline 的 `stages` 与 Role 所有权，但 native 的 `NATIVE_CARRIED_PIPELINE_STAGES` 当前只准单槽；新发射需先保证每槽两条 `ready` 各自在对应两次 TMA 完成后供 MMA 等待、生产者在复用槽前等待对应 `free`、`phase=(chunk//stages)&1`，最后排空所有槽才失效。state 的 `ready` 仍按 chunk 顺序发布，禁止下一块 MMA 越过前一块的 BF16 状态更新。四个当前 B box 合计约 26 KiB/槽，五槽的 shared 存储与 mbarrier 控制须由精确 `sm_103a` Target 资源检查和 AOT 实证；不能只因为原始 CAKE 有五槽就自动准入。缺 `free`、错 parity、过早重用 shared box、遗漏最终 drain、错误 carried-state 相位都应有各自的拒绝，而非靠另一个规则偶然拦截。

原始 CAKE 的 B300 适配 CUDA 可见五槽生产/消费/epilogue 分工、每 head 两个 M64 value slice，并在同一 kernel 内准备因子；当前 native 是单槽、每 head 一个 CTA，输入因子单独物化。二者语言都能发射 TMA/PTX，差别是 **IR 可见的所有权、相位及 emitter 的并行循环**。上述多槽设计先验证完整 H64/256 chunk 的合成输出/状态，再接真实 Q/K/G 和原位状态、逐 token BF16 舍入及 packed/tail，最后用完整 Workload 与适配 CAKE 在同一独占作业配对测量。当前提议没有性能数字，也不授权 Target 扩表或公共 pass。

### 2.15 两槽 B 超前预取：已资格化的局部重叠

`671b2a90` 把 §2.14 的第一步落为有界 native 发射：两条 Pipeline 各显式声明 `stages=2`，base/query 与 correction/output 的四个 shared B view 分别占两槽，RangeOptions 也声明两槽。共享 IR 构造与 Verifier 在不改词汇的情况下通过；旧 native 仅以 `NATIVE_CARRIED_PIPELINE_STAGES` 拒绝。新 backend 只在精确 H64/256、两条 Pipeline 都两槽且每条两次 TMA/两次 MMA 时准入，先把 chunk0 装入槽0，然后 copy warp 在 MMA 消费 chunk `j` 时将 `j+1` 的 B 装到另一槽。槽复用前生产者等相同槽上一轮的 `free`，消费者按 `(j//2)&1` 等 `ready`；carried BF16 state 仍按 `j&1` 严格顺序发布。三槽和不匹配的两条 Pipeline 有明确拒绝。可编辑[两槽时序图](figures/kda-b300-prefetch-ring.mmd)及其[SVG](figures/kda-b300-prefetch-ring.svg)只展示这条已测路径，不代表完整原始 CAKE 的五槽流水。

固定提交的 Corpus Gate 为 179/179，远端完整适用 CPU 合同 2,683 passed、16 skipped、1 项 Apple MLX 实机测试取消选择。精确 `sm_103a` AOT 使用 **55,424 字节 dynamic shared、255 寄存器、0 stack/spill**，比单槽版多约 26 KiB shared。broker-shared `gpuq-a64939fffae5` 结束并释放 GPU1 后，独立 host oracle 对同一 H64/T8192 合成输入的 67,108,864 个 BF16 输出和 1,048,576 个最终状态均为 0 超差，输入未改写。`gpuq-8e520f65c628` 在同一独占 GPU0、同一输入、交替顺序下五轮每臂各 25 个 CUPTI 冷 L2 样本、无 graph，两臂计时后输出/状态逐位复核、无其它计算 PID：单槽 pooled **2,997.654 µs**，两槽 **2,835.220 µs**，五轮比值 1.05682–1.05797，pooled **1.05729×**，最大轮内变异系数约 0.114%。原始 raw 的 `scope` 文字沿用了上一个 harness 名称，arm/commit/样本及后检查无歧义；Finding `F-2026-09-24-003` event 154 指向独立的分析记录，保留 raw 不作修改。

这个约 5.7% 的组件收益证明一块超前 B 预取有效，却仍留下 **2.835 ms**；只增加槽深不等于原始 CAKE 的五槽生产、计算和 epilogue 并行。当前 disposition：保留目标专属任务分支，**No promotion** 到共享 pass 或完整 KDA 候选。下一轮需决定是否将 copy、MMA、compute、epilogue 发射为各自持续的 loop，并验证 `free/ready` 相位、state 依赖和最终排空；上游准备值融合与逐 token BF16 状态语义仍独立缺失。

### 2.16 原始 CAKE M64 与 M128 在 B300 的同形状控制

冻结的原始 CAKE 导出同时有 M64、M128 两条 CUDA 路径；其 B300 端口只把 exact target 检查从 `sm_100a` 改为 `sm_103a`。原始 H64/T8192 路由选 **M64：每 head 两个 CTA、128 个 CTA、219,136 字节 dynamic shared/CTA**；M128 使用每 head 一个 CTA、64 个 CTA、227,328 字节 dynamic shared/CTA。为确认几何选择，`gpuq-33d167583f48` 在同一冻结输入上运行 M128，broker 释放后由独立 Workload oracle 检查全部 67,108,864 输出及 1,048,576 原位状态元素，0 超差；其它输入保持不变。M128 的 BF16 输出/状态还与先前资格化的 M64 抓取**逐位相同**。这使两条原始实现成为可配对的硬件控制，但没有资格化我们的 native emitter。

`gpuq-a9ad7fc9ec06` 在同一独占 GPU0、同一输入、两臂各用 512 个独立初始状态槽、五轮交替顺序且每轮每臂 25 个 CUPTI 冷 L2 样本下，得到 M64 pooled **456.578 µs**、M128 **493.923 µs**，M128/M64 为 **1.08179×**。两臂每轮末次输出及原位状态均与独立 oracle 资格化抓取逐位相同，输入不变、后检查无其它计算 PID。代码、CTA 数、角色映射和 shared 占用同时变化，因此不能把约 8.18% 差距单独归因于 M 值；但在这个 B300 固定形状上，直接把 M64 替换为 M128 **不会**超过原始 CAKE 路线。更早的 native M64 消费者还因重复 B 转移与半 lane 使用而比其 M128 控制慢（Finding event 90），说明几何结论不可脱离 lowering 实现继承。当前 disposition：这个对照只作为 Lab 硬件选择证据，不提升原始实现代码为 Open-Cake Compiler 结果。

### 2.17 分角色 carried loop：把块边 CTA 会合移到 kernel 边界

两槽 B 超前预取仍让 copy、MMA、compute 跟随同一块循环，base/correction 流水及每块末尾各做 CTA 会合。后继 `db979435` 不扩大 IR 词汇：在精确 H64/256、两条两槽 B Pipeline、四个 compute warp、一个 MMA warp、一个 copy warp 的结构域，把 `P` 的 2 KiB shared stage 所有权从 copy warp 改为 MMA warp。共享 IR/Verifier 接受这项 Role 与 `p_ready` producer 变更；native 的专门 `NATIVE_ROLE_PIPELINE_DOMAIN` 要求 P 只有 solve 一个读者、同作用域和发布 barrier，否则拒绝这条分角色发射。copy warp 在自己的 256 块循环用 ready/free 槽相位预取 B；MMA warp 在另一循环等 carried state、消费 base B、发布 base 完成、装 P 并发布 `p_ready`，再等 U、消费 correction B；compute warps 在第三循环做 V/beta、solve、输出与 BF16 状态更新。块内无全 CTA `__syncthreads`，三条循环结束后才共同排空、失效 barrier。可编辑[分角色时序图](figures/kda-b300-role-pipeline.mmd)及其[SVG](figures/kda-b300-role-pipeline.svg)将 B ready/free、P、U 与 carried state 的依赖分开显示。

固定提交通过 Corpus Gate 179/179 和远端完整适用 CPU 合同 2,686 passed、16 skipped、1 项 Apple MLX 实机测试取消选择；`sm_103a` AOT 为 **55,424 字节 dynamic shared、255 寄存器、0 stack/spill**。broker-shared `gpuq-77d31d7028e7` 释放设备后，独立 host oracle 在一组普通 H64/T8192 合成准备值上对全部 67,108,864 BF16 输出及 1,048,576 BF16 最终状态检查为 0 超差，输入未改写。重复启动时相对该一次抓取并不逐位稳定：频繁有 head40 从 chunk17 起出现约 66,270 个输出位值及 637 个状态位值差异，另有少数不同计数；首个保留快照对独立合成 oracle 的全部元素仍为 0 超容差，最大绝对输出/状态差分别约 `3.814697265625e-06` / `6.103515625e-05`。首个以**逐位相同**作为计时后条件的独占作业据此拒绝成绩，失败原始记录保留。此差异未定位到具体异步指令，也不能据普通输入容差通过推断高保留输入安全。

后继测量先在 CPU-only 阶段独立保存完整合成 oracle 期望值，再由 `gpuq-d399ee7aa697` 在同一独占 GPU0、同一输入、交替顺序进行五轮每臂各 25 个 CUPTI 冷 L2 样本、无 graph；每轮两臂的完整输出/状态快照均保留，租约释放后由 host **逐轮逐元素**用比 Workload 更紧的输出 `atol=0.00005`、状态 `atol=0.0005`、共同 `rtol=0.01` 复核，十份快照均为 0 超差/无非有限值、输入未改写、设备无其它计算 PID。两槽单循环 pooled **2,833.908 µs**，分角色 pooled **2,462.161 µs**，五轮比值 1.15050–1.15162，pooled **1.15098×**。这是受外部数值 oracle 约束的同范围约 15.1% 组件收益，不是位确定性或完整 KDA 资格。当前 disposition：保留有界 NVIDIA 任务分支，**No promotion** 到共享 pass 或平台；原始 CAKE 有更深的生产/计算/epilogue 重叠和片上因子准备，当前六 warp 两槽路径仍有约 2.46 ms，且逐 token BF16 舍入、原位别名与六形状未验收。

### 2.18 分角色流水后续消融与 P 暂存的发布反例

`db979435` 后的三项同卡诊断把下一轮优化范围缩小了。故意把 V 全局读取换为零的**错误**消融保留其余计算，五轮每臂 25 个冷 L2 CUPTI 样本中，有效版 2,461.426 µs、错误版 2,450.896 µs，仅相差约 10.53 µs；它只是 V 读取/地址/寄存器压力的诊断上界。故意把 496 项前代入 FMA 和 P 系数读取去掉、令 `U=RHS` 的**错误**消融，则得到有效版 2,488.402 µs、错误版 2,231.568 µs，约 257 µs 的上界；这个差别不能解释剩余约 2.2 ms。低扰动 MMA warp `clock64` 探针在 CTA0 的稳态 chunk 1–254 观察到：carried state-ready 等待 5,473 cycles，base B-ready 等待 72，base/query MMA 至完成 1,646，跨 P shared 装载的区间 9,793，U-ready 等待 172，correction B-ready 等待 76，correction/output MMA 至完成 625。区间可包含 warp 调度干扰，不是 P 指令的纯耗时；但 B-ready 等待很短，继续盲目加深 B ring 缺少依据。这三项分别由 Finding event 163、166、167 索引，均 **No promotion**。

另一种改变把输出 STORE 排在 carried state 发布后，让下一块 MMA 有机会与本块写回重叠。`efa8bbad` 通过 Corpus Gate 179/179、适用 CPU 2,688 passed、AOT 255 寄存器/0 spill、完整合成输出和状态 oracle；同 GPU 纯净配对 `gpuq-8673c413e6a4` 却从原分角色版 2,461.235 µs 变慢到 2,624.436 µs，五轮均回退 6.63%。新增的 `NATIVE_CARRIED_OUTPUT_READ_BEFORE_PUBLICATION` 禁止在 query/output TMEM 读出之前发布 state；先前的受其它 PID 干扰的配对记录只保留而不用于结论。此路线 **No promotion**，说明仅移动 publication 顺序没有获得足以抵消排程/活跃范围成本的重叠。

`fbd64306` 针对时钟探针中的 P 区间，只在精确 H64/256、MMA warp 唯一生产 P、compute warp 唯一消费 solve、BF16 32×32/B64 swizzle 的域内，把每 lane 四次 16B `uint4` 全局读取和 shared 写入替代标量路径；源地址不满足 16B 对齐时回退标量。`test_native_kda_vector_p_stage.py` 锁定准入、未对齐回退及额外读者/错误角色反例。固定提交的 Corpus Gate 为 179/179、适用 CPU 2,689 passed，B300 AOT 255 寄存器/0 spill，SASS 实际出现 `LDG.E.128` 和 `STS.128`。单独的 swizzle 往返及在 MMA warp 发布前的集成 P 回显，对抽查的三个 chunk/head 全部 1,024 个 P 位模式均正确；H64/256 全部合成输出和最终状态也对独立数值 oracle 0 超差。

但是这条发射**尚未通过 P 生命周期资格**。独占、同 GPU、交替五轮各 25 个 CUPTI 样本的 `gpuq-df9355066ce1` 测得基线 2,481.104 µs、向量版 1,950.861 µs（诊断比 1.27180×），十份完整输出/状态快照均通过独立数值 oracle、输入未改写、无其它计算 PID。向量版跨轮 BF16 输出位模式约 66.9% 不同；消费者在 `cake_wait(p_ready)` 后直接回显，首块 head0 的 P 元素 128–255 已等于下一块相同位置，而发布前回显与输入逐位相同。相同消费者探针在标量基线的三处抽查全部逐位正确。这个反例证明当前向量发射存在 P 发布/复用时序风险；不能把 1.95 ms 当成已资格化的正确性稳定性能，也不能仅凭普通输入容差推广。一个显式四 compute warp `p_free` 的源码探针通过一次完整合成数值检查，但后继消费者回显在相同位置仍有 96 个错误元素；**仅增加消费完成 barrier 没有修复它**。下一步须在同一次启动同时回显发布前、等待后、复用前的 P，定位是共享写入可见性、barrier 相位还是源码重排，再把确证的所有权、phase、合法性和反例一并纳入 Cake Schedule/Compiler。Finding event 168–171 保留原始路径和失败作业。

原始 CAKE CUDA 已写出具体的 shared ring 及生产/消费同步，现有 native CUDA 对 P 只靠 `p_ready` 和后继 U barrier 的隐含约束；本轮直接观测说明向量化会暴露该约束的不足。Triton 语言原则上也能表达向量读取和同步，但仓库现有 Triton lowering 没有这条 carried TMEM、B TMA、P 所有权与 solve 发射路径；因此差别是**现有实现的表达与分析范围**，不是宣称 CUDA 或 Triton 在原则上做不到。后续设计须同时给出 P stage 的生产、消费完成、重用和 phase 合同，不能把更宽的访存指令单独视作创新或性能保证。

### 2.19 双槽 P：把生产与消费的地址承诺写回 Schedule

单槽向量化失败后，两种只强化发布的消融都没有修复消费者所见的 P：`0c2f522f` 在向量写入后加入 `__threadfence_block` 与第二次 warp 会合，单次回显通过，但五次重复的第 5 次又有 128 个首块 P 位值错误；把 `p_ready` 改成 32 个 MMA lane 分别到达，五次重复第 5 次仍有 96 和 248 个错误元素。显式 `p_free` 探针也失败（§2.18）。这些结果不允许把某条 fence、barrier 计数或指令宽度单独称为修复。它们共同指向应优先隔离相邻 chunk 的 P 存储：在两次复用之前让当前 compute warp 完成 solve。独立源码双槽探针用 chunk parity 选择不同 2 KiB shared 槽，五次启动的 15 个消费者 P 瓦片均逐 bit 正确，末次全部合成输出/状态通过 oracle；这仍只是设计依据，Finding event 172–174 保留其失败与成功记录。

`5974e33e` 将这个地址承诺做成 Cake lowering。Schedule 中 `p_stage` 的 `stages` 从 1 变为 2，其独占 `p_smem` Allocation 从 2,048 扩为 4,096 字节；第 (j) 块的 MMA warp 将 P 写到 `p_stage + (j&1)·2048`，compute warp 在 `p_ready` 的相同 parity 等待后，从**同一槽**执行 496 项前代入读取。全局 P 仍是 BF16 `[256,64,32,32]`；16B 对齐时用 `uint4` 装载/写入，未对齐时保留标量路径。这个双槽只适用于已有的精确分角色 H64/256 域、唯一 P 生产者和 solve 读者、两个两槽 B Pipeline 以及顺序 carried state；第 (j+2) 块复用 P 槽前，MMA loop 已等第 (j+1) 块四个 compute warp 完成 `u_ready`，而每个 compute warp 的 chunk 循环仍按序执行，因此第 (j) 块的 solve 已结束。它不是允许任意异步 P producer 的通用环形缓冲合同。

共享 IR 的 `Buffer.stages` 与 Allocation 容量分析负责声明两个具体存储槽；不足 4,096 字节的反例由 `BUFFER_ALLOCATION_OVERFLOW` 拒绝。native 的 `_role_carried_domain` 只准此域内 1 或 2 槽，三槽反例由 `NATIVE_ROLE_PIPELINE_DOMAIN` 拒绝；发射器用相同 parity 选择 P 写入和 solve 读取，仍保留 `p_ready` 相位、未对齐 fallback 与 B ready/free 的旧顺序。`test_native_kda_p_double_slot.py` 固定这些正反例。没有增加共享 IR 原语、layout algebra、Target 常数或跨任务 pass；优化选择仍由 NVIDIA 任务分支负责。

固定提交通过 Corpus Gate **179/179**、远端适用 CPU 套件 **2,692 passed、16 skipped、1 项 Apple MLX 实机测试 deselected**；精确 `sm_103a` AOT 为 **57,472 字节 dynamic shared、255 寄存器、0 stack/spill**，比单槽 P 多 2 KiB shared。broker-shared `gpuq-f94f83ce1d44` 结束释放后，Cake **实际发射源码**在同一租约内五次启动，对三组 chunk/head 的 15 个消费者 P 瓦片回显全部逐 bit 相同；末次 67,108,864 个输出及 1,048,576 个最终状态对独立合成 oracle 均为 0 超差。未对齐 P 指针 mod16=2 的 broker-shared `gpuq-71bd8643bebd` 走标量 fallback，释放后完整输出/状态仍 0 超差、P 输入位模式未变。第一次未对齐 worker 因自身 shape copy 错误在 kernel 前失败并保留，不作设备结论。

`gpuq-683ba77be918` 在同一独占 GPU1、交替顺序五轮、每臂每轮 25 个 CUPTI 冷 L2/no-graph 样本中，对固定分角色单槽基线 `db979435` 与双槽后继 `5974e33e` 配对。租约释放后的独立 host replay 对十份完整输出/状态快照均 0 超差、无非有限值；输入未改写、后检查无其它计算 PID。pooled 中位数分别为 **2,481.231 µs** 与 **1,953.451 µs**，组件比 **1.27018×**，五轮均胜出，最大轮内 CV 约 0.11%；两臂第 1 与第 5 轮的 BF16 输出和最终状态都逐 bit 相同。这个合格收益归于**双槽地址隔离与向量 P 暂存的组合**，不能把 27% 全归因于多一个槽或 16B 指令。当前 disposition：保留并资格化这个精确 NVIDIA 组件映射，**No promotion** 到共享 pass、完整 KDA 候选或六形状 dispatcher。

同一固定源码的低扰动八点 MMA warp 时钟探针 `gpuq-76a9173f4e7c` 结束释放后，插桩完整输出/状态仍由独立 host oracle 检查为 0 超差；AOT 仍为 255 寄存器、0 stack/spill。在 CTA0 的稳态 chunk 1–254，七段中位数依次为 carried state-ready **5,503**、base B-ready **72**、base/query MMA 至完成 **1,648**、跨 P 暂存 **5,416.5**、U-ready **174.5**、correction B-ready **77**、correction/output MMA 至完成 **625 cycles**；段中位数之和的占比分别为 40.71%、0.53%、12.19%、40.07%、1.29%、0.57%、4.62%。相对 §2.18 的单槽向量化前探针，P 区间从 9,793 降到 5,416.5 cycles，而 state-ready 仍约 5.5k cycles。插桩区间包含 warp 调度，不是指令级归因，也不与未插桩延迟直接相减。B-ready 仍很短，下一步优先验证 carried state 的发布依赖与 P/compute 重叠；继续加深 B ring 缺少这份证据的支持。Finding event 178 保留 trace 与 host 报告。

一个顺着这份探针作出的**独立源码排程试验**把双槽 P 装载移到本 chunk 等 `state_ready` 和发射 base/query MMA 之前，试图与上一个 chunk 的 state 更新重叠；它不改变任何 P 地址、数学操作或 B Pipeline。AOT 为 255 寄存器/0 spill，单独设备抓取及同机配对中的十份完整输出/状态均通过独立合成 oracle。`gpuq-a864742ae1ac` 将固定 Cake 双槽版与这个提前 P 源码在同一独占 GPU4 上交替五轮、每臂每轮 25 个冷 L2 CUPTI 样本，pooled 中位数 **1,907.338 µs** 对 **1,897.003 µs**，提前版只快约 **10.3 µs / 0.54%**；五轮比值稳定在 1.00523–1.00550，首末轮 BF16 输出位模式逐 bit 相同。这个同范围差值比 1.9 ms 剩余差距小得多，且源码尚未由 Schedule 表达；当前 disposition：**No promotion**，不为这一个形状的半个百分点收益增加 Compiler 排程分支。Finding event 179 保留完整配对路径；它也不反证其它针对 state 依赖的结构变化。

原始 CAKE CUDA 已有更深的 shared 槽、角色流水和片上准备值；本后继仍把准备值预先物化、每 head 一个 CTA、256 次状态依赖串行，并且只在块末舍入 BF16 状态。原始 CAKE 适配的 H64/T8192 完整 kernel 在另一作业约 **456.578 µs**，不能与本组件的 1.953 ms 做同范围加速比；数值上仍有明显追赶空间。现有 native CUDA 的单槽 P 在向量化后暴露跨块混值，双槽合同把地址与相位显式留给 Schedule 和检查；现有 Triton lowering 缺 TMEM carried state、两条 B TMA 流水和这条 P/solve 发射，不是 Triton 语言原则上无法表达双槽。下一步先在真实上游 Q/K/G、原位 state、逐 token BF16 舍入与高保留输入上建立完整 Workload 候选，再与适配 CAKE 同机同范围配对；单凭本组件不能声称追上或超过 CAKE。

## 3. 对照：谁拥有哪个拒绝

| 合同/硬件选择 | 共享 IR/Verifier 的职责 | native CUDA 的职责 | 最小反例与证据 |
| --- | --- | --- | --- |
| `forward_substitute` | 形状、dtype、strict-lower 顺序、舍入边界 | P shared stage、barrier、128 行 FP32 FMA 发射 | `test_forward_substitute_ir.py`、`test_native_forward_substitute.py`；错误 stage/wait 被 `NATIVE_SOLVE_P_STAGE` 等拒绝。 |
| carried BF16 TMEM | 一个 initializer、一个 updater、读先于写、同 barrier phase | `tcgen05.st` 打包、四 warp 到达及循环 phase | `test_tmem_state_contract.py`、`test_native_two_phase_carried.py`；缺 wait/多写者不准入。 |
| 标量 `loop` 地址 | unit tile、作用域、地址范围和局部结果维度 | rank-3 TMA 坐标与两个 staged B 域 | `test_scalar_loop_access.py`、`test_native_two_phase_k128.py`；错误首轴由 `NATIVE_CARRIED_MMA_DOMAIN` 拒绝。 |
| BF16 TMEM 回读 | 同 dtype、32-bit word/双 BF16 packing、carried read-before-update | 当前 phase 等待、word load、位模式解包 | `test_tmem_state_contract.py`、`test_native_bf16_tmem_read.py`；错误 atom/wait/非 carried 源被拒绝。 |
| 旧状态衰减 | FP32 cast/mul/add 与 BF16 cast 保持独立语义 | 行拥有者逐列计算并发布下一 TMEM phase | `test_native_kda_decayed_state.py`；去掉 prefix producer 触发 `BUFFER_UNPRODUCED`。 |
| H64 标量 head / rank-4 TMA | `ProgramMap` 和 AccessMap 明示 head、chunk 与本地二维 tile | 只为 H64 两或 256 chunk 发射 4D tensor-map、TMA4 与直接 V/输出地址 | `test_native_kda_head_address.py`、`test_native_kda_chunk256.py`；错误 head、rank-5、H32、未资格化 chunk 数与漏 store 所有权各有拒绝。 |
| carried barrier 生命周期 | carried TMEM 与 producer/consumer 的 phase、读先于写保持原分析 | H64/256 的两条单槽 TMA/MMA 流水在 loop 两端初始化/失效，块内按 parity 等待 | `test_native_carried_barrier_reuse.py`；两 chunk 控制仍保留旧生命周期；B300 全输出设备证明及配对 CUPTI 见 §2.11。 |
| 无读者的末态写回 | 保持 `BUFFER_ESCAPES_LOOP`，循环出口 register 值不能暗中越界 | 仅对 H64/256、每轮同址完整覆盖的最终 STORE 发射最后一次写回 | `test_native_terminal_state_store.py`；内部 reader、错误地址与两 chunk 控制均不准入；设备及配对证据见 §2.12。 |
| 两槽 B 预取 | Pipeline.stages、四个 shared B view 和 RangeOptions 同时声明两槽，carried state 顺序不变 | 在 copy warp 中预取 `j+1`，MMA 消费 `j`；槽复用等 free、消费等 ready，各自用两轮 parity | `test_native_kda_two_stage_prefetch.py`；三槽和两条 Pipeline stage 不匹配均由 `NATIVE_CARRIED_PIPELINE_STAGES` 拒绝；设备及配对证据见 §2.15。 |
| 分角色 carried loop | P shared stage 唯一读者为 solve，Role 与 `p_ready` producer 明示 MMA warp，原 carried state 顺序不变 | copy/MMA/compute 各自推进块循环，以 ready/free、P、U、state barrier 同步，块内无 CTA 会合 | `test_native_kda_role_pipeline.py`；额外 P 读者由 `NATIVE_ROLE_PIPELINE_DOMAIN` 拒绝；重复启动与逐轮 oracle 见 §2.17。 |
| 向量 P shared 暂存 | H64/256、唯一生产/消费、BF16 32×32/B64 swizzle、对齐条件及标量 fallback | MMA warp 用 16B 全局读取和 shared 写入，后接 `p_ready`；消费者发布/复用仍有失败反例 | `test_native_kda_vector_p_stage.py`；§2.18 的等待后 P 回显失败，当前 No promotion。 |
| 双槽 P 地址隔离 | `Buffer.stages=2`、`p_smem` 4 KiB，单一 P 生产者/solve 读者与顺序 carried loop；不足容量由共享分析拒绝 | MMA 写槽 `j&1`，compute solve 读同槽；三槽由角色域拒绝，B ready/free 和 U/state 顺序维持 | `test_native_kda_p_double_slot.py`；§2.19 的五次 P 回显、未对齐 fallback 和配对 CUPTI；只资格化该 NVIDIA 组件。 |

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
| `61892f81` / `gpuq-1f1423e7d3da`、`gpuq-95ff801f95c4` | H64/256 chunk 的长循环相位和地址；完整适用 CPU 合同 2,674 passed、Corpus Gate 179/179，B300 一组 67,108,864 输出及 1,048,576 状态均通过；单组件 CUPTI 中位数 7,678.607 µs，AOT 254 寄存器、0 stack/spill | 真实准备值、逐 token BF16 舍入、packed/六形状和完整 Workload 配对延迟 |
| `92994711` / `gpuq-072c72413d0e`、`gpuq-259eacb16124` | ready/free/completion barrier 生命周期跨 256 chunk 复用；适用 CPU 合同 2,676 passed、Corpus Gate 179/179，全输出/状态设备检查通过；同 GPU 配对组件比 1.05297×，AOT 255 寄存器、0 stack/spill | 准备阶段融合、逐 token BF16 舍入、六形状、相对 CAKE 的完整配对性能 |
| `5ac55ad4` / `gpuq-4548098ec2d4`、`gpuq-5dee2913c2e2` | 最终状态只在第 256 个 chunk 写回；适用 CPU 合同 2,680 passed、Corpus Gate 179/179，全输出/状态设备检查通过；同 GPU 配对组件比 2.40637×、中位数 2,999.349 µs，AOT 255 寄存器、0 stack/spill | 原位状态别名、真实准备值、逐 token BF16 舍入、packed/六形状、相对 CAKE 的完整配对性能 |
| `671b2a90` / `gpuq-a64939fffae5`、`gpuq-8e520f65c628` | 两槽 B 预取，适用 CPU 合同 2,683 passed、Corpus Gate 179/179，H64/256 全输出及状态设备通过；同 GPU 配对组件比 1.05729×、中位数 2,835.220 µs，AOT 255 寄存器、0 spill | 完整角色级流水、原位别名、真实准备值、逐 token BF16 舍入、六形状及 CAKE 配对性能 |
| 原始 CAKE B300 端口 M64/M128 / `gpuq-a9ad7fc9ec06` | 同一 H64/T8192 输入上两版均通过独立 Workload oracle、输出/状态逐位相同；同 GPU 配对 M64 456.578 µs、M128 493.923 µs | 我们的 native M64 资格、任何 Compiler 收益或六形状迁移 |
| `db979435` / `gpuq-77d31d7028e7`、`gpuq-d399ee7aa697` | copy/MMA/compute 独立循环；适用 CPU 合同 2,686 passed、Corpus Gate 179/179；合成 H64 全输出/状态及十份配对快照均通过独立数值 oracle；同 GPU 组件比 1.15098×、中位数 2,462.161 µs，AOT 255 寄存器、0 spill | 位确定性、高保留及逐 token BF16、原位别名、真实准备值、六形状及相对 CAKE 的完整配对性能 |
| `efa8bbad` / `gpuq-8673c413e6a4` | state 提前发布后的数值正确性；纯净配对相对 `db979435` 回退 6.63%，2,624.436 µs；No promotion | epilogue 与下一块产生净重叠收益 |
| `fbd64306` / `gpuq-df9355066ce1`、`gpuq-d51bfab6a37f` | 精确域内 16B P 暂存 AOT/数值 oracle；诊断中位数 1,950.861 µs；等待后 P 回显首块 128 个元素被下一块值覆盖，位结果跨轮不稳；No promotion | P 的发布/复用正确性、正式性能收益及完整 KDA |
| `5974e33e` / `gpuq-f94f83ce1d44`、`gpuq-683ba77be918`、`gpuq-71bd8643bebd` | 双槽 P 实际 Cake 发射五次消费者回显全部逐 bit 正确；适用 CPU 2,692 passed、Corpus Gate 179/179，AOT 255 寄存器/0 spill；对齐与未对齐完整合成输出/状态通过 oracle；同机配对组件中位数 1,953.451 µs、相对 `db979435` 为 1.27018×，十份快照通过且首末轮逐 bit 稳定 | 真实准备值、逐 token BF16、原位 alias、高保留、六形状与完整 CAKE 同范围胜出 |

表中带 `gpuq-` 的数值作业使用 exact `sm_103a` 与 broker 分配；设备输出在作业终结、租约释放后由独立 host oracle 比对，输入保持性也经检查。只有 CPU-only AOT 的行不含设备结论。`d579e917` 的 full applicable CPU contracts 为 2,607 passed、5 skipped，另有 1 项本机 Apple MLX 实机测试因缺 `device_info` 接口而未作为 NVIDIA 门禁；Corpus Gate 为 179/179。GPU 程序正确仅覆盖本表对应的合成 Schedule，**不是** Workload Contract 的全形状验收。

## 5. 下一步性能闭环

| 完整 KDA 的数据边 | 当前最低证据 | 尚缺的工作 |
| --- | --- | --- |
| Q/K 归一化、decay、beta 与 32-token 耦合 | 独立准备组件在 B300 通过 H64 元素检查；两 MMA 版本单独计时 389.507 µs | 与状态/输出 CTA 融合或有证据地选择物化边界，避免七个中间张量往返。 |
| `state @ base_key` 与 P 前代入 | `61892f81` 的 H64/256 chunk 地址、公开 V/beta 到 RHS 映射在完整适用 CPU 门禁和 B300 全输出设备数值检查均通过；准备值仍为合成输入 | 接入真实上游 Q/K/G/prefix，并验证完整状态语义。 |
| `state * prefix_end + U @ final_key` | BF16 TMEM 回读/FP32 合并的两 chunk 合成状态在 B300 三种输入通过 | 接入准备组件的真实 FP32 prefix/final-key，并检验更多 chunk、尾块及原位状态别名。 |
| `state_after_update @ query` 与输出耦合 | `61892f81` 已在 B300 固定 H64/T8192 合成输入上通过公开 V/beta、token-major 输出及最终状态，AOT 0 spill | 真实准备值、逐 token 状态舍入、packed/tail 与完整 Workload 写回仍未证明。 |
| 逐 token BF16 状态舍入 | 独立 Workload oracle 和高保留失败反例 | 当前块代数只在块边界舍入；需精确路径或有证明且含 fallback 的输入 guard。 |
| 跨 chunk 流水与最终状态写回 | `92994711` barrier phase 复用获约 5.3% 配对收益；`5ac55ad4` 去掉死写回降到 2,999.349 µs；`671b2a90` 两槽 B 超前预取降到 2,835.220 µs；`db979435` 分角色循环再降到 2,462.161 µs；`5974e33e` 双槽 P 加向量暂存，在合成组件同机配对降到 1,953.451 µs；输出 sink 消融最多降约 151 µs | 优先准备值片上衔接与逐 token BF16/原位状态正确性，再检验 epilogue 重叠；以完整 Workload 的相同范围决定最终收益。 |
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
