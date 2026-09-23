# 跨硬件的可执行优化知识迁移

[技术报告](README.md) · [English](en/OPTIMIZATION_TRANSFER.md) · [消融实验设计](OPTIMIZATION_TRANSFER_ABLATION.md)

我们提出以**带适用条件的优化机制**作为跨硬件知识迁移单元。Agent 在源平台探索
operator fusion、tiling 和 memory hierarchy optimization，将可复用的语义改写沉淀为显式
Compiler pass，将使用时机与参数选择留给 Lab。目标平台复用规则，通过自己的 Target、
lowering 和实机评测决定能否应用、是否正确以及是否有收益。

本框架的研究贡献在于连接**经验发现、可执行化、目标验证与反馈**：经验既能作为 Agent
的推理材料，也能作为可调用的优化工具；成功、拒绝和负迁移共同约束后续复用。
研究问题是：这种积累能否降低新硬件上的探索成本，以及收益分别来自经验说明还是变换调用能力。

![源平台的优化经验经机制说明 E 和可调用变换 P 两个独立控制的入口进入目标 Agent；目标检查与实机验证始终保留](figures/optimization-knowledge-transfer.svg)

*图：E 控制额外的机制说明与案例，P 控制显式变换的调用权限。二者由冻结的 Study 分配，
共同使用 Run 执行与评测。图中不包含实测性能结果。*

## 一份规则，按目标实现和验证

1. **发现与提炼。** 在固定任务下保留一个机制的前后程序、正确性义务、适用条件、
   反例与观察结果；沿用 Finding 和原始证据，不建立第二套经验账本。
2. **显式改写。** 能确定性表达的部分成为带匹配与拒绝条件的 pass，返回完整候选或原因。
   通用 IR 改写共享一份实现；目标特有的指令、存储和同步仍由后端负责。
3. **目标实例化。** Agent 重新选择 tile、执行分组和 pipeline 参数。变换条件与 Target
   声明、backend preflight 交叉检查；硬件具备某项能力不等于当前 lowering 已支持它。
4. **验证与反馈。** 以目标平台的固定基线、外部 oracle 和计时协议验收候选。
   来源平台的最优参数、成本估计和加速比不自动继承。

例如，`Add → BF16 中间结果 → SiLU` 可以在中间结果私有、无外部消费者时尝试融合，
消除中间张量的全局写回与重读，同时保留 BF16 舍入。可迁移的是这条数据流改写及其条件；
具体分块与资源分配在目标平台重选。缺少必要 lowering 时报告能力缺口，正确但更慢时保留原方案。

## 国产卡迁移与目标架构协同优化

**迁移的终点是在目标架构上形成自己的 Kernel–Compiler 优化闭环。** NVIDIA 上的知识
可以作为起点，但不是所有目标必须先经过 NVIDIA 的前置条件。这里同时存在三个问题：
平台能否接入、既有机制能否帮助目标搜索、目标硬件能否产生新的优化与编译能力。
三者分别验收；“国产卡”不是一个统一 ISA 或可共享全部参数的 Target。

### 平台接入具体接到哪里

[架构章节](ARCHITECTURE.md#cake-irdsl-与-triton-的层级)区分 Program、Schedule、生成源码与
工具链。接入新硬件时，公共 Workload、typed IR、通用合法性规则与实验协议尽量复用；
平台声明并验证自己的精确 Target、指令与数值合同、代码对象、工具链、加载/启动 ABI、
host admission、计时和 profiler。一个 Target JSON 只表达已知能力，不会自动创造这些实现。

| 已有国产目标 | 当前编译与执行路线 | 不能由兼容外观推导的事实 |
| --- | --- | --- |
| Hygon BW1101，`gfx938` | Cake Schedule → Triton 源码 → DTK/HCU 环境的 Triton → AMDGCN/HSACO → HIP 加载与运行 | 与 AMD 共用部分 HIP/HSACO 机制，不代表共享硬件上限、数值验证、计时校准或性能 |
| MetaX C550，`xcore1002` | Cake Schedule → Triton 源码 → MACA Triton 工具链 → MCFATBIN 中的原生设备 ELF → MACA 加载与运行 | `GPUTarget("maca", 80, 64)` 的 `80` 是兼容 API 值，不是 NVIDIA SM80；物理 target、codegen family 与 API 身份分别检查 |

这些路线来自本仓库的 [DCU 设计](dcu-gfx938-design.md)、[C550 专题](metax-c550.md)及
[gfx938](../compiler/targets/gfx938.json) / [xcore1002](../compiler/targets/xcore1002.json)
目标声明。它们复用 Triton 源码接口，实际调用各自环境提供的编译后端，**不把 NVIDIA
PTX/CUBIN 直接搬到国产卡执行**。当前 MetaX 工具链提供 TTIR、TTGIR 和 mcfatbin；
没有的 LLVM/PTX 产物不伪造。新平台若能使用现有 emitter，可增加目标与必要的平台适配；
若现有路线无法表达所需行为，再实现有需求依据的生成机制。Triton 并非所有新硬件的必经路。

### NVIDIA 经验中哪些可以迁移

| 知识或产物 | 迁移方式 | 目标侧必须重新决定或验证 |
| --- | --- | --- |
| 算法与语义 | 复用任务数学、数据依赖、oracle 与明确的数值义务 | 目标 dtype、舍入、原子语义及完整输出；不能为了移植放宽既定容差 |
| 融合、分块、复用机制 | 提供机制说明，或调用带前提和反例的显式 pass | 数据私有性、生命周期、合法性、资源可实现性及相对本机基线的收益 |
| 经验参数 | 作为有来源的搜索候选或假设 | tile、执行组、pipeline 深度、融合边界必须在目标卡重选，不复制源最优值作为答案 |
| NVIDIA 指令与资源协议 | 保留优化意图，按目标能力重新表达；无对应能力则拒绝 | `tcgen05`、TMA、TMEM、`setmaxnreg` 等不能仅改 target 名字就成为国产指令 |
| 诊断方法 | 复用“检查数据流、定位同步、比较瓶颈”的方法 | 指标定义、可观测性、计时区间与缓存重置；不能照搬 NCU/CUPTI 数据或模型常数 |
| 源平台成绩 | 保留为来源证据与研究背景 | 目标重新建立基线、正确性、测量质量和性能证据，源加速比不继承 |

例如从 NVIDIA 观察到“融合私有中间张量可减少显存往返”，可迁移的是数据流改写及其
舍入、消费者和生命周期条件。在 C550 或 DCU 上，融合也可能增加寄存器压力、降低并发而
变慢；目标实验应允许保留未融合基线。反过来，目标卡发现的 tiling 或数据搬运机制也可
成为其他平台的候选经验，不把知识流向永久固定为 NVIDIA → 国产卡。

### 从接入到在目标架构上优化

下面是每个具体目标的推进顺序；它细化[研究路线](ROADMAP.md)，不新增一套运行模式。
已有能力按各自证据复用，无需为叙述重新执行所有阶段。

| 阶段 | 实际工作 | 结束时能回答的问题 |
| --- | --- | --- |
| 1. 建立本机基线与观测 | 让固定 Workload 经过真实编译、加载、完整正确性检查；声明计时区间、状态重置和噪声门，建立可用 profiler 证据 | 这个任务在这个目标上能否可靠运行和比较？只通过正确性时不推进性能结论 |
| 2. 实例化可迁移机制 | 在允许的参考访问范围内导入机制材料或显式变换，按目标资源重选参数并检查拒绝条件 | 该机制在目标上能否实现？一次改善属于目标局部优化；迁移贡献还需下面的受控对照 |
| 3. 直接在目标卡优化 kernel | 固定 Compiler、Workload 和工具链，生成结构不同的 Schedule/Program，静态过滤，实机核对、计时与剖析，独立确认候选 | 本机相对基线是否改善，成本多少，瓶颈是否变化？无需等待所有 NVIDIA 机制移植完 |
| 4. 用目标反馈演进系统 | 运行外根据证据修改 Target、lowering、verifier、校准或可复用 pass；提交并验证后开启后继 Run | 系统改变解决了哪个已观察缺口？不是在同一冻结 Run 中边测边改编译器 |
| 5. 扩展到应用 | 在强种子基础上检查更多形状、尾部、dispatch/fallback，再接入目标框架 | 完整 Program、模型或服务是否受益？单 kernel 加速不替代这一验收 |

阶段 3 与 4 构成后续持续交替的两个循环：**内循环在目标硬件上优化程序，外循环改进
让这种优化可表达、可诊断和可验证的系统。** 因而后续确实要直接面向特定国产架构优化，
既包括更好的 kernel，也包括该架构需要的编译与评测能力。源平台经验只是内循环可选的
输入；目标原生发现可以从自身 Workload 与硬件资料开始。

外循环按最早缺口选择负责位置，而不是看到慢就扩展 IR：

- 候选参数或调度不合适：改 Schedule/Program，选择工作归 Lab。
- IR 能表达但后端不能兑现：改对应 lowering 和 preflight。
- 重复出现可建模的非法行为：补 verifier 规则及能由该规则拒绝的反例。
- 真实硬件机制无法用现有语义表达：共同补 primitive、类型、效应、分析和 lowering。
- 估计与实测系统性不符：用该目标的测量校准；缺少覆盖继续明确报告。
- 可重复的语义优化：提炼有 guard 的显式 pass；一次性的收益可以记录 `No promotion`。

硬件特化留在相应 Target、backend、平台实现和有证据的适用范围内；共享 Compiler
不分裂成每个厂商一份。若目标优化要求修改厂商 Triton/SDK 本身，那是独立的下游工具链
工作与版本变更，不能把本项目现有 outer loop 写成已经能自动优化整个厂商编译栈。

### 当前处于什么阶段

本节依据报告已有的 [2026-09-21 平台快照](ARCHITECTURE.md#9-平台能力与代表案例)解释阶段，
不产生新实验结论。DCU 已有固定任务的运行与优化记录，但短 kernel 的测量分辨能力限制
部分性能结论；C550 已有固定单 kernel 和部分完整 Program 正确性、单 kernel 计时/
profiler，以及 M17 GEMM 的本机 tile 优化案例。完整 Program attribution 不能替代普通
Run 的完整程序性能验收，C550 的完整 provider/Ralph 优化闭环仍以平台报告的待验收边界为准。
因此平台接入已有实质工作，目标本机优化也已有局部例子；尚不能说所有阶段全部完成。

### 现有证据的四个独立范围

![机制编码、跨目标正确性、C550 本机调度收益与尚缺的迁移增益对照](figures/transfer-evidence-layers-v1.png)

*图：四格来自不同任务和收据，是证据范围的阅读图，不是同一机制从 NVIDIA 流向 C550
的实验轨迹。所列时间是固定形状单 kernel 的成对计时；`86/86` 的完整 Program 正确性
收据没有计时样本。图不提供跨卡性能比值。*

具体记录分别回答不同问题：

| 记录 | 已验证的范围 | 尚缺的证据 |
| --- | --- | --- |
| [私有 BF16/FP16 epilogue 融合 pass](EPILOGUE_FUSION_PASS.md) | 显式组合两个 Schedule、保留舍入并删除私有中间量的 store/reload；[Finding](../findings/2026-09-10-006-explicit-private-epilogue-fusion.json)保留类型、源码和 CPU 模型检查 | `verified_by` 尚为空；没有该 pass 的目标 GPU 正确性、计时或跨硬件收益 |
| [C550 的 GQA/MLA/MoE 完整正确性](metax-c550.md) | 各自精确绑定 C550 的 Workload 保留原 B300 合同的数学、形状、输入、oracle 和容差；17 个变体、86/86 case 的完整输出复核零不匹配 | 这些收据计时为 `null`；未验证把原 NVIDIA Schedule 原样移植后的性能，亦无材料/变换处理对照 |
| [C550 M17 GEMM 的 M tile 改动](metax-c550.md) | 固定 `M17/N128/K2048` 下仅将 M tile 64→32，独立 `matrix-fib-m17-tile32-confirmatory-8c0cad53-v1/result.json` 通过五类正确性 case 与测量质量门；基线/候选中位数 72.448/58.368 μs，1.241228× | 这是 `local_serialized` 范围内的显式本机 authoring 对照，不是 provider 优化 Run，也没有“给 Agent NVIDIA 经验”与不给经验的对照 |
| [E/P 四组迁移协议](OPTIMIZATION_TRANSFER_ABLATION.md) | 材料 E 和变换权限 P 的隔离、分配与审计已有软件测试 | 还没有共同目标、基线和预算下的实机迁移收益实验 |

**同一来源机制在不同目标上的实际边界。** B300-M2 的 `pairwise_sqdist` Run 在 FP32
`R=1024,K=1024,N=64` 下把 K 维按 256 分块；事件 17 的独立确认让基线与候选均通过
五类输入的前后正确性检查，CUPTI 成对计时质量通过，本机中位数为
411.7945/53.856 μs（7.646×）。这是来源卡的固定基线收益。目标 Workload 重新绑定各自
精确 Target，保持算子、张量、五类输入、oracle 与容差一致；历史 Schedule v1 的
`warps` 字段显式适配为 v2 `execution_groups`，不在 admission 时暗中翻译。

| 目标与固定来源调度 | 已到达的验证边界 | 尚不能主张什么 |
| --- | --- | --- |
| Hygon BW1101 `gfx938`，K=256 | `main@0fe3a447` 接受并 lower 来源候选与本机 starter；Hygon Triton 3.6.0 均生成 HSACO，封存与声明五 case 的 CPU 配对准入通过 | 设备尚未加载、判对或计时；八卡上观察到其他 GLM 进程时未抢占 |
| infplane AMD `gfx1151`，K=256 | 同一 Workload 的五 case 前后正确性全部通过；本机 `hip_dispatch` 新鲜确认质量通过，starter/候选为 **191.954/466.222 μs**，10 对均为 starter 更快 | 这是**确认的负迁移**，不是 B300 的 7.646× 在 AMD 上重现；gfx1151 两种计时器的绝对值尚未对齐，只解释同一 assay 内的配对关系 |
| MetaX C550 `xcore1002`，K=256 | 固定草稿源码 `023d0db4` 接受并 lower 两臂；MetaX Triton 3.6 均生成 MCFATBIN、各声明两个隐藏指针；kernel 投影后封存和 CPU 配对准入通过 | 五个容器的 MACA 锁未与宿主共享，未运行设备正确性与 MCPTI 计时；该草稿也尚未独立评审合入 |
| Apple M2 `apple_gpu_family8`，同一 `pairwise_sqdist` | Compiler 指出 Metal 当前不实现 tile loop、K 索引和跨循环归约 | 没有 Metal 二进制或设备结果；换 target 名称不会产生缺失的 lowering |

AMD 的两个预先写下的目标参数后继也没有产生合格收益：K=128 完整判对且质量门通过，
但相对同一 starter 更慢；K=64 完整判对但候选 CV 超过 0.05，显示的中位数只作描述。
K=256 编译产物报告 256 VGPR/线程与 260 字节本地存储，starter 为 126 VGPR、零本地
存储；这提示资源压力，不能单凭它证明慢的原因。来源、原始样本及失败记录分别保存在
checkout 外的 `open-cake-ir-evidence/transfer-b300-bw1101-20260923/`、
`open-cake-ir-experiments/transfer-b300-gfx1151-20260923/` 和
`open-cake-ir-experiments/transfer-b300-c550-pairwise-20260923/`。

Metal 能表达的另一个来源机制是 B300 `silu` 的执行组 1→4：其 B300 独立确认在本机为
2.432/2.112 μs（1.1515×）。同数学、输入和 oracle 的 M2 Schedule 已编译成 Metal
archive，五类输入在两臂前后均判对；但 `fixed_baseline_paired_metal_v2` 的 cohort CV
远超预定的 0.05，故**没有合格的 M2 加速结论**。原始记录在
`open-cake-ir-experiments/transfer-b300-m2-silu-20260923/`。这些实例分别证明可表达性、
正确性或负迁移，不是 E/P 材料和 pass 权限对 Agent 的因果效果。

M17 的收据名称必须含 `tile32-confirmatory`：同目录的
`matrix-fib-m17-confirmatory-8c0cad53-v1` 是原产物自比较，结论为 `close_null`，
不能代替 tile32 对固定基线的独立确认。原始收据与 86 份 Program 复核均在 checkout 外
`open-cake-ir-evidence/metax-parity-20260920/`，仓库内的链接是它们的专题阅读入口。

**“NVIDIA 知识帮助国产卡优化”仍是待验证的研究假设。** 必须在同一目标上冻结共同
Compiler/工具链、基线、作者和预算，对比有无额外机制材料与变换权限；先补齐的后端能力
由所有组共同使用。目标本机优化回答“能否优化这张卡”，E/P 研究回答“迁移知识额外带来
多少帮助”，系统后继实验回答“新编译能力带来什么变化”。三种收益分别归因，协议见下文
及[消融方法](OPTIMIZATION_TRANSFER_ABLATION.md)。

## 可消融的贡献

底层 Compiler、后端能力、oracle 与测量流程固定，只改变两个入口：

| 组别 | E：额外机制说明与案例 | P：可调用变换 | 研究作用 |
|---|---|---|---|
| E0P0 | 无 | 禁用 | 无额外迁移材料的搜索基线 |
| E1P0 | 有 | 禁用 | Agent 依据经验自行改写 |
| E0P1 | 无 | 启用 | 仅提供变换接口及使用它所需的最小合同 |
| E1P1 | 有 | 启用 | 完整的可执行经验迁移 |

E0P1 的接口本身携带知识，不能称为“无知识”。这里分离的是**额外机制材料**与
**变换调用能力**。比较固定预算内的合格优化成功率、达到终点的成本和本机基线相对性能；
来源机制的发现与适配成本另计。具体隔离、失败处理与统计量见[方法附录](OPTIMIZATION_TRANSFER_ABLATION.md)。

共同预算默认固定轮数、时间和编译/评测次数，token 保留为成本统计；固定 token 上限是
Study 可显式选择的另一种资源约束，所有组必须一致。已有平台上的正确性或局部优化收益
只说明目标验证路径可用，不构成 E/P 处理效果或跨硬件知识迁移收益。

## 实现基础与研究定位

当前已有[显式融合、输出列特化与 Triton 执行宽度变换](../src/open_cake_ir/compiler/passes.py)，
以及[带前提和反例的经验材料入口](OMOE_TRANSFER.md)。原型已接入完整 Program、独立 Run、
材料与 pass 权限隔离、共同的 Run 装配、预分配 E/P Study 和审计统计。软件测试验证了协议路径；
普通任务、incumbent 与 QSA Cake 候选已迁入公共路径。多 stage 的公共正确性执行路径已
接入 CUBIN、HSACO 和 MCFATBIN 模块。MACA 已有独立的完整 Program attribution 入口，
其有限设备观察见 [C550 报告](metax-c550.md)；普通优化 Run 的完整程序测量仍限于 Triton/CUDA，
Metal 组合尚未接入。跨硬件迁移收益仍待受控实验，自动机制提炼不在当前实现范围内。

可组合变换与调度复用已有明确基础，包括 [MLIR Transform](https://mlir.llvm.org/docs/Tutorials/transform/)
和 [Transfer-Tuning](https://arxiv.org/abs/2201.05587v2)。本设计聚焦 Agent 经验如何经显式改写与目标验证
持续积累，并用受控实验区分材料、工具和二者交互的贡献。


[Croqtile 工程对照](CROQTILE_COMPARISON.md)说明它已有 DSL、编译器变换、Agent tuner
和 skill/案例复用，不能以这些共同能力单独主张新颖性。本研究拟比较的是显式机制在目标
任务上的适用性、材料与调用权限的独立处理，以及相同验证条件下的收益和成本；相关协议
已有实现，优势与跨硬件效果仍需实验。当前对照是源码审查，不是性能排名。
