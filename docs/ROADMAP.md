# 研究路线与 Milestones

[返回项目首页](../README.md)


项目的长期目标是建设一个**开放、可复现、能够跨硬件演进的 Kernel–Compiler 协同设计平台，
逐步帮助 Agent 发现优化手段、设计高性能 Megakernel，并在真实模型与应用中验证端到端收益**。
开源实现、可复现迁移、优化方法的发现与跨后端传递、Megakernel 设计支持共同构成研究路线；
论文与公开产物围绕各阶段已经验证的结果组织。

下表是后续目标与验收条件，不表示这些能力已经完成。当前发布状态见[状态页](../reports/current/STATUS.md)，
具体成果以绑定实验的报告与原始证据为准。

| 阶段 | 核心交付 | 验收条件与范围 |
| --- | --- | --- |
| **M1：可复现的开放平台** | 开放编译器、Agent 接口、评测流程与完整案例。 | 外部使用者能以明确的环境和有限预算，复现一次 Kernel 优化，以及一次编译能力变化的效果；保留成功、失败和能力边界。 |
| **M2：可复现的跨架构迁移** | 从现有非 NVIDIA 后端中选一个，整理完整迁移案例与可重复的步骤。 | 说明可复用的语义、必须重做的硬件决策、Agent 与人工的工作及成本；在目标设备上验证正确性，并相对该设备的合理基线测量性能。第二个迁移目标用于后续检验方法的可复用性。 |
| **M3：首个 Agent 设计的多阶段融合计划** | 为一个真实子图提供融合边界、数据驻留与调度的设计支持。 | 限定单 GPU、固定子图、静态依赖与有界调度；验证完整输出，对比合理的多 Kernel 基线，并以性能剖析解释收益和失败条件。案例包含实际的跨阶段依赖与资源协调。 |
| **M4：优化机制发现与复用** | 从 Kernel 与 Megakernel 实验中提炼优化机制，将可复用部分落实为变换、选择策略或编译能力，用于第二个不同任务或子图。 | 保留单一机制的前后程序、适用条件与反例；在未参与机制开发的任务上，对比相同模型与预算下的设计成功率、性能和开发成本，记录负收益、能力开发和审查成本。Megakernel 案例同时检查同步与资源生命周期。 |
| **M5：跨后端优化传递与 Megakernel 迁移** | 将已验证的优化方法传递到另一个编译后端或硬件架构，逐步扩展到 Megakernel 设计流程。 | 区分可复用的优化原理、必须重做的硬件实现与不支持的机制；在目标后端比较有无迁移经验的搜索结果与成本，验证正确性、性能和负迁移。各目标使用自己的基线、测量协议和校准。 |
| **M6：端到端接入与加速验证** | 将已验证的 Kernel 或 Megakernel 产物接入一个真实模型、推理框架或应用。 | 固定模型与精度、输入或 token 序列、硬件、框架配置和请求负载，检查完整输出；在相同条件下测量端到端延迟或吞吐，并验证代表性、边界和回退路径。 |

**推进方式：** 每阶段固定一个核心问题、主要工作负载和主要硬件目标，并约定投入上限。
新增能力应解除当前阶段的具体阻塞；动态任务队列、全模型执行和多 GPU 通信随任务证据逐步引入。
M1–M2 可先形成公开平台、迁移报告与论文材料，M3–M4 再形成 Megakernel 的核心研究结果。
M5 与 M6 在前置产物通过验证后按需求安排；已有合格产物可以先做端到端接入，无须等待所有迁移完成。

**优化方法的发现与传递：** M4–M5 采用[可执行优化知识迁移](OPTIMIZATION_TRANSFER.md)的研究设计，
优先关注 fusion、tiling 和 memory hierarchy optimization。机制提炼与目标验证沿用现有 Finding、
Compiler 和 Lab；[E/P 消融方法](OPTIMIZATION_TRANSFER_ABLATION.md)区分额外经验材料与变换调用的贡献。
先固定一个机制与一个源/目标组合，保留负迁移和发现/适配成本，再扩展到更多后端或 Megakernel。

**端到端收益单独验收：** 分开报告 Kernel 时间、完整子图或 GPU 执行区间、模型及服务时间，
区分编译初始化与稳态执行。生成式推理按负载报告首 token 延迟、后续 token 延迟和吞吐，保留重复测量与波动。
局部加速不自动代表端到端加速；无收益和回归同样保留，并说明收益被什么开销抵消。
接入通过已验证产物与目标框架的接口完成，Compiler 继续专注于编译，Lab 继续负责搜索与实验。

The roadmap progresses from an open, reproducible platform and documented cross-architecture porting,
through agent-designed multi-stage kernels, optimization discovery and reuse, to cross-backend transfer,
megakernel porting, and end-to-end validation in a real model or application. Transfer preserves an
optimization's rationale and preconditions while revalidating its implementation and benefit on each target.
Each milestone requires its own evidence; kernel speedups, portability, and end-to-end gains are evaluated separately.

