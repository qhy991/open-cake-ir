# Croqtile 与 Open-Cake：工程范围与研究定位

[技术报告](README.md) · [English](en/CROQTILE_COMPARISON.md) · [迁移研究](OPTIMIZATION_TRANSFER.md)

审查日期：2026-09-21。范围为公开文档、编译器和 Tuner 源码及测试定义；未构建运行
Croqtile，也未复现其性能或 Agent 正确率。这里比较工程机制，不给出性能排名。
固定版本：Croqtile `8ff94800498b450ad14f27dadd798ec19c96a0dd`，教程
`81d42df73dd80929c7e363015dbcffdfeada21ac`，Tuner
`f3d590602ea64d373d5fae6420266783beaa1198`；Open-Cake 对照为 `dbc67382`。

## 比较对象与共同点

Croqtile 是面向 tile、数据搬运与并行编排的语言/编译器，另有 CroqTile Tuner 管理
Agent 调优。Open-Cake 同样包含 Compiler 和实验系统。两者都让 Agent 操作结构化
GPU 程序、利用编译反馈迭代，并在真实设备上测量；“Agent + DSL + 静态检查”不足以
作为 Open-Cake 独有创新。[C1] [T1]

教程的部分页面仍称当前目标为 CUDA；本次源码已有 HIP、CPU 及异构编排路径。
主语言是 C++ 嵌入式 `.co`，同时已有 Python 接口和 MLIR CoIR 工具。因此不能把它
概括成“仅 C++、仅 CUDA、没有 IR、没有 Agent harness”。源码注册或测试定义存在，
也不等于本次验证了这些路径的实机成熟度。[C2] [C3] [C4]

## 主要差异

| 维度 | Croqtile / Tuner：本次观察 | Open-Cake：当前实现与限制 |
| --- | --- | --- |
| 编写与表示 | `.co` 的 `__co__`、mdspan、tile、并行层级、DMA/event；Python builder 也能构造程序；CoIR 提供 MLIR 表示与变换 | 受限 Python/JSON 构造规范 Schedule，Program 负责公共 ABI、stage 与张量绑定；Workload 独立拥有数学和 oracle |
| 形状 | 有匿名/符号维度、值编号和形状推导；不能静态确定的部分可生成运行期检查 | 当前任务通常固定 Workload case、具体 tensor ABI 和封存 launch；有循环和运行时索引，不等于同一产物支持任意动态 shape |
| 数据搬运与调度 | 直接表达 DMA/TMA、warp/warpgroup 角色、事件、多缓冲；编译流水线还包含 vectorization、memory reuse、fragment layout 和 fence 处理 | 显式记录执行分组、存储、访问与同步承诺；不引入一等 layout algebra，部分自动优化由 Triton/CuTe 等后端完成 |
| 检查与反馈 | SEMA、形状/类型推导、资源与断言分析，并有 `-static-check-info` / `-target-info` JSON；运行期检查可配置 | Assessment/Finding 区分结构、硬件、数据一致性和安全问题；guidance 不改变验收。两者的诊断/测试数量不可直接换算为安全覆盖率 |
| 硬件覆盖 | 核实了 CUDA/CuTe、CoIR→PTX、HIP、CPU、hetero 的注册/实现；HIP 文件列出 gfx1030/gfx1100 | 有 NVIDIA、Apple、AMD、Hygon、MetaX 的精确 Target 与平台路径，设备和测量覆盖分层；多厂商接入不证明任意 kernel 可直接移植 |
| Agent 控制 | skill 流程有 PROFILE→IDEATE→IMPLEMENT→MEASURE→DECIDE→STORE；v2 用 TypeScript/Pi SDK 编排，控制器重新构建和测量，保存轨迹及候选快照 | Run 冻结作者权限、源码、预算和评测；Study 预分配同一 Run 引擎。模型自述不作为验收依据；冻结 Run 外处理 Compiler 演进 |
| 作者工具 | v2 会话提供 read/write/bash，可按任务配置自行构建和分析 | 受限候选 Run 冻结作者工具和写入范围，由外部控制器编译、评测；公平比较需要对齐这项权限差异 |
| 测量与接受 | v2 调用配置的 build/profile 命令并解析 TFLOPS，按相对 best 的阈值 KEEP/REJECT；具体 oracle 和计时保证取决于命令 | Workload、固定 baseline、完整输出、计时质量、独立确认及审计是明确的共同边界；并非所有平台/Program 都已完成这些验收 |
| 经验与研究 | 有 DSL skill 注入、优化案例、轮次记忆与 mock harness 比较；这些是实际存在的经验复用机制 | 拟将额外材料 E 与可调用变换 P 分开做受控研究；协议已有实现，自动机制提炼与跨硬件收益仍未验证 |

Croqtile 表格依据为 [C2] [C3] [C4] [C5] [C6] [C7] [T1] [T2] [T3]。Open-Cake 对照位置为
[架构](ARCHITECTURE.md)、[Program](../src/open_cake_ir/compiler/ir/program.py)、
[Run](../src/open_cake_ir/lab/run_spec.py)、[Evaluation](../src/open_cake_ir/evaluation/core.py)
和[平台证据](ARCHITECTURE.md#9-平台能力与代表案例)。Croqtile 的 Python `Program`
与 Open-Cake 的完整执行计划不能仅因同名就视为相同抽象。

## 源码审查中值得注意的边界

Croqtile 的语言与编译器自动处理调度细节较丰富，尤其是符号形状和数据搬运表达，
值得作为 Open-Cake 易用性研究的参照。其动态形状文档也明确存在保守推导和运行期断言；
命令行允许关闭运行期检查，故“零开销”和“全检查”必须注明各自构建条件。[C5] [C6] [D3]

Tuner 已有真实的控制器与记录机制，不能称其没有独立测量。另一方面，所审 v2
`decide.ts` 使用固定 `0.995` 相对阈值，源码注明仍需实测噪声校准；`measure.ts` 以
命令退出和 TFLOPS 解析为接口，主循环中未见 Open-Cake 式的独立终点确认步骤。
调用者可以在测量命令中实现更严格的正确性和统计检查，所以这些观察只限定于该 v2
控制器，不代表整个 Croqtile 生态没有验证。[T2]

另有一条具体静态风险：v2 baseline 构建失败后仅给出 warning，仍调用 profile 命令。
若该命令能执行遗留二进制，可能把旧产物的数值当作起始基线。候选轮的构建失败则会
阻止其正式测量。此处是源代码路径审查，尚未构造运行复现；公平对照前应验证基线产物
身份和失败行为，不能把这一风险直接归结为所有既有结果错误。[T2]

教程的 H800 FP16 GEMM、FP8/MoE 优化记录可以作为机制案例，尚不能与 Open-Cake
的 B300/C550 数字横向排名。例如 MoE 教程注明 CUDA event、50 次 warmup 和 500 次
重复；Open-Cake 的对应 NVIDIA/C550 路径使用各自冻结的 CUPTI/MCPTI 协议。
计时区间、缓存处理、形状、精度和 baseline 未对齐前，不比较公开 TFLOPS、加速比或 pass@1。[D1] [D2]

## 对本项目的影响与后续对照

Open-Cake 应将研究重点放在**可复查的诊断到改写过程、跨目标重新验证，以及 E/P
处理的可控性**，而不是声称首先把 Agent、编译器或优化经验结合起来。Croqtile 的
自动编译变换与 skill 复用也不自动等同于本文提出的跨硬件受控迁移实验；反过来，
Open-Cake 的研究协议也不能替代实际的语言易用性、性能和迁移效果结果。

可优先开展三个有界工作：

1. 借鉴 tile/DMA 的表达密度和符号推导反馈，利用已有 frontend/guidance 改善作者体验；
   以真实失败、修复轮数和 token/时间成本评估，不为对齐术语而另建语言或 layout 系统。
2. 从真实变长任务判断是否需要扩展形状表达；类型、分析、lowering 与运行期义务一起设计，
   不把删除固定 shape 检查当作动态形状支持。
3. 分开比较“相同 harness 下的表示方式”与“各自原生调优栈”。先确认同一物理 GPU 上
   双方目标与工具链均准入，再固定任务、精度、oracle、baseline、计时、模型、权限、
   预算和重复次数；额外 skill/示例必须作为处理材料记录。共同 harness 接入仍是后续工作，
   本次未启动对照实验，也未向正在运行的 clean-start/E0 任务注入这些材料。

## 固定来源

[C1]: https://github.com/LancerLab/croqtile/blob/8ff94800498b450ad14f27dadd798ec19c96a0dd/README.md
[C2]: https://github.com/LancerLab/croqtile/tree/8ff94800498b450ad14f27dadd798ec19c96a0dd/lib/Target
[C3]: https://github.com/LancerLab/croqtile/blob/8ff94800498b450ad14f27dadd798ec19c96a0dd/tools/co-py/src/croqtile/__init__.py
[C4]: https://github.com/LancerLab/croqtile/blob/8ff94800498b450ad14f27dadd798ec19c96a0dd/tools/coir/README.md
[C5]: https://github.com/LancerLab/croqtile/blob/8ff94800498b450ad14f27dadd798ec19c96a0dd/lib/pipeline.cpp
[C6]: https://github.com/LancerLab/croqtile/blob/8ff94800498b450ad14f27dadd798ec19c96a0dd/lib/command_line.cpp
[C7]: https://github.com/LancerLab/croqtile/blob/8ff94800498b450ad14f27dadd798ec19c96a0dd/lib/static_check_info.cpp
[T1]: https://github.com/LancerLab/croqtile-tuner/blob/f3d590602ea64d373d5fae6420266783beaa1198/v2/README.md
[T2]: https://github.com/LancerLab/croqtile-tuner/tree/f3d590602ea64d373d5fae6420266783beaa1198/v2/src
[T3]: https://github.com/LancerLab/croqtile-tuner/blob/f3d590602ea64d373d5fae6420266783beaa1198/.claude/skills/croq-tune/SKILL.md
[D1]: https://github.com/LancerLab/croqtile-tutorial/blob/81d42df73dd80929c7e363015dbcffdfeada21ac/docs/zh/optimization/dense-gemm-fp16-from-naive.md
[D2]: https://github.com/LancerLab/croqtile-tutorial/blob/81d42df73dd80929c7e363015dbcffdfeada21ac/docs/zh/optimization/fused-moe-fp8.md

[D3]: https://github.com/LancerLab/croqtile-tutorial/blob/81d42df73dd80929c7e363015dbcffdfeada21ac/docs/zh/documentation/symbolic-shape.md

- [C1] 项目定位；[C2] 后端实现（含 `AMDGPU/hip_target.cpp` 与 `Hetero/hetero_target.cpp`）。
- [C3] Python 接口；[C4] CoIR；[C5] 编译流水线；[C6] 运行期检查开关；[C7] 结构化静态分析报告。
- [T1] Tuner v2；[T2] 主循环、测量与决策；[T3] skill 版流程。两版实现分开解释。
- [D1] GEMM 教程；[D2] MoE 教程；[D3] 符号形状。公共教程入口：[中文网站](https://lancerlab.github.io/croqtile-tutorial/zh/)。
