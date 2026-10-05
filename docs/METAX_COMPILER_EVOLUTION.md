# MetaX Compiler 与 Agent 的迭代实验

**目标：把 Cake 建成面向 MetaX 的 kernel 编写、分析与优化工具。** Agent 使用可检查的
Python/IR 接口表达计算与硬件选择，通过实测修正候选，再把可复现的限制提交给 Compiler
维护者。共享 Compiler 继续保持厂商中立，MetaX 的事实、指令和生成限制分别由 Target、
instruction contract 和 backend 负责。沿用具体存储与访问承诺，不引入 layout algebra。

本页规定首轮任务和演进方法。它是研究计划；正式任务、预算、参考权限和执行绑定仍由
checkout 外的 Workload、Study/RunSpecification 与证据记录负责。当前没有“Agent 随
Compiler 演进持续变好”的实验结论。

## 当前基础与缺口

在 `446c00e6892619fb48fa835e9ad2eef779f54a9f`，下表 12 个已有任务工厂的 starter
及 12 个结构不同的候选均通过解析、assessment 和源码 lowering。探测未调用原生编译器、
GPU 或 provider。外部报告为 `metax-evolution-roadmap-20261005/task-panel-cpu-readiness.json`，
同目录 `probe_task_panel.py` 与 `task-panel/` 保留生成方法和完整输入。

现有基础包括固定单阶段分组展开、平方距离联合分块、pointwise 输出分块、顺序循环区域和
保留 BF16/FP16 舍入的 tiled epilogue 融合。它们支持候选探索，不保证候选更快。

仍须保留以下边界：

- `xcore1002` 没有声明同步、scan、atomic 或 `broadcast_in_dim` 能力；共享 IR 出现一个
  原语，不等于 MetaX 已准入。当前 FP8 合约是有界的补偿 SIMT 实现。
- `maxnreg`、16 个 execution groups、pipelined unroll 和 warp specialization 仍有
  未完成的路线资格。不能丢弃作者声明或借用其他厂商的行为。
- 完整多 stage Program 的普通执行与独立 attribution，不等于共同优化 Run 的计时与
  attribution 已覆盖它。首轮性能任务使用单 dispatch；多 stage 测量另行补齐。
- 短算子计时曾未通过统一 5% CV。静态通过不解除这一设备验收要求。

## 第一轮：八个性能任务、四个回归控制

每个家族预留一个开发任务、一个验证任务和一个回归控制。R 为行数，C 为列宽；平方距离
使用 R/K/N。全部使用 FP32 和精确 Target `xcore1002`，通过现有任务工厂生成独立 Workload。

| 家族 | 开发任务 | 验证任务 | 回归控制 | 探索的结构变化 |
|---|---|---|---|---|
| Pairwise squared distance | R128/K512/N64 | R256/K1024/N64 | R128/K256/N32 | N 分块、K 分块、显式展开 |
| RMSNorm | R128/C4096 | R256/C8192 | R128/C1024 | 整行驻留与两遍循环的比较 |
| Softmax | R128/C4096 | R256/C8192 | R128/C1024 | 整行驻留与 max/sum/output 三遍循环 |
| SwiGLU | R128/C4096 | R256/C8192 | R128/C1024 | 输出列分块与执行宽度 |

八个开发/验证任务构成性能面板。四个小尺寸控制检查既有行为、正确性和计时稳定性，单列
报告，不计入优化成功率分母。**这些宽度都是 2 的幂，不是 tail case。** 非整除边界的
语义测试继续由 Compiler 合同测试负责；新增 tail Workload 须先扩展其任务合同和验收。

这份清单尚未取得设备资格。源码探测为每个家族保留了由所属规则单独拒绝的反例：平方距离
的 pipelined unroll、RMSNorm 的寄存器上限、Softmax 的 warp specialization、SwiGLU
的 16 组执行宽度。反例用于解释搜索边界，不列作合法候选。

正在运行的 R512/K256/N64、R768/K256/N64 四组 pilot 独立保留，其作者已接触相关材料，
不能重新命名为本轮未见验证任务。上表“验证”仅表示计划角色；先审计历史 Run、作者
上下文、材料和生成实现的接触情况，再决定是否能称为未见形状。发现接触时，保留并报告；
可在正式冻结前另选验证形状，冻结后不能按结果换题。

开发阶段可以观察开发任务的完整诊断并改进 Compiler。验证任务仅在接口、材料、选择规则
冻结后交给新作者 Run；验证结果用于判断效果，不反馈给同轮其他作者。维护者可预先完成
基线资格检查，但必须记录接触，并与隔离作者的权限区别开。

## C0 冻结条件

只有下列条件均有证据时，才把某个干净提交称为首轮可用基础 C0：

1. 选定任务的 starter 与至少一个结构不同的完整候选可以构造、验证和 lower；反例由
   正确的规则拒绝。保持 oracle、公开 ABI 和数值顺序约束。
2. 在精确 MACA 工具链和已验收隔离环境完成原生编译。八个性能任务通过全部输入分布的
   设备正确性、独立 profiler，以及预先写下的 A/A、配对计时和统一 5% CV 要求。
   四个回归控制检查正确性与既有拒绝；其计时是诊断结果，超标继续报告为覆盖限制，
   不用于性能提升结论。
3. 在作者搜索前固定每题的强基线 bundle、参考权限、测量区间、cache reset、计时器、
   提升阈值、重复数、运行顺序、预算和停止规则；所有比较共享这些条件。
4. 固定模型及已观测身份、Claude Code/scaffold、上下文压缩规则、工具链和 Target。
   provider 资格通过，候选与测量均经过既有隔离和 MACA 锁。
5. Compiler 的相应合同测试、完整 Corpus Gate 和独立审查通过；完整源码与 Executor
   固定到提交。尚未准入的能力列入任务说明，不作为隐含搜索权限。
6. 预定面板中每项的资格状态都有记录。报告八个性能任务与四个控制的实际覆盖；未通过的
   项目不能计作优化失败，也不能被静默删除以形成“全部通过”。

资格检查失败时保存原始失败并停止该项，定位到 Compiler、工具链、Executor 或测量
合同的实际所有者。环境修复后沿用旧检查不构成验收；按仓库规则在后继环境/提交重新验收。
发生设备或测量层面的共同失败时，停止受影响批次。若需要缩小性能面板，先发布明确的
后继方案与选择理由，再开始搜索；不能将缩小后的结果冒称原 8 题结论。

## 每一轮如何积累

流程为：**冻结 Compiler、模型和 scaffold → Ralph 作者循环 → 封存证据与公开笔记 →
维护者归纳 → 独立审查后继 Compiler → 下一轮验收。** 一个 Run 内不能改 Compiler。

作者每轮提出一个可证伪假设，生成结构不同的候选，读取本 Run 的历史诊断，再经静态过滤、
共同 Evaluation 和新鲜确认。恢复或上下文压缩后也必须读取历史。记录应包含：前一候选/
Turn、改变的机制、预期现象、实际诊断、证据状态、反例，以及受阻时建议的负责模块。

`TASK.md`/`AGENTS.md` 由 [task package](../src/open_cake_ir/lab/task_package.py) 生成；
不能让作者修改它们。公开笔记写在允许提交的 candidate 源码注释或 docstring 中，并随候选
保留。它们是可审核的简短实验记录，不是私有推理过程，也不能替代 Evaluation Receipt。
不新增未经授权的可写记忆文件，不向作者暴露其他组的实现或结果。

后继任务可绑定 [learning scaffold](../contracts/scaffolds/python-artifact-optimization-learning-v1.md)。
StateCard 的 `optimization_history` 从本 Run 的已留存观察生成，包含未选中的已测候选、
拒绝与变换结果；它给出覆盖和截断数量。重放必须从更早的事件和已验证收据重建该视图。
搜索中的最佳记录仍需独立确认。作者的简短说明随原始候选源码留存，维护者要核对其引用，
不能把文字建议直接转成已证实的 Compiler 缺口。

Run 封存后，维护者检查确切候选和事件。重写规则进入 Compiler pass；适用条件和参数选择
进入 Lab recipe；缺失合法性进入 Verifier；真正缺失的表达能力进入 IR 与分析；设备执行
和计时问题进入 Executor/Evaluation。单次泛化错误或拒绝次数不自动授权新增原语。
在现有 Finding/result 中写清 promotion disposition；`No promotion` 是有效结论。

新增原语必须一起定义类型、效果、数值顺序、地址关系、合法性、lowering 和反例，并满足
P1–P8。进入后继版本后才可重新实验。旧 Run 保留原始拒绝和失败，不能回填为新版成功。

## 怎样证明变好

| 比较 | 固定内容 | 回答的问题 |
|---|---|---|
| Compiler floor | 同一批保存的候选分别在 C0/C1 编译，输入、oracle、工具链和评测相同 | Compiler 自身扩大了多少可编译范围，改变了多少资源与性能？ |
| Fresh-agent search headroom | 每版使用新作者上下文，相同任务、材料、预算和固定基线 | 新接口/诊断使 Agent 找到好解更快、更稳定吗？ |
| 跨 Run 经验效果 | 固定 Compiler，显式分配经验保留与经验重置，其他条件相同 | 改进来自积累的经验多少？ |

floor 面板至少包括未修改 starter、已确认候选和保留的被拒候选。不能只选新版擅长的
输入；旧版拒绝、新版可编译列作覆盖变化，没有旧版时间的条目不计算时延加速比。显式
pass 不会在重编译时自动调用，其价值须由作者搜索比较或另一个声明的候选比较来观察。

主要报告每题确认成功率、达到终点的轮数/编译数/设备评测数、token 与时间、相对固定
基线的确认性能，以及失败和测量缺失。相同预算、重复次数和终点规则必须预先固定。
同一题反复调参的 improvement 不能单独证明跨任务或跨版本进步。

当前 [StudyPlan](../src/open_cake_ir/lab/study_plan.py) 的四组语义是 E/P，要求共同
Compiler 和其他控制；不能把 Compiler 版本伪装成某个 E/P 组。跨版本比较先完成固定基线
准入的修正与测试，再分别冻结 C0/C1 批次及关联报告。现有
[`validate_paired_baseline`](../src/open_cake_ir/lab/admission.py) 的非 incumbent 路径
将基线与当前 starter lowering 绑定；跨版本需要明确保留旧基线的身份、来源和可执行合同，
不能为了通过该检查重新编译替换固定基线。已有独立 Run/结果记录仍是执行与证据所有者。

## 必需的矩阵计算里程碑

FP32 行算子可建立迭代方法，但产品还必须覆盖 **BF16/FP16 GEMM+bias 和带显式舍入的
epilogue**。为每种 dtype 选择具体矩阵形状，补循环累加、数值分布、固定参考实现及新鲜
确认；当前 Target 中的 dot 证据仅覆盖有界 64×64×64 诊断，不能推出任意 GEMM 资格。

[`fuse_tiled_epilogue`](../src/open_cake_ir/compiler/tiled_epilogue.py) 已能保留 producer
循环与私有 BF16/FP16 cast，生成单 dispatch 候选。先验证这个完整候选；需要多 dispatch
时，先实现并验收共同 MACA Program 测量，记录全部 stage、模块、相关性与测量区间。
不得用一个 stage 的计时代替完整计算。异步访存、同步及更细硬件控制的加入，以独立原生
正例和受阻 Schedule 为依据；不因接口关键词存在就扩展资格。

## 证据与职责入口

- [Target](../compiler/targets/xcore1002.json)、[MetaX backend](../src/open_cake_ir/compiler/backends/metax.py)、
  [Compiler 职责](contexts/compiler/CONTEXT.md)、[分支与独立评审](DEVELOPMENT_BRANCHES.md)。
- F-2026-10-03-003、F-2026-10-04-005：输出分块、固定分组展开的既有设备证据；
  [联合变换合同](SQUARED_DIFFERENCE_REWRITES.md)保留其适用边界。
- [F-2026-10-05-002](../findings/2026-10-05-002-triton-sequential-loop-regions.json)：顺序循环
  软件和离线编译证据；[F-2026-10-05-004](../findings/2026-10-05-004-maca-route-qualification-dispositions.json)：
  未完成的路线资格；[F-2026-10-05-005](../findings/2026-10-05-005-maca-program-run-measurement-coverage.json)：
  完整 Program 优化 Run 的测量缺口。
- [迁移设计](OPTIMIZATION_TRANSFER.md)、[E/P 消融](OPTIMIZATION_TRANSFER_ABLATION.md)与
  [融合合同](EPILOGUE_FUSION_PASS.md)继续维护各自的语义，本页不复制新的执行或证据协议。
