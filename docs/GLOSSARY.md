# 术语表：英文名与普通话解释

代码、合同和诊断使用下列英文名；这里是它们的统一文字定义。先读 [中文 Wiki](wiki/README.md)，遇到词再查，不必一次背完。

## Project names

### CAKE

CAKE 指论文中的系统与研究思路。项目只引用其已公开说明，不把本仓库的实现当成论文未公开源码。

### open-cake-ir

本仓库项目名。它独立实现编译器，并提供依赖该编译器的研究实验系统。

### Cake IR

项目中用于描述 GPU 执行计划的语言。精确字段和语义由本仓库的代码、Schema、Authoring Contract 与发布版本定义。

### Open Cake Compiler

独立的编译器：检查 Schedule、给出分析，并为允许的计划生成目标源码。它不负责调用 AI、申请 GPU 或宣布实验成绩。

## Compiler terms

### Schedule

执行计划：写明 Buffer、线程分工、操作依赖、循环、坐标、存储和生成方式。负责者是 Compiler 的类型与语义；它回答怎样执行，不替代 Workload 的数学题目。

### Target

明确的硬件目标及其已声明能力，由 Compiler Revision 绑定。它不是随便选一张 GPU 的名字，也不允许静默换成另一架构。

### Finding

编译器给出的一条具体诊断，带代码、位置、解释，以及是否阻止接受或生成。负责者是对应检查规则；诊断也可以只是提示。

### Assessment

一个 Schedule 在明确目标与编译器版本下的检查结果。它把结构接受与后端生成资格分开，不等于 GPU 正确性或 Lab 运行资格。

### Lowering

按确定规则生成目标源码和映射信息，或选择已核验的固定源码。负责者是 Compiler Revision；它还没有经过工具链编译或 GPU 执行。

### Compiler Revision

固定的一版编译器：绑定语义、目标、检查、分析、源码生成、语料预期和校准覆盖。负责者是发布 lock；一个正在变化的 Git 分支不是这个身份。

### Executor Revision

Lab、评测、证据工具及 Python/分析工具环境的固定组合，由已发布 descriptor 负责。
使用前分别核对实际源码和实际主机；它与 Compiler Revision 是不同身份，不代表所有 Workload 都已接通执行路径。

### Corpus

编译器的正例与反例集合，并记录每例预期表现。负责者是发布版本引用的 manifest；零散示例或普通单元测试不能替代它。

### Corpus Gate

对整套 Corpus 的发布检查，比较实际结果与已审查预期。负责者是 Gate 报告与发布批准；它不证明 GPU 计算或速度。

### Calibration

用实际测量建立的、范围明确的估算依据。负责者是绑定目标和版本的校准合同及验收记录；换机器、形状或版本，不能自动沿用估算。

## Workload and Research Lab terms

### Workload Contract

任务约定：数学、输入范围与生成方法、标准答案、容差、测试行和输出要求。它是这些事实的负责者，不是简单的 benchmark 参数表。

### Study Contract

实验约定：比较哪些写程序的环境，如何分配尝试，预算多少，读什么参考，怎样汇总结果。它定义实验设计，不是一次实际执行的可变配置。

### Claim Scope

一轮实验允许作出和使用哪类声明，由 Study 固定。当前范围区分系统资格、单产物优化和科学分析；README 中的一句话不能扩大它。

### Provider Feature Policy

AI 程序能使用哪些功能，以及怎样解释它的事件。负责者是 Authoring Environment 的固定版本；它本身不决定实验结论或授权外部操作。

### Authoring Environment

作者实际得到的完整写程序环境：工具、参考、诊断和提交方式。负责者是 Study 引用的固定定义；实验比较不能只按代码语法给环境命名。

### Campaign

一次实际执行 Study 的实验，由 CampaignLock 固定所用版本、机器条件与证据位置。它不同于 Study 定义，也不同于其中一个独立 Run。

### CampaignLock

preflight 生成的一次执行约定，固定 Workload、Compiler、Executor、provider、工具链、机器和 custody。后续主线变化不会自动修改这个 lock。

### TaskPackage

Lab 根据 CampaignLock 确定性生成的 `TASK.md` 与 `AGENTS.md`。它们向 AI 说明题目和规则，不保存可变分数、已花预算或当前最好方案。

### Ralph Controller

Lab 外部的迭代控制器，决定是否开始下一步，提供 StateCard，统计预算并记录停止原因。它不代替 AI 编写候选，也不代替评测器判断正确性。

### StateCard

由已保存 Run 状态生成的一份本轮视图：用掉多少预算、还剩多少、有哪些已知反馈。它是输入给下一轮的事实摘要，不是 AI 自己修改的规则。

### Run

Study 中一次独立分配的重复尝试，也是比较实验的单位。一个 Run 可以含多个 Turn；它不是某个终端进程或 broker job。

### Turn

Run 中一次 provider 交互及其候选提交机会。由 Run 协议定义，不另用 round 或 checkpoint 表示同一个东西。

### Candidate

已提交并固定内容的候选产物，带内容身份和来源关系。负责者是 Evidence 中的对象与事件；仍能编辑的工作区文件还不是固定候选。

### Artifact Promotion

单产物优化中，按确认性评测和既定规则选出候选。负责者是该 Study 的分析；选中一个候选不等于两组环境的统计比较，也不是生产部署。

### Endpoint

Study 预先指定、按 Run 观察的结果，如预算结束时是否已有合格候选。它不是任意一行漂亮日志。

### Estimand

科学实验开始前明确“究竟想估计什么”。它需要说明对象、比较方式和异常情况怎么处理；不是实验结束后才挑选的数字。

### Checkpoint

在约定预算处观察一次 Run：尚未到达、已到达但无合格候选、或已有合格候选。由 Run 协议负责，不预测未来成绩。

### KernelSeed

通过固定小范围确认性检查、可进入后续 portfolio Study 的候选种子。负责者是 Lab；它不自动成为通用生产算子。

## Evaluation terms

### LaunchableCandidate

已经具备明确目标、入口、启动说明及完整角色标记产物的候选，可进入共同评测。负责者是 Evaluation；只有一个输出文件名还不够。

### Evaluation Protocol

Workload 拥有并被 Study 引用的检查与测量方法，包括测试行、检查目的，以及先正确性后计时的顺序。它不定义 Study 的统计分析。

### Logical Evaluation Attempt

对一个固定候选的一次逻辑评测，保存相关 broker jobs 及受限的零工作重提规则。负责者是 Evaluation；不能用它换掉原候选或任意重试。

### Evaluation Receipt

一次候选评测留下的启动、判对、样本和处置记录，按追加方式保存。它不是一轮实验的最终结论，也不是候选可以自己填写的分数。

### Portfolio Artifact

把固定语义键映射到已封存专用候选的产物，并规定不支持输入时怎样拒绝。负责者是 Evaluation；它不是一个可随意更新的运行时形状表。

### Measurement Quality

从保留样本判断测量是否稳定，针对明确边界，由 Evaluation 负责。计时不稳定与计算答案错误是不同事实。

## Evidence and report terms

### Evidence Object

Evidence 存储中的不可变字节对象，用内容身份识别。文件路径只提供位置，不等于内容身份。

### Event Ledger

按顺序追加的事件记录，引用证据对象并保存观察到的转换。负责者是 Evidence；它不是覆盖写入的状态文件或随意拼接的日志目录。

### Terminal Archive

一个 Run 结束时的完整可复查材料，成功、失败和协议偏离都保留。负责者是 Evidence，没有另一套专门美化成功记录的归档类型。

### Integrity

审计检查归档字节、引用和事件顺序是否完整。由 Run Audit 得出；完整不代表协议遵守、答案正确或权限历史可信。

### Filesystem Custody

按当前 custody 规则核对路径的拥有者、权限和单写入者要求。由 Run Audit 检查；今天修改文件权限不能证明过去一直满足要求。

### Protocol Adherence

实际执行是否遵守固定 Study 与运行前提，由审计根据事件重建。它不同于文件完整性或候选质量。

### Run Audit

从保留证据重建完整性、custody、协议遵守和 Endpoint 观察。它是纯复查步骤，还不是 Study 的最终结果。

### Study Report

把固定分析计划应用于合格 Run Audit 后生成的报告，给出纳入情况、结果、不确定性和缺失。它只能估计事先定义的目标。

### Claim View

从已经接受的 Study Reports 生成、可删除重建的结论视图。它不单独拥有结论，也不应在 README 中手工维护另一份。

## Evidence strength and scope

这些标签说明不同范围，不是一条自动升级的阶梯。每个更强结论需要自己的记录。

| 标签 | 已说明的事 | 不能自动推断 |
| --- | --- | --- |
| `proposed` | 已审查的提案可进入实现与检查 | 已发布或 GPU 通过 |
| `released` | 对应固定版本通过发布流程 | 工作负载正确或更快 |
| `structurally_expressible` | 词汇能描述该机制 | 后端、GPU 或完整程序覆盖 |
| `lowerable` | 允许生成源码或选择固定源码 | 工具链编译或 GPU 运行 |
| `compiled` | 固定工具链生成目标产物 | 答案正确或速度更好 |
| `correctness_qualified` | 对应参考和输入检查通过 | 稳定计时或其他形状 |
| `timed` | 对应边界存在有效样本 | 已解释瓶颈或服务加速 |
| `profiled` | 有与正确候选对应的分析记录 | 无 profiler 时的速度或因果归因 |
| `operator` | 覆盖明确的算子或固定 Program | 模型或服务请求 |
| `model_forward` | 覆盖声明的模型前向路径 | 服务吞吐、延迟和可用性 |
| `serving_e2e` | 覆盖完整请求和 token 轨迹 | 其他部署或负载 |
| `not_run` | 该评测有意未执行 | 失败 |
| `unknown` | 证据不足以判断 | 一个已经证实的负结果 |
| `invalid` | 证据或协议不能支持目标解释 | 所有负结果都无效 |

延迟必须说明测量边界。加速比应同时给出对应基线和候选的绝对时间，见 [结果解读](wiki/results.md)。
