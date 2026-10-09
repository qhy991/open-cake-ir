# Evolve：Kernel 与 Compiler 协同进化设计

状态：设计提案，2026-10-09。依据源码 `76549b98`。本文中的新类型、接口和命令均为拟议设计，尚未实现。

合入前已对照 `main@2356886a` 更新实施状态。共同问题由[研究主题](RESEARCH_AGENDA.md)负责，开发与硬件独立评测遵循[Bench 标准](BENCHMARK_PROTOCOL.md)。本提案补充外循环编排设计，不替代这些现行合同；新 schema 仍需单独实现和验收。

`evolve` 将任务优化、经验审查、Compiler 修改和独立版本比较组织成有界、可恢复的工作流程。它复用现有 Run、Study、Evaluation、Evidence 与 Finding。Compiler 继续独立使用，也继续不依赖这些研究流程。

研究问题是：任务探索产生的机制进入 Compiler 后，能否让相同 Agent 在未参与开发的任务上，更可靠、更省成本地获得高性能 kernel？研究者判断机制是否成立、比较是否有效、哪些结论可以对外主张。Agent 承担候选搜索、证据整理、实现和验证。

## 1. 从使用方式开始

用户通过项目 skill 指定计划和停止边界。以下是预期交互，并非当前可运行命令。

```text
$evolve 按 /research/cake/evolution-plan.json 完成下一轮。
使用计划指定的发现任务、精确目标、参考权限和预算。
最多形成一个后继版本，完成规定的比较后停止。

$evolve 继续 /research/cake/evolution-001。
先确认哪些工作已经完成，再继续尚未尝试的分配。

$evolve 只分析 /research/cake/evolution-001 的经验与缺口。
读取现有证据，给出 Finding 和 promotion disposition，不启动新实验。
```

面向自动化只增加一个工具入口：

```text
python tools/evolve.py inspect --workspace /research/cake/evolution-001
python tools/evolve.py advance --workspace /research/cake/evolution-001 --execute
python tools/evolve.py report --workspace /research/cake/evolution-001
```

`inspect` 只读，返回下一项工作及其依据。`advance --execute` 完成一项已经获得授权且满足前置条件的机械工作，或返回需要 Agent 判断的工作包。`report` 从已保留记录生成视图。继续执行仍调用 `advance`，不另设语义不同的 resume 路径。没有执行意图时，CLI 只显示下一项工作。

Skill 在用户授权范围内循环调用这些入口。计划中的预算约束工作范围，但计划文件本身不能授予原本没有的执行权限。已获得的授权不反复询问。仅要求分析时，任何阶段都不得启动 Provider 或 GPU。

初版默认一次调用完成一轮。多轮执行必须在计划中声明轮数和累计投入上限。目标、模型、参考访问及资源预算在执行前确定，不能从上一个聊天环境中隐式继承。

## 2. 系统分工

| 负责者 | 拥有的事实 | evolve 怎样使用 |
| --- | --- | --- |
| Workload Contract | 算子语义、输入、oracle、容差 | 引用既有合同，发现冲突后交回该 owner |
| Compiler | Program、Schedule、类型、分析、pass、lowering | 在冻结 Run 外提出后继修改 |
| RunSpecification | 一次执行的版本、作者、权限、预算及评测绑定 | 所有实际搜索继续由它授权和约束 |
| Study | 分组、任务总体、控制变量、估计量和分析 | 分配版本对照，解释全部分配结果 |
| Evaluation | 正确性、计时范围、质量门、profiler | 产生独立收据，禁止由作者自报代替 |
| 独立硬件 Bench | 题目、输入生成、oracle、容差、强参考、计分与计时 | 接收固定 Compiler 生成的候选，独立判定并输出原始报告 |
| Evidence | 候选、事件、原始收据和终态 | 回放与审计事实 |
| Finding | 维护判断、实现提交、验证范围 | 决定改动归属及是否可以关闭 |
| OptimizationKnowledge | 机制材料、证据引用、变换引用 | 冻结每组作者可获得的经验 |
| Incumbent | 精确任务上已确认的实现 | 支持工程优化，不能冒充 Compiler 学到的能力 |
| Evolution workspace | 上述产物之间的轮次关系 | 推导下一项工作，引用既有结果 |

新增外循环没有自己的 Candidate、Evaluation receipt、GPU 队列、接受判定或成绩账本。每个平台继续通过自己的 Target、backend 和 Executor 声明能力。外循环不扩大任何硬件资格。

## 3. 一轮工作的具体合同

| 阶段 | 输入 | 退出条件 | 允许的后续动作 |
| --- | --- | --- | --- |
| 准备 | 精确目标、干净提交、任务与权限、预算 | 所需 Run 全部完成准入，冻结发现分配 | 开始发现 Runs |
| 发现 | 冻结的 RunSpecifications | 全部分配已有终态，或按计划确认缺失 | 已封存记录可提前审查；运行中或未知状态阻止阶段完成 |
| 审查 | 成功、失败、无收益候选及 profiler | 每项选中线索有归因、证据、最小探针与处置 | 实现、保留待议或 No promotion |
| 修改 | 已接受的 Finding proposal | 一个可说明的后继提交，或仅经验变化，或不修改 | 执行适用的软件验证和独立评审 |
| 验证 | 固定提交与原始反例 | 适用合同测试、Corpus Gate、评审及运行准入完成 | 冻结比较分配 |
| 比较 | 固定对照、后继条件、独立任务、共同合同 | 预分配单元全部可分类，保留失败和缺失 | 生成报告并记录 disposition |
| 结束 | 比较报告与维护决定 | 本轮结果引用完整，下一轮起点明确 | 停止，或在剩余授权内进入下一轮 |

冻结发生在两处：发现执行前冻结其 Run；后继提交具体化后、看到比较结果之前，冻结该轮全部比较分配。不得为等待后继提交而提前填写虚构版本，也不得根据比较结果补选对照或增加有利重复。

一轮不保证产生 Compiler 提交。仅改经验材料时，比较因素是经验；没有可晋升机制时记录 `No promotion`。若没有新处理条件，跳过没有信息增量的版本比较，并在已有结果中说明原因。

实现合入、设备资格、独立任务收益分别记录。修复正确性问题可以有价值，即使它没有加速。负性能结果不会触发自动回退或改写 Git 历史，也不会被改名为测量失败。

## 4. 经验审查与分配

审查器从同一 Run 读取完整候选关系、动作、拒绝诊断、评测和 profiler。输出只是带定位信息的审查材料，不自动生成事实成立的 Finding。

| 观察 | 最小判别工作 | 修改位置 |
| --- | --- | --- |
| 已有表达没有被正确使用 | 用最小合法 Schedule 检查作者假设 | Candidate；可复用指导归 Lab |
| 成功优化存在稳定的完整改写 | 对比前后图、访问、数值义务及反例 | Compiler explicit pass |
| 合法 IR 在指定 backend 不能生成 | 保留 assess、preflight 与 emission 的具体结果 | Backend lowering |
| 所需执行语义无法表达 | 列出现有表达及其不满足的硬件承诺 | IR、类型、效应、分析与 lowering 一起改 |
| 非法行为漏报或合法行为误报 | 证明目标规则拥有该合同，排除借来的拒绝 | Verifier |
| 估计排序持续违反可比较测量 | 检查目标覆盖和差异是否超过噪声 | 目标自己的 empirical cost model |
| 参考、oracle、输入或计时不一致 | 分离数学、候选与测量原因 | Workload 或 Evaluation |
| 只有一个更快实现，机制尚不清楚 | 保留原始证据和未决假设 | Incumbent，或 No promotion |

通用编译错误不证明 IR 不足。未知原因的生成源码 compile refusal 应先进入 triage。单个例子可以证明具体缺陷，但不能证明普遍有效的性能规则。

Finding 继续拥有来源、观察、建议、实现与验证。每个拟晋升 pass 明确前提、完整输出、拒绝理由和反例。最佳 tile 参数仍由调用者或 Lab 决定，不因一次实验而写成跨目标常数。

现有 `OptimizationKnowledge` 保存材料和 transform 引用。材料的来源可以指向 Finding，但作者只看到获准的内容，不能借来源路径访问隐藏低层实现或其他组的结果。

## 5. 工作区与唯一事实来源

源码只保存 skill、实现、测试、稳定设计和 Findings。所有实际计划、执行绑定、Run、Evidence 与实验报告放在 checkout 外。

```text
/research/cake/evolution-001/
  plan.json                         EvolutionPlan，冻结的范围与引用
  rounds/001/
    discovery.json                  准备好的 Run 引用集合，写一次
    maintenance.json                Finding、现有处置记录与后继材料引用，写一次
    comparison.json                 已冻结 Study allocation 引用，写一次
  allocations/...                   现有 Study 分配与 Run 工件
  views/round-001.json               可重建报告，引用原实验结果与 disposition
```

三个阶段 binding 只记录关系，不复制原始事件、测量数值或 owner 的合同字段。尚未具体化的阶段文件不存在。写入完整文件后原子发布；相同内容的重试返回原记录，不同内容的第二次写入被拒绝。

维护结论不在此目录创建第二份权威 `result.json`。`MaintenanceBinding.change.disposition` 指向已有调查结果的 `promotion_disposition`，或 Finding 的维护决定。现有结果的生成与维护入口是唯一写入者。`lab/reporting.py` 增加窄的读取投影，按来源 owner 的已知格式读取，缺少所需结论时返回待判断，不能凭空补一个成功状态。

当前工程结果没有一个覆盖所有历史文件的统一 schema。初版只接入实际使用的结果格式，如 `compare_rewrite_artifacts` 的 `result.json` 和 Finding record；不迁移全部旧记录。对应结果 owner 按既有追加或后继约定记录新 disposition。EvolutionReport 只引用原 locator 和结论范围，不能编辑该结论。

Finding 引用使用稳定 id，并记录选中该决定时的来源提交。后续 `implemented_in` 和 `verified_by` 由 Finding owner 追加，状态视图读取其生命周期。binding 不复制这些字段，也不因 Finding 关闭而重写；原轮次的冻结输入保持原样。

不保存权威的 `current_phase` 或 `completed: true`。阶段状态由有效绑定、原生 Run audit、Finding 和验证记录推导。删除 `views/` 后可以重建相同结论。

同一 workspace 只有一个协调写入者。任务作者、维护者和评审者各写自己的目录或 worktree；协调者在读取边界汇总。重复协调者只能读状态或等待，不能抢写他人的活动目录。

如需改变目标、任务总体、权限或研究问题，建立引用原工作区的新计划。不得覆盖已冻结 binding，也不得让计划修订隐式恢复用完的预算。

## 6. 类型与接口草案

下面是领域接口草案。引用类型表示由对应 owner 校验的 locator；实际序列化遵循 owner 的版本边界。它们不是新的 digest 目录，也不是可运行源码。

```python
@dataclass(frozen=True)
class EvolutionPlan:
    plan_id: PlanId
    driver_source: Commit
    discovery_templates: tuple[RunTemplateRef, ...]
    comparison_design: StudyPlanRef
    initial_source: Commit
    stop: EvolutionStopPolicy
    concurrency: WorkerLimit

@dataclass(frozen=True)
class DiscoveryBinding:
    round_id: RoundId
    runs: tuple[PreparedRunRef, ...]

@dataclass(frozen=True)
class SourceChange:
    successor: Commit
    findings: tuple[FindingRef, ...]
    disposition: DispositionRef

@dataclass(frozen=True)
class MaterialChange:
    knowledge: KnowledgeSelectionRef
    disposition: DispositionRef

@dataclass(frozen=True)
class NoChange:
    disposition: DispositionRef

Change = SourceChange | MaterialChange | NoChange

@dataclass(frozen=True)
class MaintenanceBinding:
    round_id: RoundId
    change: Change

@dataclass(frozen=True)
class ComparisonBinding:
    round_id: RoundId
    allocation: StudyAllocationRef

Binding = DiscoveryBinding | MaintenanceBinding | ComparisonBinding
Next = Ready | Running | NeedsJudgment | Uncertain | Complete
DispositionRef = FindingDecisionRef | ExperimentResultRef
InvocationScope = ReviewOnly | ExecuteWithinPlan

def inspect(workspace: Path) -> EvolutionView:
    """Read native authorities and derive Next; perform no writes."""
    raise NotImplementedError

def bind(workspace: Path, binding: Binding) -> BindingRef:
    """Validate owner references and publish one immutable relationship."""
    raise NotImplementedError

def advance(workspace: Path, *, scope: InvocationScope) -> Next:
    """Perform one eligible mechanical action under the existing authorization."""
    raise NotImplementedError

def report(workspace: Path) -> EvolutionReport:
    """Project audited outcomes and dispositions without accepting artifacts."""
    raise NotImplementedError
```

`SourceChange` 可以通过其已有 disposition 引用同时更新的经验选择，不把它误报成纯 Compiler 处理。材料单独变化用 `MaterialChange`，没有变化用 `NoChange`，不使用含义不清的空 successor 字段。

`EvolutionStopPolicy` 只拥有外循环的最大轮数、累计投入上限和预定停止规则。Run 内的次数、时间和确认预留仍归 Run。每次启动前，从原始 accounting 累计已用投入并预留下一分配所需额度。用量缺失时报告已知小计，不能按零计算或宣称满足无法核实的总上限。

新工程 Run 沿用共同规范的三小时默认墙钟预算，包含确认预留；token 仅记账，不设置总量、每轮限额或 token 接受门。外循环累计上限使用明确授权的时间、工具和设备资源，不借此恢复工程 token cap。科学研究若改变预算，在执行前统一声明，历史 Run 保持原合同。

`NeedsJudgment` 包含明确任务、证据引用、允许读取的材料、输出位置和完成条件。Skill 据此完成归因或实现，再调用 `bind`。它不要求程序把任意自然语言决策当作可执行命令。

`InvocationScope` 来自可信调用方的用户指令或已保留的会话授权，不从计划文件和实验内容解析。它限定 workspace、允许动作和停止边界；共同启动入口核对它。`ReviewOnly` 不允许派发。恢复时可复用仍有效的既有授权，无法确认授权范围则只返回只读状态。CLI 的 `--execute` 表达调用意图，不绕过宿主权限，也不要求新增独立审批服务。

## 7. 中断恢复与重复执行

恢复能力分两层：外循环能重建阶段；Study 或共同启动入口能确认某个分配是否已经尝试。两层都不恢复已经死亡 Run 的内部搜索状态。

当前 `execute_study` 会在发现已有尝试时拒绝整批执行。将它改为按原始分配逐项核对，只执行尚未尝试的单元。新增 `lab/dispatch.py` 作为唯一启动声明 owner，提供 `dispatch_once(prepared_run, scope, runtime_factory, execute_run)`。它只管理派发尝试，不管理 Run 内部状态。

`study_execution.execute_study`、`launch_task.py` 和 evolve 的具体工具组合都调用 `dispatch_once`；matrix 继续调用共同单任务入口。Runtime factory 的构造调用和 Run 执行均置于它的声明保护之后，不能先构造 Provider 或申请 GPU，再调用启动保护。

初版 evolve 要求已存在可用的 Provider、工具链及主机资格。缺少资格时返回前置工作，按既有资格入口独立完成；不在 Run 准备中隐式发起 live qualification。这些资格工作若有成本，仍进入总投入记录。这里的派发保证覆盖已准备好的 Run，不声称给所有历史资格工具增加了重试语义。

启动 owner 在任何可能消耗 Provider 或 GPU 资源的动作之前，原子创建分配自己的启动声明。声明绑定原 RunSpecification 和唯一 attempt id。并发调用中只有一个获得声明；其他调用读原分配。失败后的声明不能被删除来制造“从未启动”。

| 观察到的事实 | 恢复动作 |
| --- | --- |
| 没有启动声明，也没有执行证据 | 可在原授权和预算内启动 |
| 启动声明存在，原进程可确认仍在运行 | 附着或等待，不再启动 |
| Run 已封存且审计通过 | 读取原结果，推进未尝试的分配 |
| 有明确的启动失败、终止或 preemption | 保留为原分配结果；按计划处理缺失 |
| 声明存在，但进程和结果都无法确定 | 标记 Uncertain，不自动重投 |
| 原始证据损坏或 custody 不满足 | 报告前置条件，不修复权限来通过检查 |

初版接受保守恢复：即使进程在写声明后、真正启动前死亡，该分配也不会自动重跑。若环境提供持久的 job 查询，才可以凭同一 attempt id 查清状态。PID 消失和日志为空都不能证明没有发生付费调用。

这提供的是不重复派发的保守保证，不承诺分布式 exactly-once。自动跨主机接管和死亡 Run 内续跑不在初版范围。

工程后继可以在新的授权与身份下重新尝试，但保留原失败、成本及参考访问。科学分配不做事后补跑。运行失败后，预分配的其他单元仍可按计划继续。GPU 释放和 preemption 始终遵守安装的 gpu-infra lease lifecycle。

## 8. 同目标版本比较如何接入 Study

保留 `matched_search` 作为唯一 Study kind。将现有 Study 内部的 E/P 特定约束与公共分配、执行、审计分开，使用两个封闭的设计变体：

```python
SearchDesign = TransferEP | RevisionComparison

@dataclass(frozen=True)
class RevisionComparison:
    conditions: tuple[Condition, Condition]
    controls: MatchedControls
    population: TaskSplitRef
    analysis: RevisionAnalysis
```

`TransferEP` 继续要求原来的四组、材料约束和源目标关系。`RevisionComparison` 固定同一精确目标，声明两个条件的允许差异；不能用任意差异字典放宽控制变量。初版比较初始锚点与当前后继，父版本对比仅在执行前另行声明。

Study 模板固定任务总体、角色、预算规则、终点、对照选择规则和分析。每轮只创建该 Study 下的 allocation cohort，不按 Compiler commit 复制 Study successor。cohort 在执行前一次性确定全部 Run，条件、重复编号、随机顺序及实际绑定随后不可改。

具体 Compiler、Executor、Provider 资格和材料仍写入各 RunSpecification。allocation 记录其完整成员及角色并校验匹配关系，不另存一份可独立修改的版本值。准备输出必须能证明“这些 Run 是执行前分配的”，不能事后收集有利 Run 组成比较。

新 schema 在现有 Study owner 内处理。若 Run assignment 需要区分 cohort，则以显式 schema 后继增加 `allocation_id`，并校验 Study、allocation 和 Run 成员关系。沿用既有身份机制，不新增 hash-only 账本。旧 schema 与已冻结研究在原提交回放，当前创建入口一次性迁移，兼容保留在真实输入边界。

协调器固定在计划中的 `driver_source` 提交运行。每个 Run 的执行与语义审计在该 Run 所属的干净提交中启动独立进程，不能把旧代码导入当前 Python 进程，也不能用今天的审计器重判旧记录。协调器只接收版本明确的报告投影和原始 locator。

首个正式版本比较的 C₀ 取自新分配协议、启动恢复与报告接口验收后的提交。这样 C₀ 与后继都支持同一研究执行合同。更早提交若没有该接口，只能经单独审查的外部适配或已有 artifact 对照接入；不得修改旧 checkout 以伪造协议支持。

需要修改的重点是 `study_plan.py` 的约束、`study_execution.py` 的逐分配推进及 `study_analysis.py` 的分析选择。公共 outcome 读取可复用，E/P 的系数不能用于版本比较。当前任务去重、参考权限、预算和确认门继续适用。

## 9. 评测的两条主线

版本评测固定各平台独立 Bench 的 commit，当前维护入口包括 c550-bench、metal-bench 和 bw1100-bench。Bench 的 oracle、计分与计时不能依赖 Cake 的 Verifier、Lab 或 Provider 才能判定。Lab 的 Evaluation 负责开发搜索与证据对接，不能用内部 Workload 结果替换 Bench 的独立判定。报告同时绑定 Compiler 与 Bench 版本；Bench 或参考合同改变时开始新的比较段。

| 主线 | 输入与控制 | 结果解释 |
| --- | --- | --- |
| 固定程序 | 固定同一 Program、目标和工具链；分别编译；共同评测 sealed artifacts | 表达覆盖、正确性及生成质量变化 |
| 固定预算搜索 | 未参与改动的任务；相同模型、scaffold、预算和材料；新会话和独立重复 | 版本变化是否改善获得合格 kernel 的能力 |

固定程序评测直接走现有 Compilation 与 Evaluation 的 owner，不伪装成有 Agent 的搜索 Run。需要补一个有界的共同 artifact 评测入口；不能直接把当前限 B300 特定 ABI 的工程脚本包装成跨平台能力。

旧版无法表示或生成的程序标为该版覆盖缺口。固定程序延迟对比只在两版都能执行的预定公共子集成立，同时报告全集覆盖。不得把新版 IR 翻译成旧版可接受形式后称为同输入比较，也不得给拒绝任务填无限 speedup。

固定预算搜索允许两版用各自合法表达实现同一 Workload。记录全部分配的成功率、首次正确成本、确认性能和失败类别。相同种子不意味着 LLM 路径相同，必须保留独立重复与任务间差异。

每个任务固定外部基线 artifact，并在同一 assay 内重新配对测量。不能仅复用历史延迟。发现阶段可以继承已授权 incumbent，版本曲线的基线不随 incumbent 移动。基线跨版本 handoff 仍需通过既有 ABI、目标和权限准入；无法准入时先明确接口缺口，不替换对照。

若研究目标是纯 Compiler 效果，材料和选择策略保持固定。新增 pass 的调用接口属于 Compiler 能力，最小接口说明随能力提供；额外案例材料另算处理变量。对经验本身的归因可在后续独立设计中复用 E/P，不要求每轮运行一个无边界的全因素实验。

### Compiler 与 Executor 的源码绑定

当前两者身份都来自 checkout commit。比较两个提交时不能只写“Compiler 变了”，也不能手填一个旧 Executor id。

初版搜索比较默认报告“系统版本处理效果”，列明 Compiler、Executor、作者工具、经验和工具链的实际差异。只有评测与其余控制可证明保持一致时，才收窄为 Compiler 归因。检查源码差异有助于审查，但不能代替共同评测资格。

固定程序路径优先把两版产物交给一个有资格的共同评测环境。若目标的 artifact 接口不能跨版本使用，就报告限制，不引入不受检查的旧版本加载模式。

### 统计与成本

主要终点是冻结预算内经新鲜确认的合格优化成功率，按任务等权。性能报告逐任务的确认比值与重复分布；成功子集的汇总明确标为条件性结果。基础设施缺失与协议内失败区分，全部预分配单元保留。

正式置信区间和重复数由 Study 在执行前规定。任务不足的试点只报告观察与范围。多轮滚动面板属于自适应开发轨迹，不能把每轮显著性当成独立确认。

成本分开报告发现、维护实现、评审、校准、搜索与独立确认。若要声称 evolve 比继续搜索更省成本，另设固定 Compiler、继续搜索的对照，并匹配包括维护在内的总资源条件。不能把预先写好的 pass 当作零成本。

## 10. 独立任务与作者隔离

计划使用三个任务集合：发现集用于修改；滚动验证集用于观察版本回归，允许影响下一轮；最终测试集用于未参与开发的确认。未见形状和未见算子族分别声明。现有规范化语义、shape 和 ABI 去重继续使用，但它不证明任意数学等价。

最终测试由独立评测执行环境持有。维护 Agent 不挂载测试任务包、结果目录和作者会话。比较作者只收到其条件允许的材料，不能继承维护 Agent 或另一条件的记忆、检索缓存与 incumbent。靠提示词要求“不要看”不构成隔离。

把最终测试交给“同一个可读取全部目录的 agent 的子代理”也不构成独立隔离。若环境无法限制读取，仍可做工程回归，但不能声称盲测或 E/P 隔离成立。已存在于公共 checkout 或已被维护者读取的任务，按实际暴露情况声明。

每次最终测试在选定版本和分析前完成冻结。结果一旦向维护者开放并用于修改，该集合转为已暴露材料；后继泛化主张需要新的保留集或预注册的顺序检验方案。不能通过换 workspace、Run id 或 agent 名字恢复“未见”身份。

初版建议每轮只跑滚动验证，在计划结束或预定里程碑运行一次最终测试。这个选择允许观察持续进步，也避免每轮消耗最终测试的独立性。

隔离由现有 Lab Provider 资格和 workspace admission 负责验收，执行由平台的 OS 进程或账户边界负责。沿用 `claude_isolation.py` 的 allowlist 工作区与实际读取探针；`author_home.py` 的新 home 只能隔离会话和技能，不能单独证明文件读取隔离。共同准入检查在 Study dispatch 前完成。

最终测试的任务、输入和结果由独立账户或主机上的评测 custodian 持有。它只接收已选定的版本和冻结的 Study allocation，按原合同运行后发布报告。维护进程没有该目录的读取权限，作者工作区也不能访问其他条件。证明材料来自实际隔离探针与已存在的资格记录，不以新建一份“隔离已启用”布尔字段代替。没有这个执行环境时，初版只开放滚动工程验证。

## 11. Skill 与代码的最小组织

| 位置 | 变更 | 范围 |
| --- | --- | --- |
| `skills/evolve/SKILL.md` | 新增 | 用户入口、诊断方法、阶段交接与停止条件 |
| `tools/evolve.py` | 新增 | CLI、具体 TaskLab 组合与受控派发 |
| `lab/evolution.py` | 新增 | 计划引用、不可变 binding、状态推导 |
| `lab/study_plan.py` | 修改 | 明确 E/P 与版本比较设计、稳定模板与分配关系 |
| `lab/study_execution.py` | 修改 | 按原分配逐项检查，调用共同派发入口 |
| `lab/study_analysis.py` | 修改 | 共享 outcome 读取，按明确设计计算估计量 |
| `lab/dispatch.py` | 新增 | `dispatch_once` 拥有唯一启动声明；既有入口共同调用 |
| Lab Provider 资格与 `claude_isolation.py` | 沿用与窄扩展 | OS 读取隔离、作者材料及实际探针的准入 |
| `lab/reporting.py` | 窄扩展 | 从已有调查结果与 Finding 读取 disposition，不新增写入者 |
| 现有 Findings 与 Knowledge | 有证据需要时窄改 | 保持一个事实来源 |

`lab/evolution.py` 不导入 `tasks`、维护 agent SDK 或 GitHub 客户端。具体任务组合留在工具层。它不按时间步骤拆成 prepare、mine、promote 等大量转发模块。模块按拥有的事实划分。

首版 skill 的必要指令草案如下：

```text
name: evolve
description: 在 open-cake-ir 中执行有界的 Kernel–Compiler 进化轮次，
  从已封存实验提出维护改动，并完成预定版本比较；也可只审查现有证据。

先读取 workspace 的计划和 inspect 结果，确认用户授权范围。
从原始 owner 恢复上下文，不把上一段聊天或缓存状态当作事实。
对 NeedsJudgment 检查成功、拒绝、无收益和 profiler 证据。
先确定 owner，再提出最小改动及反例；不从错误次数推断 IR 不足。
维护任务使用独立 worktree，按现有分支规范集成。
修改 Compiler 前检查 P1–P8，修改后按适用合同验证并取得独立评审。
执行前读取 gpu-infra 的 lease lifecycle，沿用现有 GPU 分配。
遇到 Uncertain 不重投。把它交回启动 owner 核对。
比较作者使用隔离的新上下文与组内材料。
每轮记录 promotion disposition；No promotion 与负收益正常结束。
计划用尽、前置条件不满足或需要新的研究决定时，保留证据并停止相关工作。
```

部署时 skill 链接本仓库的 canonical owners，不复制 AGENTS、Target 能力表或 GPU 租约规则。上述待实现入口就绪前，不把草案安装为可自动执行的 skill。

## 12. 验收与实施顺序

每个实现单元先形成 commit，再在固定提交的独立 worktree 验证。Compiler 改动执行完整 Corpus Gate 和独立评审；CPU fixture 不能代表 GPU 性能验收。

评审和集成遵循[开发分支规范](DEVELOPMENT_BRANCHES.md)。沿用已有检查、PR 与评审记录，不为每次变更新建 release 文档或机器校验 reviewer 字段。

| 单元 | 交付 | 可观察的验收条件 |
| --- | --- | --- |
| 1. 接齐证据审查 | 复用已修复的当前 Run 诊断读取，补 triage 与成功候选审查输入 | 现有读取回归继续通过；未知 compile error 不被当成已证明 verifier 缺陷 |
| 2. 只读外循环 | `inspect`、binding 与报告投影 | 删除视图可重建；相冲突的第二次 binding 被拒绝；No promotion 能正常结束 |
| 3. 分配恢复 | 共同启动声明、逐项执行 | 并发启动只有一次派发；崩溃窗口不重投；已完成和未知单元都不被重跑 |
| 4. 版本研究 | RevisionComparison、cohort 绑定、分析 | 执行前分配；不匹配控制被拒绝；所有失败、缺失与旧版不支持案例留在报告 |
| 5. 有界 artifact 对照 | 指定一个精确目标的共同评测接入 | 两版本 artifact 对同一固定基线通过同一正确性和测量路径 |
| 6. evolve skill | 编排前述入口，独立 Agent 行为演练 | 清空会话后能继续；不泄漏材料、不重复付费、不热改 frozen Run |
| 7. 首轮实机试点 | 一个目标、一个有界机制、计划好的任务集 | 从实验到 Finding、后继提交与独立比较全链可复查，保留负结果和成本 |

恢复验收在写启动声明前后、派发前后、运行封存前后注入故障。测到的保证是“不自动重复派发”，不能把没有造成重复的几个成功样例称为 exactly-once。

研究验收应主动尝试：给测试任务改名、移动 incumbent 分母、让一组继承维护会话、改变容差、增加后验重复、把旧版拒绝从分母删掉，以及把 Executor 改动标成纯 Compiler 效果。每项都必须被拒绝或明确降级结论。

## 13. 设计取舍

选择由 skill 协调判断、由既有 owner 执行、从产物推导进度的方案。外循环状态不会与 Run 终态竞争。用不可变阶段 binding 保存尚不能从事实中推导的决定。

采用持久控制器方案中“启动前留下意图、派发不明时禁止重投”的要求，但将它放在共同启动 owner，而不是 evolve 自己的事件账本。初版不要求新建工作流服务或分布式 supervisor。

纯 prompt 串命令虽然代码少，却把失败恢复和研究约束留给每次聊天重新判断。全功能 workflow engine 能自动转移更多状态，却引入与 Run、Study、Finding 重复的事实和修复方式。两者都没有被选为初版。

接受的代价是：极端崩溃可能留下一个无法自动恢复的分配；一些科学选择仍需研究者判断；跨版本共同评测须逐目标资格验证。这些限制比重复付费、隐式重跑或过度归因更容易审查。

## 14. 现有代码与决定依据

- [Compiler 独立与双循环的原始决定](adr/0001-compiler-first-with-dependent-lab.md)。
- [Program、Run、Study 与 Knowledge 的职责](adr/0076-programs-and-transfer-runs.md)。
- [取消无校准结构排序与旧 Portfolio Study](adr/0071-retire-uncalibrated-ranking-and-portfolio-study.md)。
- [共同研究问题与预算](RESEARCH_AGENDA.md)及[独立硬件 Bench 规范](BENCHMARK_PROTOCOL.md)。
- [维护 Agent 已有的归因与 Compiler 演进规则](../contracts/scaffolds/kernel-reproduction/AGENTS.md)。
- [批量任务入口](../tools/launch_task_matrix.py)与[单任务入口](../tools/launch_task.py)。
- [唯一 Run 执行路径](../src/open_cake_ir/lab/execution.py)与[Lab 接口](../src/open_cake_ir/lab/core.py)。
- [当前 Study 设计](../src/open_cake_ir/lab/study_plan.py)、[分配与执行](../src/open_cake_ir/lab/study_execution.py)、[分析](../src/open_cake_ir/lab/study_analysis.py)。
- [诊断汇总](../tools/summarize_diagnoses.py)、[诊断归属](../src/open_cake_ir/lab/routing.py)、[Run 输入结构](../src/open_cake_ir/lab/run_spec.py)。
- [输出维分块进入 Compiler 的已有案例](../findings/2026-10-03-003-squared-distance-output-tiling.json)。
- [GPU 结果与经验迁移的现有范围](OPTIMIZATION_TRANSFER.md)及[E/P 研究合同](OPTIMIZATION_TRANSFER_ABLATION.md)。

## 15. 设计验证记录

本文只新增设计，没有修改运行代码、创建实际 Run 或启动 GPU。设计经过两个独立候选比较和交叉审查。后续实现若改变接口、结果语义或职责，须在本提案的对应位置同步记录已接受的变化；不能让实现和交接文档各自保留一套合同。

交叉审查后，明确了可信调用范围、共同 `dispatch_once` owner、实际 OS 隔离、只引用原处置结果，以及不可变 binding 与 Finding 生命周期的关系。复核未发现该范围内仍未解决的主要设计问题。文档本地链接、Python 草案语法和图解生成已检查；这些检查不证明运行实现或实验效果。

合入时，当前 Run 诊断读取已由 `27def9a2` 修复并随主线集成，记录见 [F-2026-10-05-008](../findings/2026-10-05-008-diagnosis-run-authority.json)。单元 1 剩余工作是让未知编译拒绝保留待归因状态，并接入成功候选的审查材料。此后先证明只读重建与保守恢复，再接入付费执行。
