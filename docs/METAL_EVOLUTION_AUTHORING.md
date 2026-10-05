# Metal 演进实验：TASK.md / AGENTS.md 与 Run 后提炼

状态：平台规划及技能观察边界已合入 `metal@229c1293`；演进文本 treatment 尚未启用到实验。配套 [总计划](METAL_EVOLUTION_PLAN.md)。
实际 TASK.md / AGENTS.md 继续由 `lab/task_package.py` 从冻结 Run 权威生成。
禁止直接修改已经交付的文件，也不手写第二份参数权威。

## 执行绑定、任务目录与技能材料

2026-10-05 用户确定后续 Ralph 作者使用 Codex 的 `gpt-6.1-sol`、`xhigh`。
将下面的显式输入传给实验入口，并由既有 owner 写入 Run 的 provider authority：

```json
{"harness": "codex", "model": "gpt-6.1-sol", "effort": "xhigh"}
```

不是浮动默认模型，不替换成其他模型，也不把其他模型的 qualification 借给它。
CLI/executable、模型、effort、工具与作者环境的两轮资格仍需通过；initial/resume 保持
同一绑定，独立重复使用新 Run。当前仍未启动这组正式 Metal 实验。

目录采用“任务 → Compiler treatment → 独立重复”，都在源码 checkout 外。例如：

```text
metal-evolution/
  manager/                         # 管理输入；schema v2
    TASK.md AGENTS.md experiment.json
    cells/rmsnorm-c0-r01/           # 每个 cell 独立材料快照
      TASK.md AGENTS.md scaffold.md
      references/ author-skills.tar  # 包为后继策略的可选材料
    cells/softmax-c0-r01/
      TASK.md AGENTS.md scaffold.md
      references/
    launches/<cell-id>/            # 保留发送内容、transport 状态
  rmsnorm/C0/r01/
    run-inputs/AGENTS.md            # 节点收到的该 cell scaffold
    run-source/                    # 固定提交的 detached worktree
    run/                           # node.workspace 指向这里
      workload.json starter.py run.json runtime.json
      qualification-workspace/     # 独立资格任务
      qualification-workspace-author-home/
      actors/<run-id>/             # Lab 生成的作者 TASK/AGENTS 与候选
      actors/.codex-homes/<run-id>/ # 私有状态/凭据，不进入报告
      actors/.user-homes/<run-id>/  # 技能包后继的私有 HOME；正式使用仍待资格
      builds/ baseline/ infra-evaluations/
      run-evidence/ report.json
  rmsnorm/C0/r02/                   # 新 Run，不能复用 r01 的可写状态
  softmax/C0/r01/
```

这是后继输入的布局约定，不是已经生成并完成的实验。环境版本可以共享固定安装，
各任务不必重复安装 Compiler；进程状态、可写缓存、候选和证据则要按 Run 分开。
Ralph 与 GPU Infra/Evaluation 保持现有唯一执行路径，不引入第二套调度器或评测器。

**同步边界不能省略。** 旧 `metal@a2e62b08` 已有 task/actor/evidence 独立目录，但没有
ADR 0081 的 Codex author-home 后继；本计划分支已吸收 `main@354a670f` 的每 Run auth-only home。
主线的该 home 只复制私有凭据，允许 CLI 自带 system skills，检查并拒绝该 CODEX_HOME 内的 user skills/plugins；
它不是每任务任意技能包安装能力，也不是完整文件读取隔离。PR #319 已完成 CPU 集成检查
与平台合并；真实作者与设备资格仍待验证。旧基点的 Run 在旧提交回放，不能按新实现解释。

新管理输入采用共享 `kernel_experiment.py` 的 schema v2，
实现见 [PR #317](https://github.com/qhy991/open-cake-ir/pull/317)，已合入 `main@354a670f` 并被本计划分支吸收。
每 cell 必填自己的
`references`，可选完整 `agents_md`，各自产生 TASK/AGENTS/资料/scaffold 快照；选中
哪个 cell 就只发送那份 scaffold。共同资料须显式列在每个使用者的列表中。schema v1
保留历史共享资料语义；新实验不用它来声称任务材料隔离。源码快照投递的独立实现不
表示候选 MSL 已进入下一轮反馈，后者仍由下文 G4 的 Lab 后继负责。

第一阶段的技能说明通过已审阅 `agents_md` 与显式参考材料交付；前者是完整任务规范，
后者是数据，不会安装脚本或取得工具权限。通用 Cake 使用规则与每任务数值/结构提示
分开审阅后组成该 cell 的完整规范。C0/C1、独立重复和经验消融中保持相同材料边界，
不得把维护者读过的其他任务经验隐式带入作者。

原生 skill 环境是另一项需要资格验证的后继。当前 main 仍保留宿主 `HOME`；Codex
还会从 `$HOME/.agents/skills`、工作目录祖先、admin 与 system 来源发现技能，见
[官方技能加载规则](https://learn.chatgpt.com/docs/build-skills#where-codex-loads-local-skills)。
仅检查 `CODEX_HOME/skills` 不足以证明全部发现来源受控。后继需绑定允许的技能说明、
脚本依赖及调用权限，观察实际 catalog/输入，并以宿主、邻任务和非授权插件的哨兵材料
验证 initial/resume 都未混入；不以 agent 口头自报作为证明。既有 auth-only 策略不容许
塞入 user skills 后继续沿用原资格；也不因这一项完成就开放 clean-start。

本机静态接口审计还发现版本差异：bundled Codex CLI 为 `0.159.2`，其导出的
`skills/list` 参数只有 `cwds` / `forceReload`，没有最新文档中的
`perCwdExtraUserRoots`；`extraRoots` 只增加来源，不排除默认来源。响应 schema 能表示
`path`、`scope`、`enabled`、`pluginId` 与错误，但该静态审计没有执行查询或观察真实 exec
加载的 catalog。help/schema 命令退出 0，同时保留 PATH alias 的 EPERM 警告；未修复
宿主后重跑。schema、独立 app-server 的查询和 agent 自报都不能充当实际作者输入证明。
见 [官方技能接口](https://learn.chatgpt.com/docs/app-server#skills)。

后续独立 CPU 软件探针使用临时 HOME/CODEX_HOME、哨兵技能和本机确定性 Responses
服务，不使用真实凭据或模型。v1 在网络边界自检失败，CLI 尚未启动；独立语法诊断保留在
`codex-skill-exec-probe-v1/profile-parse-diagnostic.json`。修正脚本后的 v2 通过了
网络边界、独立 Git 仓库和 CLI 版本检查，但 initial exec 在 45 秒内未观察到发往测试服务的请求，
按预定上限终止，没有执行 resume 或无 Git 仓库的第二案例。只读检查同一 exec 的 retained
context，已见 `host_skills.instructions` 与 `world_state.host_skills.body` 中的任务/私有 HOME
技能目录哨兵，未见技能正文哨兵。这里的 retained context 尚无实际投递证据，不能声称已投递模型，
也不能声称两轮隔离、完整来源覆盖或模型使用通过。超时原因仍未确定，没有调整环境重跑。
原始软件探针与只读投影分别见外部台账的 `codex-skill-exec-probe-v2/report.json` 和
`codex-skill-exec-inspection-v2/report.json`；它们不构成 Provider qualification。

最小后继应沿现有 `author_home` / `ProviderInvocation` / qualification owner：为每 Run
绑定私有用户 HOME 与明确技能包，initial/resume 使用相同绑定，保留既有独立
CODEX_HOME；旧 auth-only 策略不改，也不把 user skills 塞入其禁止的位置。完整包保留原生
SKILL.md、脚本、引用资料和二进制资源，依赖说明继续由技能自身文档持有，不新增另一种
依赖清单语言；文件投递不自动安装依赖、插件或授予工具权限。允许先完成包绑定、投递、
漂移拒绝与 fixture 证据重建的软件实现和 CPU 检查；真正启用此策略仍须同一次 exec 的实际 catalog /
输入观测以及独立 qualification，旧 auth-only 收据不能放行。
祖先、admin、system 和插件来源仍需检查；发现未知来源、重复/缺失技能或漂移时拒绝。
目前已有 retained context 的观察入口，尚未验证它与实际请求及 resume 的关系，不能用旁路
进程补出“隔离通过”。实现完成、CPU 检查通过、原生输入资格、模型实际使用和设备收益
分别报告；前两项不计为已经完成用户要求的独立 skill 环境。

通用软件后继已提交为 [PR #325](https://github.com/qhy991/open-cake-ir/pull/325)，当前待完整
CI 与 main 集成。固定 `8a937973` 的 31 项技能测试及 33 项管理器测试通过；本机没有 ruff，
完整证据资格 fixture 交 CI，未安装工具或修改环境。现有材料入口接纳完整原生 tar 包，
v2 cell 保存并发送自己的快照；两个私有 home、初次/续轮绑定和漂移检查沿用 Lab owner。
实现将在 fixture qualification 中保留原包并支持重建投影；该完整证据 fixture 的运行结果待 CI。
正式 Run/Campaign/回放仍明确拒绝新策略。
使用方法和准确边界由共享 `docs/KERNEL_REPRODUCTION.md` 持有，平台不复制实现。
本地结果见 `native-skill-preparation-8a937973/report.json`，不替代上面的原生输入验收。

技能资格的最小验收步骤：

1. 固定 CLI、模型/effort、技能名单/版本、依赖、参考范围、工具与预算，绑定到既有 Run owner。
2. 留存过滤秘密后的配置/路径解析和实际 catalog、已加载正文与引用资源；CLI 若无法观察
   相应输入，就报告未验证并先补观测，不能用模型自报补齐。
3. 在可丢弃的资格环境检查允许技能，以及来自用户 HOME、祖先、admin、插件与邻任务的
   哨兵材料；未覆盖的来源明确列出，不改现有宿主状态使检查通过。
4. initial/resume 保持相同绑定；技能漂移拒绝继续或转入新资格。不同 Run 不继承前一 Run 状态。
5. 分别报告目录/状态分离、实际作者输入符合声明、文件读取限制；前两项不证明第三项。
   完成这些软件与真实 CLI 资格后，才比较技能对 Cake 探索和设备收益的影响。

借鉴依据是 [KDA-Internal e99a6ce1 的 workspace builder](https://github.com/qhy991/KDA-Internal/blob/e99a6ce1f0ea69101bb03904b00d39ade988e708/src/kda/tasks/workspace.py)：
每次 setup 创建独立 repo，仅复制当前任务资源，并测试其他任务资源未投递。
其 [运行职责](https://github.com/qhy991/KDA-Internal/blob/e99a6ce1f0ea69101bb03904b00d39ade988e708/docs/running-experiments.md)
将 flow、模型、重复与 agent 容器交给 Humanize2；当前 KDA 源码没有证明 author skill
的完整安装/挂载流程。因此借鉴最小任务投递、固定 runtime 和独立 judge，不把 GPU
容器的 HOME 设置当成 author 隔离证明，也不将 CUDA 镜像与 NCU 路线照搬到 Metal。

## 完整 scaffold 的准备边界

当前 `--agents-md` 与 `cells[].agents_md` 都完整替换默认 scaffold；系统不会展开 Markdown
链接、include 或 extends。把短附录单独传入会遗漏 Metal 能力约束。

本次提供 [演进附录组件](../contracts/scaffolds/metal-evolution-appendix-v1.md)。冻结前，从同一
固定提交读取完整 `python-artifact-optimization-metal-bundle-v1.md` 与该附录，按此顺序拼接，
在 checkout 外独占创建一份完整文件，再交给现有 `--agents-md` 或 v2 cell 的 `agents_md`。
准备后不追加；实际 Run 仍由现有 scaffold 绑定和 TaskPackage owner 验证、投递、回放。
附录只增加行为要求，不重复形状、模型、预算等参数，不授予源码访问或工具权限。
未实际交付生成代码的 Run 只能记录缺失，不能被算作低层代码驱动的优化验收。

已从固定 `1b2920b7` 顺序组合完整文件，在源码外独占创建并通过现有 `task_run_inputs`
做了 D2 的 CPU 绑定检查：模型为 `gpt-6.1-sol/xhigh`，完整说明与报告列出的预算上限被绑定。
confirmatory 配额仍为 8，执行逻辑最终只确认一个提名；正式冻结时再核对完整 RunSpecification。
记录位于外部规划根目录的 `authoring-treatment-1b2920/composition-report.json`，完整
文件为同目录 `AGENTS.md`。这是初版 CPU 观察，不是已冻结 Run 或真实 TaskPackage 交付。
管理预算入口已由 PR #321 实现并集成，参数及完整验证范围见总计划。

当前 CLI 作者的正常生命周期保留上轮 `candidate-set.py`，resume 要求该文件存在，
沿用原 thread 与 cwd；读取提交和封存原始文件不会把它清空或移走。但下一请求不会重新
投递上轮作者源码/注释，`generated_source` 也是 Compiler 输出，不是研究笔记。
因此附录增加“更新前按既有权限读取当前文件”的要求，只携带少量有效观察，再替换本轮
有序候选；不把旧完整候选累加以越过候选上限。现有文件可读不等于 agent 已读取，G4
仍需真实两轮行为证据。上述 `1b2920b7` 初版组合文件没有这条后继要求，不能直接作为
最终 treatment。后继已从固定 `8cc264f0` 重新组合，并通过原有 input owner 绑定
`generated_source_v1`、精确模型、预算与每轮 3 次搜索；报告和完整文件分别为
`authoring-treatment-8cc264f0/composition-report.json` 与同目录 `AGENTS.md`。它包含上述
重读要求，可作为待验收的完整任务材料；仍没有创建正式 Run 或实际作者/设备资格。
正式冻结时引用这份完整文件及实际 Run 权限，不能只传附录。

## TASK.md 负责“做什么、如何判断”

保留当前 renderer 的 Workload、oracle、预算、候选输出、基线和完整参考权限投影。
以下内容作为未来 scaffold / 任务投影的说明，不重复手写容差和硬件参数：

> 在本 Run 固定形状与 FP32 数值语义下，提交结构不同的完整候选，目标是获得外部确认的性能改善。
> TASK 中的 Workload、预算、评测协议、授予材料和变换 API 是唯一任务权威。
> 每轮候选应有可检验的结构假设；改名、注释变化或单独改变 group 数不算结构不同。
> 主动探索已交付的 Cake frontend/API 与合法组合，说明候选使用的 primitive 和操作区域，
> 以及预期生成的访存、归约、工作划分或算术结构；不能把 starter 当作 Compiler 的能力上限。
> 分别判断 Cake 能否表达、lowering 是否实现、设备是否获益。只引用实际交付且允许查看的
> 候选低层证据；没有相应代码或计数器时记为未验证，不用耗时反推指令实现。
> 正确性失败先修复正确性；计时不稳定保留 unknown；不从逻辑存储估计推断物理寄存器或 spill。
> 最终提名和确认由控制器管理。作者摘要不能证明收益或改变终止条件。

作者 Run 接收哪些经验，由已有 `knowledge.materials` 控制；哪些工具可调，由
`knowledge.transformations` 控制。E0 不挂载经验，P0 实际拒绝变换访问，不能仅提示“不要使用”。
第一阶段绑定 `reference_access=known_kernel_reproduction`：当前参考权限 gate 只在这一类别
接纳现有 Python starter 和新 Metal scaffold。可描述为 starter-informed 优化，但这不能
替代实际权限枚举，不宣称 clean start。以后改为 clean start，需要另审受限 scaffold 与
不完整参考，不能只改一个标签。
E0 表示没有额外经验材料，不代表 P1 API 不含机制描述。材料 schema 只约束结构与字段，
不能证明文字中没有隐藏的目标实现或结果；冻结前要审查实际交付文本及其引用的访问权限。
维护者已经阅读的源码与候选不能通过当前会话隐式传给独立作者。

## AGENTS.md 负责“怎样进行一轮”

保留现有 renderer 的只写 `candidate-set.py`、同线程、冻结 Compiler、Lab 评测及证据规则。
默认 bundle 与旧 `candidate-set.json` 是不同 treatment；显式单源码模式 `candidate.py` 也另算。
当前 renderer 已要求读取 Run 内 `optimization_history` 并在允许的源码里留下简短公开笔记；
下面补充的是 Cake 机制与低层证据要求，不把这些基础能力重新列为尚未实现。
第一阶段可用一个后继 Metal scaffold 加入下面的简短要求，避免改公共 envelope/schema。
目前它只是一份设计，尚未部署到任何 Run。

> 先读上一轮 StateCard 的候选拒绝、correctness、measurement、profile 和剩余额度。
> 选择一个明确可反驳的优化假设，说明它改变什么执行结构，以及什么观察会否定它。
> 阅读与假设相关的已交付 Cake API、Target 约束和诊断；先检验已有 canonical primitive 的
> 合法组合，再判断是否存在表达能力缺口。探索通过原有候选提交与预算进行，不额外调用工具。
> 若已交付该候选的 MSL/编译层证据，定位 operation/region，比较预期与实际实现；否则写明
> 缺失的证据层，继续可执行的候选探索。每轮留下结果摘要，不以 API 数量或笔记数量计进步。
> 对有反馈的上一假设给出简短结果摘要：支持、反驳或证据不足，并说明下一步改变。
> 这些只需要可供实验审计的摘要，不要求输出内部推理过程。
> 给每个 Schedule 函数体开头添加下面的简短注释，正文仍是完整合法 frontend 程序。
> 函数外笔记只保证留在完整原文件，不能假定也进入逐候选源码。
> 只更新 `candidate-set.py`，不创建 MEMORY.md、诊断 JSON 或 Findings 文件，不增加提交字段。
> 引用实际可见的轮次、候选 ID、诊断码或收据字段。不可见的 event sequence 留给维护者解析。
> 若反馈没有指明对应候选，就记录归因未知，不凭提交顺序猜测哪项改动获益。
> 区分候选错误、疑似 verifier/lowering 缺口、测量不可用和环境故障。重复出现不自动证明根因。
> 不修改既有预算、Compiler 或 TASK/AGENTS；无法改进时提交诚实的边界说明，不假造成功。

建议每个 Schedule 函数体内的注释（所有字段都是作者声明，非评测事实）：

```python
# experiment: h02
# parent: <visible candidate id or initial starter>
# change: <one concrete structural change>
# cake_expression: <relevant API/primitive and operation/region>
# lowering_check: <expected mechanism; visible evidence location or unverified>
# prediction: <observable outcome; uncertainty allowed>
# prior_observation: <turn and visible diagnostic/receipt field, or none>
# falsifier: <what would reject this hypothesis>
# lesson_status: <supported / refuted / unknown, bounded to available evidence>
# suggested_owner: <candidate / verifier / lowering / cost_model / evaluation / lab / none>
```

注释不是新 IR、不是评分字段，也不能让相同 Schedule 被算成两个结构候选。
在 `a2e62b08` 的离线传输/规划探针中，注释完整保留、Schedule 不变，同轮 search 计划折叠为
一个程序；原始源码字节变化仍产生不同提交身份。这不等于消除构建成本：当前先构建整组，
随后规划 search，且去重只作用于本轮。跨轮重复仍可能消耗评测额度，须在维护总结中单独统计。
源码和既有原生输出可供 Run 后整理；消息作者和 CLI 作者都通过同一 source transport 保留摘要。
这些是旧 envelope 的探针结论。当前 bundle 从装饰器到函数末行切片：函数体内注释保留，
装饰器前注释仅在 sealed `provider_source_file` 保留。`d3967a0d` 的独立 CPU probe 已验证
投影位置以及注释不同而 Schedule 相同；它没有启动 provider 或证明完整 seal custody/作者学习。
PR #316 已实现从先前事件与收据派生的有界 `optimization_history` 和独立回放，供下一轮
及上下文压缩后读取；它不自动证明作者理解了负结果，也不授予跨 Run 经验。不能新增一个
会自行改写政策的长期记忆文件。

当前 bundle 的 `cake.transform(...)` 附近也可以留注释，完整原文件会保留；已解析的 transform
动作字段和结果 Program JSON 仍不带这些文字。P1 笔记应明确标注原动作位置、parent 与
transformation，由维护者同原文件和动作事件对齐；不额外新增 schema，也不为了留下笔记
绕过变换 API。文件笔记与逐候选源码是两个保留范围，E/P 两组的记录机会保持一致。

## Metal 低层证据与交付合同

任务要建立“Cake 表达 → 生成实现 → 设备观察”的对应。层级上，MSL 是源码；Metal IR
（离线工具链可保存为 `.air`）才是中间表示；更下层是 Apple GPU 机器指令。
PTX 是 NVIDIA 的虚拟 ISA，Metal IR 只在编译链位置上与之相近，不代表存在同等的公开
文本指令接口。不能把 MSL 或 `.metallib`/binary archive 容器称为可读的 Metal PTX。
依据：[Apple 编译流程](https://developer.apple.com/documentation/metal/metal-libraries)、
[Apple 离线 IR 文件说明](https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Dev-Technique/Dev-Technique.html)、
[PTX 定义](https://docs.nvidia.com/cuda/parallel-thread-execution/)。

首轮以生成 MSL 为必需的代码检查层：对照 work mapping、load/store、重算/复用、归约、
SIMD-group 操作和数值契约，只检查该候选实际包含的机制。AIR/原生指令与资源观察作为
精确工具链支持时的后续能力，不把获得反汇编作为首轮启动的必要条件，也不凭空填充。
源码级观察不能证明最终指令、物理寄存器或带宽瓶颈；最终收益仍由合格的 oracle、测量
与相应覆盖的 profiler 支持，MSL 看起来更短不等于设备更快。

**源码交付的软件后继已合入 main。** [PR #320](https://github.com/qhy991/open-cake-ir/pull/320)
在 `main@473f8cad` 引入显式 `generated_source_v1`；本计划任务分支已吸收，尚未启动新 Run。
单任务用 `--generated-source-feedback`，schema v2 cell 用 `generated_source_feedback: true`；
默认不启用，只接纳 Cake 的 `known_kernel_reproduction`，不扩大 baseline、其他 arm、
clean-start 或工具权限。完整 scaffold 的说明文字不会自动授予这一开关。

`MetalArtifactBuilder` 保存的 `lowered_source` 现在可以随已封存且已搜索的候选进入
下一实际请求。仍由 MSL 经 `MTLDevice.makeLibrary` 构建，不额外导出 AIR。每项保留
source Turn、candidate、stage、精确 Target、lowering route、语言和原始行号；源码中的
`CAKE_OP` 标记原样保留，未声称另有完整物理指令映射。多阶段使用既有 Program artifact
owner，折叠为一个 kernel 的 Program 仍保留作者 stage 身份；**这没有开放多阶段 Metal
执行**，其现有运行 owner 仍拒绝该路线。

投递完整 stage 文本，不截取半段代码：每候选源码 32 KiB、每 Turn 源码 64 KiB、每候选
最多 32 stages；包含元数据的单候选序列化视图另限 64 KiB，总视图受冻结候选数约束。
按原提交/stage 次序分配，超限、缺失、未封存或未搜索均有明确原因；不以其他候选或
baseline 代码填空。完整正文不会重复进入 `optimization_history`。

交付正文进入现有请求 bundle，独立 replay 从正确时间点的候选 artifact 重建，并用
冻结 Compiler 校验 authored candidate 与生成源码的关系。反例覆盖真正不同的 MSL
自洽替换、多候选/多 stage 错绑、大小边界、缺失、initial/resume 请求以及故障和末轮
没有后续交付。只有封存源码或反馈记录，不等于存在下一次作者请求。

固定 `03d9cac5` 的 10 项非 custody CPU 检查通过，完整 CI 在 Python 3.10/3.11/3.12
各 2698 通过、35 跳过，包含 EvidenceStore 的两轮控制器用例。记录为外部规划根目录
`generated-source-feedback-03d9cac5/report.json`。本机先前 custody EPERM 保持原样；
没有提权重跑该用例，也没有真实 provider、原生编译或 GPU 验收。

G4 剩余的是在适用门通过后，用 D2/D6 的真实两轮作者路径检查：允许的候选 MSL 确实
进入请求；作者准确引用一个真实区域，比较预期与实现，并据此改变或撤回假设。分别报告
交付、解释准确度、行为变化与设备收益。维护者可读代码、CPU bundle 测试和提示词要求
都不能代替这些行为证据；没有设备收益也可产生有效反证。

公共复现任务的 canonical 要求维护在
`contracts/scaffolds/kernel-reproduction/AGENTS.md`，通过已有 `--agents-md` 显式绑定。
通用规范后继见 [PR #315](https://github.com/qhy991/open-cake-ir/pull/315)，提交 `b2befbf5`；
本计划分支已包含该共享后继。Metal 默认 bundle scaffold 不会因该文件修改而自动更新；
本页文本与低层投递均需接入
后继 Run。修改 scaffold 与源码可见性应先作为新 treatment 冻结，再在后续 Compiler
对照中保持一致，不能把提示词/材料的变化归因于 Compiler。

## G4 前置：共享反馈后继的最小合同

`a2e62b08` 的两轮 CPU fixture 已复现：日志中的第二个候选胜出，第二轮收到数值与 profile，
却没有候选身份；现有回放照样通过。源码还显示未胜出的已评测候选没有下一轮反馈。
以下共享 Lab 修复现已纳入本计划分支，接下来验证集成和真实作者。它不是 Compiler primitive 或 Metal 特例。

1. 反馈引用原始源轮次、提交中的候选位置及已有候选身份；不计算新身份、不暴露被禁止的源码。
   胜出项有明确标识。一次反馈不能混用 A 的耗时和 B 的 Findings/profile。
2. 为所有已搜索候选投影有界结果，包括错误/不稳定/慢于胜出者的项；未搜索、语义折叠与
   构建拒绝分别说明，不能把“没选中”显示为数值失败或默认正确。
3. 字段由既有候选事件和收据派生，作者不能改写；计时精度、profile 覆盖域和截断标记保留。
   公共 live 和 replay 使用相同合同，另用独立反例验证它，旧 Run 按旧提交回放。
4. 控制器验收至少含：第二候选胜出；构建拒绝/可运行但错误/正确胜出混合；同轮折叠；
   评测上限导致未搜索；profile 不可用；末轮没有下一次交付。按实际已交付信息审查作者。
5. 篡改胜出身份、交换两候选反馈、删除已评测 peer 或把 unknown 改为成功，回放都必须拒绝。
   明确这种验证证明信息交付与一致性，仍不证明作者学习或设备收益。

复用 `tests/contracts/test_diagnosis_feedback.py::DiagnosisRunTests` 与
`tests/contracts/test_lab.py::QualifiedCandidateSelectionTest` 的 CPU doubles，不引入另一套
Ralph；真实 provider/GPU 接入在对应门通过后单独执行。

该合同已在独立共享后继 `5453a4f5` 实现，`6a646140` 的 71 项相关 CPU 合同通过。
另一个广泛回归的工作树写权限前提未满足，仍为环境未验证。完整结果见外部
`feedback-binding-successor/report.json`。共享修复已通过三版本 CI 并由 PR #313 合入
`main@af1b18ba`；本计划分支经 `354a670f` 已包含该实现，但没有部署本页 scaffold 或启动新 Run。
读反馈时按 `source_turn` 与每项既有候选身份对应注释，`candidate_index` 是首次动作的原始
位置，遇到拒绝变换时可能不连续。`selected` 不等于成功：全部构建拒绝时它只是诊断选择。
`omitted_findings` / `text_truncated` 表示有界投影的损失；需要完整诊断由维护者回到 Evidence，
作者不得因为省略而推断没有其他限制。没有 receipt 的项不填写正确性、时间或 profile。
零轮停止、未完成轮次和末轮终态均需保留真实时间顺序；确认结果不反向成为搜索时已知信息。
软件验证证明归属与一致性，真实作者是否据此改变策略仍属于 G4 行为验收。

## Run 后维护任务模板

该任务在已封存 Run 后，由有权读取完整 Evidence 的维护环境执行，不在候选作者的写权限里执行。
输入必须是精确 Run 路径与对应固定源码，不能用移动中的 main 重放历史。

> 读取给定 Run 的独立 audit、原始候选、构建诊断、搜索/归因/确认收据、停止原因和实际 usage。
> 先检查 archive integrity 和 filesystem custody，分别报告；不恢复 mode bits 来制造历史 custody。
> 用现有 summarize_diagnoses --compiler-gaps 汇总保留路由、当前码表提示与变换拒绝线索；
> 不重写旧事件分类，不把当前 triage 提示当作当时的诊断或独立根因证明。
> 将每个候选的简短假设与实际反馈对齐，明确哪些声明得到支持、哪些被反驳、哪些未被检验。
> 区分提出假设、收到反馈、下一轮改变行为这三个时间点；没有后续交付证据时，不推断作者已经学习。
> 统计相同程序的跨轮重试、同轮折叠及实际构建成本；区分预先声明的噪声复测与无效重复。
> 按机制与实际拒绝条件聚类，不按任务名字或错误文本相似度直接合并根因。
> 在既有 Finding 合同下提出最小建议或 No promotion；每条引用精确 workspace、Run、事件或对象路径。
> 保留失败、慢化、缺失和环境问题；不得把没有 counter 数据解释为 occupancy 或带宽瓶颈。
> 不修改 Compiler，不启动新 GPU/provider 实验，不自动授予下一 Run 经验，也不批准自己的提案。

维护摘要应覆盖以下内容（是已有记录中的文字要求，不是新增生产 schema）：

| 内容 | 必须回答的问题 |
| --- | --- |
| 观察域 | 哪个提交、设备、任务、shape、dtype、输入分布与测量协议？什么没测？ |
| 证据 | 候选、对照和原始事件在哪里？缺失项是什么？ |
| 假设检验 | 修改了什么？有什么支持与反证？只改一处还是有混杂？ |
| 归属 | 作者错误、工具限制、lowering、verifier、cost model、Evaluation 或 Lab？ |
| 复现 | 几个独立来源？同一次 Run 的重复不能充当独立复现 |
| 建议 | 最小修复/机制、适用前提、反例、P1–P8、预计受益范围 |
| 决定 | proposed / deferred / rejected 或 No promotion；实现与验证尚缺什么？ |

不创建第二个 CompilerProposal 数据库。机制有复用价值时通过既有 OptimizationKnowledge
携带解释和反例；其 references 是审计定位，不等于允许作者访问对应低层实现。
确定性变换由现有 pass API 授权，Candidate 仍走相同 verifier 与 Evaluation。

## 接入验收与开发边界

第一轮绑定后继 scaffold 与已实现的源码反馈开关；适用资格通过后，用两个代表任务检查
Cake 探索与低层对照的实际行为。只有文本而缺少源码时，可验收可见反馈的利用，不能
据此验收“作者结合低层实现优化”。后继接入按以下边界推进：

1. 平台 scaffold：`task/metal-*`，外部路径通过已有 `--agents-md` 冻结机制接入；
   先核对入口实际参数与权限，不能把本文整页当成未经处理的 author scaffold。
2. 若需要 renderer / StateCard / 回放变更：`task/core-* → main`，保持旧证据与旧提交可回放。
3. Run 后维护采用既有 Findings owner，生成报告在 checkout 外；只有经过整理的 Finding 进入源码。

接入时必须验证：两轮反馈真实交付；假设注释按新提交保留且不新增结构候选；未写额外文件；
作者声称成功不能改变确认结果；末轮无后续反馈也可由维护者整理；E0/P0 的权限隔离真实生效；
反例触发的是拟定 guard 而非借来的无关规则。环境故障不做“反复失败”性能机制统计。

验收时从预先指定的两轮 Run 逐项审计：上一轮诊断确实存在于下一轮可见 StateCard；作者引用
没有编造；候选变化与声明相符；收据支持的结论没有越过测量域；末轮记录由维护者补齐。
分别报告交付成功、记录准确、行为改变和设备收益，不合并成一个“学习成功”分数。
错误假设被证据否定并停止重复，也是合格的积累；例如本次 Softmax 重算探针的逻辑峰值槽位
从 65 变为 66，就应撤回“降低逻辑峰值”的具体判断，但设备性能仍是 unknown。

“Agent 会不会有意识积累”只能用行为检验：下一轮是否依据已有反馈改变候选、是否重复已证伪假设、
记录是否准确、独立后继 Run 是否更省预算。要求写总结本身不保证学习，更不保证 Compiler 改进有效。
