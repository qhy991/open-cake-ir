# Lab implementation map / 实现导航

单次执行入口是 `Lab.preflight_run/execute_run/audit_run`，消费独立的 `RunSpecification`。
`Lab.preflight/execute/audit` 接受现有 matched-Study 输入，先投影为同一种 Run，再执行和汇总。
公共导入仍为 `open_cake_ir.lab`；所有 Run 共用一个搜索、预算和终态循环。

| 职责 | 主要入口 | 边界 |
| --- | --- | --- |
| Run 准入与 Study 输入绑定 | `core.py`、`preflight.py`、`run_spec.py`、`contracts.py`、`bindings.py`、`admission.py` | Run 冻结执行闭包；Study 预先指定条件与分析 |
| 搜索与预算 | `execution.py`、`candidate_filter.py`、`selection.py`、`ralph.py`、`run_completion.py` | 全部候选完成过滤后才评测；预算和终态有明确归属 |
| 作者与构建 | `providers.py`、`message_provider.py`、`provider_documents.py`、`provider_events.py`、`provider_invocation.py`、`toolchains.py`、`build.py` | 协议数据、进程调用和工具链各自负责自己的规则 |
| 证据与回放 | `archive.py`、`evaluation_writer.py`、`replay/`、`reporting.py` | 写入与独立核验分离；报告使用审计后的结果 |
| 任务接线 | `environments.py`、`pairing.py`、`task_package.py`；任务层 `TaskLab` | 通用引擎不导入具体任务实现 |

## Admission

`Lab.preflight_run` → `RunSpecification.load` → `TaskLab._validate_run` → exact
Compiler/Executor、Workload、provider、baseline 与设备声明准入。

工程 Run 的 `assignment` 为 `null`，无需 Study 或估计量。研究分配在 Run 开始前绑定 Study
及 condition id；条件名称与 `authoring.environment_kind` 分开，不能改变候选解析和编译路线。
Run 冻结公共评测、预算、作者环境、参考材料和终点规则；不持有 Study 的统计分析计划。

现有 Study 模板由 `StudyContract.load` 解析、绑定为 `CampaignLock`，再由
`CampaignLock.run_specification` 转换成上述执行输入。旧格式编码在 run id 中的分组只在这个
输入边界解读；执行器与回放不再拆解 run id。原有历史证据仍由产生它的提交回放。

`core.py` 保留六项依赖，任务准入钩子不增加状态容器。身份或配置不符时，在 provider、GPU
和 Evidence writer 副作用之前拒绝。编译与评测后端能力没有因 Run 合同变化而自动扩大。

## Execution and replay

`execution.py::_execute_run` 是独立 Run 与 Study 分配 Run 共用的唯一执行循环。`evaluation_writer.py` 负责一次评测的归档事务；
选择策略、预算和故障阶段仍归调用方所有。`provider_documents.py` 拥有协议常量，内部使用者直接导入。

每轮只产生搜索观察与反馈。`search_completed` 冻结搜索停止时的预算状态；随后
`candidate_nominated` 按搜索延迟选择一个预算内候选，同值取较早轮次。确认重用该候选的
封存产物，产生一份新鲜收据；确认失败或无收益不再换候选。没有合格搜索候选时明确记录空提名。
终态确认事件使用 `source_turn` 指向原始轮次，不伪造新的作者 Turn。搜索 checkpoint 只报告
`best_search_latency_ms`；最终端点和产物晋升使用独立确认。阈值视图记录全部搜索消耗和确认完成
时间，不把较早候选的生成成本当作整个 Run 的成功成本。

`budget.maximum_compilations` 限制原生 kernel 源码编译入口的调用次数：每个 Program stage、
Triton 对齐版本和 NVCC 的 PTX/CUBIN 调用各自领取额度，失败调用同样计数。静态拒绝、
反汇编和预制 host 工具的准备不计为 kernel 编译。`CompilationRecorder` 在实际入口前向
Ralph 领取额度；没有额度就不调用工具链，并将候选记录为预算拒绝，不推断实现缺陷。
回放从 `compilation_started/completed/refused` 重建计数、顺序和对应候选，拒绝提名前后
越界调用或把预算拒绝变成可执行产物。`launch_task.py --max-compilations` 显式冻结该限额。

`budget.confirmation_wall_time_seconds` 在总 wall budget 中预留固定确认额度；搜索额度由
总量减去预留量得到，不能另存一份配置。Ralph 在 Turn 边界停止搜索，关闭后禁止继续作者调用、
编译或搜索评测。确认阶段从 `search_completed` 开始计时，包含提名、确认、必要归因与归档，
不能借用提前结束搜索所剩的时间。`launch_task.py --confirmation-seconds` 可指定额度，生成器默认
预留总量的十分之一并将具体数值冻结到 Run 输入。

Turn 和 Evaluation 的在途操作由既有 adapter timeout 限制，不声称可抢占编译器或 GPU。
实际超限始终记录在 Ralph 的 `budget_exceeded` 中，列出已观察到的 token、作者、搜索或确认
时间超限；正常端点据此禁止预算内成功。故障 Run 保持结果缺失，同时保留预算诊断和实际成本，
token 用量未知时区分已知小计。阈值报告使用经过审计的同一投影。正式 E/P Study 的冻结分配与
统计验收仍待完成。

`replay/__init__.py` 组织独立回放，`artifacts.py`、`attempts.py`、`candidates.py`、
`provider.py`、`selection.py`、`outcomes.py` 和 `refusals.py` 核对各自的原始记录。
它们不能调用 live execution 或 Evaluation writer 来证明自身正确。
每个 ledger 绑定自己的 Run authority。`audit_run` 返回存储审计和独立语义回放结果；
`reporting.py` 从这些 Run 结果投影原有 matched-Study 报告，不产生新的证据。

## Changes

输入检查顺序、失败类别、预算计数、日志顺序和冻结格式都是行为合同。
源码变更提交为新的 clean commit；Campaign 固定的旧提交保持不变，历史记录由原提交的工具回放。
新代码进入规则的实际 owner，不增加转发函数、隐式注册表或通用状态字典。

The existing Study facade remains an input boundary. Independent Runs carry their own frozen
authority and use the same engine and independent replay. A new source commit
changes the identity of future execution; it does not rewrite historical evidence.

`tasks.compose.run_runtime_factory` 统一装配每个 Run 的 provider、构建环境和评测器。
生产装配要求真实 live qualification，并核对冻结的 Compiler、Executor、工具链、broker 和基线。
消息作者的 runtime `provider` 对象为空，不声明 CLI 可执行文件或作者目录；其消息和历史按 Run
隔离。CLI 作者每个 Run 使用新目录。旧 Campaign 入口只适配预分配和原有报告，保留它自己的
比较约束；普通 Run 不需要比较组或估计量。

命令 `open-cake-ir lab run preflight|execute|audit --run <run.json>` 接收独立 Run；执行另需
`--runtime-config` 和 `--evidence-root`。`tools/launch_task.py` 直接准备并冻结 `run.json`，
执行后写入 `run-evidence/` 和来自独立审计的 `report.json`；`kernel_experiment.py` 的各节点
沿用这个入口。基线来源、incumbent 选择、后端 assay 和完整输入验证保留原有准入规则。
旧 Study 模板仅投影共同的任务控制项，普通任务不再写入 Study 或 Campaign Lock。
`report_task_efficiency.py` 可读取新 Run workspace，也保留旧 Campaign 的 opt-in 报告约束。

`promote_task_incumbent.py --run <run.json>` 对工程 Run 完成独立审计后，沿用现有实质收益、
receipt、custody 和当前基线链检查追加晋升；`--campaign-lock` 保留旧输入。新旧记录可以在
同一 registry 链上读取，完整 Program 与原生 kernel 均可供下一轮准备使用。`confirmed_artifact`
只是报告中的已确认产物，不表示已写入 registry 或取得预算内成功。Study 分配仍由其研究策略负责。

## Preassigned E/P Studies

`StudyPlan` 保留 `matched_search` 类型，冻结四个 E/P 组合、知识版本、源发现/目标适配/测试划分，
以及各测试任务的 Run 模板。标签只是分配身份，不选择候选语言或后端。准备阶段校验共享控制、
同一 operator 的 family 标签一致性和 canonical workload/case 去重；未见形状与未见算子族分别报告。
机制文本是否包含测试答案仍属于内容审查，结构检查不代替该判断。

`Lab.prepare_study` 在外部目录写入计划及每个预分配 Run；`execute_study` 通过调用方提供的
runtime factory 调用同一个 `execute_run`，不引入另一套搜索状态。已尝试分配不得覆盖或自动替换，
启动前失败也有绑定原 Run 身份的记录。`audit_study` 只采用匹配预分配且通过独立回放的结果。
`tools/transfer_study.py prepare|execute|audit` 提供准备、执行和审计入口；execute 指定
`--runtime-config` 并使用上述共同装配。prepare 和 audit 不启动 provider 或 GPU。

报告按任务等权计算材料、pass 与交互的成功率差，缺失保留在分母并给出范围与已观测子集敏感性。
配对 task bootstrap 需要预注册的先导依据和最小任务数，单任务及软件资格测试不生成总体区间。
首次正确成本来自真实评测完成事件；材料交付与显式 pass 使用分别报告。`system_qualification_only`
只产生协议描述，不能输出科学主估计量。发现、适配和维护成本以原始证据引用报告，缺失不当作零。

## Complete Program candidates

Cake 作者可以提交完整 `Program`。公共 ABI 在 Workload 边界绑定一次，叶子 Schedule
只描述自己的参数；Triton builder 为每个 stage 保留独立编译产物和实际 launch metadata。
父候选封存 stage bundle，回放核对原始作者 Program、冻结 Compiler lowering 和编译产物。

只有一个 stage、无视图或重命名绑定且参数顺序与公开 ABI 完全一致时，复用该平台已有的
单 kernel builder，保留原始 Program 候选身份。Triton 的普通和 alignment 变体保留同一
编译记录；CuTe 的二进制符号由已有编译合同验证；Metal 保留原 lowering 的线程和共享存储
声明。回放同时核对作者程序、生成源码、公开 ABI 与物理 launch，不能只凭封存完整性接受。

Program 与单 kernel 共用 oracle 和 Evaluation receipt。物理 kernel/module 计数来自封存
stage，计时覆盖完整有序调用；NCU 输出按 dispatch 保留各 stage 的指标，不把逐 stage
指标冒充整体性能估计。每个计时 cohort 结束后释放它自己的输出与中间张量。

需要多次 launch、非 identity 绑定或视图映射的 Program，在公共 Evaluation 中使用同一
构建封存、存储与顺序执行路径，现有模块 loader 覆盖 CUBIN、HSACO 和 MCFATBIN。
公共 Torch 接口在加载前检查公开张量的形状、dtype、设备与连续性，准备时检查中间值和
singleton views；启动前重新检查类型及存储身份，防止同一 tensor 对象被重新绑定。

完整 Program 的计时及 attribution 目前仍只有 Triton/CUDA 路径。HIP/MACA 的既有仪器
只覆盖单次 dispatch，不能借给多 stage 程序。优化环境在编译前拒绝这些组合；设备 worker
与 receipt reader 同样拒绝候选或基线借用单 kernel 测量。非 CUDA 组合可通过
`Compiler.lower_program` → 共享的 `build_program_candidate` → 公共 Evaluation 验证正确性。
该构建接口同时服务优化环境与 `tools/qualify_tensor_program.py` 的原生张量验证；它只拥有
source/ABI/封存，不决定测量策略。非 CUDA 组合尚未接入计时优化 Run，也不改变 Target 的单
kernel timer 声明。Metal 组合继续明确拒绝。软件合同测试不赋予实机正确性、计时或迁移
收益资格。显式动作、知识授权与消息作者
隔离见下节。QSA Cake 候选已使用同一 Program 构建、封存与执行路径；其旧节点描述只作输入
适配，原生 CUDA 参考保留自己的固定 ABI。QSA 的真实参考形状与视图已通过 CPU 编译替身和
虚拟存储合同检查，另有小型融合 oracle 回归；正式硬件消融尚未验证。

## Knowledge and explicit author actions

`knowledge.py` 校验版本化机制说明及原始证据引用。Run 分别冻结 `knowledge.materials`
与 `knowledge.transformations`；解释材料不要求对应一个 pass，工具授权也不自动交付经验材料。
Compiler 的变换声明拥有 API 与 guard；知识记录不再保存硬件支持表或复制测量结果。

`actions.py` 将作者请求解析成新的候选字节或拒绝。父引用只能来自此前本 Run 已产生的候选，
或 `reference_inputs.baseline_programs` 中明示授权的完整实现；同一批动作不互相授予父引用。
P0 在解析父程序和调用 Compiler 之前拒绝变换。直接提交的既有输入继续作为 submit 处理。

`author_actions_resolved` 留存动作、父引用、变换和结果对象；构建过滤只处理实际产生的候选。
全部动作被拒绝时留下无候选的 Turn observation，不伪造 Candidate。独立回放从原始动作、
冻结权限与此前候选重新推导结果。材料计入 provider 输入，拒绝的请求也占 per-turn 上限，
改写与构建时间计入同一 Run 的 wall budget。

服务端调用权限与材料交付不等同于证明 CLI 作者无法读取主机文件。受限作者使用下面的消息
入口；E/P 分配和分析使用上述 StudyPlan，真实迁移收益仍需正式实验验证。

## Message-only authoring

`message_provider.py::ResponsesRunProvider` 接入同一 Run 引擎。模型只收到冻结任务材料和本 Run
的消息历史，以 JSON 提交候选或显式动作；请求没有文件、shell、web 或可执行回调工具。
每个 Run 独立保留原生 response output，包括 reasoning 与 assistant phase；不共享服务端
conversation 或 previous-response 引用。独立回放重建请求并核对候选、上下文及原生 token 用量。

正式实例只接受 `ResponsesHTTPTransport`，准入再次检查实际传输和资格绑定；自定义传输只用于
CPU 协议测试。`tools/qualify_message_provider.py` 留存两轮原生请求与响应，资格区分
`live_two_turn_message_provider` 和 `zero_gpu_contract_fixture_only`。显式执行该工具会调用 API；
fixture 标签本身不模拟网络。当前验证使用假传输，没有真实模型或 GPU 资格。

消息作者通过共同的 `run_runtime_factory` 接入 `Lab.execute_run` 和 `lab run execute`；原有 CLI
provider 继续服务其既有工程用途，不因上述消息协议测试获得正式消融隔离资格。

## Agent-led reproduction

For reference-kernel reproduction, bind the reusable
[AGENTS.md](../../../contracts/scaffolds/kernel-reproduction/AGENTS.md) as the arm's
scaffold. The launcher accepts `--agents-md`; the bound instructions are delivered in
the generated `AGENTS.md` on every provider turn. See the
[workflow and input guide](../../../docs/KERNEL_REPRODUCTION.md).
