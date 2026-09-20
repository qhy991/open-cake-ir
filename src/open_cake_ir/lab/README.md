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

## Complete Program candidates

Cake 作者可以提交完整 `Program`。公共 ABI 在 Workload 边界绑定一次，叶子 Schedule
只描述自己的参数；Triton builder 为每个 stage 保留独立编译产物和实际 launch metadata。
父候选封存 stage bundle，回放核对原始作者 Program、冻结 Compiler lowering 和编译产物。

Program 与单 kernel 共用 oracle 和 Evaluation receipt。物理 kernel/module 计数来自封存
stage，计时覆盖完整有序调用；NCU 输出按 dispatch 保留各 stage 的指标，不把逐 stage
指标冒充整体性能估计。每个计时 cohort 结束后释放它自己的输出与中间张量。

当前完整 Program 执行适配实现于 Triton/CUDA 路径；其他 code object 在构建及执行入口
明确拒绝，静态 Program 表示不因这个适配范围受限。软件合同测试不赋予实机正确性、
计时或跨架构收益资格。显式动作、知识授权与消息作者隔离见下节；正式消融仍在本任务中实施。

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
入口；正式 E/P Study 的冻结分配、统计分析和统一终态确认尚未完成。

## Message-only authoring

`message_provider.py::ResponsesRunProvider` 接入同一 Run 引擎。模型只收到冻结任务材料和本 Run
的消息历史，以 JSON 提交候选或显式动作；请求没有文件、shell、web 或可执行回调工具。
每个 Run 独立保留原生 response output，包括 reasoning 与 assistant phase；不共享服务端
conversation 或 previous-response 引用。独立回放重建请求并核对候选、上下文及原生 token 用量。

正式实例只接受 `ResponsesHTTPTransport`，准入再次检查实际传输和资格绑定；自定义传输只用于
CPU 协议测试。`tools/qualify_message_provider.py` 留存两轮原生请求与响应，资格区分
`live_two_turn_message_provider` 和 `zero_gpu_contract_fixture_only`。显式执行该工具会调用 API；
fixture 标签本身不模拟网络。当前验证使用假传输，没有真实模型或 GPU 资格。

当前消息作者已接入 `Lab.execute_run` Python API；生产 CLI/管理器接线仍待完成。原有 CLI
provider 继续服务其既有工程用途，不因上述消息协议测试获得正式消融隔离资格。

## Agent-led reproduction

For reference-kernel reproduction, bind the reusable
[AGENTS.md](../../../contracts/scaffolds/kernel-reproduction/AGENTS.md) as the arm's
scaffold. The launcher accepts `--agents-md`; the bound instructions are delivered in
the generated `AGENTS.md` on every provider turn. See the
[workflow and input guide](../../../docs/KERNEL_REPRODUCTION.md).
