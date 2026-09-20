# Lab implementation map / 实现导航

单次执行入口是 `Lab.preflight_run/execute_run/audit_run`，消费独立的 `RunSpecification`。
`Lab.preflight/execute/audit` 接受现有 matched-Study 输入，先投影为同一种 Run，再执行和汇总。
公共导入仍为 `open_cake_ir.lab`；所有 Run 共用一个搜索、预算和终态循环。

| 职责 | 主要入口 | 边界 |
| --- | --- | --- |
| Run 准入与 Study 输入绑定 | `core.py`、`preflight.py`、`run_spec.py`、`contracts.py`、`bindings.py`、`admission.py` | Run 冻结执行闭包；Study 预先指定条件与分析 |
| 搜索与预算 | `execution.py`、`candidate_filter.py`、`selection.py`、`ralph.py`、`run_completion.py` | 全部候选完成过滤后才评测；预算和终态有明确归属 |
| 作者与构建 | `providers.py`、`provider_documents.py`、`provider_events.py`、`provider_invocation.py`、`toolchains.py`、`build.py` | 协议数据、进程调用和工具链各自负责自己的规则 |
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

## Agent-led reproduction

For reference-kernel reproduction, bind the reusable
[AGENTS.md](../../../contracts/scaffolds/kernel-reproduction/AGENTS.md) as the arm's
scaffold. The launcher accepts `--agents-md`; the bound instructions are delivered in
the generated `AGENTS.md` on every provider turn. See the
[workflow and input guide](../../../docs/KERNEL_REPRODUCTION.md).
