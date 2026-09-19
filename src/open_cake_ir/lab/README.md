# Lab implementation map / 实现导航

公开入口是 `Lab.preflight/execute/audit`。内部按五组职责阅读；公共导入仍为
`open_cake_ir.lab`，模块归组不增加转发层或第二份运行状态。

| 职责 | 主要入口 | 边界 |
| --- | --- | --- |
| Study 准入与绑定 | `core.py`、`preflight.py`、`contracts.py`、`bindings.py`、`admission.py` | 解析一次，先检查，再解析执行依赖 |
| 搜索与预算 | `execution.py`、`candidate_filter.py`、`selection.py`、`ralph.py`、`run_completion.py` | 全部候选完成过滤后才评测；预算和终态有明确归属 |
| 作者与构建 | `providers.py`、`provider_documents.py`、`provider_events.py`、`provider_invocation.py`、`toolchains.py`、`build.py` | 协议数据、进程调用和工具链各自负责自己的规则 |
| 证据与回放 | `archive.py`、`evaluation_writer.py`、`replay/`、`reporting.py` | 写入与独立核验分离；报告使用审计后的结果 |
| 任务接线 | `environments.py`、`pairing.py`、`task_package.py`；任务层 `TaskLab` | 通用引擎不导入具体任务实现 |

## Admission

`Lab.preflight` → `StudyContract.load` → `TaskLab._validate_study` → execution binding
resolution → `CampaignLock`.

`preflight.py` 是唯一 Study 解析入口。任务层接收同一个已解析对象，检查计时覆盖和执行模式，
随后才解析依赖。`core.py` 保留六项原有依赖，检查钩子不增加状态容器。
身份和配置不符时，在进程、GPU 和 Evidence writer 副作用之前拒绝。

## Execution and replay

`execution.py` 组织作者调用、构建、过滤与评测。`evaluation_writer.py` 负责一次评测的归档事务；
选择策略、预算和故障阶段仍归调用方所有。`provider_documents.py` 拥有协议常量，内部使用者直接导入。

`replay/__init__.py` 组织独立回放，`artifacts.py`、`attempts.py`、`candidates.py`、
`provider.py`、`selection.py`、`outcomes.py` 和 `refusals.py` 核对各自的原始记录。
它们不能调用 live execution 或 Evaluation writer 来证明自身正确。
`reporting.py` 只投影已审计事实，不产生新的证据。

## Changes

输入检查顺序、失败类别、预算计数、日志顺序和冻结格式都是行为合同。
源码变更提交为新的 clean commit；Campaign 固定的旧提交保持不变，历史记录由原提交的工具回放。
新代码进入规则的实际 owner，不增加转发函数、隐式注册表或通用状态字典。

The public facade remains stable. Parse each Study once, admit task-specific constraints before
resolving execution bindings, and keep replay independent of live writers. A new source commit
changes the identity of future execution; it does not rewrite historical evidence.

## Agent-led reproduction

For reference-kernel reproduction, bind the reusable
[AGENTS.md](../../../contracts/scaffolds/kernel-reproduction/AGENTS.md) as the arm's
scaffold. The launcher accepts `--agents-md`; the bound instructions are delivered in
the generated `AGENTS.md` on every provider turn. See the
[workflow and input guide](../../../docs/KERNEL_REPRODUCTION.md).
