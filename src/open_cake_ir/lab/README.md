# Lab 代码维护 / Lab maintenance

Lab 保留 `Lab.preflight/execute/audit` 公开入口。内部按职责分工，避免让一个模块同时决定
配置是否合法、启动进程、写入证据和解释结果。500 行是检查入口，不是正确性或质量证明。

| 职责 / Responsibility | Owner |
| --- | --- |
| 公开 Lab 门面（preflight/execute/audit 生命周期入口） | `core.py` |
| Study 解析与 Campaign Lock 投影 | `preflight.py` |
| Study 与 Campaign 公共记录及持久化文档校验 | `contracts.py` |
| 外部执行定位器一次性解析进 Campaign Lock | `bindings.py` |
| 内容绑定的 Executor Revision（所有 Study 变体共享） | `executor.py` |
| Provider、预算、评测协议的声明检查 | `admission.py` |
| Provider 配置面向 preflight/execution/replay 的单一投影 | `provider_policy.py` |
| 已绑定运行依赖的实际身份检查 | `execution_admission.py` |
| 运行顺序、Turn、预算、故障阶段 | `execution.py` |
| provider、toolchain 与 broker 命令的进程组监督 | `process.py` |
| 运行局部 checkpoint 投影与可达性 | `checkpoints.py` |
| 终局观察策略（与 token 检查点分离） | `endpoints.py` |
| 故障留存和终止检查点 | `run_completion.py` |
| 类型化外部/协议偏差到唯一终局路径的映射 | `faults.py` |
| 全集构建、过滤和拒绝记录 | `candidate_filter.py` |
| 一次评测的预算计数、收据校验与归档 | `evaluation_writer.py` |
| 逻辑 evaluator 调用回放（不推断物理设备工作） | `evaluation_lifecycle.py` |
| Campaign 聚合与描述性 claim 投影 | `reporting.py` |
| 输出 custody（Lab 与其 CLI 共享） | `custody.py` |
| Provider 数据记录、候选信封与字节边界 | `provider_documents.py` |
| JSONL 事件、辅助活动与终止消息解析 | `provider_events.py` |
| Claude 原生 provider 的 Turn 边界件（事件契约、用量核算） | `claude.py` |
| 初次与恢复调用的命令及 host 绑定 | `provider_invocation.py` |
| Provider 进程执行与 adapter；兼容的公开导出 | `providers.py` |
| Authoring Environment 与构建输入/输出接口 | `environments.py`、`build.py` |
| Triton 内核编译（Linux 文件系统 jail 下监督执行） | `triton_build.py` |
| CuTe 工具链（文件系统隔离，汇入公共 CUBIN 启动边界） | `cute_build.py` |
| Metal 二进制归档构建（BuildRequest 接缝处） | `metal_build.py` |
| Metal 原生 host 绑定（Executor admission 边界处） | `metal_host.py`、`metal/` |
| 任务自有 Python starter 绑定（不引入第二种 Schedule 语言） | `python_reference.py` |
| Broker 配置解析与实际进程运行 | `runtime_config.py`、`runtime.py` |
| 证据写入 | `archive.py` |
| 独立回放顺序 | `replay.py` |
| 回放中的 provider、candidate、selection、outcome | `replay_provider.py`、`replay_candidates.py`、`replay_selection.py`、`replay_outcomes.py` |
| Broker attempts 与原始 artifact/receipt 校验 | `replay_attempts.py`、`replay_artifacts.py` |
| 纯咨询性的候选排序与资格判定 | `selection.py` |
| 每条诊断送往必须改变的owner（candidate/verifier/cost model/IR） | `routing.py` |
| 被拒成员的有界作者反馈 | `diagnoses.py` |
| 仅由 StateCard 已有反馈导出的作者指南 | `rubrics.py` |
| 同线程 Ralph 迭代的预算与 StateCard 控制 | `ralph.py` |
| 同后端已知基线配对（走既有 Lab 路径） | `pairing.py` |
| Authoring Environment 的参考可见性策略（S5 per-arm） | `reference_access.py` |
| 由已解析 Campaign 权威派生的两文件 Agent 接口 | `task_package.py` |
| 私有文档原语与规范策略值（不对外） | `_documents.py`、`_policies.py` |
| 公开名字的原地重导出（唯一允许的转发层） | `__init__.py` |

新代码应进入实际负责该规则的模块。公开类型和函数可以直接重导出同一个对象；不要再包一层
转发函数，不用隐式注册表、反射分派或通用状态字典绕过职责边界。

输入检查的顺序、失败类别、预算计数、日志顺序和冻结格式均是行为合同。尤其要保留：

- 配置和身份不符时，在进程、GPU、Evidence writer 副作用之前拒绝。
- 全部候选完成构建/过滤后才花费 GPU 时间。
- search、confirmatory、attribution 共用写入事务，但各自的调用条件和选择策略仍有原来的 owner。
- 独立回放不能调用 live execution 或 Evaluation writer 来证明自身正确。
- Lab 文件属于 Executor source closure；改变它们需要 successor，旧 descriptor 和旧 Git 回放保持不变。

**English.** The public Lab facade stays stable. Internal modules own admission, live orchestration,
provider event parsing, process invocation, artifact writing and independent replay separately.
Share the live Evaluation-to-evidence transaction rather than maintaining three copies. Keep
selection policy and fault-stage ownership with the caller. Preserve rejection order, budget and
clock accounting, raw artifact formats, and public import identity. Replay must remain independent
of live writers. Use a source-bound Executor successor for changed Lab code; never rewrite an old
descriptor to accommodate a refactor. File length is a review signal, not an acceptance test.
