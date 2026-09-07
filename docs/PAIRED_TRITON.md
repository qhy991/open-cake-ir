# 同后端 Triton 已知基线优化

本轮沿用 `matched_search` 的 AuthoringEnvironment、候选封存、共同 Evaluation、预算和
追加式 Evidence。`open_cake` 臂支持 JSON Schedule 或现有受限 Python IR 前端；
`native_triton` 臂只接受已声明的 kernel-only Triton 子集。两臂从同一 Compiler lowering
提取的 kernel 开始，固定 Workload、case、ABI、oracle、后端、工具链和预算。这个处理条件
比较 agent 搜索效果，不代表完整 Triton、clean-start 或仅语法的因果比较。

所有 Run 使用 Ralph：先读只读的 `TASK.md`、`AGENTS.md`，再根据每轮 StateCard 更新 `candidate-set.json`，由外部控制器管理预算和停止。不再读取旧 prompt 模板。

稳定科学 Study 模板为 `contracts/studies/matched-search-triton-optimization-template.json`。
每臂三个预先安排的独立 Run，候选失败是观察结果，外部故障是 missing，不补跑；资格率和
条件确认延迟保留各自含义。Provider 的新输出 schema 需重新资格验证，模板中的 pending
引用没有 live 权威。Compiler/Executor 绑定由 CampaignLock 解析，预算和 Evidence owner 不变。

ABI 只取自 `WorkloadContract.tensor_abi(case_id)`；新边界没有算子到形状表。IR 源码只解析、
不执行，规范化为一个 Schedule，并在诊断中保留源码位置。Native 只接受固定 imports、单一
`@triton.jit` kernel 和声明的编译/launch 元数据；pointer 顺序必须遵守 Workload ABI。
对可信 Compiler baseline 的 kernel 提取是显式准备步骤，用户 native 源码则整份验证，不能
静默忽略 host 修改。详细支持语法见 [英文契约](en/PAIRED_TRITON.md)。

两臂共享需要 Linux bubblewrap 的监督构建器：只读 runtime/Compiler 挂载，独立临时目录、
HOME/cache、网络 namespace，只有构建目录可写。协调器不 import native Python；缺少隔离或
运行时依赖即报告 harness fault，不回退为普通 subprocess。新 manifest 投影 Workload ABI，
可信 launcher 复用既有 CUDA Driver 生命周期。共同 correctness assay 使用独立 materializer/
oracle，逐元素比较并验证输入不变；真实性能仍需共同的 fresh confirmation、CUPTI 与 profiler。

`Lab.audit` 的 paired view 包括所有预先安排的 repetition，保留失败、未资格和 missing。
`lab audit --threshold-ms <数值>` 从已审计 Evidence 提取首次 fresh confirmation 达到阈值的
turn、provider tokens 和已有 wall-time 观察；历史数据没有时间戳时返回 unknown。该查询是
描述性视图，不制造实验结果或改变科学估计。

R1 只有 CPU 契约与明确的 fixtures。Linux 隔离 canary、真实 Triton/GPU 编译、全 case 正确性、
共同 CUPTI 测量、profiler、provider 资格与框架 E2E 均为 R2 pending。旧 Compiler/Executor
release、审批和历史 Evidence 不改写；新源码需独立审查后的 successor release 才能 live 使用。
