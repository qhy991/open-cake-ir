# C550 Agent 优化任务的准备与验收

开发池为现有 catalog 中可在 `xcore1002` 生成源码的 54 个任务 ID。任务的数学定义、
完整形状、输入分布、oracle 和容差由原 Workload 负责；54 项不是 54 个独立算子族。
`cake_tinygemm2` 仍由其任务定义限制为 B300，不在此开发池中。

## 统一入口

新实验显式选择 `tools/launch_task.py --backend triton-metax --metax-timing graph-mean10-events`。
普通任务沿用其任务自有 starter。GEMM+Bias 使用 MACA 工程 starter：逐列输出，保留完整 K
归约、FP32 产品及全部原始输出；按幅度分组并补偿合并，处理已观察到的抵消误差。
这是一个新的固定基线。它的收益不能与旧大驻留基线的分数混用。

`native-mean10-events` 保留原生逐次提交控制；`mean10-events` 保留 Python 事件路径；
`legacy` 保留 MCPTI。每条路线绑定自己的计时区间和原始记录。冻结 Run 保持原路线。

## 图事件测量的工作量

- 每个样本的图只有一次目标 kernel 启动，使用不同的预先初始化输出。
- 每张图依次执行四倍声明 L2 大小的 FP32-one 重置、开始事件、目标启动、结束事件。
- 受控非阻塞 stream 由执行器创建和释放；指针校验、图捕获与实例化均在正式采样之前。
- 每块十一轮预热。候选/基线、基线/候选两块各五次，共每臂十个正式样本，取均值。
- 正式计时不启用 profiler。独立 MCPTI 归因提供设备正确性与资源诊断。
- 这是一条单 dispatch 的评测路线；不授予多 kernel Program 计时资格。

重置动作和图中顺序由可信执行器保证，事件路径不声称观测了外部 dispatch 或证实缓存
已经清空。每臂十个样本的数量、有限正值、区间与设备身份均必须成立。

## 什么情况下可以启动 Agent

CPU/source、基线编译、作者隔离、设备映射、全 Workload 正确性、同产物 A/A 和独立
profiler 全部通过后，才将任务标为可启动。A/A 用事先声明的 1.05 倍边界，不重抽样求通过。
失败记录有明确分类：基线数值错误、启动资源拒绝、计时不稳定、采集故障或尚未验收。
不能用“已经生成源码”替代任务的设备验收。

每任务一次独立 Claude Code / GLM-5.3 Run，三小时包含十八分钟确认预留，token 只记账。
TaskPackage 提供原 Workload、当前 starter、允许的变换、生成源码反馈和研究记录要求。
设备按既有 MACA 本机锁分配。已经开始的九项保留自己的版本；新准备不发起重复作者 Run。

## 当前验证范围

原生提交后继 `00737e7f` 的完整控制仍在收集。RMSNorm 已通过新的同产物 A/A 与独立
归因；SiLU 等短任务仍保留不稳定结果。逐列但未补偿的 GEMM+Bias 已能启动，但在
`mixed_magnitude` 上有36个不合格输出；其余四种分布通过。这些是补偿和图提交后继
需要解决的具体问题，不是已完成的全量资格。

图提交和补偿 starter 在 `28a53cd2` 通过113项受影响的软件合同；完整196项 Corpus
保持匹配。实际54项设备验收和逐项状态保存在 checkout 外，只有完成验收的任务才能启动。
CPU 验证不构成54项性能测量资格。

- 原生控制：`c550-2:/root/open-cake-runs-reviewed/c550-evolution-parallel-author-20261006/native-events-task-readiness-final-20261006/`
- 图提交后继：`c550-2:/root/open-cake-runs-reviewed/c550-evolution-parallel-author-20261006/graph-events-task-readiness-20261006/`

以上目录保留 baseline、Workload、准备和设备控制的原始产物。新的 RunSpecification、
证据与计划仍由已有 Lab/Run 所有者负责；任务状态是它们的读取投影。
