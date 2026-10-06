# C550 Agent 优化任务的准备与验收

开发池是现有 catalog 中可在 `xcore1002` 生成源码的54个任务 ID。原 Workload 继续负责
数学定义、完整形状、输入分布、oracle 和容差。54项不是54个独立算子族。
`cake_tinygemm2` 由任务定义限制为 B300，不在此池中。

## 一个明确的启动入口

新修复显式选择 `tools/launch_task.py --backend triton-metax --metax-timing gated-mean10-events`。
任务获得固定 starter、完整 Workload、允许的变换、生成源码反馈及研究记录要求。
普通任务保持任务自有 starter；GEMM+Bias 使用逐列输出、幅度分组和补偿合并的 MACA
工程 starter，保留完整 K 归约、FP32 产品、全部输出和原容差。

这是新的固定基线，不能把其收益与旧大驻留起点的分数混用。
`torch-reset-mean10-events` 保留默认流双填充控制；`native-mean10-events` 保留 D32 原生控制；`mean10-events` 保留 Python 事件；
`legacy` 保留 MCPTI。冻结记录在其原提交重放，各路径的计时区间与结果分开展示。

## 十次采样的具体含义

- 每个实现共十个正式样本，取算术平均：候选/基线、基线/候选两块，各五次。
- 每块十一轮预热另计。每次目标启动使用不同的初始化输出。
- 参数检查和指针打包在采样前完成。执行器创建独立目标 stream 和门控 stream。
- 门控 stream 的 host callback 只等待主机标志，不调用设备 API；开始、目标、结束命令
  全部排队后才释放标志。目标 stream 等待门控事件后开始，结束事件同步后读取耗时。
- 回调两秒内未释放则拒绝该次采样；失败仍释放两个 stream 和相关事件。
- 开始事件前，在目标 stream 两次填充四倍声明 L2 大小的 FP32-one 缓冲区。
  这两次填充均在测量区间外；动作和顺序保留在协议与原始记录中。
- GPU 开始前已提交完整采样命令；原生非门控路线仍可能包含提交间隙。
  各任务通过本路线的 A/A 与独立归因后才授予启动资格。
- 正式计时不启用 profiler。独立 MCPTI 提供一次设备归因与资源诊断。

该路径声明可信执行器的动作与顺序，不声称已经观测缓存清空或外部 dispatch。
有限正值、完整样本、设备和 launch 身份均为硬门。回调错误不会被 ctypes 忽略；
部分失败样本及完成调用数保留。当前路线只覆盖一个 dispatch，不授予完整 Program 资格。

## 从“任务存在”到“Agent 可启动”

CPU/source、基线编译、作者隔离、设备映射、完整正确性、同产物 A/A 和独立 profiler
全部通过后，任务才可启动。A/A 采用事先声明的1.05倍边界，不重抽样求通过。
每项保留明确状态：已准备、可启动、数值失败、启动资源拒绝、计时不稳定、采集故障、未验收。
源码生成和 CPU 测试各自不构成设备资格。

每任务一次 Claude Code / GLM-5.3 Run，三小时含十八分钟确认预留，token 只记账。
设备由既有 MACA 本机锁分配。已开始的九项保留自己的版本，新准备不重复发起它们的作者 Run。

## 为什么需要这些修复

原 Python 事件控制有36项 A/A 不稳定。原生 D32 提交改善了部分任务，RMSNorm 两臂
18.330/18.355微秒，并通过独立归因；SiLU 等仍有残留差异，因此需要新的重置/提交控制。

GEMM+Bias 原产物私有内存21572字节，首次启动报 `mcErrorMemoryValueTooLarge`。
逐列基线已能启动，但混合幅度分布有36个不合格输出，其他四类通过。补偿后继用于
修复抵消误差，原 oracle 和容差保持。

图提交探索在实际节点检查处拒绝，尚未执行目标 kernel，未获得任务资格。
图代码保留在 `21f07b36` 历史与对应外部目录；当前启动入口已移除该未验收路线。

## 验收来源

`eeff134f` 通过117项受影响合同，完整196项 Corpus保持通过。固定实现 `fa322406` 的
54项 CPU准备和完整 Run绑定完成。RMSNorm与SiLU已通过全数值/A/A/独立归因，两臂差异
分别约1.2%与0.85%。GEMM+Bias已在前一个直接事件后继通过全部五类输入和独立归因；
门控后继继续接受自己的验收。全量逐项设备验收继续
保存在 checkout 外。只有正式完成设备验收的任务可计入可启动集合。

- 当前后继：`c550-2:/root/open-cake-runs-reviewed/c550-evolution-parallel-author-20261006/gated-event-task-readiness-20261006/`
- 原生控制：`c550-2:/root/open-cake-runs-reviewed/c550-evolution-parallel-author-20261006/native-events-task-readiness-final-20261006/`
- 图拒绝证据：`c550-2:/root/open-cake-runs-reviewed/c550-evolution-parallel-author-20261006/verified-graph-task-readiness-20261006/`

原始 baseline、Workload、收据、拒绝与计划保留。RunSpecification、运行和审计继续由
已有 Lab/Run 所有者负责；本页提供协议和阅读入口。
