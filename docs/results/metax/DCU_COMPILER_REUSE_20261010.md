# 从 DCU 开发结果选择 C550 Compiler 改进

## 后续设备进展

执行组选择已完成 43 候选、215 项 C550 原正确性检查；有限域实现与审查见
[执行组资格记录](WIDTH_QUALIFICATION_20261010.md)。目前没有性能结论，FP32 contraction
到 IEEE MMA 的独立变换仍未对 C550 开门。下文源码对照保留当时的固定版本范围。

## 结论

可以复用。优先复用完整变换、适用条件和反例，再为 C550 选择参数并建立设备证据。
DCU 的默认参数、指令资格和加速比不能直接移植。

本次比较 `metax@439b4b5d` 与 `main@bdd52ce1`。当时维护分支 `dcu@72dcba90`
和 `metax` 的 Compiler 目录没有差异。最近两项 DCU 开发成果已经通过
[PR464](https://github.com/qhy991/open-cake-ir/pull/464) 和
[PR465](https://github.com/qhy991/open-cake-ir/pull/465) 进入 `main`，尚未同步到这两个平台分支。
任务分支以保留祖先的 merge 准备 MetaX 软件后继；冻结实验不改动，C550 变换资格门不放宽。

## 已执行的 C550 源码对照

对统一目录的 55 个任务，使用各任务现有默认形状、真实 Workload 工厂和 Compiler：

- 两版均有 54 个任务可构造并降级；TinyGEMM 仍在工厂边界拒绝 C550。
- 52 个任务减少了未使用的 constexpr 参数，总数从 349 降到 202。
- 54 个任务的 Workload、Schedule、kernel 主体 AST、指针 ABI、grid 和编译选项一致。
- 剩下的 constexpr 值没有变化。14 项上游针对性合同在固定 `bdd52ce1` 通过，
  Corpus Gate 204/204 通过，没有刷新期望。
- 在 C550 的 RMSNorm、GEMM、GEMM+SiLU、转置 GEMM+bias 上，执行组选择和新的
  contraction 变换均明确返回 `target_route`，没有把 gfx938 资格当成 C550 资格。

这些是 CPU 构造、检查和发射证据。54 个可降级任务不等于 54 个新设备实验已通过。
本轮没有原生编译、GPU 时间、provider 或加速比。

## 应当优先做什么

| 优先级 | 来源机制 | MetaX 当前边界 | 应验证的独立问题 |
|---|---|---|---|
| 高 | 显式执行组选择 `specialize_triton_warps` | 基础 Schedule 能表达，便捷变换的资格集尚无 xcore1002 | 固定 RMSNorm / fused-add RMSNorm，保持运算、cast 位置与 tile，仅比较 1/2/4 组；先原正确性与资源，再合格计时 |
| 高 | FP32 broadcast multiply + K sum → tiled IEEE MMA | xcore1002 已声明 `triton.dot.fp32_ieee`，但新 pass 只接受 gfx938 | 首选 A[M,K] × B[N,K]，避免叠加转置资格；再做 KN、尾部及混合幅值反例。保持原 dtype、oracle 和点式尾部 |
| 中 | 保留舍入边界的 tiled epilogue fusion | 共享 pass 已在 MetaX 代码中；完整 Program 计时仍有独立门槛 | 检查已正确 Gate-Up 的私有中间值与消费者是否满足融合守卫；若适用，比较原完整 Program 与融合候选，不把某个叶 kernel 的时间当整题收益 |
| 低成本同步 | 删除未用 constexpr 参数 | 本次 54 任务源码对照已通过 | 验证 MACA 原生编译接口；不要预设设备 kernel 更快 |
| 后续 | store-loop pipeline 深度选择 | 当前仅对 gfx938 开放；MACA 实际实现和资源变化未验 | 固定 store region，改变一个深度参数，检查真实发射、资源与完整输出；不默认更深更快 |

### 为什么不把 constexpr 精简作为主要速度实验

DCU 的观测是含输出分配和同步的普通调用；PR464 自身也未声称稳定的一般加速。
C550 当前独立评测通过 `metax_driver.py` 加载封存的 `mcfatbin` 并调用原生入口，
不在每次样本中重新走生成的 Python/Triton JIT 包装器。源码参数精简有接口和维护价值，
但不能据此预测本路径的设备延时下降。

### 为什么 MMA 值得做，但不能直接开门

PR465 把 Agent 已发现的完整计算改写做成了显式工具，保留 ABI、Workload 和尾部计算，
由调用者选择 M/N/K tile、组数和 pipeline。它改变归约顺序；已有 DCU 的混合幅值
GEMM+bias 反例在两臂都失败，说明原 oracle 仍是必要边界。

C550 初轮应使用纯 FP32 GEMM / GEMM+SiLU / NT GEMM+bias 开发任务。
不能拿这个 pass 直接替换 BF16 Gate-Up，也不能删去已存在的数值补偿或改用未准入 TF32。
“Target 声明支持 IEEE dot”只说明存在一个可用部件，不证明新的完整改写已通过设备验证。

## 最小、有解释力的后续资格

1. 执行组：两个 H4096 开发任务各 1/2/4 组，共六个候选；固定 cast 次序、分块、输入、
   参考和全部其他参数。另保留小形状控制与不合法资源/同步承诺反例。
2. MMA：先一个 NT 正例及尾部，再扩展至普通 KN 和 SiLU；加入混合幅值、强抵消、
   alias、额外消费者和舍入前置反例。每个候选按原比较器独立验收。
3. 通过设备正确性不自动开放变换。记录精确 C550 源码、工具链和资源证据，经审查后在
   后继提交中扩展证据集。tile 默认选择仍由 Lab 决定。
4. 性能只能使用独立合格的计时协议与新基线。当前 Program 计时问题不靠降低标准解决；
   若计时未准入，只报告正确性、代码生成、资源与失败，不宣布收益。

这些实验回答“已知机制能否成为 C550 可用的 Compiler 工具”。没有 Agent 对照组时，
不能回答“工具是否减少搜索成本”；没有独立 Bench 时，不能宣布泛化收益。

## 重做本次 CPU 对照

使用干净的固定 checkout 和各自源码；报告写在 checkout 外：

```sh
python tools/benchmarks/c550/review_dcu_reuse.py --source-root /tmp/metax-439b4b5d --output /tmp/metax-review.json
python tools/benchmarks/c550/review_dcu_reuse.py --source-root /tmp/main-bdd52ce1 --output /tmp/main-review.json
python tools/benchmarks/c550/review_dcu_reuse.py --compare /tmp/metax-review.json /tmp/main-review.json --output /tmp/reuse-comparison.json
```

输出是派生审查记录，不是新的 Run、Study 或评测执行器。比较器逐任务检查真实对象及 AST，
不使用文件哈希代替语义对照。来源资格分别见 PR464、PR465、
[执行组变换 PR312](https://github.com/qhy991/open-cake-ir/pull/312)、
[融合 PR310](https://github.com/qhy991/open-cake-ir/pull/310)、
[store-loop PR375](https://github.com/qhy991/open-cake-ir/pull/375)。

## 软件集成验收与后续合并边界

任务分支的保祖先集成提交 `499a3816` 已通过 27 项受影响合同、204 项 Corpus Gate
及 115 个发射源码快照。Compiler 目录与已审查 `main@bdd52ce1` 相同。源快照变更沿用
上游独立采纳提交，本任务没有重新生成期望，也没有修改 Target 或变换资格集。

这不替换原 Bench 的 `b2180ca2`。今后合入那个实验集成时，还有两个明确的记录冲突：
`F-2026-10-10-001` 在 main 指 constexpr，在 Bench 分支指 Mapping 边界；
`F-2026-10-10-002` 分别指 MMA 改写和不可变 Evaluation 序列化。需要在后继中明确
区分记录并更新引用，历史冻结提交按原身份保留；不能把 Git 无文本冲突当作证据整合完成。
