# C550 执行组：正确性与资源的有界检查

本轮检验 DCU 已有执行组选择机制在 C550 上的一部分适用域。公开工具提交
`792c4863`，实际执行提交 `9114271b` 只绑定相同 C550 环境的主机记录。
使用现有 Task Workload、隔离原生构建器和 tensor correctness evaluator，没有 provider、
搜索 Run、性能样本或变换晋升。[资格入口 PR474](https://github.com/qhy991/open-cake-ir/pull/474)

## 固定条件

- RMSNorm H4096、fused-add RMSNorm H4096：batch 64，各测试 1、2、4 个执行组。
- RMSNorm H128：batch 32，测试 1、4 个执行组，作为小形状控制。
- 每个候选完整检查其原 Workload 的五种输入：primary、zeros、near_zero、alternating、mixed_magnitude。
- 仅改变 `execution_groups`。Workload、Schedule 的其他字段、cast 顺序、kernel 主体 AST、
  指针 ABI、grid、常量及其他编译选项保持不变。每个执行组的 lane 宽度来自 xcore1002 Target。

这不是独立 c550-bench 的 16×10 检查，也不是全部形状或执行组数量的资格。

## 实际结果

8 个候选，40 次检查全部通过；输入保持不变，零输出不匹配，一次 kernel 调用/检查，
零 fallback。模块关闭、worker 退出，专用容器已停止，物理锁已释放。

以下为驱动报告的静态分配，两个 H4096 任务分别得到相同资源数值：

| 任务 | 执行组 | 寄存器/线程 | 私有内存分配字节 | 动态共享内存字节 |
|---|---:|---:|---:|---:|
| RMSNorm H4096 | 1 | 256 | 336 | 0 |
| RMSNorm H4096 | 2 | 224 | 0 | 8 |
| RMSNorm H4096 | 4 | 128 | 0 | 16 |
| fused-add RMSNorm H4096 | 1 | 256 | 336 | 0 |
| fused-add RMSNorm H4096 | 2 | 224 | 0 | 8 |
| fused-add RMSNorm H4096 | 4 | 128 | 0 | 16 |
| RMSNorm H128 | 1 | 48 | 0 | 0 |
| RMSNorm H128 | 4 | 44 | 0 | 8 |

CPU：本机 17 项合同、新环境 27 项合同、204 Corpus 与 115 发射快照通过；
八个候选均完成原生编译。设备阶段没有计时器或 profiler，所以没有加速比或动态 spill
流量结论。私有分配为零也不能推出最佳速度；更多执行组增加了共享内存及协作成本。

## 对 Compiler 的意义

这一对照把执行组数量从先前同时修改 cast/cache 的 Agent 候选中单独分离出来。
它证明：在这两个固定 H4096 Workload 上，单独改变工作分配可以降低静态寄存器与
私有内存分配，且保留原正确性。H128 控制没有相同的私有内存改善，不能把 4 组设为通用默认。

可沉淀的机制仍是显式的工作分配改写；选择 1/2/4 或其他组数应由任务、资源反馈和实测决定。

## 晋升处置

**本轮 No promotion。** 三个独立只读审查均未发现这八个候选正确性证据的矛盾。
但是，当前检查直接构造普通 Schedule，没有在 C550 上调用仍被拒绝的生产 pass。
把 xcore1002 直接加入整个 `specialize_triton_warps` 证据集，会额外开放 8/16 组、
pointwise、MMA、固定循环等本轮没有覆盖的域。因此生产门保持关闭。

下一步补齐额外执行组和结构类型的原正确性、资源与负例；在后继中把真正 pass 的输出
与已验收的显式候选对照，再决定目标适用域。全部历史源码与结果保持冻结。

launch receipt 中 `module_unloaded=false` 记录调用时刻；per-case 的 `module_closed=true`
记录随后 `finally` 清理。两者是不同时间点，不能把前者误读为终态模块泄漏。

相关工作：[DCU 到 MetaX 复用分析](DCU_COMPILER_REUSE_20261010.md)、
[Issue473](https://github.com/qhy991/open-cake-ir/issues/473)。
