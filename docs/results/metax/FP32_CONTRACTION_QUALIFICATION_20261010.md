# C550 FP32 contraction 到 IEEE MMA 的资格检查

本轮迁移 DCU 在 PR465 中形成的显式 Compiler 改写：将 FP32 broadcast multiply +
K sum 转为分块 IEEE MMA，保留公共张量、原 Workload 和点式尾部。主跟踪为
[Issue477](https://github.com/qhy991/open-cake-ir/issues/477)，实现见
[PR478](https://github.com/qhy991/open-cake-ir/pull/478)。这与执行组选择的资格分开。

## 实现边界

- 后继 `4a82d689` 只为 C550 准备 `A[M,K] × B[N,K]` 路径。合法 KN 输入被
  `operand_layout` 拒绝；gfx938 仍保留原来的两种布局。
- 不新增 IR 操作或放宽 Target。NT MMA 发射内部的 `tl.trans` 是已有 dot 实现，
  与另一个独立 `transpose` IR 操作的资格不同。
- 保留原 storage、别名、私有中间值、依赖、输出和 epilogue 守卫。改写结果重新通过
  assessment/lowering，16 组仍由 C550 后端拒绝。
- 归约顺序会变化，不宣称位级等价。原外部 oracle 决定数值正确性，变换只生成候选。

## 固定设备准备

原 `aka_gemm_nt_bias` 工厂要求 M/N 可被 8 整除，并要求 N/K 为二次幂。因此本轮使用：

| 实现 | M | N | K | 改写参数 |
|---|---:|---:|---:|---|
| 原始乘法/归约 | 24 | 32 | 64 | 原 starter |
| IEEE MMA | 24 | 32 | 64 | tile16×16×16、4组、stage1 |
| IEEE MMA | 8 | 8 | 128 | 同上，M/N 小于 tile |
| IEEE MMA | 8 | 16 | 32 | 同上，两个 K 迭代 |

每个 Program 检查原 primary、zeros、near_zero、alternating、mixed_magnitude；
原 `atol=rtol=2e-5` 不变。没有修改 oracle、输入 seed 或误差阈值。
奇数 M/N/K、residual、SiLU 的独立 CPU 语义测试不能被记为设备覆盖。

## 软件与原生检查

固定 `4a82d689` 通过 30 项本机合同、204 Corpus 和 115 发射快照，没有刷新期望。
四个 Program 已在新的、固定提交的 C550 环境完成原生构建；构建阶段无 GPU、网络或密钥。
现有 `qualify_tensor_program.py` 拥有构建。本轮没有新的搜索/Run owner。
原版本选择该工具的 Program 评测入口时，原实现 primary 在 CPU 准备阶段拒绝：原任务
只有标准库 materializer，没有 tensor-native materializer。失败发生于申请 GPU 之前，
没有设备调用或数值结论。旧环境停止，失败保留。

后继 `cc7cd5ab` 先在 CPU 准备全部原输入与参考，再通过已有单 kernel ABI、
`PreparedTensorCase`、`evaluate_tile_validation_case` 与 MACA native loader 检查。
单 kernel Program 已由既有构建器封成 Tensor ABI；没有修改任务 materializer、oracle
或 Workload。新入口明确 physical/runtime/PCI，按物理锁执行，每例独立进程并关闭模块。
后继的 30 项 CPU 合同、204 Corpus、115 发射快照通过；独立审查确认入口与实际 ABI 一致。

初始准备中的 M5/M1 被原任务契约拒绝，随后保留原契约改用 M8。新测试另修正了
ProgramStage 的读取接口，并区分 IR transpose 与已有 MMA 内部转置。这些失败全部发生于
CPU 准备，未运行 GPU，也不是数值验收失败。

## 设备与晋升状态

后继 `cc7cd5ab` 的原实现对照和三个改写候选均通过五类原输入，共 20 次原正确性
检查，没有放宽比较器。20 次调用、零 fallback、输入未变；实际 target/PCI 已核对，模块、worker、物理锁
和专用容器均已回收。独立审查核对原始收据后支持有限域晋升。该任务不报告性能收益。
后续每个候选仍需自己的原 oracle 和合格计时；接口允许的参数范围不等于本轮逐一测量
的组合，特别是其他 tile、组数、stage 与 epilogue。


| 实现 / M×N×K | 寄存器/线程 | 私有字节 | 动态共享字节 | 五类输入最大绝对误差 |
|---|---:|---:|---:|---:|
| 原实现 24×32×64 | 104 | 0 | 256 | 3.8147e-6 |
| MMA 24×32×64 | 32 | 0 | 2048 | 1.1444e-5 |
| MMA 8×8×128 | 32 | 0 | 2048 | 9.5367e-6 |
| MMA 8×16×32 | 32 | 0 | 2048 | 2.8610e-6 |

资源是驱动报告的静态分配。原 starter 与改写还改变了分块、执行组和归约顺序，不能把
分配变化单独归因于一条 MMA 指令，也不能把共享内存增加或寄存器减少当作加速证据。

## 组合版本与晋升处置

`7b9dddb4` 保祖先吸收已验收的 MetaX 执行组能力后，通过 60 项组合合同、204 Corpus、
115 发射快照。四个 Program 的 Workload、Program、投影原生 kernel 输入及全部编译要求
与设备受测 `cc7cd5ab` 完全一致；因此没有重复设备检查。

**有限 NT FP32 contraction→IEEE MMA 候选生成能力已获得设备证据并完成组合验证，
可通过 PR 晋升。** 这不是整个参数空间的正确性证明。奇数尾部和 residual/SiLU 仅有本轮
CPU 语义证据；每个新候选仍须通过原 Workload。KN 转置路径维持拒绝。
