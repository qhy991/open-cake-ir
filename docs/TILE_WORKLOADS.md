# 三个独立 Tile Workload

本轮为现有 RMSNorm、GEMM+bias 和 indexed gather Corpus 示例建立正式
Workload。契约拥有数学定义、数值和索引边界、输入输出 ABI、验证 case 及容差；
Evaluation 拥有独立 CPU oracle 和输入物化；`examples/paired_triton/prepare.py`
拥有示例 Schedule 的 case 专门化。Compiler 仍只接收完整 Schedule。

最小范围是连续、行主序、有限值的独立算子，输入不变，输出不与输入别名。
不包括模型集成、任意步长、动态运行时形状、训练或服务语义。
已有 Corpus 中的零值 metadata digest 不提供 Workload authority。

| 契约 | primary 输入 | primary 输出 | 数学定义 |
| --- | --- | --- | --- |
| `rmsnorm-fp32-v1.json` | x `[8,512,128]`、gamma `[128]`，FP32 | y `[8,512,128]`，FP32 | `x * gamma / sqrt(mean(x*x, last_axis) + 1e-6)` |
| `gemm-bias-bf16-fp32-v1.json` | a `[512,256]`、b `[256,256]`，BF16；bias `[256]`，FP32 | c `[512,256]`，FP32 | `sum_k a[m,k] * b[n,k] + bias[n]`，b 按 `[N,K]` 存储 |
| `indexed-gather-bf16-v1.json` | expert_rows `[4,8,16]`，BF16；expert_ids、row_ids `[8,8]`，INT32 | gathered_rows `[8,8,16]`，BF16 | 同一位置的 expert/row ID 成对索引；任一 ID 越界则整行正零 |

原始 Schedule 中 RMSNorm 的逐元素和归约中间值都是 FP32；GEMM 在 K loop
中累加 BF16 dot 的 FP32 结果，最后加 FP32 bias。两者未指定硬件归约树，
RMSNorm 的 rsqrt 也允许后端近似，因此 oracle 给出高精度数学结果并按契约
容差比较，不能要求和 CPU 某个归约顺序逐位一致。Gather 没有数值归约，要求
BF16 逐位相等，合法索引保留符号零，非法索引返回正零；负数不作 Python 式回绕。

本轮验收是三个契约可正式加载、微型数学反例和边界输入可在 CPU 上验证、
ABI 可直接供通用物化消费，并且 IR 与 native 源码由同一次 Compiler lowering
关联。GPU 编译、正确性、计时、profiler 和框架端到端验收全部属于 R2 待验证；
容差是预先声明的验收规则，尚无本轮设备校准结论。

共用入口是 `WorkloadContract.tensor_abi(case_id)`、
`tile_workloads.materialize_case(workload, case_id)` 和
`tile_workloads.reference_outputs(workload, case_id, inputs)`。ABI 按契约输入顺序、
随后输出顺序返回 `TensorABI(name, shape, dtype, mode)`；所有符号维度只从该 case
解析，不猜算子名字、不执行表达式。物化与 oracle 使用已经按 dtype 舍入的行主序
扁平 Python 列表。旧契约不被自动推断出新 ABI，也不改变旧的 admission。

`examples/paired_triton/prepare.py` 的 `baseline_schedule` 专门化现有 seed，
`prepare_baseline` 在新建的仓库外目录写出 `baseline.schedule.json`、
`baseline.triton.py` 和 `preparation.json`。native 源码就是配套 IR 的 Compiler
输出，是公开共同优化起点；它不构成 clean-start 对比的独立创作证据。
小型 GEMM case 只有 2–15 个输出，K=65/66 覆盖尾部与两次循环，避免完整大矩阵
Python oracle，也不绕过现有 Compiler 的单次 TileLoop 拒绝规则。

完整命令、输入验证边界和 A/B 接口见 [English reference](en/TILE_WORKLOADS.md)。
本修改是 Evaluation 后继版本工作；历史 Executor 钉住的 `workload.py` 原始字节
和历史契约仍由各自旧版本保留，新运行需要独立评审的 Executor 后继版本。
