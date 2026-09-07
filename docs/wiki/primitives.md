# IR 基本操作：拼出计算的积木

[中文首页](../zh-CN/README.md) · [English](../en/wiki/primitives.md) · [中英文对照](../README.md)

这些是执行计划的基本操作，不是一张“所有 GPU 算子已经支持”的清单。
同一个操作仍受类型、形状、访问方式、目标和后端限制；请对完整计划运行 `assess`。

[返回 Wiki](README.md) · [组合成常见算子](operators.md) · [代码中的操作定义](../../src/open_cake_ir/compiler/ir/operations.py)

| 需要做什么 | 操作 |
| --- | --- |
| 取数、存答案 | [load](#load)、[store](#store) |
| 对应位置计算或换精度 | [elementwise](#elementwise)、[cast](#cast) |
| 改变临时值的分组形状 | [reshape](#reshape) |
| 矩阵乘加和收尾 | [mma](#mma)、[epilogue](#epilogue) |
| 合并一组数、找编号、选前几名 | [reduce](#reduce)、[reduce_argmin](#reduce_argmin)、[top_k](#top_k) |
| 前缀累计、展开编号 | [scan](#scan)、[index_expand](#index_expand) |
| 分批计算注意力权重 | [online_softmax](#online_softmax) |
| 多组工作共同更新计数器 | [atomic_rmw](#atomic_rmw) |

## load

**读取数据。** 例如从输入表取出某一行，放进参与计算的临时 Buffer。
读哪个位置由 AccessMap 决定；它可以使用工作编号，也可以使用输入中的 INT32 索引。
多个索引按位置配对，不会自动变成所有组合。

普通全局读取和 TMA 数据搬运有不同要求。`reuse` 说明数据打算重复使用还是只读一次。
它不是一个“保证加速”的开关。

例子：[按索引读取](../../corpus/schedules/indexed-gather-b8-smoke.json)。
该示例的无效索引由掩码处理，读出的无效位置为零；不能据此推断所有访问方式都允许越界。

读取的结果形状必须与实际访问一致：只取一个数，不能贴上“128 个数”的标签再让 FMA 自动复制。
全是标量坐标时，单值结果使用 `[1]`。若要一组数，就要用合法的向量访问明确取到它们。
见 [这条检查的实际修复案例](../ACCESS_DOMAIN_REPAIR_20260906.md)。

## store

**把结果写回。** `y_tile → y` 表示把暂存的答案放进输出表。
地址由 AccessMap 定义，结果类型和舍入要符合规则。

写 `state` 还须证明不会有两份工作争写同一位置。
普通状态写要求工作坐标把状态分开；另一种已支持的索引写方式使用原子计数器分配唯一位置。
一句“我的索引肯定不重复”不能替代这个证明。

例子：[普通状态更新](../../corpus/schedules/state-store-b8-smoke.json)、[按预留位置写入](../../corpus/schedules/reservation-owned-store-b8-smoke.json)。
只更新状态的计划可以返回空元组，改变仍保留在传入的状态里。

Q8_1 的 typed store 按记录登记顺序接收 `d:fp16`、`s:fp16`、`qs:int8[32]`。
当前约束只允许一个 program 在 TileLoop 外写入，避免多组工作覆盖同一份量化记录。

## elementwise

**对每个对应位置分别计算。** 例如 `[1,2] + [3,4] = [4,6]`。
`op` 选择数学操作；形状相同并不代表任何类型组合都合法。

| `op` | 含义 | 小例子 |
| --- | --- | --- |
| `square` | 平方 | `3 → 9` |
| `abs` | 绝对值 | `−3 → 3` |
| `round` | 按声明的规则舍入为整数值 | `−2.5 → −3`，中点远离零 |
| `divide_no_nan` | 分母为零时返回零，否则做除法 | `6÷3 → 2`，`6÷0 → 0` |
| `rsqrt` | 平方根的倒数 | `4 → 1/2` |
| `exp` | 指数，常用于 softmax | `0 → 1` |
| `relu` | 负数变零，非负数保留 | `[-2,3] → [0,3]` |
| `tanh` | 把数压向 −1 到 1 之间 | `0 → 0` |
| `add` | 加 | `2+3 → 5` |
| `sub` | 减 | `2−3 → −1` |
| `mul` | 乘 | `2×3 → 6` |
| `div` | 除 | `6÷3 → 2` |
| `fma` | 融合乘加 | `2×3+4 → 10` |

浮点数能存的精度有限。FMA 对 `a×b+c` 只做一次最终舍入；先乘再加可能舍入两次，最后几位会不同。
本项目的 FMA 明确要求三个同形状 FP32 寄存器输入和 `ptx.fma.rn.f32`，不允许用标量或广播字段省掉输入。
`tanh` 也要声明目标指令合同，不会自动把精确要求换成近似指令。
`round` 要求 `rounding=nearest_away_from_zero`。这些量化组合操作仍受精确 Target、
FP32 类型和形状规则约束，不能仅凭名称绕过完整计划检查。

例子：[FMA](../../corpus/schedules/fma-b8-smoke.json)、[嵌套 FMA](../../corpus/schedules/fma-chain-b8-smoke.json)、[ReLU](../../corpus/schedules/relu-b8-smoke.json)。

## cast

**改变数字的存储类型。** 例如把一组 BF16 数转换为 FP32，方便后续计算。
增加存储位数不会找回之前舍入时丢掉的信息；减少位数则可能再次舍入。

它不是改变表的尺寸，也不是挪动数据到另一块 GPU。
允许哪些类型转换，由当前 Compiler 和后端检查。
例子：[类型转换](../../corpus/schedules/cast-b8-smoke.json)。

量化 producer 继续使用同一个 `cast`：FP32 到 FP16 明确声明
`rounding=nearest_even, overflow=ieee`；FP32 到 INT8 使用
`rounding=toward_zero, overflow=forbid`。舍入和溢出策略成对出现，不能只写一个。

## reshape

**把同一批临时值重新分组。** 例如 4 个元素可以从 `[4]` 改看成 `[2,2]`，
元素总数、顺序和类型保持不变。它不执行全局内存拷贝，也不声明新的 layout。
Verifier 检查输入、结果和目标约束；量化 producer 用它把一组值按 32 个一组组织。

## mma

**做矩阵乘加。** 一行 `[2,3]` 与一行 `[4,5]` 对应相乘再求和，得到 `2×4+3×5=23`。
扩展到多行，就一次产生一张结果表。项目的秩二收缩约定中，两个输入的最后一维都代表被求和的 K 维。

输入类型、累加类型、分块大小和指令合同都必须写清楚。
例如 FP32 输入是否允许 TF32 乘法精度，需要显式声明；不能默默放宽数值要求。
一个计划可以包含多个明确的 MMA 节点，依赖关系说明它们怎样组合。

例子：[矩阵乘加偏置](../../corpus/schedules/gemm-bias-b1-smoke.json)、[显式 TF32](../../corpus/schedules/fp32-tf32-mma-b1-smoke.json)。

## epilogue

**矩阵计算的固定收尾步骤。** 这里保留两种已有公式：

- `centroid_sq_minus_two_dot`：用于聚类距离比较的收尾。
- `bias_add_bf16_round`：加偏置后舍入到 BF16。

它不是“随便写一个函数”的位置。能用基本操作组合的计算，优先读懂组合本身。
例子：[聚类收尾](../../corpus/schedules/flash-kmeans-assignment-full.json)、[固定线性层源码路线](../../corpus/schedules/tinygemm2-stage4-split-k.json)。

## reduce

**把一组数合成一个数。** `sum` 求和：`[2,5,1] → 8`；`max` 找最大值：`[2,5,1] → 5`。
`axis` 说明合并哪一维；对表的列求和，会得到每行一个结果。
`scope` 说明需要哪个范围的线程合作。

普通归约使用 `algorithm=backend`。`xor_tree_32` 则明确要求常驻 FP32 值的最后一维为 32，
按 XOR 偏移 16、8、4、2、1 的顺序合并，不跨 TileLoop。
这个顺序属于数值合同，不能随意换成另一种求和树。

归约与后面的 `scan` 不同：归约只给合并结果，scan 保留沿途结果。
例子：[softmax 中的最大值和求和](../../corpus/schedules/softmax-b8-smoke.json)。

## reduce_argmin

**找最小值的位置。** `[4,1,8] → 1`，结果是从 0 开始的编号，不是最小值 `1` 的数值含义。
换成 `[4,2,8]`，输出仍是编号 `1`。

这里比较 FP32 值，返回 INT32 位置。真实任务中“两个候选同样好”怎样判对，由任务约定决定。
例子：[最近聚类中心](../../corpus/schedules/flash-kmeans-b32-smoke-v2.json)。

## top_k

**保留最大的 k 个数及原编号。** `[4,9,9,1]` 取前两名，得到值 `[9,9]` 和编号 `[1,2]`。
相同值按较小原编号优先，结果按值降序排列。

支持的常驻输入包括 FP32 分数与有符号 INT32 值；浮点输入有明确 NaN 策略。
分批 top-k 还需保留前一批的候选状态，不能把常驻版本的限制直接套过去。
例子：[常驻 top-k](../../corpus/schedules/top-k-b8-smoke.json)、[分批 top-k](../../corpus/schedules/top-k-streaming-b8-smoke.json)。

## scan

**保留每一步的累计结果。** 前向求和把 `[2,5,1]` 变成 `[2,7,8]`；反向则是 `[8,6,1]`。
当前扫描操作是 `sum`，方向用 `forward` 或 `reverse` 表示。

扫描中的顺序和浮点舍入属于实际数值行为；不要用一个普通求和结果替代整列累计输出。
例子：[前向累计](../../corpus/schedules/chunk-cumsum-b8-smoke.json)、[反向累计](../../corpus/schedules/chunk-cumsum-reverse-b8-smoke.json)。

## index_expand

**把一个块编号展开成多个元素编号。** 示例中 `scale=4、extent=4`，块编号 `2` 对应 `[8,9,10,11]`。
它只生成编号，随后还要通过 `load` 读取那些位置的数据。

无效块编号使用声明的 sentinel（示例为 `−1`），不能当成普通有效位置。
例子：[编号展开](../../corpus/schedules/index-expand-b8-smoke.json)。

## online_softmax

**分批处理分数，同时累计归一化所需的信息。** 不必把整张分数表同时放进快速存储。
每批输入是分数和对应的 value；状态包括当前最大分数、指数总和、加权累加值和归一化输出。
新一批可能提高最大值，因此旧累计值也要按新的尺度调整。

例如三个分数都相等，value 为 `[10,20,30]`，最终加权平均为 `20`。
分成两批处理也应遵守同一数学与舍入约定。
例子：[分批 softmax](../../corpus/schedules/online-softmax-b8-smoke.json)。

## atomic_rmw

**不可分割地读、改、写一个共享数，并拿到旧值。** 好比排队领号：计数器原来为 5，
一次加 1 拿到旧号 5；下一次拿到 6，不会都拿到 5。

当前形式是 INT32 原子加，内存顺序 `relaxed`、作用域 `device`。
它只保证这个原子更新的约定，不代表所有其他读写都自动完成同步。
例子：[原子预留](../../corpus/schedules/atomic-reservation-b8-smoke.json)。
