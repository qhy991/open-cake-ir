# 常见算子：它们到底在算什么

**算子是一项明确的计算；一个算子可以由多个 IR 基本操作组成，也可能需要多个 GPU kernel。**
下面先看输入输出和小例子，再看项目中的完整文件。例子中的小数用于解释数学，不是 GPU 测试成绩。

[返回 Wiki](README.md) · [基本操作](primitives.md) · [完整任务合同](workloads.md)

## 矩阵乘法、偏置与分组矩阵乘法

矩阵可以先理解成一张数字表。矩阵乘法把一组输入与多组权重分别做“对应相乘，再相加”。
例如输入 `[2,3]`、权重 `[4,5]`，输出是 `23`；再加偏置 `1`，就是 `24`。
神经网络的线性层经常使用这种计算。

数据太大时分块，算完一个 K 块继续累加。分组矩阵乘法处理多组大小可能不同的任务；
ragged 表示每组有效行数可以不同，补齐的区域不能被当成真实数据。
block-scaled 形式为低精度数据配上分块缩放系数，数据与缩放块的对应关系必须正确。

项目示例：[GEMM+bias](../../corpus/schedules/gemm-bias-b1-smoke.json)、[不等长分组](../../corpus/schedules/ragged-grouped-gemm-b1-smoke.json)、[分块缩放](../../corpus/schedules/block-scaled-gemm-b1-smoke.json)。
完整固定线性层任务见 [TinyGEMM2](workloads.md#tinygemm2)。

## 仿射计算：每行乘一个数，再加一个数

例如某行输入 `[2,3]`，scale 为 `4`，bias 为 `1`，输出就是 `[9,13]`。
NCHW 平面仿射中的每个 `(batch,channel)` 有自己的系数；不能把不同 batch 的同一 channel 混用。
这类计算常见于归一化的收尾等位置，使用 [FMA](primitives.md#elementwise)时还须保留融合舍入规则。

项目有一份 [8 个输出的完整 GPU 检查实例](../AFFINE_PARENT_B200_CANARY_20260906.md)，
列出了每个输入、系数、答案和验证边界。它帮助理解怎么证明一个固定样本算对，不是通用性能成绩。

## RMSNorm：按一行的整体大小缩放

输入是一行数和每个位置的权重。先平方、求平均，再用这个平均值的平方根缩放原数，最后乘权重。
保护数 `epsilon` 避免除以零。

`y = x / sqrt(mean(x²) + epsilon) × weight`

例如 `[3,4]` 的平方平均是 `12.5`。权重都为 1、暂忽略很小的保护数时，输出约为 `[0.849,1.131]`。
这不会先减去平均值。

项目用 `square → reduce(sum) → rsqrt → mul` 等基本操作拼出来，
见 [RMSNorm 计划](../../corpus/schedules/rmsnorm-b8-smoke.json)。

## LayerNorm：先减平均，再调整尺度

输入也是一行数，但先算平均值，减掉平均值后，再按方差缩放；最后乘权重并加偏置。
方差可以理解成“偏离平均值的程度”：先求差，再平方，再求平均。

`y = (x − mean(x)) / sqrt(mean((x − mean(x))²) + epsilon) × weight + bias`

例如 `[1,3]` 的平均是 `2`，减去后为 `[-1,1]`；权重为 1、偏置为 0 时，答案接近 `[-1,1]`。
见 [LayerNorm 计划](../../corpus/schedules/layernorm-b8-smoke.json)。

## Softmax：把分数变成比例

输入是一行分数，输出是非负权重，总和接近 1。分数越高，一般分到的权重越多。
相同分数 `[0,0]` 得到 `[0.5,0.5]`。

计算时先减去行最大值，再求指数和总和，最后相除。减最大值用于控制数值范围，数学上的比例不变。
见 [Softmax 计划](../../corpus/schedules/softmax-b8-smoke.json)。
注意力中还会用 [online_softmax](primitives.md#online_softmax)分批累计。

## ReLU 与 SwiGLU：决定信号通过多少

ReLU 把负数变成零：`[-2,3] → [0,3]`，见 [示例](../../corpus/schedules/relu-b8-smoke.json)。

SwiGLU 有两路输入：一路控制通过程度，另一路提供被调整的值。
常见写法是 `sigmoid(x) × x × value`；当 `x=0` 时，结果为零。
本项目示例用明确的 `tanh` 指令合同组合 sigmoid 的数学表达，再做逐元素乘法。
见 [SwiGLU 计划](../../corpus/schedules/swiglu-b8-smoke.json)。

## RoPE：按给定角度旋转一对数

RoPE 给模型提供位置信息。对一对数 `(x1,x2)`，以及已经给出的 `cos、sin`：

```text
y1 = x1 × cos − x2 × sin
y2 = x1 × sin + x2 × cos
```

例如 `(1,0)` 配 `cos=0、sin=1`，得到 `(0,1)`。
当前示例接收旋转系数，用乘、减、加完成旋转；没有把求正弦余弦伪装成已支持的 IR 基本操作。
见 [融合 RoPE 计划](../../corpus/schedules/rope-b8-fused.json)。

## Flash-KMeans：每个点找最近的中心

输入是许多个点和候选中心，输出是每个点对应的中心编号。
例如点 `3`，中心为 `[0,5]`：平方距离是 `[9,4]`，因此输出编号 `1`。

多维时把各维的平方距离相加。找最近中心时，可以省略对所有中心都相同的点自身平方项，
所以计划里不一定直接出现完整距离公式的每一项。
矩阵乘法算大量点与中心的关系，后续操作找最小距离的位置。

[计算计划](../../corpus/schedules/flash-kmeans-b32-smoke-v2.json)与 [Workload 合同](workloads.md#flash-kmeans)分别说明怎样算、怎样判对。

## 索引、补齐、加权合并与状态更新

这些计算常和大算子一起出现，读写位置出错也会破坏答案。

| 计算 | 小例子 | 项目中的完整计划 |
| --- | --- | --- |
| 按索引读取 | 数据 `[10,20,30]`，编号 `[2,0]` → `[30,10]` | [indexed gather](../../corpus/schedules/indexed-gather-b8-smoke.json) |
| 无效行补零 | 只使用声明的有效前缀，补齐部分不当成输入 | [ragged zero pad](../../corpus/schedules/ragged-zero-pad-b1-smoke.json) |
| 加权合并 | `0.25×10 + 0.75×20 = 17.5` | [weighted combine](../../corpus/schedules/kda-weighted-combine-b8-smoke.json) |
| 状态相加 | 旧 `[1,2]` 加更新 `[3,4]`，原状态变 `[4,6]` | [state store](../../corpus/schedules/state-store-b8-smoke.json) |
| 累计和 | `[2,5,1] → [2,7,8]` | [scan](../../corpus/schedules/chunk-cumsum-b8-smoke.json) |
| 前几名 | 选值，并保留原位置，方便再读取数据 | [top-k](../../corpus/schedules/top-k-b8-smoke.json) |

状态必须按约定原地更新，未选中的位置保持不变；并不是所有 store 都允许任意并发写入。

## 注意力：先找相关信息，再加权组合

可以把 query 理解为“现在想查什么”，key 为“每条信息的检索特征”，value 为“实际要取的信息”。
先比较 query 与 key 得到分数，用 softmax 分配权重，再对 value 加权相加。
例如三个权重相等，value 为 `[10,20,30]`，输出就是 `20`。

- **DSA/稀疏 MLA decode：** 已给出稀疏编号，只读取有效候选，再做评分、softmax 和加权组合。
- **QSA prefill：** 先对信息分块、压缩和评分，再选部分块，最后做有因果限制的注意力。
  因果限制指不能读取“未来位置”的信息。该任务的输入从投影之后开始。
- **Kimi-K3 KDA decode：** 每次接收新一步信息，更新卷积和递归状态，再做带门控的归一化。
  可以把状态理解为不断更新的笔记；更新规则必须使用真正的任务公式，不能随意替换。
- **Kimi megaop：** 在上述核心前后接上投影和本地输出投影，边界到跨卡 AllReduce 之前为止。

这几种名字对应不同合同，不能互用成绩。具体输入、边界与当前接入情况见 [Workload 目录](workloads.md)。
