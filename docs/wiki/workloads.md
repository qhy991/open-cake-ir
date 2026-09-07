# Workload 目录：项目里有哪些完整任务定义

[中文首页](../zh-CN/README.md) · [English](../en/wiki/workloads.md) · [中英文对照](../README.md)

Workload 是一道题的完整约定：给什么输入，必须输出什么，怎样判对，怎样测量。
它比“矩阵乘法”这个名称更具体。相同名称、不同形状或精度，可能是两道不同的题。

[返回 Wiki](README.md) · [先理解数学](operators.md) · [合同原目录](../../contracts/workloads/README.md)

**这里列出的是合同，不是“所有任务都已接通 Lab、通过 GPU 或达到最佳性能”的排行榜。**
选择哪一版由具体 Study 决定，不能只看文件名里最大的版本号。

## 独立 Tile Workload：归一化、矩阵乘与索引读取

这三个合同把已有 Corpus 示例明确为独立算子，分别固定输入域、tensor ABI、
数学参考与逐元素判对规则。

| 合同 | 计算与输出 |
| --- | --- |
| [RMSNorm FP32 v1](../../contracts/workloads/rmsnorm-fp32-v1.json) | 沿最后一维归一化并乘 gamma，输入与输出为 FP32。 |
| [GEMM+bias BF16/FP32 v1](../../contracts/workloads/gemm-bias-bf16-fp32-v1.json) | BF16 的 A、B 按 `A @ B.T + bias` 计算，bias 和输出为 FP32。 |
| [Indexed gather BF16 v1](../../contracts/workloads/indexed-gather-bf16-v1.json) | expert/row ID 成对选择 BF16 行；任一 ID 越界时输出整行正零，负数不回绕。 |
| [RMSNorm FP32 v2](../../contracts/workloads/rmsnorm-fp32-v2.json) | 同一数学定义与 6 个用例，明确绑定 B300 `sm_103a`。 |
| [GEMM+bias BF16/FP32 v2](../../contracts/workloads/gemm-bias-bf16-fp32-v2.json) | 同一数学定义与 5 个用例，明确绑定 B300 `sm_103a`。 |
| [Indexed gather BF16 v2](../../contracts/workloads/indexed-gather-bf16-v2.json) | 同一索引语义与 4 个用例，明确绑定 B300 `sm_103a`。 |

B300 的 Python 起点和单独的实验资格要求见 [B300 指南](../B300.md)。

形状、容差、共用 ABI 和独立 CPU oracle 见 [Tile Workload 指南](../TILE_WORKLOADS.md)。
[基线准备入口](../../examples/paired_triton/README.md)从同一 IR 生成对应的原生 Triton
源码，供明确声明的共同优化起点使用。当前交付范围是合同、CPU 参考与源码准备；
GPU 编译、正确性、计时、profiler 和框架验收仍待 R2 验证。

## Flash-KMeans

- **输入：** 一批点和聚类中心，BF16 存储；距离计算与累加遵守 FP32 约定。
- **输出：** 每个点最近的中心编号。
- **判对：** 独立距离参考与合同的同距规则；不能临时要求所有同距情况都返回同一编号。
- **合同：** [v1](../../contracts/workloads/flash-kmeans-assign.json)、[v2](../../contracts/workloads/flash-kmeans-assign-v2.json)。
- **代码入口：** [flash_kmeans.py](../../src/open_cake_ir/evaluation/flash_kmeans.py)、[教学工具](../../examples/gpu/flash_kmeans_quickstart.py)。

v2 是后继合同，旧实验继续使用它原来固定的版本。学习 Compiler 时可以先看 [对应计划](../../corpus/schedules/flash-kmeans-b32-smoke-v2.json)。

## TinyGEMM2

- **输入：** 固定形状的输入、权重和偏置；任务是小批量线性层。
- **输出：** `input × weightᵀ + bias` 后按约定舍入到 BF16。
- **判对：** 独立 FP32 线性层参考，再执行约定的 BF16 舍入和容差检查。
- **合同：** [v1](../../contracts/workloads/tinygemm2-stage4.json)、[v2](../../contracts/workloads/tinygemm2-stage4-v2.json)。
- **区别：** v2 还固定具体输入、权重、偏置与参考结果的物化内容；只保持随机种子不一定足以复现相同字节。

其保留的 [Schedule](../../corpus/schedules/tinygemm2-stage4-split-k.json)使用 `checked_cuda_asset`：
它选择经过核验的固定源码，不代表编译器能从任意新计划生成这一整类算子。

## DSA：稀疏 MLA 单步解码

- **输入：** query、缓存的压缩 KV、位置相关部分，以及要读取的稀疏编号。
- **输出：** 对有效候选完成评分、softmax 与加权读取后的注意力结果；没有有效候选时按合同输出零。
- **范围：** DeepSeek-V3.2 的这个固定单 GPU 任务，含来源捕获和生成的测试行；不是整个模型服务。
- **合同：** [DSA v1](../../contracts/workloads/dsa-attention-sparse-mla-decode-v1.json)。

合同还保留了来源文字与实际缩放常量不一致的记录：使用它指定的执行值，不能自行按旧说明重算一个常量。
任务验证入口由 [WorkloadContract](../../src/open_cake_ir/evaluation/workload.py)负责。

## QSA：长序列的筛选与注意力

- **输入：** 已完成投影的 `q、k、v、index_q、index_k`。
- **计算：** 完整因果块 → 平均压缩索引 key → LayerNorm → 按索引 head 评分 → 选块 → 展开 token → 所选 token 的因果注意力。
- **输出：** 完整 QSA 任务范围内的结果，不含前面的投影、RoPE、缓存更新或服务调度。
- **合同：** [QSA prefill v1](../../contracts/workloads/qsa-prefill-t32768-v1.json)。
- **执行说明：** 多个阶段需要完整 Program；只测一个 top-k 或 attention kernel 不能替代完整 Program 计时。
  [专用评测工具](../../tools/evaluate_qsa_candidate.py)保留这个边界。

较小检查形状与目标形状分别在合同中列出。它们是任务声明的几何，不等于某个模型 checkpoint 的完整配置。

## Kimi-K3 KDA：带状态的单步核心

- **输入：** 当前一步的信息、门控参数、卷积状态、递归状态和缓存位置。
- **计算：** 卷积更新 → KDA 递归更新 → 带 sigmoid 门控的 RMSNorm。
- **输出：** 当前步结果，并按合同更新选中状态；未选中位置保持不变，填充行不能乱写状态。
- **合同：** [KDA fused decode v1](../../contracts/workloads/kimi-k3-kda-fused-decode-v1.json)。

该合同有多种本地 head 数和活动行情况，不能拿其他 prefill、chunk 或完整模型成绩替代。
输入输出包含状态，判对时要检查状态池和连续两次调用，不能只看最后一张输出表。

## Kimi-K3 megaop：加上核心前后的投影

- **输入：** 当前层输入、投影权重、卷积与递归状态等完整参数。
- **计算：** qkvg 与门控投影 → 上述状态核心 → 本地输出投影。
- **边界：** 单张 B200、本地 rank 的模块；到跨卡 BF16 TP AllReduce 之前停止。
- **合同：** [v1](../../contracts/workloads/kimi-k3-kda-decode-megaop-b200-v1.json)、[v2](../../contracts/workloads/kimi-k3-kda-decode-megaop-b200-v2.json)。
- **区别：** v1 保留前后基线夹住候选的计时协议；v2 使用成对计时和置信区间。不得事后互换判分标准。

这里已接入合同加载与边界校验。**单靠注册不提供完整 megaop Lab evaluator，也不产生 GPU 正确性或速度结果。**
具体形状、oracle、输出存储和状态规则都在各自合同里。

## 为什么有些示例不在这张合同表里

Softmax、RoPE 等还以 [Corpus Schedule](../../corpus/manifest.json)或独立测量任务出现。
[状态更新](../../examples/gpu/state_store_b200_correctness/README.md)和 [FMA](../../examples/gpu/fma_b200_correctness/README.md)
也有自己的固定 GPU 任务。

它们并不会因为存在一个 JSON 或 README，就自动成为可由任意 Study 调用的完整 Workload。
新增任务时，要分别交付语义合同、可调用的评测路径和需要的验证记录。
