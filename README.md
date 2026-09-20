# open-cake-ir

**Agent 驱动的 GPU Kernel 搜索与 Compiler 演进。**

Agent 用 Python 编写显式执行计划，在固定任务与编译器版本下迭代 Kernel；
编译诊断和 GPU 评测帮助定位问题，维护 Agent 再将可复用的改进落实到编译器，供后续任务使用。
也支持从已有优秀实现出发，提取执行机制并用 Cake 改写。

A research system for agent-driven GPU kernel search and compiler evolution.

[技术报告 / Technical report](docs/README.md) · [中文文档](docs/zh-CN/README.md) · [English](docs/en/README.md) · [当前状态](reports/current/STATUS.md)

## 关键算子：与 CAKE 对比

**2026-09-20：TinyGEMM2 已完成三个形状的严格数值验证，性能仍明显落后于 CAKE；其余三个算子尚未完成完整改写与实测。**
以下是 NVIDIA 开发分支的测量快照，硬件为 **B300-M2 / NVIDIA B300 SXM6 AC（`sm_103a`）**。
CAKE 一列来自论文提交到 [FlashInfer PR #4274](https://github.com/flashinfer-ai/flashinfer/pull/4274) 的固定 stage4 实现，在同一设备重新测量；这里没有使用论文的 B200 耗时。

| 算子 | 形状 B / N / K | CAKE（µs） | Open-Cake（µs） | Open-Cake 耗时 / CAKE 耗时 |
|---|---|---:|---:|---:|
| TinyGEMM2，BF16 + bias | 1 / 128 / 720 | 2.720 | 99.969 | 36.72× |
| TinyGEMM2，BF16 + bias | 16 / 1024 / 1024 | 3.040 | 106.016 | 34.87× |
| TinyGEMM2，BF16 + bias | 64 / 4096 / 3072 | 21.216 | 239.969 | 11.31× |
| KDA prefill | — | — | — | 未实测：缺完整循环状态计算与评测链路 |
| KDA decode | — | — | — | 未实测：缺完整状态、检查点与 GQA 契约 |
| Alpha-MoE | — | — | — | 未实测：缺完整 FP8 路由融合计算 |

耗时越低越好，最后一列 **1× 表示持平，超过 1× 表示 Open-Cake 更慢**。
三条 TinyGEMM2 数据属于同一个算子。Open-Cake 统一展示 `num_stages=2` 请求，未逐行挑选最快配置；前两个形状没有 K 循环，因此该参数不形成两级流水线。
测量源码为 [`539c6c81`](https://github.com/qhy991/open-cake-ir/commit/539c6c81e739e83a045ffd9ba361bad213cba82e)，CAKE 源码固定于 [`b38a0d68`](https://github.com/flashinfer-ai/flashinfer/commit/b38a0d686ee0d0be7a07b2ea559f2af14d2adf59)。

双方先通过同一个外部参考的 BF16 逐位相等检查，并保留独立 CPU 数学检查。
Open-Cake 的 2/4 两种配置共 **30/30** 通过，CAKE 共 **15/15** 通过。
计时使用 CUPTI，每次样本前清空 L2，关闭 CUDA Graph 和 PDL；五轮交替顺序，每轮每实现 25 个样本。
表中耗时为五轮中位数的中位数，比值为配对轮次比值的中位数；只统计 GPU Kernel 区间。
完整配置、波动、4-stage 结果、失败记录与原始证据位置见[复现报告](docs/NVIDIA_CAKE_REPRODUCTION.md)。

### Open-Cake 如何演进

| TinyGEMM2 阶段 | 改动与观察 | 验证结果 |
|---|---|---|
| 初始改写（`9aaace45`） | 单累加器未对齐参考实现的归约顺序，配置为 4/8 stages | 22/30 数值检查通过，不能进入性能比较 |
| 分组累加（`3235b947`） | 表达四路 K256 累加与固定合并顺序 | 短 K 的单次循环被编译器拒绝，尚无数值结论 |
| 短 K 去循环（`8fd0d5c9`） | 用带边界掩码的无循环执行计划处理短 K | 前 21 项数值检查通过，随后大形状 8-stage 超出共享内存限制 |
| 可运行配置（`539c6c81`） | 保留失败候选，选择 2/4 stages 完成同条件比较 | 30/30 数值检查通过，得到上表的首组性能数据 |

当前进步是数值语义、编译能力与评测链路的补齐；还没有多版本实测支持“性能持续提升”的结论。
原实现的 TMA 数据搬运、warp 分工与流水线、完整形状分发和框架端到端验证仍待补齐。
后续结果沿用相同形状、精度和计时边界，注明源码与硬件；详细报告保留历次结果，README 展示有证据的最新对照。

## 快速开始

需要 Python 3.10+，从源码安装：

```bash
git clone https://github.com/qhy991/open-cake-ir.git
cd open-cake-ir
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

按[入门教程](docs/GETTING_STARTED.md)完成无需 GPU 的检查与源码生成。
GPU 构建、运行和测量需要对应目标的工具链与设备环境。

## 继续阅读

- [系统架构](docs/ARCHITECTURE.md)：Kernel 搜索与 Compiler 演进如何衔接。
- [Python 前端](docs/zh-CN/PYTHON_FRONTEND.md)：如何编写执行计划。
- [已有 Kernel 改写](docs/KERNEL_REPRODUCTION.md)：参考实现、实验输入与验证流程。
- [概念与算子导读](docs/wiki/README.md)：从任务理解 IR 和评测结果。
- [研究路线](docs/ROADMAP.md)：跨架构迁移、优化复用与端到端验证目标。

硬件指南、设计决策与历史材料见[文档目录](docs/README.md)。

## 参与与引用

[贡献说明](CONTRIBUTING.md) · [平台与分支维护](docs/DEVELOPMENT_BRANCHES.md) · [安全问题](SECURITY.md) · [引用信息](CITATION.cff)

现有中英文文档共同构成本仓库的技术报告，作者为秦海岩（Haiyan Qin）。
报告入口提供推荐引用和 BibTeX；引用具体章节时使用提交永久链接。
测量结果另注明其源码提交、精确硬件目标、工作负载和计时范围。
联系：Haiyan Qin <haiyanq@buaa.edu.cn>。

本项目独立探索 [CAKE 论文](https://arxiv.org/abs/2608.12629v1)的部分思路，属于源码研究预览，并非官方实现。
自有代码与文档采用 [Apache-2.0](LICENSE)；第三方许可见 [NOTICE](NOTICE) 与[第三方说明](THIRD_PARTY_NOTICES.md)。
