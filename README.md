# open-cake-ir

**Agent 驱动的 GPU Kernel 搜索与 Compiler 演进。**

Agent 用 Python 编写显式执行计划，在固定任务与编译器版本下迭代 Kernel；
编译诊断和 GPU 评测帮助定位问题，维护 Agent 再将可复用的改进落实到编译器，供后续任务使用。
也支持从已有优秀实现出发，提取执行机制并用 Cake 改写。

A research system for agent-driven GPU kernel search and compiler evolution.

[技术报告 / Technical report](docs/README.md) · [中文文档](docs/zh-CN/README.md) · [English](docs/en/README.md) · [当前状态](reports/current/STATUS.md)

<!-- hardware-results:start -->
## 按硬件查看成果

2026-09-20 整理：展示已有确认结果与最佳实现维护记录。**加速比 = 各自固定基线耗时 ÷ 候选耗时**，超过 1× 表示更快。
各平台的输入与计时协议不同，图中各面板使用独立刻度；这些是选定任务的结果，不是跨硬件排名，也不是相对厂商最优库的成绩。

![按硬件分组的代表性确认结果](docs/results/overview.svg)

| 硬件 | 收录范围 | 阅读入口 |
|---|---|---|
| NVIDIA B300 / B200 | B300 服务实验 27 个任务、29 次尝试，含 `pairwise_sqdist` 7.646×；另含 CTA 宽度验证与 CAKE 对照。B200 保留正确性边界 | [NVIDIA](docs/RESULTS.md#nvidia) |
| Apple M1 Pro / M4 / M2 | M1 Pro 固定形状的历史确认、M4 两项晋升记录；M2 未通过稳定性的结果单列 | [Apple](docs/RESULTS.md#apple) |
| Hygon DCU BW1101 | 29 个任务中 27 个首个合格结果，含加速、变慢、未检出差异及计时分辨能力待查的条目 | [DCU](docs/RESULTS.md#hygon-dcu) |
| AMD gfx1151 | 两种设备计时器尚未对齐，暂不列合格性能 | [AMD](docs/RESULTS.md#amd) |

[完整任务与证据目录](docs/RESULTS.md) · [可筛选页面源码 / 下载后打开](docs/results/index.html) · [最佳实现如何维护](docs/zh-CN/TASK_INCUMBENTS.md)

目录区分“Campaign 内最佳合格候选”“有晋升记录”“历史确认”和“尚未合格”；每项可追溯到输入、版本、基线与原记录。
同一 Campaign 的确认历史保留在详情中；当前没有足够的跨代数据为每个任务绘制长期晋升曲线。

### 与原始 CAKE 的距离

TinyGEMM2 在同一 B300 上与 CAKE 的固定 FlashInfer 导出比较，Open-Cake 通过 30/30 严格数值检查，已测配置耗时仍约为 CAKE 的 11–37 倍。
KDA prefill、KDA decode、Alpha-MoE 尚无完整改写实测。[查看形状、计时协议和演进过程](docs/NVIDIA_CAKE_REPRODUCTION.md)。
<!-- hardware-results:end -->

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
