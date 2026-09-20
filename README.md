# open-cake-ir

**Agent 驱动的 GPU Kernel 搜索与 Compiler 演进。**

Agent 用 Python 编写显式执行计划，在固定任务与编译器版本下迭代 Kernel；
编译诊断和 GPU 评测帮助定位问题，维护 Agent 再将可复用的改进落实到编译器，供后续任务使用。
也支持从已有优秀实现出发，提取执行机制并用 Cake 改写。

A research system for agent-driven GPU kernel search and compiler evolution.

[技术报告 / Technical report](docs/README.md) · [中文文档](docs/zh-CN/README.md) · [English](docs/en/README.md) · [当前状态](reports/current/STATUS.md)

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

[贡献说明](CONTRIBUTING.md) · [安全问题](SECURITY.md) · [引用信息](CITATION.cff)

现有中英文文档共同构成本仓库的技术报告，作者为秦海岩（Haiyan Qin）。
报告入口提供推荐引用和 BibTeX；引用具体章节时使用提交永久链接。
测量结果另注明其源码提交、精确硬件目标、工作负载和计时范围。
联系：Haiyan Qin <haiyanq@buaa.edu.cn>。

本项目独立探索 [CAKE 论文](https://arxiv.org/abs/2608.12629v1)的部分思路，属于源码研究预览，并非官方实现。
自有代码与文档采用 [Apache-2.0](LICENSE)；第三方许可见 [NOTICE](NOTICE) 与[第三方说明](THIRD_PARTY_NOTICES.md)。
