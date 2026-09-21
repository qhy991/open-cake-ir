# open-cake-ir

**Agent 驱动的 GPU Kernel 搜索与 Compiler 演进。**

Agent 用 Python 编写 Schedule（单个 Kernel 的执行计划），用 Program 组合完整的多阶段计算；
Compiler 检查并生成目标源码，独立评测核对正确性和性能，实验反馈再推动候选与编译器改进。
支持从任务定义开始探索，也支持参考已有优秀实现进行 Cake 改写。

A research system for agent-driven GPU kernel search and compiler evolution.
[English reading guide](docs/en/README.md) · [中文阅读入口](docs/zh-CN/README.md)

## 快速导航

| 你想做什么 | 直接入口 | 能看到什么 |
|---|---|---|
| 第一次了解或运行项目 | [快速开始](#快速开始) · [入门教程](docs/GETTING_STARTED.md) | 安装、无需 GPU 的检查与源码生成 |
| 阅读或引用技术报告 | [技术报告](docs/README.md) · [完整中英文目录](docs/catalog.md) | 架构、IR、方法、结果章节与引用方式 |
| 查看 FlashInfer 改写与外部实现差距 | [逐任务实验综述](docs/results/nvidia/FLASHINFER_STATUS.md) | starter 身份、配对性能、正确性、失败边与未完成项 |
| 查看各硬件成果和使用方法 | [实验结果](#按硬件查看成果) · [硬件指南](#硬件指南) | 已收录观察、平台工具链和验收范围 |
| 编写 Kernel、发起优化或改写已有实现 | [Python 前端](docs/zh-CN/PYTHON_FRONTEND.md) · [Lab 任务流程](docs/wiki/experiments.md) · [改写指南](docs/KERNEL_REPRODUCTION.md) | 输入格式、任务用途、参考材料、预算与评测 |
| 修改实现或定位问题 | [源码结构](#源码与文档结构) · [系统架构](docs/ARCHITECTURE.md) · [开发流程](docs/DEVELOPMENT_BRANCHES.md) | 模块职责、扩展位置、分支与测试要求 |

当前能力与版本见[生成状态页](reports/current/STATUS.md)；具体实验的结论以其绑定的源码、硬件、Workload 和计时协议为准。

## 系统怎样工作

Workload 定义计算与标准答案；Agent 提交完整 Schedule / Program；Compiler 检查并生成源码；
Evaluation 核对输出、计时与 profiler；Evidence 保存原始观察，供审计和后续改进使用。
Compiler 可以独立使用，Research Lab 负责组织优化或受控研究。

| 核心部分 | 用途 | 进一步阅读 |
|---|---|---|
| Compiler | IR、合法性检查、显式变换、目标代码生成与静态分析 | [IR 导读](docs/IR_GUIDE.md) · [精确编写契约](compiler/AUTHORING_CONTRACT.md) |
| Research Lab | 组织 Agent、参考访问、材料与变换授权、预算和候选迭代 | [Lab 架构](docs/contexts/lab/CONTEXT.md) · [任务流程](docs/wiki/experiments.md) |
| Evaluation | 独立检查正确性、配对计时及性能归因 | [评测职责](docs/contexts/evaluation/CONTEXT.md) · [验收规则](docs/ACCEPTANCE_GATES.md) |
| Evidence | 保存原始产物、回放记录与支持结论的依据 | [证据职责](docs/contexts/evidence/CONTEXT.md) · [结果阅读](docs/wiki/results.md) |

研究主线是[跨硬件的可执行优化知识迁移](docs/OPTIMIZATION_TRANSFER.md)：把发现的机制提炼为带前提的显式变换，
在目标平台重新选择参数并验证。已有有限 pass 和研究流程作为基础，迁移收益仍按[受控消融](docs/OPTIMIZATION_TRANSFER_ABLATION.md)验证。

## 快速开始

需要 Python 3.10+，从源码安装：

```bash
git clone https://github.com/qhy991/open-cake-ir.git
cd open-cake-ir
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

先检查仓库内的 FMA 执行计划，无需 GPU：

```bash
.venv/bin/python -m open_cake_ir.cli compiler assess --format text \
  --revision compiler/revision.json corpus/schedules/fma-b8-smoke.json
```

[完整入门教程](docs/GETTING_STARTED.md)继续介绍源码生成与反例检查。
真实 GPU 构建和评测还需对应工具链与设备环境，按下面的硬件指南准备。

<!-- hardware-results:start -->
## 按硬件查看成果

各平台维护自己的发布数据，main 汇总已合入的版本。观察条目包括形状、实验集合和历史尝试，**不是任务总数**。

| 硬件 | 维护分支 | 已收录观察 | 结果页面 |
|---|---|---:|---|
| NVIDIA | [nvidia](https://github.com/qhy991/open-cake-ir/tree/nvidia/docs/results/nvidia) | 134 | [B300 多任务与 CAKE 对照；B200 正确性记录](docs/results/nvidia/README.md) |
| Apple | [metal](https://github.com/qhy991/open-cake-ir/tree/metal/docs/results/metal) | 4 | [M1 Pro / M4 / M2；历史确认、晋升与稳定性边界](docs/results/metal/README.md) |
| Hygon DCU | [dcu](https://github.com/qhy991/open-cake-ir/tree/dcu/docs/results/dcu) | 32 | [BW1101；首个合格结果、后续运行与失败记录](docs/results/dcu/README.md) |
| AMD | [amd](https://github.com/qhy991/open-cake-ir/tree/amd/docs/results/amd) | 1 | [gfx1151；计时边界待解决](docs/results/amd/README.md) |

[完整结果与原始证据](docs/RESULTS.md) · [FlashInfer 逐任务对比](docs/results/nvidia/FLASHINFER_STATUS.md) · [CAKE 对照与改写](docs/NVIDIA_CAKE_REPRODUCTION.md)

[交互目录（下载后打开）](docs/results/index.html) · [发布数据维护流程](docs/RESULTS_MAINTENANCE.md)

**加速比 = 各自固定基线耗时 ÷ 候选耗时**，超过 1× 表示更快。正确性、计时质量与性能收益分别记录；各平台协议不同，不作跨硬件排名。

<details>
<summary>展开代表结果图（各平台独立刻度）</summary>

![按硬件分组的代表性确认结果](docs/results/overview.svg)

图表展示选定的历史观察；完整表格保留失败、无显著差异和未实测记录。各自任务基线不等于厂商最优库。

</details>
<!-- hardware-results:end -->

## 硬件指南

| 平台 | 运行与开发入口 |
|---|---|
| NVIDIA B200 / B300 | [B300 入门](docs/B300.md) · [原生 CUDA / PTX](docs/zh-CN/NATIVE_CUDA.md) · [Triton 配对](docs/PAIRED_TRITON.md) · [CuTe DSL 配对](docs/zh-CN/PAIRED_CUTE.md) |
| Apple Metal | [中文指南](docs/metal.zh-CN.md) · [English guide](docs/metal.md) |
| AMD | [任务入口与当前边界](docs/amd-task-entrypoints.md) |
| Hygon DCU | [设计与运行路径](docs/dcu-gfx938-design.md) · [设备结果](docs/dcu-gfx938-results.md) |
| MetaX C550 | [当前路径与验收范围](docs/metax-c550.md) |

指南说明各自的接入与验证范围；MetaX 的入口在这里，发布数据尚未纳入上方四个平台的汇总表。

## 源码与文档结构

| 位置 | 内容与入口 |
|---|---|
| [src/open_cake_ir/compiler/](src/open_cake_ir/compiler/) | IR、Verifier、后端、pass 与性能分析；[实现导读](docs/IR_GUIDE.md) |
| [src/open_cake_ir/lab/](src/open_cake_ir/lab/) | Agent 与实验组织；[模块导航](src/open_cake_ir/lab/README.md) |
| [src/open_cake_ir/evaluation/](src/open_cake_ir/evaluation/) · [evidence/](src/open_cake_ir/evidence/) | 公共评测、执行与证据存储；[职责地图](CONTEXT-MAP.md) |
| [src/open_cake_ir/tasks/](src/open_cake_ir/tasks/) · [contracts/](contracts/) | 具体任务实现与语义契约；[任务目录](docs/wiki/workloads.md) |
| [compiler/targets/](compiler/targets/) · [runtime/hosts/](runtime/hosts/) | 精确硬件声明与主机环境 |
| [tools/](tools/) · [tests/](tests/) · [corpus/](corpus/) | CLI 工具、合同测试和编译器验证用例；[贡献说明](CONTRIBUTING.md) |
| [docs/](docs/README.md) | 技术报告和使用文档；[完整目录](docs/catalog.md) · [概念与算子导读](docs/wiki/README.md) |
| [findings/](findings/) · [reports/current/](reports/current/STATUS.md) | 问题与改进记录、从权威输入生成的当前状态 |

README 提供快速入口；[报告首页](docs/README.md)组织章节与引用；[文档总目录](docs/catalog.md)收纳详细专题与历史材料。
文档职责由 [Context map](CONTEXT-MAP.md)维护，新增内容按已有负责位置补充。

## 参与与引用

- **参与开发：** [贡献说明](CONTRIBUTING.md) · [平台与分支维护](docs/DEVELOPMENT_BRANCHES.md) · [设计决策](docs/adr/README.md) · [研究路线](docs/ROADMAP.md)。
- **引用报告：** [报告题名、作者、推荐引用与 BibTeX](docs/README.md#引用--citation) · [机器可读引用](CITATION.cff)。中英文文档共同构成本仓库的技术报告；引用具体章节使用提交永久链接，实验结果另注明其自身版本与测量范围。
- **问题与许可：** [安全问题](SECURITY.md) · [Apache-2.0](LICENSE) · [NOTICE](NOTICE) · [第三方说明](THIRD_PARTY_NOTICES.md)。

作者：秦海岩（Haiyan Qin），联系：<haiyanq@buaa.edu.cn>。
本项目独立探索 [CAKE 论文](https://arxiv.org/abs/2608.12629v1)的部分思路，属于源码研究预览，并非官方实现。
