<p align="center">
  <img src="docs/figures/open-cake-ir-mark.svg" width="112" alt="open-cake-ir：代码括号中的三层执行计划" />
</p>

<h1 align="center">open-cake-ir</h1>

<p align="center"><strong>让 GPU 执行计划可编写、可检查、可复查。</strong><br />
An open research system for agent-driven GPU kernel and compiler co-evolution.</p>

<p align="center">
  <a href="docs/open-cake-ir-technical-report.pdf">技术报告 PDF</a> ·
  <a href="docs/open-cake-ir-technical-report.tex">TeX 源码</a> ·
  <a href="docs/GETTING_STARTED.md">快速开始</a> ·
  <a href="docs/RESULTS.md">实验结果</a> ·
  <a href="docs/en/README.md">English</a>
</p>

智能体用 Python 编写单内核的 Schedule，或用 Program 组合多阶段计算。Compiler 检查
执行计划并生成目标源码；独立评测依据外部答案和目标设备的测量协议确认候选。反复出现
的问题再推动编译器改进。项目独立探索 [CAKE 论文](https://arxiv.org/abs/2608.12629v1)
的研究思路，属于研究预览。

[中文阅读入口](docs/zh-CN/README.md) · [项目当前状态](reports/current/STATUS.md)

## 快速导航

| 你想做什么 | 直接入口 | 能看到什么 |
|---|---|---|
| 第一次了解或运行项目 | [快速开始](#快速开始) · [入门教程](docs/GETTING_STARTED.md) | 安装、无需 GPU 的检查与源码生成 |
| 阅读或引用技术报告 | [PDF 正文](docs/open-cake-ir-technical-report.pdf) · [TeX 源码](docs/open-cake-ir-technical-report.tex) · [引用与配套专题](docs/README.md) | 设计、实例、方法、结果及其证据范围 |
| 查看 FlashInfer 改写与外部实现差距 | [逐任务实验综述](docs/results/nvidia/FLASHINFER_STATUS.md) | starter 身份、配对性能、正确性、失败边与未完成项 |
| 查看各硬件成果和使用方法 | [实验结果](#按硬件查看成果) · [硬件指南](#硬件指南) | 已收录观察、平台工具链和验收范围 |
| 编写 Kernel、发起优化或改写已有实现 | [Python 前端](docs/zh-CN/PYTHON_FRONTEND.md) · [Lab 任务流程](docs/wiki/experiments.md) · [改写指南](docs/KERNEL_REPRODUCTION.md) | 输入格式、任务用途、参考材料、预算与评测 |
| 修改实现或定位问题 | [源码结构](#源码与文档结构) · [系统架构](docs/ARCHITECTURE.md) · [开发流程](docs/DEVELOPMENT_BRANCHES.md) | 模块职责、扩展位置、分支与测试要求 |

**报告与当前状态：**PDF 由仓库中的 TeX 编译，封面标注它所读的源码快照；
[报告索引](docs/README.md)提供引用方式和持续维护的专题文档。当前实现见
[生成状态页](reports/current/STATUS.md)。具体实验仍按各自绑定的源码、硬件、Workload
和计时协议解释。

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

需要 Python 3.10+。下面先离线检查 Softmax，再生成可阅读的 Triton 源码；这些步骤不需要 GPU：

```bash
git clone https://github.com/qhy991/open-cake-ir.git
cd open-cake-ir
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
CAKE_DEMO_DIR=$(mktemp -d)
.venv/bin/open-cake-ir compiler assess --format text examples/python/softmax.py
.venv/bin/open-cake-ir compiler lower --format text examples/python/softmax.py \
  --output "$CAKE_DEMO_DIR/softmax.py"
```

检查结果中的“结构检查：通过”与“生成代码：允许”说明这份计划通过了当前软件门；
生成文件在 `$CAKE_DEMO_DIR/softmax.py`。[技术报告的 Softmax 实例](docs/open-cake-ir-technical-report.pdf)
逐项对照 Python Schedule 和生成的 Triton 代码。[完整入门教程](docs/GETTING_STARTED.md)
从更小的 FMA 示例解释诊断。设备正确性和性能还需对应工具链与 GPU 环境，见下方硬件指南。

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
| [compiler/](src/open_cake_ir/compiler/) · [Target 文档](compiler/targets/) | IR、Verifier、后端和精确硬件声明；[IR 指南](docs/IR_GUIDE.md) |
| [lab/](src/open_cake_ir/lab/) · [evaluation/](src/open_cake_ir/evaluation/) · [evidence/](src/open_cake_ir/evidence/) | 候选搜索、外部评测和运行证据；[职责地图](CONTEXT-MAP.md) |
| [tasks/](src/open_cake_ir/tasks/) · [contracts/](contracts/) | Workload、oracle 和任务实现；[任务目录](docs/wiki/workloads.md) |
| [examples/](examples/) · [corpus/](corpus/) · [tests/](tests/) | 可读示例、编译器语料和合同测试；[贡献说明](CONTRIBUTING.md) |
| [docs/](docs/README.md) · [findings/](findings/) · [reports/](reports/) | 报告索引、问题记录与[当前状态](reports/current/STATUS.md)；[完整目录](docs/catalog.md) |

更细的模块边界见 [Context map](CONTEXT-MAP.md)，平台操作与历史材料见[文档总目录](docs/catalog.md)。

## 参与与引用

- **参与开发：** [贡献说明](CONTRIBUTING.md) · [平台与分支维护](docs/DEVELOPMENT_BRANCHES.md) · [设计决策](docs/adr/README.md) · [研究路线](docs/ROADMAP.md)。
- **引用报告：** [PDF 正文](docs/open-cake-ir-technical-report.pdf) · [TeX 源码](docs/open-cake-ir-technical-report.tex) · [推荐引用与 BibTeX](docs/README.md#引用--citation) · [机器可读引用](CITATION.cff)。引用具体论点请记录所读提交；实验结果还需注明各自的源码版本与测量范围。
- **问题与许可：** [安全问题](SECURITY.md) · [Apache-2.0](LICENSE) · [NOTICE](NOTICE) · [第三方说明](THIRD_PARTY_NOTICES.md)。

作者：秦海岩（Haiyan Qin），联系：<haiyanq@buaa.edu.cn>。
本项目是独立研究实现，不是 CAKE 论文的官方源码。
