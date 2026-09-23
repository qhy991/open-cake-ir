# open-cake-ir 技术报告 / Technical Report

**中文题名：open-cake-ir：Agent 驱动的 GPU Kernel 与编译器协同演进技术报告**

**English title: open-cake-ir: A Technical Report on Agent-Driven GPU Kernel and Compiler Co-Evolution**

作者 / Author：秦海岩（Haiyan Qin） · 2026

本页是仓库技术报告的统一入口。报告由现有中英文技术文档组成，覆盖系统架构、IR、
编译与验证、实验方法及有明确证据范围的结果；章节沿用原路径持续维护，不另复制一份正文。
中文与英文是同一报告的阅读版本，部分页面为导读或摘要，并非逐句翻译；历史材料保留原日期与结论。

This is the canonical entry point for the repository's technical report. The existing Chinese
and English documents form its chapters, covering architecture, IR, compilation and verification,
experimental methods, and results within their stated evidence scope. Chapters remain at their
existing paths and evolve with the repository. The language editions describe the same report;
some companion pages are guides or summaries rather than literal translations. Historical material
retains its original dates and conclusions.

[中文阅读入口](zh-CN/README.md) · [English reading guide](en/README.md)

## 研究定位与当前证据 / Research position and evidence

本报告研究 Agent 如何通过显式 GPU 程序表示和外部反馈，同时推进 Kernel 优化与编译能力演进。
核心设计包括：在固定 Compiler 下搜索并确认候选；将诊断归属到候选、分析或表达能力，
通过独立提交与后继实验验证系统变化；将可复用机制表示为带适用条件的显式变换，
并用独立的材料与工具权限研究其跨硬件复用。报告把国产卡工作分成三条相互关联但分别验收的线：
目标平台接入、在目标架构上的直接优化，以及 NVIDIA 机制材料是否额外降低目标搜索成本。
前两者已有软件实现和有明确范围的设备案例；迁移协议已有实现，自动机制提炼及跨硬件迁移收益仍待验证。

报告正文新增[平台能力与代表案例](ARCHITECTURE.md#9-平台能力与代表案例)，分别说明
编译、正确性、测量及完整优化闭环。MetaX 的设备结果由 [C550 专题](metax-c550.md)维护；
其单 kernel 优化证据不代表完整 Program 的自动优化或跨硬件迁移已经验收。

The report studies agent-driven kernel optimization and compiler evolution through explicit GPU
programs and external feedback. It connects frozen-compiler search, diagnosis-driven implementation
changes validated in successor runs, and reusable guarded transformations with independently
controlled explanation and tool access. Software paths and bounded device cases exist; automated
mechanism extraction and controlled cross-hardware transfer benefits remain unverified.
See the [platform/evidence overview](en/ARCHITECTURE.md#platform-capabilities-and-representative-evidence)
and the [C550 evidence chapter](metax-c550.md) for the distinct qualification boundaries.

## 从哪里开始 / Choose a reading path

| 阅读目的 / Purpose | 路线 / Route |
|---|---|
| 初次了解 / First visit | [项目 README](../README.md) → [系统概览](ARCHITECTURE.md) → [无需 GPU 的入门](GETTING_STARTED.md) |
| 查实验与外部差距 / Inspect evidence | [硬件结果](RESULTS.md) → [FlashInfer 逐任务综述](results/nvidia/FLASHINFER_STATUS.md) → 各行的 Finding 与原始报告 |
| 编写或优化 Kernel / Author a kernel | [Python 前端](zh-CN/PYTHON_FRONTEND.md) → [IR 指南](IR_GUIDE.md) → [Lab 任务流程](wiki/experiments.md) |
| 研究或引用 / Research and citation | 下方章节 → [研究设计](OPTIMIZATION_TRANSFER_ABLATION.md) → [引用方式](#引用--citation) |
| 维护代码 / Contribute | [职责地图](../CONTEXT-MAP.md) → [开发分支](DEVELOPMENT_BRANCHES.md) → [贡献说明](../CONTRIBUTING.md) |

## 报告章节 / Report chapters

| 章节 / Chapter | 中文入口 | English | 内容范围 / Scope |
|---|---|---|---|
| 系统架构 / Architecture | [系统概览](ARCHITECTURE.md) | [Architecture](en/ARCHITECTURE.md) | Compiler、Lab、Evaluation、Evidence 的职责与依赖 |
| IR 与编写 / IR and authoring | [IR 与 Schedule 图解](IR_GUIDE.md#阅读前irschedule-ir-与-cake-ir) · [FMA 实例](IR_GUIDE.md#fma同一公式两种不同的信息) · [Python](zh-CN/PYTHON_FRONTEND.md) | [IR guide](en/IR_GUIDE.md) · [Python](en/PYTHON_FRONTEND.md) | Schedule、Program、数据与执行计划；精确字段见 [Authoring Contract](../compiler/AUTHORING_CONTRACT.md) |
| 检查与验收 / Verification | [验收规则](zh-CN/ACCEPTANCE_GATES.md) · [结果导读](wiki/results.md) | [Acceptance](ACCEPTANCE_GATES.md) · [Results](en/wiki/results.md) | 合法性、编译、正确性、计时质量与结论边界 |
| 实验方法 / Experiment methods | [Lab 流程](wiki/experiments.md) · [改写指南](KERNEL_REPRODUCTION.md) | [Lab workflow](en/wiki/experiments.md) | 任务用途、参考访问、材料与变换授权、预算和评测 |
| 优化知识迁移 / Optimization knowledge | [机制设计](OPTIMIZATION_TRANSFER.md) · [现有证据图](OPTIMIZATION_TRANSFER.md#现有证据的四个独立范围) · [消融方法](OPTIMIZATION_TRANSFER_ABLATION.md) | [Transfer design](en/OPTIMIZATION_TRANSFER.md) | 带前提的显式变换、目标参数选择与待验证的迁移收益 |
| 实验结果 / Experimental results | [硬件汇总](RESULTS.md) · [FlashInfer 综述](results/nvidia/FLASHINFER_STATUS.md) · [MetaX C550](metax-c550.md) | [English evidence overview](en/ARCHITECTURE.md#platform-capabilities-and-representative-evidence) · [NVIDIA summary](results/nvidia/FLASHINFER_STATUS.md#english-reading-summary) | 固定 Workload、starter 身份、配对结果、失败记录和 profiler 证据 |
| 实现与维护 / Implementation | [当前状态](../reports/current/STATUS.md) · [分支流程](DEVELOPMENT_BRANCHES.md) | [Context map](../CONTEXT-MAP.md) | 源码与执行绑定、平台维护、模块负责位置 |
| 相关工程 / Related engineering | [Croqtile 对照](CROQTILE_COMPARISON.md) | [Croqtile comparison](en/CROQTILE_COMPARISON.md) | 固定源码审查、重叠能力、差异与公平比较条件 |
| 研究路线 / Roadmap | [后续目标](ROADMAP.md) | [Roadmap (Chinese)](ROADMAP.md) | 迁移、优化复用与端到端验证的研究目标 |

## 详细材料 / Detailed references

- **按平台运行：** [文档总目录](catalog.md)集中列出 NVIDIA、Apple、AMD、Hygon DCU、MetaX 的指南和结果入口。
- **查概念与算子：** [中文 Wiki](wiki/README.md) · [English Wiki](en/wiki/README.md) · [统一术语表](GLOSSARY.md)。
- **查设计原因：** [ADR 目录](adr/README.md)是完整设计决策索引。
- **查历史调查与数据：** [总目录](catalog.md)保留日期、原路径及中英文对照；历史观察不替代[当前状态](../reports/current/STATUS.md)。

本页组织章节，完整专题与历史清单由总目录维护；文档归属见 [Context map](../CONTEXT-MAP.md)。
The report entry organizes chapters; the catalog owns detailed reading links and historical material.

## 引用 / Citation

推荐引用下面的技术报告。机器可读引用元数据由仓库根目录的
[`CITATION.cff`](../CITATION.cff) 维护，其中 `preferred-citation` 指向本报告，根级条目描述软件。

秦海岩. *open-cake-ir：Agent 驱动的 GPU Kernel 与编译器协同演进技术报告*. 2026.

Qin, Haiyan. *open-cake-ir: A Technical Report on Agent-Driven GPU Kernel and Compiler
Co-Evolution*. 2026. open-cake-ir project.

```bibtex
@techreport{qin2026opencake,
  author      = {Qin, Haiyan},
  title       = {{open-cake-ir}: A Technical Report on Agent-Driven GPU Kernel and Compiler Co-Evolution},
  institution = {open-cake-ir project},
  year        = {2026},
  type        = {Technical report},
  url         = {https://github.com/qhy991/open-cake-ir/blob/main/docs/README.md},
  note        = {Living technical report; Chinese and English editions}
}
```

本报告随 Git 提交演进。引用具体论点时，请记录所读提交及章节路径，并在 GitHub 文件页按 `y`
取得包含完整提交号的永久链接，替换上面的滚动 `main` URL；在引用的 note 中注明提交。
引用实验结果时，还需注明该结果自身绑定的源码版本、精确目标、工作负载及计时范围，
不能以阅读报告的版本替代实验版本。代码使用可另引用 `CITATION.cff` 的软件条目。

The report evolves with Git commits. For a specific claim, record the chapter path and the
commit you read. Press `y` on its GitHub file page to obtain a commit permalink, replace the
rolling `main` URL above, and identify that commit in the citation note. For measurements, also
cite the experiment's own source revision, exact target, workload, and timing boundary; the report
revision does not replace the experiment revision. Cite the software entry separately when needed.

这是作者维护的仓库技术报告，目前未声明 DOI、期刊或会议发表及同行评审状态。
This is an author-maintained repository technical report; no DOI, journal or conference
publication, or peer-review status is claimed. Original text is covered by the repository's
[Apache-2.0 license](../LICENSE); third-party material retains its [own notices](../THIRD_PARTY_NOTICES.md).
