# open-cake-ir 技术报告 / Technical Report

**报告正文 / Report:** [BW1100 增补版 PDF](open-cake-ir-technical-report.pdf) ·
[TeX 源码 / TeX source](open-cake-ir-technical-report.tex)

**推荐中文引文题名：open-cake-ir：跨硬件优化机制复用与编译器协同演进技术报告**

**Recommended English citation title: open-cake-ir: A Technical Report on Cross-Hardware Optimization Reuse and Compiler Co-Evolution**

作者 / Author：秦海岩（Haiyan Qin） · 2026

本次更新日期为 2026-10-10。报告保留 `677eb444` 的历史实现章节，新增 BW1100 多轮协同演进实证：53 题开发、两项 Compiler 机制、独立 Bench 的逐题观察与契约修正。被测源码与合入源码分别列出。新增机制清单与 Agent 发现、授权、调用和反馈的源码分析；原生 Lab 与外部 hmz 的接入状态分别说明。

PDF 是由仓库中的 TeX 编译得到的技术报告，封面注明所读源码快照；引用具体实验仍需追溯
该实验自己的执行版本和测量记录。本页连接报告正文、引用信息和持续维护的中英文专题文档。
2026-09-28 增补的 [B300 MoE 案例](WEAVE_CASE_STUDY.md)单独说明实验源码、证据和未合入主线的边界。
专题页面提供架构、接口、方法与最新结果的详细入口，不充当另一份报告正文；历史材料保留
原日期与结论。

The October 10 revision adds a BW1100 co-evolution case study while preserving the historical implementation snapshot. It separates development-starter improvements, community-baseline Bench observations, and unresolved fresh-search effects. Precision and output-lifetime exclusions remain visible. The report traces native Lab API delivery and action permissions, and identifies the external hmz adapter's discovery gap. The PDF includes the same additions.
The TeX-compiled PDF states its source snapshot on the cover.
This page links the report, citation guidance, and living Chinese and English companion docs.
Those guides track implementation and evidence by topic; historical records retain their own
source and measurement boundaries.

[中文阅读入口](zh-CN/README.md) · [English reading guide](en/README.md)

## 研究定位与当前证据 / Research position and evidence

本报告以 Compiler 为核心，研究优化发现怎样被整理为带条件的可执行机制，在目标硬件上
重新实例化，并降低后续任务的搜索与适配成本。研究分别检验表示与诊断、材料与变换权限、
原生生成自由度及系统后继的作用。Compiler 拥有表示、检查、改写和 lowering；独立原生
探索由 Lab 组织，发现经审查后按责任层晋升，并在后继实现与任务上重新验证。

2026-10-06 的研究定位更新以[共同研究主题与硬件实验约定](RESEARCH_AGENDA.md)为入口，
包含近邻路线、强原生与非 LLM 对照、贡献边界、各硬件分支方向及新 Run 的三小时预算约定。
新增设计不改写主体快照或历史成绩。已有局部设备案例和工程 Run 可提供机制与失败证据；
跨目标经验及可执行工具的增量收益，仍由与该主张匹配的受控研究判定。

The report centers the Compiler as the foundation for conditional, executable optimization reuse
across hardware. It separates representation/diagnostic effects, material/tool effects, lowering
freedom and successor-system effects. Qualified native exploration supplies evidence for reviewed
promotion; both the re-expressed implementation and its reuse require fresh validation.
The [shared agenda](en/RESEARCH_AGENDA.md) defines the common question, hardware contributions and
new-Run budget convention. Historical implementations and results retain their original boundaries.

## BW1100 增补的阅读重点 / BW1100 addendum

- 53/53 开发终点与 265 个用例通过，表示完成该轮原有流程，不表示 53 题全部加速。
- 显式 FP32 contraction 改写把重复手写机制变成受约束工具；constexpr 清理简化生成接口。数值反例与 A/A 噪声单独说明。
- 十题历史 Bench 保留逐题社区 baseline；GQA 精度违例与 RMSNorm 返回值生命周期违例从原轮性能解释中排除。
- 新 12 题比较用于判断固定版本下的新搜索效果；未完成结果不进入报告性能表。
- 已实现的十个变换入口按七类机制说明。原生 Lab 从注册表向任务包交付 API 并检查调用授权；外部 hmz 目前主要依赖文档与人工提示，尚未完整接入自动能力清单和动作追踪。

The evidence supports reusable mechanism implementation and bounded device gains. It does not yet establish a whole-suite Compiler effect, a Cake-versus-native authoring advantage, or cross-hardware transfer gains.

## 从哪里开始 / Choose a reading path

| 阅读目的 / Purpose | 路线 / Route |
|---|---|
| 初次了解 / First visit | [项目 README](../README.md) → [系统概览](ARCHITECTURE.md) → [无需 GPU 的入门](GETTING_STARTED.md) |
| 查实验与外部差距 / Inspect evidence | [硬件结果](RESULTS.md) → [FlashInfer 逐任务综述](results/nvidia/FLASHINFER_STATUS.md) → 各行的 Finding 与原始报告 |
| 编写或优化 Kernel / Author a kernel | [Python 前端](zh-CN/PYTHON_FRONTEND.md) → [IR 指南](IR_GUIDE.md) → [Lab 任务流程](wiki/experiments.md) |
| 研究或引用 / Research and citation | 下方专题 → [研究设计](OPTIMIZATION_TRANSFER_ABLATION.md) → [引用方式](#引用--citation) |
| 维护代码 / Contribute | [职责地图](../CONTEXT-MAP.md) → [开发分支](DEVELOPMENT_BRANCHES.md) → [贡献说明](../CONTRIBUTING.md) |

## 配套专题 / Companion reading

| 专题 / Topic | 中文入口 | English | 内容范围 / Scope |
|---|---|---|---|
| 系统架构 / Architecture | [系统概览](ARCHITECTURE.md) · [面向 Agent 的接口](ARCHITECTURE.md#面向-agent-的设计如何起作用) | [Architecture](en/ARCHITECTURE.md) · [Agent-facing interface](en/ARCHITECTURE.md#why-the-interface-is-agent-facing) | Compiler、Lab、Evaluation、Evidence 的职责与依赖 |
| IR 与编写 / IR and authoring | [IR 与 Schedule 图解](IR_GUIDE.md#阅读前irschedule-ir-与-cake-ir) · [FMA 实例](IR_GUIDE.md#fma同一公式两种不同的信息) · [Python](zh-CN/PYTHON_FRONTEND.md) | [IR guide](en/IR_GUIDE.md) · [Python](en/PYTHON_FRONTEND.md) | Schedule、Program、数据与执行计划；精确字段见 [Authoring Contract](../compiler/AUTHORING_CONTRACT.md) |
| 检查与验收 / Verification | [验收规则](zh-CN/ACCEPTANCE_GATES.md) · [结果导读](wiki/results.md) | [Acceptance](ACCEPTANCE_GATES.md) · [Results](en/wiki/results.md) | 合法性、编译、正确性、计时质量与结论边界 |
| 实验方法 / Experiment methods | [Lab 流程](wiki/experiments.md) · [改写指南](KERNEL_REPRODUCTION.md) | [Lab workflow](en/wiki/experiments.md) | 任务用途、参考访问、材料与变换授权、预算和评测 |
| 优化知识迁移 / Optimization knowledge | [机制设计](OPTIMIZATION_TRANSFER.md) · [现有证据图](OPTIMIZATION_TRANSFER.md#现有证据的四个独立范围) · [消融方法](OPTIMIZATION_TRANSFER_ABLATION.md) | [Transfer design](en/OPTIMIZATION_TRANSFER.md) | 带前提的显式变换、目标参数选择与待验证的迁移收益 |
| 实验结果 / Experimental results | [硬件汇总](RESULTS.md) · [FlashInfer 综述](results/nvidia/FLASHINFER_STATUS.md) · [MetaX C550](metax-c550.md) | [English evidence overview](en/ARCHITECTURE.md#platform-capabilities-and-representative-evidence) · [NVIDIA summary](results/nvidia/FLASHINFER_STATUS.md#english-reading-summary) | 固定 Workload、starter 身份、配对结果、失败记录和 profiler 证据 |
| 实现与维护 / Implementation | [当前状态](../reports/current/STATUS.md) · [分支流程](DEVELOPMENT_BRANCHES.md) | [Context map](../CONTEXT-MAP.md) | 源码与执行绑定、平台维护、模块负责位置 |
| 相关工程 / Related engineering | [Croqtile 对照](CROQTILE_COMPARISON.md) | [Croqtile comparison](en/CROQTILE_COMPARISON.md) | 固定源码审查、重叠能力、差异与公平比较条件 |
| 共同研究主题 / Shared agenda | [研究问题与硬件分工](RESEARCH_AGENDA.md) | [Shared agenda](en/RESEARCH_AGENDA.md) | 全部分支的比较轴、贡献边界与实验约定 |
| 研究路线 / Roadmap | [后续目标](ROADMAP.md) | [Roadmap (Chinese)](ROADMAP.md) | 迁移、优化复用与端到端验证的研究目标 |

## 详细材料 / Detailed references

- **按平台运行：** [文档总目录](catalog.md)集中列出 NVIDIA、Apple、AMD、Hygon DCU、MetaX 的指南和结果入口。
- **查概念与算子：** [中文 Wiki](wiki/README.md) · [English Wiki](en/wiki/README.md) · [统一术语表](GLOSSARY.md)。
- **查设计原因：** [ADR 目录](adr/README.md)是完整设计决策索引。
- **查历史调查与数据：** [总目录](catalog.md)保留日期、原路径及中英文对照；历史观察不替代[当前状态](../reports/current/STATUS.md)。

本页连接 PDF 与配套专题；完整专题和历史清单由[文档总目录](catalog.md)维护，模块归属见
[Context map](../CONTEXT-MAP.md)。
The [catalog](catalog.md) owns the complete companion and historical index.

## 引用 / Citation

推荐引用上方的 PDF 技术报告。机器可读引用元数据由仓库根目录的
[`CITATION.cff`](../CITATION.cff) 维护，其中 `preferred-citation` 指向本报告，根级条目描述软件。

秦海岩. *open-cake-ir：跨硬件优化机制复用与编译器协同演进技术报告*. 2026.

Qin, Haiyan. *open-cake-ir: A Technical Report on Cross-Hardware Optimization Reuse and Compiler Co-Evolution*. 2026. open-cake-ir project.

```bibtex
@techreport{qin2026opencake,
  author      = {Qin, Haiyan},
  title       = {{open-cake-ir}: A Technical Report on Cross-Hardware Optimization Reuse and Compiler Co-Evolution},
  institution = {open-cake-ir project},
  year        = {2026},
  type        = {Technical report},
  url         = {https://github.com/qhy991/open-cake-ir/blob/main/docs/open-cake-ir-technical-report.pdf},
  note        = {TeX-compiled technical report; cite the report commit and page}
}
```

本报告随 Git 提交演进。引用具体论点时，请记录所读 PDF 的提交与页码，并在 GitHub
文件页按 `y` 取得包含完整提交号的永久链接，替换上面的滚动 `main` URL。
引用实验结果时，还需注明该结果自身绑定的源码版本、精确目标、工作负载及计时范围，
不能以阅读报告的版本替代实验版本。代码使用可另引用 `CITATION.cff` 的软件条目。

The report evolves with Git commits. For a specific claim, record the PDF page and the
commit you read. Press `y` on its GitHub file page to obtain a commit permalink and replace the
rolling `main` URL above. For measurements, also
cite the experiment's own source revision, exact target, workload, and timing boundary; the report
revision does not replace the experiment revision. Cite the software entry separately when needed.

这是作者维护的仓库技术报告，目前未声明 DOI、期刊或会议发表及同行评审状态。
This is an author-maintained repository technical report; no DOI, journal or conference
publication, or peer-review status is claimed. Original text is covered by the repository's
[Apache-2.0 license](../LICENSE); third-party material retains its [own notices](../THIRD_PARTY_NOTICES.md).
