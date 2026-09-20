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

## 章节与材料 / Chapters and supporting material

当前状态由源码生成；历史报告保留其当时的结论。
Start with the system overview, then follow the implementation or evidence relevant to your task.

- [系统概览 / Architecture](ARCHITECTURE.md) · [English](en/ARCHITECTURE.md)
- [入门与运行 / Getting started](GETTING_STARTED.md)
- [概念导读 / Wiki](wiki/README.md) · [English](en/wiki/README.md)
- [当前状态 / Current status](../reports/current/STATUS.md)
- [研究路线 / Roadmap](ROADMAP.md)
- [设计决策 / Decisions](adr/README.md)
- [完整阅读目录与历史材料 / Full catalog and history](catalog.md)

术语由 [Glossary](GLOSSARY.md) 定义，文档归属见 [Context map](../CONTEXT-MAP.md)。

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
