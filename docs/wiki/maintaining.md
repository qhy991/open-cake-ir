# 文档怎样保持准确

[中文首页](../zh-CN/README.md) · [English](../en/wiki/maintaining.md) · [中英文对照](../README.md)

这套 Wiki 面向第一次接触项目的人。正文用短句和小例子，代码名保留原拼写，读者可以继续追到实现。

[返回 Wiki](README.md)

## 每种信息只放一个负责位置

| 信息 | 放哪里 |
| --- | --- |
| 系统用途、职责与数据流 | [ARCHITECTURE](../ARCHITECTURE.md) |
| 正式术语与负责者 | [GLOSSARY](../GLOSSARY.md) |
| 上手和计算讲解 | 本 Wiki 与 [GETTING_STARTED](../GETTING_STARTED.md) |
| 精确数学、形状、判对和测量 | [Workload 合同](../../contracts/workloads/README.md)与 Study |
| IR 的字段与允许形式 | [Authoring Contract](../../compiler/AUTHORING_CONTRACT.md)、代码与 Corpus |
| 当前发布状态 | 由 lock 和 Executor inventory [生成](../../reports/current/STATUS.md) |
| 一次实验的过程和结果 | 仓库外 Evidence，以及绑定该实验的报告 |
| 为什么改变长期设计 | [ADR](../zh-CN/adr/README.md) |

Wiki 解释这些材料，不成为另一份合同或当前版本数据库。
旧教程和调查结论要标明当时的范围，不把日期较旧的记录写成今天的运行保证。

## 写一个算子说明时

说明它解决什么问题，输入输出是什么，用一组小数字展示计算，然后链接完整计划或合同。
需要知道精度、状态更新或硬件限制时，在对应段落写清楚；不把一页塞满所有历史失败。
数学小例子不算 GPU 证据，也不要为叙述方便省略会改变答案的舍入顺序。

新增 IR 基本操作时，补 [基本操作](primitives.md)的对应节；
新增 Workload 合同时，补 [任务目录](workloads.md)中的用途、边界和链接。
文档测试会提醒遗漏操作名、合同文件或坏链接，但读者能否理解仍需要逐页检查。

## 修改后运行的检查

从仓库根目录运行：

```bash
.venv/bin/python -m unittest tests.contracts.test_documentation tests.contracts.test_cli
.venv/bin/python tools/render_current_status.py --check
git diff --check
```

它们检查 Wiki 的导航、引用、基本操作和 Workload 覆盖、入门命令，以及 CLI 的 JSON/中文输出。
它们不会验证新的 GPU 性能，也不会替代 Compiler Corpus Gate。

修改了文档里的可执行命令，要实际执行它。示例应写到新临时目录；明确哪个命令故意失败、期待什么原因。

## 发布时

普通讲解改动不需要重发 Compiler。修改绑定的工具或运行时源码，则走它自己的后继发布流程。
当前状态变化后运行 `tools/render_current_status.py --write`，不要手工写一个新版本号。
保留 JSON 供工具使用，中文输出是同一结果的阅读视图。

文档随主仓库提交和发布；本 Wiki 不维护另一套手工复制的站点或版本表。

## 中英文怎样对应

[总目录](../README.md)把每篇原有文档对应到中文和英文。已有正文保留路径，缺失中文在 `docs/zh-CN/` 补阅读版，原有中文的英文对照放在 `docs/en/`。每篇对照页链接原文；历史长表与逐项证据留在原负责位置。

中文先讲“解决什么问题”，再用小例子说明“怎样做”，最后标明“检查到哪里”。代码名、命令、字段和数字不随意翻译。导读可以调整顺序和压缩技术长表，但须明确它是阅读版，不能自称完整逐句译文。

更新稳定说明时，同时核对语言对照页、目录和链接。历史报告的翻译保留日期、原结论及后继关系，不把旧失败写成已通过，不把旧通过冒充当前版本；不改写原实验记录。新增原文或语言页时，补总目录的对应行。
