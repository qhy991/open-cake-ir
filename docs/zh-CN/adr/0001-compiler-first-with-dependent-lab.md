# ADR 0001：编译器是核心，Lab 使用它

[设计目录](README.md) · [英文决定原文](../../adr/0001-compiler-first-with-dependent-lab.md)

**原文状态：已接受。** 本页解释这条决定；后来的发布审查方式另见 [0052](0052-independent-agent-release-review.md)。

## 为什么这样分

只做编译器，就没法研究 AI 怎样借助反馈改进程序；把实验运行器当系统中心，又容易为每次实验复制一套实现。项目需要两者，但职责必须清楚。

## 决定

Compiler 独立接收计划、检查并生成代码。Research Lab 固定一版 Compiler，再比较完整的 Cake 和直接 CUDA 写程序环境。依赖永远单向：Lab 能使用 Compiler，Compiler 不认识实验、AI 工具、Workload、证据或研究结论。

像考试时固定评分尺子一样，候选在一个固定编译器版本内改进。编译器本身只在实验之间经过提案、完整语料检查和发布审查后升级。比较的是作者获得的整套环境，不能只叫“语法切换”。
