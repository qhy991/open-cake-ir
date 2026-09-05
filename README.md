# open-cake-ir

**把 GPU 的计算计划写清楚，先检查，再生成代码，再验证。**

GPU 算得快，但写出又对又快的程序很难。同一个计算，换一种切块、读数或协作方式，速度就可能改变。
open-cake-ir 用一份结构化的“执行计划”描述这些选择，让人和 AI 都能修改，并让编译器指出问题。

这里的 IR 是“中间表示”：它比最终机器代码好读，又比一句“把矩阵乘快一点”明确。
项目独立重建了 [CAKE 论文](https://arxiv.org/abs/2608.12629v1)中的部分思路，并非论文未公开实现的复制品。

## 从这里开始

- **第一次看项目：** [中文 Wiki](docs/wiki/README.md)，按问题找答案。
- **想马上试用：** [不需要 GPU 的入门教程](docs/GETTING_STARTED.md)。
- **想知道系统有什么用：** [系统全貌](docs/ARCHITECTURE.md)。
- **想看算子怎么算：** [算子图解](docs/wiki/operators.md)和 [IR 基本操作](docs/wiki/primitives.md)。
- **想看现在发布了什么：** [自动生成的当前状态](reports/current/STATUS.md)。

## 一条完整路径

```mermaid
flowchart LR
    A["任务：要算什么"] --> B["执行计划：怎样分工"]
    B --> C["编译器检查"]
    C --> D["生成目标源码"]
    D --> E["工具链编译并在 GPU 运行"]
    E --> F["核对答案，再测速度"]
```

编译器可以单独使用，负责读计划、检查规则和生成源码。
研究实验系统（Research Lab）负责让 AI 在固定任务和预算下提出候选方案。
Evaluation 核对答案与测量；Evidence 保存原始记录，让别人能复查。

“计划通过检查”“GPU 答案正确”“速度提高”是三个不同结论。
每个结论都应能找到对应记录，见 [怎样读结果](docs/wiki/results.md)。

## 给维护者

[文档维护](docs/wiki/maintaining.md)说明每类信息写在哪里，以及怎样检查示例和链接。
正式术语见 [GLOSSARY](docs/GLOSSARY.md)，执行流程见 [RUNBOOK](docs/RUNBOOK.md)，
设计决策见 [ADR](docs/adr/README.md)，代码职责见 [CONTEXT-MAP](CONTEXT-MAP.md)。

当前版本只由下面两个文件决定，README 不另存一份版本表：

- [`compiler/revision.lock.json`](compiler/revision.lock.json)：Compiler。
- [`inventory/EXECUTOR_REVISIONS.json`](inventory/EXECUTOR_REVISIONS.json)：Executor。
