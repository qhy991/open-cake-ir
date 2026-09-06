# 旧顶层设计入口：现在应该读哪里

[中文目录](README.md) · [英文原文](../TOP_LEVEL_DESIGN.md)

这个旧路径因历史链接而保留。[ADR 0047](adr/0047-documentation-separates-stable-history-and-current-views.md) 已将它的说明职责交给下列材料：

| 想查什么 | 阅读位置 |
| --- | --- |
| 系统做什么、各部分怎样合作 | [系统全貌](../ARCHITECTURE.md) |
| 一个词的统一定义 | [术语表](../GLOSSARY.md) |
| 各模块负责什么 | [职责地图](CONTEXT-MAP.md) |
| 当前 Compiler 和 Executor | [自动状态页](../../reports/current/STATUS.md) |
| 精确任务和实验约定 | [合同目录](../../contracts/) |
| 为什么作出长期设计选择 | [设计记录](adr/README.md) |
| 当时迁移到哪一步 | [历史清单](../../inventory/)和 Git 历史 |

原设计确立了编译器为核心、Lab 单向依赖、每个事实一个负责者，以及候选改进与编译器演进分开等原则。[ADR 0001](adr/0001-compiler-first-with-dependent-lab.md) 和 [ADR 0002](adr/0002-portfolio-as-second-study-variant.md) 保留决定依据。

新的实验结果不追加到这个入口：实验产生 Evidence 和 Study Report，新的长期决定写后继 ADR。
