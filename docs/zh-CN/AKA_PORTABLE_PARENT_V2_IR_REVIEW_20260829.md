# 100 个便携父算子的历史 IR 审查

[中文目录](README.md) · [英文原始记录](../AKA_PORTABLE_PARENT_V2_IR_REVIEW_20260829.md)

本页用中文解释原文的背景、决定和结论范围；逐项长表、精确来源及原始记录由链接中的原文负责。

**终态阅读版：2026-08-29。对应固定 v2 减 v1 的 100 条新来源。**

这次 Sol/max 池完成 `100/100` 项、没有 worker 失败，最多 30 个并发、各项独立、不重试。完成指审查流程结束，不是生成 Cake 程序在 GPU 上算对。

| 主要分类 | 数量 | 普通话解释 |
| --- | ---: | --- |
| `schedule_candidate_lowerable` | 13 | 模型计划通过固定 draft 的检查与生成 |
| `schedule_gap_candidate` | 68 | 审查者认为完整父项还缺计划能力 |
| `program_redirect_candidate` | 15 | 问题归多次启动或程序组合 |
| `portfolio_redirect_candidate` | 4 | 问题归输入分派和专用方案选择 |

每项都先变成 `runnable_unqualified` 父记录，父验证器接受了可运行源码包，但便携“曾合格”投影没有重新证明节点保管。所有结果仍是 `reviewer_claimed`、`gpu_test=not_run`。这组题来自历史父项不合格记录的重建，不能代表全部 AKA 的覆盖率。

词面信号为运行时参数 65、控制／谓词 43、索引／scatter 40、数值／转换 32、多启动 25、归约／scan 24、向量／线程映射 8。它们互有重叠，既不是 227 个独立缺口，也不能按频率直接批准新操作。

逐父项来源、合同、负责人和模型提出的能力描述保留在[原文逐项表](../AKA_PORTABLE_PARENT_V2_IR_REVIEW_20260829.md#per-parent-results)。进入新 IR 提案需至少两个独立完整来源共同要求同一不可替代约定，再同步类型、规则、分析、生成与正反语料，经完整 Gate、外部发布审查及目标正确性。
