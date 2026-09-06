# AKA 合格父项审查数据：每份文件能说明什么

[中文目录](../../README.md) · [英文原文](../../../data/aka-qualified-ir-v6-review-20260904/DATASET.md)

这是外部只追加证据的可发布阅读投影，不是原始模型事件、完整 GPU 张量或可修改实验状态。

| 文件 | 内容和边界 |
| --- | --- |
| phase-a-review.jsonl | 677 项：439 静态审查被确定性检查器接受，237 保留审查／结构拒绝，1 项模型前基础设施失败 |
| lab-terminal-results.jsonl | 57 项静态准入：56 在固定实例 B200 正确性、memcheck、racecheck、完整输出独立重算后有效；1 写作拒绝、GPU 没跑 |
| ir-gap-clusters.json | 326 个接受的 ir_gap 项、296 个候选名称的聚类提案 |
| l000214-authoring-rejection.json | 唯一未进入动态运行的 Lab 终态 |

[原数据目录](../../../data/aka-qualified-ir-v6-review-20260904/)中的 manifest 记录来源；小型准入、核验、节点、环境和聚类摘要放 evidence 子目录，原始大材料仍在外部。

聚类不批准新操作；56 项成功只证明固定实例。所有项目 `performance_measured=false`，Lab 集训练资格为假。交接用身份值只回答内容是否对应，不证明语义或速度。
