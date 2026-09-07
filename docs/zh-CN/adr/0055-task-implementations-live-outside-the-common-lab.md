# ADR 0055：任务实现从通用 Lab 中分离

[英文决定](../../adr/0055-task-implementations-live-outside-the-common-lab.md)

仓库拥有者于 2026-09-07 要求把 QSA 等任务专用实现归到任务目录。
Lab 与通用 Evaluation 不导入具体任务。任务层负责严格合同校验、参考实现、输入准备、
启动 ABI 和专用诊断，再通过静态组合使用通用 Ralph 引擎。

K-means portfolio 整体回到 K-means 目录，并共用一个 ExactShape。
QSA 旧续轮接口删除，下一轮状态由 Ralph 统一生成。QSA 的固定参考源码原样搬迁，
历史合同、发布和证据不改写；Executor 后继版本纳入递归任务源码闭包。

目录和使用方法见[任务说明](../../TASKS.md)。
