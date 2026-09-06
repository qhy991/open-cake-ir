# ADR 0029：生成路线只说明后端和入口，不兼职定义题目

[设计目录](README.md) · [英文原文与完整证据](../../adr/0029-lowering-route-is-not-a-workload-profile.md)

**原文状态：2026-08-25 在 Compiler v24 接受。**

旧 `metadata.profile` 名称同时选择后端、函数入口、接口标签和特定题目检查，一处名字负责太多事情。新计划只用顶层 `lowering={backend,entry_point}`。后端为 `triton`、`cutlass_cute_dsl` 或 `checked_cuda_asset`，入口是函数符号，不是算子身份。

操作、Buffer、地址映射和 Target 决定程序是否合法；后端 preflight 只管生成限制；真实参数顺序从全局 Buffer 推导。Workload 和它的消费者定义输入含义、形状、标准答案与容差。可选 Workload 内容绑定对 Compiler 是不透明引用，它不解析题目。

能用已有操作组合的新算子不需要增加后端表行。固定 CUDA 资产是受限例外：具体入口选择固定源码、精确计划语义与资产前提。TinyGEMM2 仍 `generated=false`；错误的四分片求和或尾部公式使路线不允许生成。

不支持的后端拼写是结构错误；后端缺操作或数据类型是定位的生成限制；未知 Workload 内容引用本身不改变 Compiler 判断。`metadata.profile` 没有新别名，旧版本只在冻结源码中保留。历史迁移 Gate `32/32` 不能让旧 GPU 观察自动成为新版证据。
