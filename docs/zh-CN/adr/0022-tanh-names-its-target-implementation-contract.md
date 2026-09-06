# ADR 0022：tanh 必须说清使用哪种实现

[设计目录](README.md) · [英文原文与完整证据](../../adr/0022-tanh-names-its-target-implementation-contract.md)

**原文状态：已接受，2026-08-25。**

相同数学函数可以有不同精度和耗时的实现。计划因此明确写 `{"op":"tanh","instruction":{"contract":"libdevice.tanh.f32"}}`，不能让后端偷偷替作者选。该字段当时只允许且必须用于 tanh；不能挂在加法上，或把矩阵乘指令冒充 tanh。

Target 决定接受哪种指令约定，Verifier 在操作位置拒绝不可用或类型不符的约定，Triton 按指定方式生成，外部 oracle 负责数值判对。省略字段会恢复隐藏选择，所以不能省略。

原文记录的一次临时近似指令探测已经编译并启动，却有 `2,304/524,288` 个元素超过原先 `1e-5` 绝对误差，最大约 `3.24e-5`。它不是保留的性能证据。不能借用 KDA 完整 FP8 MoE 更宽的容差来让独立 FP32 SwiGLU 通过。

正例保留 `libdevice.tanh.f32` 和原误差；近似 `tanh.approx.f32` 反例可解析，但在当时目标下以 `TARGET_INSTRUCTION_UNSUPPORTED` 阻止生成。形状错误另有反例。此决定不增加全局快速数学或通用精度开关，未补齐 KDA 的近似机制或完整程序。
