# ADR 0027：后端做不到的事，在检查阶段就说明

[设计目录](README.md) · [英文原文与完整证据](../../adr/0027-backend-preconditions-are-assessment-findings.md)

**原文状态：已接受，2026-08-25。**

如果计划合法，但选定代码生成器不支持其中公式，应在 `Compiler.assess` 中明确出现诊断，而不是等开始生成后才崩溃。

每个生成器拥有一个 `preflight(schedule,target)`，返回 `BackendPrecondition(code,path,message)`。检查结果使用同一份规则，生成器直接调用也先查同样前提；不在编译器另一处复制判断。此类 Finding 可保留结构 `accepted=true`，同时令 `lowering_eligible=false`。

例如 CuTe DSL 当时只实现 `centroid_sq_minus_two_dot`，不能把 `bias_add_bf16_round` 静默生成为质心公式。TinyGEMM2 固定源码则明确要求后者，不能接受前者冒名。直接调用生成器遇到失败抛 `EmitError`。

验收覆盖公式互换和 Triton/CuTe 的角色、循环、存储、流水线等前提，并确保已有正例输出不变。这不承诺所有后端支持任意合法操作，也没把固定 TinyGEMM2 变成可自由生成的源码。
