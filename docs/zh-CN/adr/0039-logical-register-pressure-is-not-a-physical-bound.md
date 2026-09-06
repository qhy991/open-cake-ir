# ADR 0039：逻辑寄存器压力不能当成真实寄存器下界

[设计目录](README.md) · [英文原文与完整证据](../../adr/0039-logical-register-pressure-is-not-a-physical-bound.md)

**原文状态：提案；实现、语料准备与发布／GPU 权限分开。**

计划里“同时活着多少数据”，与编译器最终为每条线程分配多少物理寄存器，没有可靠的固定大小关系。QSA 旧观察中 tile128 的逻辑值 72、物理值 156；tile256 的逻辑值 140、物理值反而是 116，直接推翻了“逻辑数一定更小”。

后端可能加临时值，也可能分摊到线程、复用、重算或改用其他存储。因此保留 `logical_register_pressure_per_thread` 作为未校准的结构特征，移出物理寄存器驻留、maxnreg 容量和不可能驻留的硬阻止。

线程数、明确声明的共享内存和 tensor memory 仍可推导安全上限；后端隐式分配会使上限偏松。物理寄存器在编译或 profiler 前是 unknown。`residency.registers_per_thread` 仍传给后端作为 maxnreg 等约定；没有真实消费者的 `allow_spill` 删除。

验收需把原来仅因逻辑代理数过大而拒绝的计划允许生成，同时保留确切资源超限拒绝，分析、Corpus 预期和完整 Gate 一起审查。后继执行器实际 profile 后才谈晋升。不加 QSA 特例，也不能把缺物理证据写成零。
