# ADR 0025：不等长分组矩阵乘法可以用已有操作组合

[设计目录](README.md) · [英文原文与完整证据](../../adr/0025-ragged-grouped-gemm-is-primitive-composition.md)

**原文状态：2026-08-25 在 Compiler v20 接受，后继 v21 补通用地址范围规则。**

每个组各做一次矩阵乘法，行容量固定、有效行数不同。组与输出块分配已经由 ProgramMap 负责，坐标由 AccessMap 负责，有效行由 `valid_extent` 负责，K 方向遍历由 TileLoop 负责，乘加由 MMA 负责。因此不需要再发明 `grouped_gemm` 或专家描述符。

最小例子用四组 `A[4,16,32]`、`B[4,16,32]`，输出 `C[4,16,16]`。每组一个 program，每次累加 16 个 K 元素，共两步。无效 A 行补零，独立参考先遮住无效数据再做批量矩阵乘法。

把 B 的组数改小，会让某些 program 访问不存在的组。后继通用规则从 `ProgramAxis.tile_count` 推导所需范围，以 `ACCESS_PROGRAM_EXTENT_MISMATCH` 拒绝；无需强迫无关 Buffer 尺寸完全相同，也不禁止多出的未使用行。

历史 Gate 为 `29/29`，B200 比较 1,024 个输出，最大误差约 `1.91e-6`，在 `1e-5` 之内，未计时。这只证明分组乘法算术组合，未包含完整路由、第二个 GEMM、量化、scatter、持久取任务、PDL 或完整 KDA。
