# FMA 在 B200 上的正确性：两个 kernel、十二组输入

[中文目录](README.md) · [英文原始记录](../AKA_FMA_B200_CORRECTNESS_20260904.md)

本页用中文解释原文的背景、决定和结论范围；逐项长表、精确来源及原始记录由链接中的原文负责。

**历史目标检查：2026-09-04，固定 Compiler v41 与 `[8,128]` 形状。**

普通 FMA 与带单独舍入乘法的嵌套 FMA，各六组输入，共十二个 workload。正确性、memcheck、racecheck 三阶段都通过，合计 36 份完整输出、36,864 元素。独立重算中 30,333 项逐位相等，6,531 项按 quiet-NaN 类别判断；NaN 符号和载荷不属于约定。

六类输入覆盖相消、正负零、次正规数、溢出、Inf/quiet-NaN/signaling-NaN 组合、固定随机 binary32 位模式。oracle 用整数精确乘加再一次 RN-even 舍入，不调用候选、CUDA 或 Triton 生成答案；嵌套顺序也明确。输出先填与预期类别相反的哨兵，避免漏写恰好混过 NaN 检查。

judge 还检查指针互不重叠、输入位不变、输入输出指针不变、返回所给输出对象，以及形状和 FP32 类型。memcheck 明确零错误，racecheck 零风险、错误和警告。

一个正式固定节点提交，三份 broker job 各完成对应阶段，无 GPU 重试或改路由。第一次收集 SSH 超时保留 `fetch_failed`，之后只读取同一已终止结果，没有重跑 GPU。备份中 AppleDouble 附加文件导致的拒绝也保留，新的独立备份目录通过完整输出复查。

关键结论是**两个固定实例的数值与内存／竞争检查通过**。`frontier_eligible=false`、`performance_measured=false` 符合该任务设计。十二组输入不是十二个新 AKA 父算子；没有任意形状、完整父等价、框架、服务或训练资格。运行、核验及清理收据见[原始数据目录](../data/fma-v41-b200-correctness-20260904/)。
