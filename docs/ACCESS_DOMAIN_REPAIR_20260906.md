# 从 FMA 再审修复读取与向量检查

本轮把 [v41 父算子再审](AKA_FMA_PARENT_REAUDIT_V41_20260906.md)当成问题线索，
先在 `e4cf49e` 复现，再修复当前 Compiler。历史 12 行记录保持原样，不能改标为当前版本或 GPU 成绩。

## 四个已复现的问题

| 输入 | 修复前 | 修复后 |
| --- | --- | --- |
| 只读一个位置，却把寄存器结果声明为 128 个数 | 接受并生成，FMA 中可能发生隐式复制 | `LOAD_ACCESS_SHAPE_MISMATCH`，拒绝计划 |
| 声明不存在的第 99 维 | 抛出 IndexError | `ACCESS_DIMENSION_MISMATCH`，返回位置明确的诊断 |
| 用长度 8 的维度定义向量，却拿它访问实际长度 4 的维度 | 接受并生成，存在读越界 | `ACCESS_DIMENSION_COORDINATE_RANGE`，拒绝计划 |
| Triton 向量区间包含 3 个元素 | 接受并生成，工具链编译才报错 | `TRITON_ARANGE_RANGE_UNSUPPORTED`，结构可成立但不允许该后端生成 |

读取必须产生它声明的值形状。全是标量坐标时，单值寄存器仍写成 `[1]`，不会被误拒绝。
全局有 9 列也可以用长度 4 的向量分块并掩码处理尾部；检查针对实际生成的向量，不要求整个输入大小是二次幂。
非零起点按 `end-start` 检查，匹配本次检查的 Triton 3.7.1 实现。

原有 72 条 Corpus 预期未改。新增一个合法单值读取例子和四个对应反例，并补充寄存器秩、未知引用、
掩码分块和非零起点等公开 API 回归。没有新增 splat、reshape 或运行时标量语法。

## 历史再审记录怎样使用

原报告记载 4 个静态/编译候选、5 个运行时标量相关缺口、1 个语义拒绝和 2 个创作失败。
本轮对照原 Phase A 固定合同检查了四个候选的公式与扁平地址关系：
逐 NC 平面仿射、逐通道 NCHW 仿射、NHWC 仿射，以及先内后外的两层分组 FMA。
它们在修复后的检查器中仍允许生成；被拒绝的 GroupNorm 计划现在会在生成前得到形状诊断。

这不证明四个候选在 GPU 上正确，也不代表完整动态 ABI、流、状态返回或所有形状都已覆盖。
原报告中的编译记录是历史元数据；投影验证器不核验它未携带的 PTX/CUBIN 原始字节。

当前投影验证器需要明确的 v41 源码目录，使用其中的真实 CLI 在新进程里重放检查和生成，
不会把当前 Compiler 的结果冒充 v41：

```bash
CAKE_V41_REPLAY=$(mktemp -d)
git worktree add --detach "$CAKE_V41_REPLAY/source" \
  d9d56e835cf96eecf70e0259b65bc1b20c4f6f0d
.venv/bin/python tools/verify_aka_fma_v41_reaudit.py \
  --data docs/data/aka-fma-v41-reaudit-20260906 \
  --compiler-root "$CAKE_V41_REPLAY/source" \
  --output "$CAKE_V41_REPLAY/verification.json"
```

新输出用 v2 schema 明示“投影一致性与冻结 Compiler 重放”范围；原始 v1 验证记录仍保留。
它拒绝把摘要或编译记录的 GPU/性能标志升级为通过，输出文件也不覆盖旧文件。

运行时 FP32 标量和显式 splat 仍是独立提案，需要单独确定接口、规则和实现。
