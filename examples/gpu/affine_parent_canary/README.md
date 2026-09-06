# 仿射父算子的第一次 B200 正确性检查

这个任务只检查原父算子的固定 `batch-sensitive-n2-c2-s2` 样本：2 个 batch，
每个 2 个 channel，每个平面 2 个数，共 8 个输出。

每个输出计算 `Y[n,c,s] = fma(X[n,c,s], scale[n,c], bias[n,c])`。
scale 和 bias 按 `(n,c)` 平面选择，不能只按 channel 选择。
计划原样取自 [v41 再审记录](../../../docs/AKA_FMA_PARENT_REAUDIT_V41_20260906.md)，
用源码提交 `26fbf8f` 的已发布 Compiler v43 检查并生成，不手改候选代码。

输入逐项重现父任务 harness 的确定性整数/二次幂公式。
独立参考使用 batch/channel/spatial 三层循环及精确分数运算；该样本每个精确结果均可用 FP32 表示，
因此能逐位核对融合乘加。这个性质只对所选样本成立，不扩展为任意浮点 FMA 的通用参考。

`prepare.py` 创建新 task 和 candidate 包，需要明确的远端 Python 与工作目录。
它本身不提交 GPU。task 只有一个 shared correctness stage，GPU 由 broker 分配。
`canary.py` 在受控 stage 中检查全部 8 个输出、16 个输入元素不变、输出前后两个 guard，
以及返回值仍指向传入输出。guard 不是完整内存 sanitizer，记录不声称通过 memcheck/racecheck。

评测产生 `complete-output.json`。在本地用下面的命令重新计算标准答案，不再运行 GPU：

```bash
.venv/bin/python examples/gpu/affine_parent_canary/canary.py \
  --verify /absolute/path/to/complete-output.json
```

核对失败会返回非零退出码；缺少完整数组、改变原输入或放宽 claim 也会拒绝。
指针与运行 ABI 是 judge 的实机观察，离线复算不能重新证明它们。
本任务没有计时阶段，不建立性能、所有形状、完整动态父 ABI、模型或 serving 资格。

一次实际 B200 运行及其完整输出见 [本次结果记录](../../../docs/AFFINE_PARENT_B200_CANARY_20260906.md)。
该记录绑定固定提交；再次准备包不会自动获得它的 GPU 验收。
