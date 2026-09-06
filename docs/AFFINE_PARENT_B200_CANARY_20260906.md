# 仿射计算：一次完整的 B200 正确性检查

**本次固定样本通过：8 个输出逐位正确，16 个输入元素保持不变，输出的两个 guard 与返回接口检查通过。**
这是一次新的 GPU 正确性记录，使用 Compiler v43；没有计时，也没有运行内存 sanitizer。

## 这道题算什么

想象有两本作业（batch），每本有两行题（channel），每行两个数（spatial）。
每行有自己的“乘多少”和“加多少”，即 scale 与 bias：

`Y[n,c,s] = fma(X[n,c,s], scale[n,c], bias[n,c])`

FMA 表示融合乘加，只在最后舍入一次。行号包含 batch：第二本作业不能误用第一本同一行的系数。
这也是为什么选择这个小样本；把系数错误地只按 channel 选择，会让 4 个答案不一致。

实际输入与输出如下。每个精确结果都能由 FP32 保存，所以可以逐位比较：

| batch | channel | 位置 | 输入 x | scale | bias | CPU 与 GPU 相同的答案 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | -3.65625 | -0.75 | -0.28125 | 2.4609375 |
| 0 | 0 | 1 | -2.5 | -0.75 | -0.28125 | 1.59375 |
| 0 | 1 | 0 | -1.34375 | 0.0625 | 0.25 | 0.166015625 |
| 0 | 1 | 1 | -0.1875 | 0.0625 | 0.25 | 0.23828125 |
| 1 | 0 | 0 | 0.96875 | 0.875 | -0.125 | 0.72265625 |
| 1 | 0 | 1 | 2.125 | 0.875 | -0.125 | 1.734375 |
| 1 | 1 | 0 | 3.28125 | -0.25 | 0.40625 | -0.4140625 |
| 1 | 1 | 1 | -3.59375 | -0.25 | 0.40625 | 1.3046875 |

## 检查了什么

1. 输入完全按原父任务的确定性公式重建，形状固定为 N=2、C=2、S=2。
2. 候选 Schedule 原样取自已提交的 12 个 FMA 父算子再审记录，用已发布 v43 生成源码，没有手改 GPU 代码。
3. GPU 执行时，输出预填不能等于正确答案的 NaN，并放置前后两个 guard。
4. 独立参考使用 batch/channel/spatial 三层循环和精确分数运算，检查每个输出；同时检查输入未修改、输出接口与 guard。
5. 取回完整输入/输出位模式后，在本地重新计算。复算得到 0 个输出差异、0 个输入改动。

这个样本的分数结果恰好可用 FP32 表示，不意味着分数参考已经实现任意浮点 FMA。
两个 guard 也不等于完整的 memcheck/racecheck。离线复算验证数值与数组，指针/接口仍是实机 judge 记录。

## 固定来源与实际执行

| 对象 | 固定身份 |
| --- | --- |
| 原 AKA 输入来源 | `387aa7faf521a0b72c994ff15a7638cd7e6a8583` 的 l000075 NCHW plane affine 父任务 |
| 所选输入行 | `batch-sensitive-n2-c2-s2` |
| Compiler 与 Schedule 来源 | `26fbf8f3f2bc691707f82d0e79e7327386861feb`，Compiler v43 |
| 新任务/参考代码 | `ed35501a982a74d45dad26f6ec1851083cdb2073` |
| GPU Infra | `ad7b9009143878fb1548ab521ec9fa787e16040b`，0.17.0 |
| 实际 GPU | NVIDIA B200，compute capability 10.0；broker 分配物理 GPU 0 |
| Broker job | `gpuq-9fd4ca406482`，shared，一个 GPU |
| Run | `open-cake-affine-n2c2s2-v43-b200-correctness-v1-3cb2d72125ce` |

节点原始结果为 `completed / valid`。`frontier_eligible=false`，因为任务没有计时阶段。
只提交了一次，没有重试或换节点；任务完成后 broker job 已释放。已有 daemon/broker 保持运行，未停其他任务或修改旧目录权限。

## 可复查材料

- [完整输入/输出](data/affine-parent-v43-b200-canary-20260906/complete-output.json)
- [节点 run 结果副本](data/affine-parent-v43-b200-canary-20260906/run-result.json)
- [correctness stage receipt 副本](data/affine-parent-v43-b200-canary-20260906/stage-receipt.json)
- [本地独立复算](data/affine-parent-v43-b200-canary-20260906/verification.json)
- [任务说明与复算命令](../examples/gpu/affine_parent_canary/README.md)

这些小文件是节点证据和本次复算的便携副本，不接管原节点的生命周期。
完整 route、task/candidate snapshot、日志和镜像保留在外部实验目录
`/Users/haiyan-infiniai/Documents/Codex/2026-09-06/open-cake-ir-affine-canary/`。

## 能下什么结论

这一个固定样本、这一份候选与 v43 生成代码在 B200 上算对了。
它不证明全部 12 个父算子通过，也不覆盖其他形状、动态宿主 ABI、流/状态返回协议、性能或模型服务。
原 v41 再审记录仍保留“GPU 未运行”，本次通过是独立的新记录，不回写旧结论。
