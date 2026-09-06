# 无 GPU 的编译资源分析

`Compiler.profile` 和 `tools/report_schedule_profile.py` 可以在运行 kernel 之前提供
资源反馈。普通静态分析只需要 Python 3.10+；编译资源采集需要支持目标的 Triton 和
CUDA binary utilities，但无需 GPU，也不会初始化 Triton kernel handle。
已经采集的报告可以连同产物带到 macOS 等无 CUDA 环境复用。

## 三种输入阶段，同一份反馈

| 输入 | 可以得到什么 | 仍然不能得到什么 |
| --- | --- | --- |
| Schedule | 运算量、声明资源上界、同步和索引访问风险 | 后端寄存器和隐式共享内存 |
| 编译后的 CUBIN 与 launch 元数据 | 寄存器、静态／动态共享内存、栈和 local allocation，以及更紧的驻留上界 | 实际占用率、stall 百分比、缓存命中、运行时间 |
| 已保留的报告与对应产物 | 不加载 CUDA、Triton、PyTorch 的本地复用 | 另一套工具链重新编译后的资源分配 |

寄存器数量来自 `cuobjdump --dump-resource-usage`，动态共享内存来自编译元数据。
两者都不需要加载或执行 GPU kernel。`STACK` 和 `LOCAL` 保留 binary utility 给出的
字节含义，不能当作运行时 spill 次数或内存流量。
[NVIDIA binary utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)
定义了这些输出字段。

驻留分析仍报告**上界**。实际寄存器和共享内存分配改善了输入，但分配粒度、driver
保留空间、shared-memory carveout、barrier、隐式 tensor memory 和调度可能进一步
降低并发度。不能把上界直接读成耗时或实际利用率。

`work.scheduled_transfer_payload` 另外计算显式 LOAD/STORE 在整个计划中的逻辑载荷，
包含每个分块和循环执行次数，并保留逐操作明细。例如 GEMM 的 A 会随 N 分块重复读取，
B 会随 M 分块重复读取；只统计输入 Buffer 的大小看不到这些复用成本。
访问掩码可能减少载荷，所以这里使用上界；persistent 计划按遍历的工作块计数，动态
停止循环使用真实可推导的 trip 分布。无法建模的访问（如原子读改写）明确返回未知。
这个数字**不包含后端复制、合并、缓存命中和事务粒度的效果**，不是实际 L2/DRAM
字节数，也不能直接除以 HBM 带宽当作运行时间。终端的 `IR read MiB<=` 显示该范围。

## 使用

以下命令在仓库根目录运行，使用已经满足依赖的 Python 环境。输入既可以是 JSON，
也可以是由当前 Python 前端接受的 `.py` Schedule。

只做静态分析：

```bash
python3 tools/report_schedule_profile.py --json \
  corpus/schedules/gemm-bias-b1-smoke.json
```

在具备工具链的 Linux 环境中编译和检查；输出目录必须是仓库外的新目录：

```bash
CUDA_VISIBLE_DEVICES= python3 tools/report_schedule_profile.py \
  --compile-to /tmp/cake-compiled-profile \
  --cuobjdump /usr/local/cuda/bin/cuobjdump \
  corpus/schedules/gemm-bias-b1-smoke.json
```

工具保留 `report.json`，以及按报告行编号的 `0000/lowered.py`、`kernel.ptx` 和
`kernel.cubin`。本轮采集支持 `sm_100a`、不使用多 CTA cluster 的 Triton 路线；CuTe DSL、NVCC 路线和
辅助 global scratch 尚未接入这个采集器。批量报告为不支持的后端保留静态分析，并
明确列出未采集的范围；编译错误仍会中止采集，不替换目标或伪造数值。

把整个输出目录带到无 GPU、无 CUDA 工具链的机器后，可复用原编译观察：

```bash
python3 tools/report_schedule_profile.py --json \
  --compiled-report /tmp/cake-compiled-profile/report.json \
  corpus/schedules/gemm-bias-b1-smoke.json
```

复用会核对保留的源码和 CUBIN 身份，再匹配当前生成源码、目标、kernel entry 与
launch 线程数；不按 Schedule 显示名称猜测匹配。记录携带原编译器与检查器版本。
它是该工具链和该二进制的观察，不保证另一套编译器会分配相同资源。
报告的数值来源是受信任的采集器；身份核对不等于对外部篡改数值的独立机器认证。

同一观察可以进入已有 agent 反馈：

```bash
python3 tools/project_qsa_feedback.py compiler \
  --revision compiler/revision.lock.json \
  --compiled-report /tmp/cake-compiled-profile/report.json \
  corpus/schedules/gemm-bias-b1-smoke.json
```

Python 调用通过 `compiler.profile(assessment, compiled_resources=resources)`。
公开方法先重放 Assessment，防止把修改过的 Schedule 与旧编译观察组合。
省略 `compiled_resources` 就只做静态分析；不会隐式申请 GPU 或启动编译工具链。

## 与 cost model 的关系

这条路径补充后端真实资源特征，帮助定位瓶颈、比较候选和建立后续耗时模型。
它保留已否定的 wave 假设和旧校准结论，未改变公开排序的校准覆盖。
耗时、吞吐率和 stall 比例的数值预测需要独立的目标校准；最终候选仍由外部正确性及
匹配实测验收。当前工作继续推进有适用范围和误差证据的性能估算，而不把资源上界
直接作为运行时间。
