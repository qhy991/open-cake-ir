# 原生 CUDA/PTX：把执行计划直接变成 GPU 代码

[English](../NATIVE_CUDA.md) · [具体设计与范围](../NATIVE_CUDA_DESIGN.md) · [中文首页](README.md)

`native_cuda` 根据 Schedule 自己生成 CUDA C++ 和 PTX。它负责线程分工、共享内存地址、
张量内存、搬运描述符和同步。NVIDIA 的 nvcc/ptxas 继续负责编译。B200 使用 `sm_100a`，
B300 使用 `sm_103a`；两者分别编译和评测。

如果所有候选都装得进一个 tile，就省去候选循环，并写 `across_loop=false`。
此时只支持从零开始的完整候选范围，因此 tile 内下标就是全局候选下标。
原生 CUDA 和 Triton 共用范围推导：4 个真实候选即使放在 64 列 tile 中，补零的列也不能获选。

可以把一个 stage 理解为装数据的格子。搬运组先等格子空闲，再填入数据；矩阵乘组等待
数据到齐，发出矩阵乘指令；等全部读这个格子的矩阵乘完成后，才把格子还给搬运组。
两个组可以同时工作。格子的数量由 Schedule 决定，编译器不能用逐步串行等待冒充流水线。

## 可以直接生成代码的例子

例子在 `examples/schedules/native/`：

| 文件 | 计算和硬件分工 |
| --- | --- |
| `gemm-bias.json` | 矩阵乘后加偏置，使用 TMA 搬运和 tcgen05 矩阵乘 |
| `gemm-bias-k65-tail.json` | K=65 的尾部，显式在 GPU 中补零并写入共享内存 |
| `two-mma.json` | 两次独立矩阵乘，各自保留累加器，再明确相加 |
| `kmeans.json` | 分批算距离，跨 tile 保留最小距离的下标，平局取较小下标 |
| `kmeans-partials.json` | 两次矩阵乘分别算不同 K 区间，各自读回结果，再明确相加 |

这些例子共用操作生成器。K65 的连续 BF16 行宽不满足 TMA 的对齐要求，因此例子明确选择
普通 global→shared 搬运。没有在主机端偷偷补齐输入，也没有把搬运成本移出 kernel。

KMeans 的核心例子还接收 `centroid_sq`，沿用现有运行时的准备接口。评测必须说明这个
平方和怎样生成、是否计入成本；核心 kernel 的时间不等于原始“两输入任务”的完整时间。
原始 Workload 仍要求下标完全相等，距离“足够接近”不能代替正确下标。

新例子仍搬运完整的 K64 数据块。第一个 MMA 写
`k_ranges=[[0,16],[32,48]]`，第二个写 `[[16,32],[48,64]]`。
每对数字表示当前输入块里的半开区间：包含起点，不包含终点。两个输入使用相同的坐标。
区间必须有序、不重叠且不越界；相邻区间会合并，完整覆盖会变回省略字段的旧形式。
原生指令还要求端点对齐到已声明的指令 K 步长。即使第一个区间从 16 开始，
它也必须初始化自己的累加器。两个结果使用独立的张量内存、完成信号和读回操作，
全部读者完成后才允许复用输入格子。

对应的 `examples/schedules/triton/kmeans-partials.json` 保留完整 K128 输入，
从中分别选出四段偶数/奇数 K16 区间，执行两次 K64 BF16 dot，再把两个 FP32 结果相加。
每个结果经过一条不透明的寄存器移动，保留独立舍入边界，防止后续编译器把相加吸收到
第二次 dot 的累加器里。首批只支持 NVIDIA `sm_100a`/`sm_103a`、BF16、K128 输入、
64 个选中元素，以及至少 16 且为 2 的幂的 M/N。不支持沿 K 循环累加；其他选中长度、
指令契约及 CuTe/Metal 会明确拒绝。原有候选范围规则继续约束 argmin。

两部分合起来的乘加工作量与完整收缩相同，另计明确写出的 ADD。搬运量不变，第二份
累加器和寄存器占用仍会计入分析。旧诊断程序算对过，不代表这份新生成程序已经算对。
新程序必须重新通过原始下标检查，之后才能进入计时和 profiler 验证。

## 从公共接口生成

在项目目录中执行下面的命令。输出目录必须已经存在，并且位于 checkout 外：

```sh
PYTHONPATH=src python3 -m open_cake_ir.cli compiler assess \
  --revision compiler/revision.json examples/schedules/native/gemm-bias.json
PYTHONPATH=src python3 -m open_cake_ir.cli compiler lower \
  --revision compiler/revision.json examples/schedules/native/gemm-bias.json \
  --output /absolute/external/output/gemm-bias.cu
```

这里明确使用开发草稿。独立批准并发布后，改用 `compiler/revision.lock.json`。
生成代码不会启动 GPU。生成物的 `toolchain_requirements` 给出真实参数顺序、线程数、
共享内存字节数和精确编译参数；`source_map` 把生成行映射回操作与资源声明。
具体 Python 调用和 C 接口见[英文接口说明](../NATIVE_CUDA.md#host-interface)。

编译必须使用生成物给出的 `--gpu-architecture=compute_100a --gpu-code=sm_100a`，
或对应的 103a 配对。不要改用可能附带普通架构代码的 `-arch` 简写。

## 怎样理解通过与拒绝

公开测试入口是：

```sh
PYTHONPATH=src python3 -m unittest tests.contracts.test_emit_cuda -v
```

它检查源码生成、双 MMA、尾部、stage/tile/角色变化和局部拒绝。首期一个 epilogue 线程
负责一行，store 如实声明 `coalesced=false`。不支持的布局、角色预算、缓存/reuse 提示、
循环选项或精度会返回带位置的 Finding，不会静默忽略或换成另一条指令。算术只读写
FP32 寄存器值，全局输入必须先 load；标量必须能表示为有限 FP32。store 只支持每线程
独占一行的矩阵或逐行 argmin 结果，地址必须包含全部会变化的 ProgramMap 轴。
复制给多个线程的寄存器向量没有唯一写者，当前明确拒绝直接存储。

评测工具的 CPU 合同测试是
`PYTHONPATH=src python3 -m unittest discover -s tests/native_cuda -v`，需要 NumPy 和
CPU PyTorch。它检查参数、保存的输入、精确下标、准备值，以及模拟的计时和 profiler
进程边界。`tools/native_cuda_evaluate.py` 使用 `prepare`、`compile`、`run`、`verify`
四步，共用 checkout 外的 `--evidence-root`。当前 12 个规定用例绑定 `sm_103a`；
准备阶段要求已发布 Compiler，实际编译和设备运行另需对应授权与环境。

KMeans 的准备检查从保存的 BF16 质心和 FP32 平方和字节恢复精确数值。
它先确认每项平方可在正常 FP32 范围内精确表示，且求和不会溢出，再检查可精确求和
情形或由 FP32 加法推导的误差界；超出模型范围会明确拒绝。该检查不要求 CPU/GPU
采用相同求和树，也不证明某个 CUDA 求和顺序。输入生成方法与精确 INT32 下标 oracle
保持原合同。验证器变化后需要新的源码绑定 GPU 证据，不能把旧运行改记为通过。

检查通过、编译成功、GPU 算对、计时有效、profiler 有证据是不同结果。完整 Workload 的
所有规定输入仍要经过外部 oracle。当前交付状态由 checkout 外的 delivery ledger 负责，
本页不保存第二份进度或性能结论。
