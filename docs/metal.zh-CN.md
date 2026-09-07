# 用 Python 编写和测量 Apple Metal 程序

编写 Python Schedule，读取定位到程序区域的编译器反馈，再检查生成的 Metal。
Python 与 JSON 使用同一个规范 Schedule。

| 精确设备 | Schedule 的 `target` 与命令行 `--target` |
| --- | --- |
| Apple M1 Pro | `apple_gpu_family7` |
| Apple M2 | `apple_gpu_family8`（命令行默认值） |

运行时同时检查设备名与 Metal family，同 family 的其他型号也不会自动成为替代目标。
示例默认声明 M2，下面演示如何选择 M1 Pro。

[逐元素示例](../examples/python/metal_elementwise.py)、[行归约](../examples/python/metal_row_sum.py)
和[带权重 RMSNorm](../examples/python/metal_rmsnorm.py) 都使用现有张量前端。
RMSNorm 由 square、sum、标量算术、rsqrt 和乘法组合而成：

```python
from open_cake_ir.compiler import Compiler, frontend

compiler = Compiler.load(".", "compiler/revision.lock.json")
source = frontend.read_schedule("examples/python/metal_rmsnorm.py")
document = {**source.document, "target": "apple_gpu_family7"}
assessment = compiler.assess(document)
for finding in assessment.findings:
    print(finding.code, finding.path, finding.message)
    print(source.location_for(finding.path))
if assessment.lowering_eligible:
    lowered = compiler.lower(assessment)
    print(lowered.source)
    print(dict(lowered.toolchain_requirements))
```

Agent 修改公式或具体调度决定，读取 Findings，修正对应声明，再检查下一次 lowering。
Source map 将生成代码中的操作关联回 Schedule。静态接受、成功编译、输出正确性和测量资格
分别记录，不互相替代。

## 执行和数值范围

当前 lowering 将每个展平元素分配给 `index % 32` 对应的 lane，私有槽位为 `index // 32`。
32 个 lane 均参与受支持的 SIMD 集合操作，包括尾部；program 坐标继续拥有互不重叠的输出区域。
`(1,)` 标量结果可以参与张量算术，奇数宽度无须调用方补齐。编译器检查其支持的访存、形状、
存储、广播与归约承诺，对未实现的声明返回定位明确的拒绝。每 lane 存储分析只是建模结果，
不代表实测寄存器、spill、驻留量或带宽。

[通用 Swift runner](../tools/metal/runner.swift) 消费当前 SIMD 启动元数据及明确的串行参考/回放接口，
不依据算子名称选择执行逻辑。[Python 边界](../tools/metal/adapter.py) 从已评估的 Schedule 推导
缓冲区形状、类型、大小和启动信息。Runner 检查精确设备与 pipeline 限制，在同一进程复用
设备、队列、pipeline 和缓冲区，在计时外重置输出为 NaN，采用串行 dispatch 顺序，检查完成状态与输入未被修改。

源码通过 `MTLDevice.makeLibrary` 编译，显式使用 MSL 2.3、安全数学、精确数学函数并关闭
contraction。需要 Apple Silicon、macOS 15+ 和已有的 `xcrun swiftc`，不依赖独立的 `xcrun metal`。
Metal 仍允许非规格化数清零和不同 FP32 舍入行为，不能据此宣称 IEEE/PTX 位级等价。
参见 Apple [编译选项](https://developer.apple.com/documentation/metal/mtlcompileoptions)及
[MSL 规范](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)。
主机可执行文件统一使用 `swiftc -O` 构建一次、供所有比较臂复用；
`swift-build-command.json` 记录实际构建参数。Metal 数学选项保持独立。

## 正确性和测量

独立发布审查与适用的本机 GPU 授权完成后，在 checkout 中运行；输出根目录必须是所有项目
worktree 之外的绝对路径：

```sh
env PYTHONPATH=src python3 tools/metal/check_correctness.py \
  --target apple_gpu_family7 --output-root /absolute/external/metal-correctness
env PYTHONPATH=src python3 tools/metal/benchmark.py \
  --target apple_gpu_family7 --output-root /absolute/external/metal-measurements
```

每次调用先创建新的外部 receipt，再验证已提交、清洁的运行时源码，以及已审查发布且绑定所选目标的
Compiler；通过后才构建主机 runner 或提交 GPU 工作。已发布版本若未绑定该目标，须按
[Compiler 发布流程](adr/0052-independent-agent-release-review.md)产生后继版本。

正确性覆盖 60 个逐元素/sum/max 组合和 35 个 RMSNorm 组合。
[RMSNorm 契约](../tools/metal/rmsnorm.py) 统一拥有公式、输入范围、种子、epsilon、容差和形状，
覆盖零值、受限正常范围输入、epsilon 主导的小输入、负权重与零权重，以及宽度
1、7、32、65、257、1024、4096。独立高精度 CPU oracle 的 RMSNorm 容差固定为 `atol=rtol=2e-5`。

[测量协议](../tools/metal/benchmark.py) 在固定主形状 `(128, 1024)` 上比较四种数学等价公式 DAG：
canonical、weight first、prescaled square 和 scale weights first。
不同 FP32 计算顺序须通过同一 oracle 容差，任何一种都不预设加速。
手写串行与 SIMD 参考具有明确源码来源，使用相同输入字节和 oracle。
本轮属于允许查看已知实现的复现/优化。

所有 treatment 在同一进程中准备。仅使用参考程序的 pilot 从有限的 2 次幂中选择批量 dispatch 数，
随后固定这个数量，完成一次随机匹配搜索和两次独立匹配确认。A/A 对照使用独立构建的同源参考
pipeline/缓冲区，经过与 A/B 相同的选择、绑定和调度路径。数据保持 warm，不执行 cache flush，
也不继承 NVIDIA CUPTI 测量语义。

| 字段 | 实际区间 |
| --- | --- |
| 冷构建 | Swift 主机构建、设备/队列创建、主机准备及 library/pipeline 构建；可能命中系统缓存 |
| Warmed host call | encode、提交到完成等待；不含输出重置、oracle 检查和文件 I/O |
| GPU command buffer | 命令完成后的 `GPUEndTime - GPUStartTime` |
| 摊销 dispatch | command-buffer 区间除以实际 dispatch 数，不是纯 kernel latency |

预先固定的工程资格规则要求 A/A 配对中位数比率位于 `[0.95, 1.05]`，相关各臂相对 IQR
不超过 10%，且搜索与两次确认中的增益均超过 5%；否则报告 inconclusive 或无实质增益。
保留全部原始样本、顺序、warmup 和批量数量，不裁剪样本。计时器、执行或正确性失败返回非零退出码。
每个 pilot 和普通批量样本都附带自身的输出/输入验证；验证在计时外、下一次 dispatch 覆盖缓冲区前完成，
并与 profile 的验证分开记录。
参见 Apple 的 [GPU command-buffer 时间戳](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime)。

普通计时结束后，单独采集设备支持的 compute-stage `GPUTimestamp` 观察，记录实际能力枚举、
解析出的原始计数和缺失/失败原因。计数不转换成主机时间，也不称作 kernel cycles；不推导
occupancy、带宽或指令计数。参见 [Apple counter 采样](https://developer.apple.com/documentation/metal/sampling-gpu-data-into-counter-sample-buffers)。

`feedback.json` 提供候选处置、定位明确的 Findings、搜索/确认结果，并复用现有 Lab 的拒绝路由词汇。
两个 Apple 目标均在 GPU dispatch 前保留成本模型未排名原因：没有经过校准的 Apple ranker，
不估计 occupancy、物理寄存器、spill、带宽或延迟，也不推断 ranking inversion。这些是本机产物评估，不构成
完整 Study/provider campaign、框架集成、serving 或端到端结果。以下可移植测试不提交 GPU 工作：

```sh
env PYTHONPATH=src python3 -m unittest \
  tests.contracts.test_metal tests.contracts.test_metal_runtime tests.contracts.test_metal_benchmark
```

`test_metal` 在存在 C++ 编译器时，用原生 C++ adapter 验证生成代码体的 CPU 语义；
这些测试不证明 Metal 编译、设备正确性或性能。编译器变更的审查依据与有界迭代计划见
[M1 Pro 后继版本设计](metal-m1-pro-design.md)。
