# 用 Python 编写 Apple Metal 程序

使用现有张量前端编写 Python Schedule，先评估，再检查生成的 Metal 源码。
首个目标为 `apple_gpu_family8`，精确设备为 `Apple M2`；其他 Apple GPU 不会自动作为替代目标。

[逐元素示例](../examples/python/metal_elementwise.py) 在 `(3, 37)` 张量上计算
`(x + y) * 2`；[行求和示例](../examples/python/metal_row_sum.py) 将 `(5, 65)` 归约为
`(5,)`。Python 和 JSON 都进入同一个规范 Schedule，无须手写 Metal。

```python
from open_cake_ir.compiler import Compiler, frontend

compiler = Compiler.load(".", "compiler/revision.lock.json")
source = frontend.read_schedule("examples/python/metal_row_sum.py")
assessment = compiler.assess(source.document)
for finding in assessment.findings:
    print(finding.code, finding.path, finding.message)
    print(source.location_for(finding.path))
if assessment.lowering_eligible:
    lowered = compiler.lower(assessment)
    print(lowered.source)
    print(dict(lowered.toolchain_requirements))
```

Agent 也沿用这个循环：修改 Python 表达式或具体调度决定，读取定位到程序区域的 Findings，
修正对应声明，再检查 lowering，最后才申请设备验证。静态评估不证明 GPU 正确性或速度；
当前未实现的语义与调度承诺会被拒绝。

## 当前执行边界

这是**每个 program tile 只有一个活跃 lane** 的正确性原型。每个线程组启动 32 个线程，
lane 0 串行处理私有 FP32 数据，并按索引递增顺序归约。它尚未实现优化的 SIMD 归约。
后端支持其建模范围内可组合的 FP32 算术与有限输入 sum/max 归约，并限制私有存储量和
全局缓冲区数量。示例以一行作为一个 program tile，store 显式设置 `coalesced=False`，
reduce 显式设置 `across_loop=False`。支持奇数行宽，无须调用方补齐；更宽的 program tile、
跨循环归约、共享内存协作、张量操作以及未支持的访存承诺会被拒绝。

[Swift 适配器](../tools/metal/runner.swift) 通用地消费 lowering 的缓冲区顺序与启动元数据；
[Python 边界](../tools/metal/adapter.py) 从已评估的 Schedule 推导类型、形状和字节数。
适配器检查精确设备、GPU family 和 pipeline 限制，通过 `MTLDevice.makeLibrary` 编译源码，
将输出初始化为 NaN，等待命令完成并保留输入输出字节。需要 macOS 15+、Apple Silicon 和
已有的 `xcrun swiftc`，不依赖独立的 `xcrun metal` 工具。

编译显式采用 MSL 2.3、安全数学和精确数学函数，生成源码关闭 contraction。Metal 仍允许
非规格化数清零及不同 FP32 舍入模式；当前 oracle 以有限正常范围输入和零值、明确容差验证，
不宣称 IEEE/PTX 位级等价。参见 Apple 的[编译选项](https://developer.apple.com/documentation/metal/mtlcompileoptions)
及 [MSL 规范第 1.6.3、8 节](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)。

## 正确性证据

适用的发布门禁通过、且获得本机 GPU 执行授权后，在 checkout 中运行：

```sh
env PYTHONPATH=src python3 tools/metal/check_correctness.py \
  --output-root /absolute/external/metal-checks
```

输出根目录必须是所有项目 worktree 之外的绝对路径。每次调用创建新的 receipt 目录，
记录已提交且清洁的运行时源码身份，验证已获独立审查的 released Compiler，再通过公共
`assess`/`lower` 路径检查 60 个组合：逐元素、行求和与同一 Python 归约模板的 max 变体，
五种形状（行宽 1、7、32、65、257）及四类确定性输入分布。
外部 CPU oracle 检查输出长度、有限值和误差；字节比较检查输入未被修改。
失败返回非零退出码并保留记录，静态结果、编译与命令完成、输出正确性分别记录。
结果仅覆盖列出的本机输出样例，不代表计时、profiler、框架集成或端到端性能验证。

以下可移植主机契约测试不提交 GPU 工作：

```sh
env PYTHONPATH=src python3 -m unittest tests.contracts.test_metal_runtime
```

本阶段没有经过校准的 Apple 计时模型。应读取编译器报告的覆盖限制，不能将缺失估计解释为
某项资源没有成本。
