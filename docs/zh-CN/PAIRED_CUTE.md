# 在 B300 上比较 Cake 与原生 CuTeDSL

[English](../en/PAIRED_CUTE.md)

使用 `contracts/studies/matched-search-cute-b300-gemm-optimization-template.json`。
它沿用现有 matched_search 和 Ralph 流程。两组分别是 `open_cake` 和
`native_cute_dsl`，从同一个 GEMM+bias kernel 开始，都用 `cutlass_cute_dsl`
编译到精确的 `sm_103a`。Study 保存稳定规则；实际 Compiler、Executor、provider、
运行配置和固定基线只绑定到外部 CampaignLock。新增模板不等于获准启动六次科学实验。

数学任务是 `C = A @ B.T + bias`。A、B 为 BF16，bias、输出为 FP32。
现有 `gemm-bias-bf16-fp32-v2` Workload 独立拥有 oracle 和误差要求
`atol = rtol = 0.001`。五组形状 M×N×K 为 512×256×256、3×5×65、
2×3×65、1×2×65、2×3×66，分别检查主形状、尾部、小输入、零值和抵消输入。
Compiler 的主 Schedule 不伪造 Workload 哈希，Lab 在使用时绑定真实元数据。

## 编写候选

完整 Python 例子在 `examples/python/b300_cute_gemm_bias.py`。普通 Compiler CLI
可检查它：

```sh
PYTHONPATH=src python -m open_cake_ir.cli compiler assess --revision compiler/revision.lock.json examples/python/b300_cute_gemm_bias.py
```

Cake 组提交 JSON Schedule，或含 `python_source` 的对象。受限 Python 前端只解析，
不运行用户的主机代码，并保留报错位置。后端固定为 `cutlass_cute_dsl`。
首版支持一个 warp、BM=16、BN 为 8 的倍数、BK 为 16 的倍数、K>BK，以及单级 K 循环。
M/N/K 尾部有掩码，无效输入在集体 MMA 前补零。它显式使用寄存器和 BF16 warp MMA。
不支持寄存器上限或驻留率承诺；`coalesced=False` 不承诺合并写出。

原生组修改 TASK 中的 `candidate-baseline.cute.json`。候选只有四个字段：
`kernel_source`、`grid`、`block`、`dynamic_shared_memory_bytes`。
附带的 `candidate.schema.json` 说明格式。两组都写现有 `candidate-set.json` 信封，
保留分配的 arm。生成源码中的来源注释只说明初始基线，可以在原生组优化其 kernel。

CuTe 源码只允许固定的 cutlass、cute、warp 导入和一个 `@cute.kernel`。
四个有序 `cute.Pointer` 参数来自 Workload：A、B、bias、输出。禁止新增导入、
辅助函数、任意 Python 调用、主机编译与主机启动。编译前检查整个模块。
block 必须为 `[32,1,1]`，动态共享内存必须为 0。grid 可以改为三个正整数，
但是否覆盖所有输出由外部正确性检查决定。额外编译选项会被拒绝。

## 运行与证据

运行配置仍是 provider、toolchain、broker 三部分。CuTe 的 toolchain 只接受
`python`、`bubblewrap`、`runtime_roots`、`cuobjdump`、`cutlass_version`、
`timeout_seconds`；[英文说明](../en/PAIRED_CUTE.md)给出完整字段示例。
填写实际部署路径，SDK 固定为 4.5.2。Executor 要同时绑定 nvidia-cutlass-dsl、
nvidia-cutlass-dsl-libs-base、nvidia-cutlass-dsl-libs-cu13 三个包。
Linux bubblewrap 编译环境不挂载作者工作区，不暴露 NVIDIA 设备，拒绝驱动初始化。
缺少依赖会报告环境失败，不会切换架构或绕过隔离。

`tools/qualify_codex_provider.py` 配合
`contracts/providers/codex-cute-optimization-output-schema-v1.json`，通过两轮零 GPU
任务检查两组的实际 provider 信封。它不验证 kernel 正确性。随后将 qualification
receipt、anchor、运行配置和已封存基线作为外部执行绑定交给
`lab preflight --execution-bindings`。所有新 lock、证据和报告放在源码目录外。

两组共用 `open_cake_ir.lab.cute_build` 的 `CuTeToolchainBuilder` 和
`IsolatedCuTeCompiler`。最终 launch manifest 使用 SDK 实际生成的符号；此 GEMM Study 只有四个
指针，没有隐藏参数。原始提交、源码、诊断与产物进入现有追加式 archive。
GPU 上继续使用统一 oracle、冷 L2 的配对 CUPTI 计时和单独 profiler。
源码测试通过或 CPU 编译成功都不等于 GPU 正确、性能合格或 IR 更优。


## 通用 FP32 SIMT lowering

无 MMA 的 Schedule 现在可选择 `cutlass_cute_dsl`，由操作结构进入单 warp
SIMT 路线。支持 FP32 load/store、add/sub/mul/div、square/relu/rsqrt/exp/exp2/
reciprocal/tanh、显式广播和单 tile sum/max reduction。目标必须精确匹配
`sm_100a` 或 `sm_103a`。这条路线可表达多输入和多输出；已有 BF16 register-MMA
GEMM+bias 的研究合同仍由上文的 Workload 和 Study 管理。

Schedule 使用一个 `warps=[0]` role、tile=1 的非 persistent ProgramMap、global
输入/输出和 register 中间值，以及 PROGRAM/DIMENSION AccessMap。元素 i 放在
lane i%32、slot i//32。设归约轴之后各维度的乘积为 `inner`。当 `inner` 是 32
的倍数时，每个归约项已与对应输出处在同一 lane。后端生成按输出 slot 和归约长度
嵌套的 constexpr 循环，源 slot 为
`(outslot // (inner//32)) * extent * (inner//32) + outslot % (inner//32) + k * (inner//32)`。
每个 lane 按 `k` 递增顺序合并，sum 从 FP32 零开始，max 从负无穷开始。该分支
无需跨 lane 归约，也不会为每个输出元素展开一套标量归约。它直接由固定条带映射
推导，不增加 Schedule 控制项、变换 pass 或可选归约算法。`CUTE_SIMT_EXECUTION`
诊断说明两条路径，局部归约的源码注释记录尾部跨度。其他跨度（包括 31 和 33）
保留本 lane 合并后执行全 warp collective 的原路径。跨 lane 广播对所有 source
slots 一致地 shuffle，尾部 lane 仍参与这些 collective。全局拷贝为
标量，store 声明 `coalesced=false`。不支持的 cache/reuse、residency、pipeline、
barrier、循环及存储声明会被拒绝。逻辑 live slots 上限是实现边界，不是物理寄存器
或 spill 的预测。

编译接口接受完整、有序的 FP32 pointer signature，并核对每个 PTX/CUBIN 参数的
数量、偏移、类型对齐、地址空间及精确目标。源码准入仍限制为固定 imports 和单个
kernel。CPU source-model 测试检查广播、各轴归约、跨 lane slot 和输出写入；它不
模拟 CuTe 编译器、GPU 数学近似或物理资源分配。此能力的实际 SDK 编译、GPU
correctness、timing 和 profiler 验收仍需各自的证据，不能由静态/CPU 测试代替。
