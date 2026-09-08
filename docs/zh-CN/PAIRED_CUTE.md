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
`IsolatedCuTeCompiler`。最终 launch manifest 使用 SDK 实际生成的符号，只有四个
指针，没有隐藏参数。原始提交、源码、诊断与产物进入现有追加式 archive。
GPU 上继续使用统一 oracle、冷 L2 的配对 CUPTI 计时和单独 profiler。
源码测试通过或 CPU 编译成功都不等于 GPU 正确、性能合格或 IR 更优。
